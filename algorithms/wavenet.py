import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(CustomConv1d, self).__init__()
        k = np.sqrt(1 / (in_channels * kernel_size))
        weight_data = -k + 2 * k * torch.rand((out_channels, in_channels, kernel_size))
        bias_data = -k + 2 * k * torch.rand((out_channels))
        self.weight = nn.Parameter(weight_data, requires_grad=True)
        self.bias = nn.Parameter(bias_data, requires_grad=True)
        self.dilation = dilation
        self.padding = padding

    def forward(self, x):
        xp = F.pad(x, (self.padding, 0))
        return F.conv1d(xp, self.weight, self.bias, dilation=self.dilation)

class wavenet_cell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(wavenet_cell, self).__init__()
        self.conv_dil = CustomConv1d(in_channels, out_channels * 2, kernel_size, padding, dilation)
        self.conv_skipres = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, x):
        h_prev, skip_prev = x
        f, g = self.conv_dil(h_prev).chunk(2, 1)
        h_next, skip_next = self.conv_skipres(torch.tanh(f) * torch.sigmoid(g)).chunk(2, 1)
        
        return (h_prev + h_next, skip_prev + skip_next)

class wavenet(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, N, kernel_size):
        super(wavenet, self).__init__()
        # Embedding layer for time series ID
        if len(d_emb)>0:
            self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
            d_emb_tot = d_emb[:, 1].sum()
        else:
            self.emb = nn.ModuleList([])
            d_emb_tot = 0
        self.upscale = nn.Linear(d_lag + d_cov + d_emb_tot, d_hidden)
        # Wavenet
        wnet_layers = nn.ModuleList([wavenet_cell(
                    d_hidden, d_hidden, 
                    kernel_size, padding=(kernel_size-1) * 2**i, 
                    dilation = 2**i) for i in range(N)])  
        self.wnet = nn.Sequential(*wnet_layers)
        # Output layer
        self.loc = nn.Linear(d_hidden, d_output)
        self.scale = nn.Linear(d_hidden, d_output)
        self.epsilon = 1e-6
        
    def forward(self, X_emb, X_cov, X_lag, d_outputseqlen, domain=None):  
        dim_seq = X_lag.shape[0]

        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(X_emb[:, :, i])
            x_emb.append(out)
        if self.emb:
            x_emb = torch.cat(x_emb, -1)
        # Concatenate inputs

        if self.emb:
            h = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            h = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)

        h = self.upscale(h)
        # Apply wavenet
        _, h = self.wnet((h.permute(1, 2, 0), 0))
        # Output layers - location & scale of the distribution
        output = h[:, :, -d_outputseqlen:].permute(2, 0, 1)

        y_hat_loc = self.loc(output)
        y_hat_scale = F.softplus(self.scale(output)) + self.epsilon
        return y_hat_loc, y_hat_scale 