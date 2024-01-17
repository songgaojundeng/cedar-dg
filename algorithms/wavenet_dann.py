import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import  Normal
from torch.autograd import Function

"""
adapted from 
wavenet and https://github.com/fungtion/DANN
"""


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha


# This implementation of causal conv is faster than using normal conv1d module
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

class wavenet_dann(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_dim, d_output, d_hidden, N, kernel_size):
        super(wavenet_dann, self).__init__()
        if len(d_emb)>0:
            self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
            d_emb_tot = d_emb[:, 1].sum()
        else:
            self.emb = nn.ModuleList([])
            d_emb_tot = 0
        self.upscale = nn.Linear(d_lag + d_cov + d_emb_tot, d_hidden)
        # feature 
        N_feature = N 
        feature_layers = nn.ModuleList([wavenet_cell(
                    d_hidden, d_hidden, 
                    kernel_size, padding=(kernel_size-1) * 2**i, 
                    dilation = 2**i) for i in range(N_feature)])  
        self.feature_model = nn.Sequential(*feature_layers)

        forecast_layers = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        ])  
        self.forecast_model = nn.Sequential(*forecast_layers)

        # Output layer
        self.loc = nn.Linear(d_hidden, d_output)
        self.scale = nn.Linear(d_hidden, d_output)
    
        self.domain_classifier = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU(),nn.Linear(d_hidden, d_dim))

        torch.nn.init.xavier_uniform_(self.domain_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[2].weight)

        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, X_emb, X_cov, X_lag, y, d_outputseqlen, domain, alpha=1.0):  
        dim_seq = X_lag.shape[0]
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(X_emb[:, :, i])
            x_emb.append(out)
        if self.emb:
            x_emb = torch.cat(x_emb, -1)
        if self.emb:
            h = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            h = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)

        h = self.upscale(h)
        _, feat = self.feature_model((h.permute(1, 2, 0), 0))
        feat = feat.permute(2, 0, 1)

        reverse_feature = ReverseLayerF.apply(feat, alpha)

        feat_forecast = self.forecast_model(feat)

        y_hat_loc = self.loc(feat_forecast[-d_outputseqlen:])
        y_hat_scale = F.softplus(self.scale(feat_forecast[-d_outputseqlen:])) + 1e-6


        if self.training:
            d = domain
            _, d_target = d.max(dim=1)
            d_hat = self.domain_classifier(reverse_feature) 
            d_hat = d_hat.mean(0)
            loss_d = F.cross_entropy(d_hat, d_target, reduction='mean')
        else:
            loss_d = 0
        loss_d = 0
        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = - y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()

        total_loss = loss_d + loss_y
        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale