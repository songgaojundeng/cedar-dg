import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from lib.mmd import MMD_loss

"""

"""

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

class wavenet_cedar(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, N, kernel_size, mmd_type='rbf', gamma=1.0, loss_dis='abs', reg='dd'):
        super(wavenet_cedar, self).__init__()
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

        if torch.cuda.is_available():
            self.cuda()

        if mmd_type == 'linear':
            self.mmd_loss = MMD_loss('linear')
        elif mmd_type == 'rbf':
            self.mmd_loss = MMD_loss('rbf')
        self.gamma = gamma
        if loss_dis == 'abs':
            self.loss_dis = torch.abs
        else:
            self.loss_dis = torch.square
        
        self.reg = reg
 
    def forward(self, X_emb, X_cov, X_lag, y, d_outputseqlen, domain=None):  
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
        h_pred = h[:, :, -d_outputseqlen:].permute(2, 0, 1)
        y_hat_loc = self.loc(h_pred)
        y_hat_scale = F.softplus(self.scale(h_pred)) + self.epsilon

        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = -y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()
        loss_y_tensor = -y_hat_distr.log_prob(y[-d_outputseqlen:]).mean(0) # (batch, 1)

        if self.training:
            uni_domains = torch.unique(domain)
            n_uni_domains = uni_domains.size(0)
            
            domains_rep_tensor = []
            domains_loss_tensor = []
            domains_loss_std_tensor = []
            for domain_i in uni_domains: # for each domain_i, get avg embedding
                indices = (domain == domain_i).nonzero(as_tuple=True)[0] # e.g. [0,1] torch.LongTensor
                domain_rep = h_pred[:,indices].mean(1) # (d_outputseqlen, hid)
                domains_rep_tensor.append(domain_rep)
                domains_loss_tensor.append(loss_y_tensor[indices].mean())
                if self.reg == 'ddd':
                    domains_loss_std_tensor.append(loss_y_tensor[indices].std(unbiased=False))

            domains_rep_tensor = torch.stack(domains_rep_tensor,dim=0) # (uni_dom, d_outputseqlen, N*hid)  
            domains_loss_tensor = torch.stack(domains_loss_tensor,dim=0) # (uni_dom)
            
            domain_mmd_matrix = self.mmd_loss(domains_rep_tensor.view(n_uni_domains,-1), domains_rep_tensor.view(n_uni_domains,-1), return_matrix=True)
            # Expand dimensions to facilitate broadcasting
            expanded_losses = domains_loss_tensor.unsqueeze(1)
            transposed_losses = expanded_losses.transpose(0, 1)
            # Compute the pairwise differences using broadcasting
            pairwise_loss_diff_matrix = self.loss_dis(expanded_losses - transposed_losses)
            
            if self.reg == 'ddd':
                domains_loss_std_tensor = torch.stack(domains_loss_std_tensor,dim=0) # (uni_dom)
                expanded_losses_std = domains_loss_std_tensor.unsqueeze(1)
                transposed_losses_std = expanded_losses_std.transpose(0, 1)
                sum_std = expanded_losses_std + transposed_losses_std
                sum_std_norm = 1/(sum_std+1)
            else:
                sum_std_norm = 1.
            penalty = torch.sum(domain_mmd_matrix * pairwise_loss_diff_matrix * sum_std_norm) / ((n_uni_domains-1) * (n_uni_domains))
            penalty = self.gamma * penalty
        else:
            penalty = 0.

        total_loss = loss_y + penalty
        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale 
