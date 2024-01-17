import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from lib.mmd import MMD_loss

"""

"""
class deepar_cedar(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, mmd_type='linear', gamma=1.0, loss_dis='abs', reg='dd'):
        super(deepar_cedar, self).__init__()
        if len(d_emb) > 0:
            # Embedding layer for time series ID
            self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
            d_emb_tot = d_emb[:, 1].sum()
        else:
            self.emb = nn.ModuleList([])
            d_emb_tot = 0

        # Network   
        lstm = [nn.LSTM(d_lag + d_cov + int(d_emb_tot), d_hidden)]
        for i in range(N - 1):
            lstm += [nn.LSTM(d_hidden, d_hidden)]    
        self.lstm = nn.ModuleList(lstm)
        self.drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(N)])
        # Output distribution layers
        self.loc = nn.Linear(d_hidden * N, d_output)
        self.scale = nn.Linear(d_hidden * N, d_output)
        self.epsilon = 1e-6
 
        if torch.cuda.is_available():
            self.cuda()

        if mmd_type == 'linear':
            self.mmd_loss = MMD_loss('linear')
        elif mmd_type == 'rbf':
            self.mmd_loss = MMD_loss('rbf') # test

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
        # Concatenate x_lag, x_cov and time series ID
        if self.emb:
            inputs = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            inputs = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)
        # DeepAR network       
        h = []
        for i, layer in enumerate(self.lstm):
            outputs, _ = layer(inputs)
            outputs = self.drop[i](outputs)
            inputs = outputs
            h.append(outputs)
        h = torch.cat(h, -1) # (dim_seq[d_inputseqlen+d_outputseqlen], batch, N*hid)
        h_pred = h[-d_outputseqlen:]  # (d_outputseqlen, batch, N*hid)
        # Output layers - location and scale of distribution
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
            indices_lists = []
            for domain_i in uni_domains:
                indices = (domain == domain_i).nonzero(as_tuple=True)[0] # e.g. [0,1] torch.LongTensor
                indices_lists.append(indices)
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
            domain_discrepancy_loss_penalty = torch.sum(domain_mmd_matrix * pairwise_loss_diff_matrix * sum_std_norm ) / ((n_uni_domains-1) * (n_uni_domains))
            penalty = self.gamma * domain_discrepancy_loss_penalty
        else:
            penalty = 0.
        total_loss = loss_y + penalty

        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale 
    
 