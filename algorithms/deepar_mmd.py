import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from lib.mmd import MMD_loss
import itertools as it

"""
adapted from 
deepar
"""

class deepar_mmd(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, mmd_type='linear', gamma=1.0):
        super(deepar_mmd, self).__init__()
        if len(d_emb)>0:
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
            self.mmd_loss = MMD_loss('rbf')
        self.gamma = gamma
            
 
    def forward(self, X_emb, X_cov, X_lag, y, d_outputseqlen, domain=None):
        dim_seq = X_lag.shape[0]
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(X_emb[:, :, i])
            x_emb.append(out)
        if self.emb:
            x_emb = torch.cat(x_emb, -1)
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
        h = torch.cat(h, -1) # (dim_seq(+d_outputseqlen), batch, N*hid)
        h_pred = h[-d_outputseqlen:]
        # Output layers - location and scale of distribution
        y_hat_loc = self.loc(h_pred)
        y_hat_scale = F.softplus(self.scale(h_pred)) + self.epsilon

        # mmd loss
        if self.training:
            uni_domains = torch.unique(domain)
            n_uni_domains = uni_domains.size(0)
            domains_rep_tensor = []
            for domain_i in uni_domains:
                indices = (domain == domain_i).nonzero(as_tuple=True)[0] # e.g. [0,1]
                domain_rep = h_pred[:,indices].mean(1) # (d_outputseqlen, hid)
                domains_rep_tensor.append(domain_rep)
            domains_rep_tensor = torch.stack(domains_rep_tensor,dim=0) # (uni_dom, d_outputseqlen, N*hid)  
            uni_domains_index_list = [i for i in range(n_uni_domains)]

            tuples = list(it.combinations(uni_domains_index_list,2))
            a, b = map(list, zip(*tuples))
            n_pairs = len(a)
            source = domains_rep_tensor[a].view(n_pairs,-1)
            target = domains_rep_tensor[b].view(n_pairs,-1)
            loss_mmd = self.mmd_loss(source, target) * self.gamma
        else:
            loss_mmd = 0.

        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = -y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()
        total_loss = loss_y + loss_mmd
        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale 
    
 