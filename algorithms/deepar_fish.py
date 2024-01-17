import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


"""
adapted from 
deepar and https://github.com/YugeTen/fish
"""

class deepar_fish(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N):
        super(deepar_fish, self).__init__()
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

    def forward(self, X_emb, X_cov, X_lag, d_outputseqlen):
        dim_seq = X_lag.shape[0]
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
        h = torch.cat(h, -1)
        # Output layers - location and scale of distribution
        loc = self.loc(h[-d_outputseqlen:])
        scale = F.softplus(self.scale(h[-d_outputseqlen:]))
        return loc, scale + self.epsilon
    
    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))