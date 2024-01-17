import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Function

"""
adapted from 
deepar and https://github.com/fungtion/DANN
"""

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class deepar_dann(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_dim, d_output, d_hidden, dropout, N):
        super(deepar_dann, self).__init__()
        if len(d_emb)>0:
            self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
            d_emb_tot = d_emb[:, 1].sum()
        else:
            self.emb = nn.ModuleList([])
            d_emb_tot = 0

        # Network   
        feature_layers = [nn.LSTM(d_lag + d_cov + int(d_emb_tot), d_hidden)]
        self.feature_model = nn.ModuleList(feature_layers)
        self.feature_model_drop = nn.ModuleList([nn.Dropout(dropout)])

        forecast_layers = nn.ModuleList([])
        for i in range(N - 1):
            forecast_layers += [nn.LSTM(d_hidden, d_hidden)]    
        self.forecast_model = nn.Sequential(*forecast_layers)    
        self.forecast_model_drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(N-1)])

        # Output distribution layers
        self.loc = nn.Linear(d_hidden * (N-1), d_output)
        self.scale = nn.Linear(d_hidden * (N-1), d_output)

        self.domain_classifier = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),nn.Linear(d_hidden, d_dim))
        torch.nn.init.xavier_uniform_(self.domain_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[3].weight)

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
            inputs = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            inputs = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)

        feat = []
        for i, layer in enumerate(self.feature_model):
            outputs, _ = layer(inputs)
            outputs = self.feature_model_drop[i](outputs)
            inputs = outputs
            feat.append(outputs)
        feat = torch.cat(feat, -1)

        reverse_feature = ReverseLayerF.apply(feat, alpha)

        h = []
        for i, layer in enumerate(self.forecast_model):
            outputs, _ = layer(feat)
            outputs = self.forecast_model_drop[i](outputs)
            feat = outputs
            h.append(outputs)
        h = torch.cat(h, -1)
        y_hat_loc = self.loc(h[-d_outputseqlen:])
        y_hat_scale = F.softplus(self.scale(h[-d_outputseqlen:])) + 1e-6

        if self.training:
            d = domain
            d1, d2 = reverse_feature.size(0), reverse_feature.size(1)
            reverse_feature = reverse_feature.contiguous().view(d1*d2, -1)
            d_hat = self.domain_classifier(reverse_feature) 
            d_hat = d_hat.view(d1, d2, -1)
            d_hat = d_hat.mean(0)
            _, d_target = d.max(dim=1)
            loss_d = F.cross_entropy(d_hat, d_target, reduction='mean')
        else:
            loss_d = 0.

        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = -y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()

        total_loss = loss_d + loss_y
        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale 
        