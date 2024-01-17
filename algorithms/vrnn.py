import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch.distributions import Normal

"""
adapted from
https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = torch.finfo(torch.float).eps # numerical logs

class vrnn(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, N, dropout=0.1, bias=False):
        super(vrnn, self).__init__()
        if len(d_emb) > 0:
            self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
            d_emb_tot = d_emb[:, 1].sum()
        else:
            self.emb = nn.ModuleList([])
            d_emb_tot = 0

        self.x_dim = d_lag + d_cov + d_emb_tot
        self.hid_dim = d_hidden
        self.z_dim = d_hidden
        self.n_layers = N
        self.d_output = d_output

		#feature-extracting transformations
        self.phi_x = nn.Sequential(
			nn.Linear(self.x_dim, self.hid_dim),
			nn.ReLU(),
			nn.Linear(self.hid_dim, self.hid_dim),
			nn.ReLU())
        self.phi_z = nn.Sequential(
			nn.Linear(self.z_dim, self.hid_dim),
			nn.ReLU())

		#encoder
        self.enc = nn.Sequential(
			nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim),
			nn.ReLU(),
			nn.Linear(self.hid_dim, self.hid_dim),
			nn.ReLU())
        self.enc_mean = nn.Linear(self.hid_dim, self.z_dim)
        self.enc_std = nn.Sequential(
			nn.Linear(self.hid_dim, self.z_dim),
			nn.Softplus())

		#prior
        self.prior = nn.Sequential(
			nn.Linear(self.hid_dim, self.hid_dim),
			nn.ReLU())
        self.prior_mean = nn.Linear(self.hid_dim, self.z_dim)
        self.prior_std = nn.Sequential(
			nn.Linear(self.hid_dim, self.z_dim),
			nn.Softplus())

		#decoder
        self.dec = nn.Sequential(
			nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim),
			nn.ReLU(),
			nn.Linear(self.hid_dim, self.hid_dim),
			nn.ReLU())
        self.dec_std = nn.Sequential(
			nn.Linear(self.hid_dim, self.d_output),
			nn.Softplus())
        self.dec_mean = nn.Linear(self.hid_dim, self.d_output)

		#recurrence
        self.rnn = nn.GRU(self.hid_dim + self.hid_dim, self.hid_dim, N, bias, dropout=dropout)

    def forward(self, X_emb, X_cov, X_lag, y, d_outputseqlen):

        seq_dim = X_lag.shape[0]
        
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(X_emb[:, :, i])
            x_emb.append(out)
        if self.emb:
            x_emb = torch.cat(x_emb, -1)
        # Concatenate inputs
        if self.emb:
            x = torch.cat((X_lag, X_cov[:seq_dim], x_emb[:seq_dim]), dim=-1)
        else:
            x = torch.cat((X_lag, X_cov[:seq_dim]), dim=-1)
        # print(x.shape,'x')
        batch_dim = x.size(1)
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        pred_y_loss = 0
        h = Variable(torch.zeros(self.n_layers, batch_dim, self.hid_dim))
        if torch.cuda.is_available():
            h = h.cuda()
        for t in range(seq_dim):
            phi_x_t = self.phi_x(x[t])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t) # current x, reconstructed

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            cur_nll_loss = self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            if t >= seq_dim - d_outputseqlen:
                pred_y_loss +=  cur_nll_loss
                all_dec_mean.append(dec_mean_t)
                all_dec_std.append(dec_std_t)
        
        all_dec_mean = torch.stack(all_dec_mean, dim=0)
        all_dec_std = torch.stack(all_dec_std, dim=0)
        y_hat_distr = Normal(all_dec_mean, all_dec_std)

        return kld_loss + nll_loss, pred_y_loss, y_hat_distr, all_dec_mean, all_dec_std
            

    def forecast(self, X_emb, X_cov, X_lag, y, d_outputseqlen):
        dim_seq = X_lag.shape[0]
        
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(X_emb[:, :, i])
            x_emb.append(out)
        if self.emb:
            x_emb = torch.cat(x_emb, -1)
        if self.emb:
            x = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            x = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)
        seq_dim, batch_dim = x.size(0), x.size(1) 

        with torch.no_grad():

            h = Variable(torch.zeros(self.n_layers, batch_dim, self.hid_dim))
            if torch.cuda.is_available():
                h = h.cuda()
            y_loss, pred_y_loss = 0, 0
            all_dec_mean, all_dec_std = [], []

            for t in range(seq_dim):
                if t >= seq_dim - d_outputseqlen:
                    x[t] = dec_mean_t
                phi_x_t = self.phi_x(x[t])
                
                #encoder
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                #sampling and reparameterization
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                phi_z_t = self.phi_z(z_t)

                #decoder
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_std_t = self.dec_std(dec_t) # current x, reconstructed

                #recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
                #computing losses 
                cur_nll_loss = self._nll_gauss(dec_mean_t, dec_std_t, x[t])
                y_loss += cur_nll_loss
                if t >= seq_dim - d_outputseqlen:
                    pred_y_loss +=  cur_nll_loss
                    all_dec_mean.append(dec_mean_t)
                    all_dec_std.append(dec_std_t)
        
  
            all_dec_mean = torch.stack(all_dec_mean,dim=0)
            all_dec_std = torch.stack(all_dec_std,dim=0)
            y_hat_distr = Normal(all_dec_mean, all_dec_std)
            return pred_y_loss, y_hat_distr, all_dec_mean, all_dec_std
    



    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.hid_dim))
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            
            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass
 

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
 
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + math.log(2*math.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
 