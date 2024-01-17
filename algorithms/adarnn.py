import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.mmd import MMD_loss
from torch.distributions import Normal

"""
adapted from
https://github.com/jindongwang/transferlearning/blob/master/code/deep/adarnn/base/AdaRNN.py
"""

class TransferLoss(object):
    def __init__(self, loss_type='cosine', input_dim=512):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type == 'mmd_lin' or self.loss_type =='mmd' or self.loss_type =='linear':
            mmdloss = MMD_loss(kernel_type='linear')
            loss = mmdloss(X, Y) 
        return loss


class adarnn(nn.Module):
    """
    model_type: 'AdaRNN'
    """

    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, len_seq, mmd_type='linear', gamma=1):
        super(adarnn, self).__init__()
        self.num_layers = N
        self.hid_dim = d_hidden
        self.n_output = d_output
        self.model_type = 'AdaRNN'
        self.trans_loss = mmd_type
        self.len_seq = len_seq
        if len(d_emb)>0:
            self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
            d_emb_tot = d_emb[:, 1].sum()
        else:
            self.emb = nn.ModuleList([])
            d_emb_tot = 0
        in_size = d_lag+d_cov+d_emb_tot

        features = nn.ModuleList()
        for hidden in [self.hid_dim for i in range(self.num_layers)]:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        self.fc_loc = nn.Linear(self.hid_dim, self.n_output)
        self.fc_scale = nn.Linear(self.hid_dim, self.n_output)
        # self.fc_out = nn.Linear(self.hid_dim, self.n_output)
        self.epsilon = 1e-6

        gate = nn.ModuleList()
        for i in range(self.num_layers):
            gate_weight = nn.Linear(
                len_seq * self.hid_dim*2, len_seq)
            gate.append(gate_weight)
        self.gate = gate

        bnlst = nn.ModuleList()
        for i in range(self.num_layers):
            bnlst.append(nn.BatchNorm1d(len_seq))
        self.bn_lst = bnlst
        self.softmax = torch.nn.Softmax(dim=0)

        self.gamma = gamma
        self.init_layers()

    def init_layers(self):
        for i in range(self.num_layers):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def process_gate_weight(self, out, index):
        i =  out.shape[0] // 2 
        x_s = out[0: i]
        x_t = out[-i: ]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze() # seq_len,
        return res
    
    def gru_features(self, x, predict=False):
        x_input = x.permute(1,0,2).contiguous() # batch, seq, feat
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'AdaRNN') else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])
        return fea_list_src, fea_list_tar
    
    def forward(self):
        pass

    def forward_pre_train(self, X_emb, X_cov, X_lag, y, d_outputseqlen, len_win=30):

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
            x = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            x = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)
        out = self.gru_features(x)
        fea = out[0]
    
        y_hat_loc = self.fc_scale(fea[:, -d_outputseqlen:, :])
        y_hat_scale = F.softplus(self.fc_scale(fea[:, -d_outputseqlen:, :])) + self.epsilon
        y_hat_loc = y_hat_loc.permute(1,0,2).contiguous()
        y_hat_scale = y_hat_scale.permute(1,0,2).contiguous()
        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = - y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()

        out_list_all, out_weight_list = out[1], out[2]
        # print('out_list_all',len(out_list_all), 'out_weight_list',len(out_weight_list))
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,))
        if torch.cuda.is_available():
            loss_transfer = loss_transfer.cuda()
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            h_start = 0 
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (
                        self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        total_loss = loss_y + self.gamma * loss_transfer
        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale, out_weight_list 


    # For Boosting-based
    def forward_Boosting(self, X_emb, X_cov, X_lag, y, d_outputseqlen, weight_mat=None):
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
            x = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            x = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)

        out = self.gru_features(x)
        fea = out[0]
       
        y_hat_loc = self.fc_scale(fea[:, -d_outputseqlen:, :])
        y_hat_scale = F.softplus(self.fc_scale(fea[:, -d_outputseqlen:, :])) + self.epsilon
        y_hat_loc = y_hat_loc.permute(1,0,2).contiguous()
        y_hat_scale = y_hat_scale.permute(1,0,2).contiguous()
        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = - y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,))
        if torch.cuda.is_available():
            loss_transfer = loss_transfer.cuda()
        if weight_mat is None:
            weight = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq))
            if torch.cuda.is_available():
                weight = weight.cuda()
        else:
            weight = weight_mat
            
        dist_mat = torch.zeros(self.num_layers, self.len_seq)
        if torch.cuda.is_available():
            dist_mat = dist_mat.cuda()
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        total_loss = loss_y + self.gamma * loss_transfer
        return total_loss, loss_y, y_hat_distr, y_hat_loc, y_hat_scale, dist_mat, weight 

    # For Boosting-based
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    def predict(self, X_emb, X_cov, X_lag, y, d_outputseqlen):
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
            x = torch.cat((X_lag, X_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        else:
            x = torch.cat((X_lag, X_cov[:dim_seq]), dim=-1)

        out = self.gru_features(x, predict=True)
        fea = out[0]
        y_hat_loc = self.fc_scale(fea[:, -d_outputseqlen:, :])
        y_hat_scale = F.softplus(self.fc_scale(fea[:, -d_outputseqlen:, :])) + self.epsilon
        y_hat_loc = y_hat_loc.permute(1,0,2).contiguous()
        y_hat_scale = y_hat_scale.permute(1,0,2).contiguous()
        y_hat_distr = Normal(y_hat_loc, y_hat_scale)
        loss_y = - y_hat_distr.log_prob(y[-d_outputseqlen:]).mean()
        return y_hat_loc, y_hat_scale
    

    def transform_type(self, init_weight):
        weight = torch.ones(self.num_layers, self.len_seq)
        if torch.cuda.is_available():
            weight = weight.cuda()
        for i in range(self.num_layers):
            for j in range(self.len_seq):
                weight[i, j] = init_weight[i][j].item()
        return weight
    
 