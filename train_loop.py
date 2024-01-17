import torch
import time
import numpy as np
import torch.utils.data as torchdata
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from itertools import cycle
import torch.autograd as autograd

from lib.utils import calc_metrics, fish_step
import copy, random

'''
for real data check data_min and data_max
'''
""" Loop to calculate output of one epoch"""

def loop_basic(model, data, optimizer, batch_size, id_samples, train, metrics, scaling):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)
    # Quantile forecasting
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    # Initiate dimensions and book-keeping variables
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov, dim_lag = data.d_emb, data.d_cov, data.d_lag
    
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')
    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0
    n_samples_dist = 1000
    # Datamax
    data_max = 1e4
    data_min = -1e4 if data.name.startswith('same') else 0
    # Loop
    start = time.time()
    for i, (X, Y) in enumerate(data_generator):
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        # Permute to [seqlen x batch x feature] and transfer to device
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2) # X torch.Size([150, 128, 15]) Y torch.Size([30, 128, 1])
        Y = Y[-dim_outputseqlen:]
        # Fill bookkeeping variables
        y_tot[:, i*batch_size:j] = Y.detach().numpy()
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        # Create lags and covariate tensors
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY[:,[-1]]
        else:
            scaleY = torch.tensor([1.0])

        X_emb = X[:, :, :dim_emb].long()
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]
 
        X_emb, X_cov, X_lag, Y = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device)
        scaleY = scaleY.to(device)
        if train:
            optimizer.zero_grad()
            mean, variance = model(X_emb, X_cov, X_lag, dim_outputseqlen) # (#out, #batch, 1), (#out, #batch, 1)
            distr = Normal(mean, variance) 
            loss_batch = -distr.log_prob(Y).mean()
            loss_batch.backward()
            optimizer.step()
        else:                        
            with torch.no_grad():
                mean_prev = X_lag[dim_inputseqlen, :, [-1]].clone().detach()
                for t in range(dim_outputseqlen):
                    X_lag[dim_inputseqlen + t, :, [-1]] = mean_prev
                    mean, variance = model(X_emb, X_cov, X_lag[:dim_inputseqlen+t+1], t + 1)
                    mean_prev = mean[-1].clone().detach().clamp(data_min, data_max)
                distr = Normal(mean, variance) 
                loss_batch = -distr.log_prob(Y).mean()

        # Append loss, calculate quantiles
        loss += loss_batch.item()
        yhat = distr.sample([n_samples_dist])
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q.detach().cpu().numpy()
        
    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    yhat_tot = np.clip(yhat_tot, data_min, data_max)

    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
            
    return model, loss, yhat_tot, y_tot, x_tot, df


def loop_dann(model, data, optimizer, batch_size, id_samples, train, metrics, scaling, epoch=0, epochs=1):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)
    # Quantile forecasting
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    # Initiate dimensions and book-keeping variables
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov,  dim_lag = data.d_emb, data.d_cov,  data.d_lag
    
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')
    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0
    
    n_samples_dist = 1000
    # Datamax
    data_max = 1e4
    data_min = -1e4 if data.name.startswith('same') else 0

    # Loop
    start = time.time()
    len_data_generator = len(data_generator)

    for i, (X, Y) in enumerate(data_generator):
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2)
        y_tot[:, i*batch_size:j] = Y[-dim_outputseqlen:].detach().numpy()
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY
        else:
            scaleY = torch.tensor([1.0])
        domain = F.one_hot(X[0, :, 0].long(),num_classes=data.d_dim).float()  
        X = X[:, :, 1:]
        X_emb = X[:, :, :dim_emb].long()
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]

        X_emb, X_cov, X_lag, Y, domain = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device), domain.to(device)
        scaleY = scaleY.to(device)
        if train:
            p = float(i + epoch * len_data_generator) / epochs / len_data_generator
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            optimizer.zero_grad()
            loss_batch, loss_y, distr, mean, variance = model(X_emb, X_cov,X_lag, Y, dim_outputseqlen, domain, alpha)
            loss_batch.backward()
            optimizer.step()
            y_loss_batch = loss_y

        else:   
            with torch.no_grad():
                mean_prev = X_lag[dim_inputseqlen, :, [-1]].clone().detach()
                for t in range(dim_outputseqlen):
                    X_lag[dim_inputseqlen + t, :, [-1]] = mean_prev
                    loss_batch, loss_y, distr, mean, variance = model(X_emb, X_cov, X_lag[:dim_inputseqlen+t+1], Y, t + 1, domain)
                    mean_prev = mean[-1].clone().detach().clamp(data_min, data_max)
                y_loss_batch = loss_y         

        # Append loss, calculate quantiles
        loss += y_loss_batch.item()
        yhat = distr.sample([n_samples_dist]) #[1000, 30, 8, 1]
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q.detach().cpu().numpy()
        
    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    yhat_tot = np.clip(yhat_tot, data_min, data_max)

    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
            
    return model, loss, yhat_tot, y_tot, x_tot, df


def loop_mmd(model, data, optimizer, batch_size, id_samples, train, metrics, scaling):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov,  dim_lag = data.d_emb, data.d_cov,  data.d_lag
    
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')
    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0
    n_samples_dist = 1000
    data_max = 1e4
    data_min = -1e4 if data.name.startswith('same') else 0
    
    # Loop
    start = time.time()
    for i, (X, Y) in enumerate(data_generator):
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2)
        
        y_tot[:, i*batch_size:j] = Y[-dim_outputseqlen:].detach().numpy()
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY
        else:
            scaleY = torch.tensor([1.0])
        domain = X[0, :, 0]
        X = X[:, :, 1:]
        X_emb = X[:, :, :dim_emb].long()
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]

        X_emb, X_cov, X_lag, Y, domain = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device), domain.to(device)
        scaleY = scaleY.to(device)
        if train:
            optimizer.zero_grad()
            loss_batch, loss_y, distr, mean, variance = model(X_emb, X_cov, X_lag, Y, dim_outputseqlen, domain)
            loss_batch.backward()
            optimizer.step() 
            y_loss_batch = loss_y
        else:          
            with torch.no_grad():
                mean_prev = X_lag[dim_inputseqlen, :, [-1]].clone().detach()
                for t in range(dim_outputseqlen): 
                    X_lag[dim_inputseqlen + t, :, [-1]] = mean_prev
                    loss_batch, loss_y, distr, mean, variance = model(X_emb, X_cov, X_lag[:dim_inputseqlen+t+1], Y, t + 1, domain)
                    mean_prev = mean[-1].clone().detach().clamp(data_min, data_max)
                y_loss_batch = loss_y

        # Append loss, calculate quantiles
        loss += y_loss_batch.item()
        yhat = distr.sample([n_samples_dist]) #[1000, 30, 8, 1]
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q.detach().cpu().numpy()
        
    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    yhat_tot = np.clip(yhat_tot, data_min, data_max)
    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
            
    return model, loss, yhat_tot, y_tot, x_tot, df


def loop_vae(model, data, optimizer, batch_size, id_samples, train, metrics, scaling):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov, dim_lag = data.d_emb, data.d_cov, data.d_lag
    
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')

    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0

    n_samples_dist = 1000
    data_max = 1e4
    data_min = -1e4 if data.name.startswith('same') else 0
    
    start = time.time()
    for i, (X, Y) in enumerate(data_generator):
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2) # X torch.Size([150, 128, 15]) Y torch.Size([30, 128, 1])
        y_tot[:, i*batch_size:j] = Y[-dim_outputseqlen:].detach().numpy()
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY
        else:
            scaleY = torch.tensor([1.0])

        X_emb = X[:, :, :dim_emb].long()
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]
 
        X_emb, X_cov, X_lag, Y = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device)
        scaleY = scaleY.to(device)
        if train:
            optimizer.zero_grad()
            loss_batch, loss_y, distr, mean, variance = model(X_emb, X_cov, X_lag, Y, dim_outputseqlen)
            loss_batch.backward()
            optimizer.step() 
            y_loss_batch = loss_y
        else:                        
            with torch.no_grad():
                loss_y, distr, mean, variance = model.forecast(X_emb, X_cov, X_lag, Y, dim_outputseqlen)
                y_loss_batch = loss_y

        loss += y_loss_batch.item()
        yhat = distr.sample([n_samples_dist])
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q[:,-dim_outputseqlen:].detach().cpu().numpy()
        
    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    yhat_tot = np.clip(yhat_tot, data_min, data_max)
    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
            
    return model, loss, yhat_tot, y_tot, x_tot, df


def domain_batching(domains, X, Y, batch_size): 
    # batchify each domain
    domains_batches_left = {}
    domains_batches = {}
    domain_ids, inverse_indices = torch.unique(domains.long(), sorted=True, return_inverse=True)
    
    for i, did in enumerate(domain_ids):
        sel_idx = (inverse_indices==i).nonzero(as_tuple=False)
        sel_X = X[sel_idx]
        sel_Y = Y[sel_idx]
        splitted_X = torch.split(sel_X, batch_size)
        splitted_Y = torch.split(sel_Y, batch_size)
        domains_batches_left[did.item()] = len(splitted_X)
        domains_batches[did.item()] = [splitted_X, splitted_Y] 
    return domains_batches, domains_batches_left



def loop_fish(model, data, optimizer, batch_size, id_samples, train, metrics, scaling, meta_lr=0.01, params=None):
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size=len(id_samples))
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov,  dim_lag = data.d_emb, data.d_cov,  data.d_lag
    
    loss = 0
    start = time.time()
    for _, (all_X, all_Y) in enumerate(data_generator):
        domains_batches, domains_batches_left = domain_batching(all_X[:, 0, 0], all_X, all_Y, batch_size)
        opt = getattr(optim, 'Adam')
        model.train()
        opt_inner_pre = None
        domains_batches_left_copy = copy.deepcopy(domains_batches_left)
        domains = list(domains_batches_left_copy.keys())
        i = 0
        meta_steps = 1
        reload_inner_optim = True
        while sum([v > 0 for v in domains_batches_left_copy.values()]) > meta_steps:
            i += 1
            # permute domains
            random.shuffle(domains)
            # model_inner = copy.deepcopy(model)
            model_inner = type(model)(*params).to(device) 
            model_inner.load_state_dict(model.state_dict())
            model_inner.train()

            opt_inner = opt(model_inner.parameters()) 
            if opt_inner_pre is not None and reload_inner_optim:
                opt_inner.load_state_dict(opt_inner_pre)
        
            # inner loop update
            for domain in domains_batches:
                if domains_batches_left_copy[domain] == 0:
                    continue

                cur_batch_id = random.randint(0,domains_batches_left_copy[domain]-1)
                X, Y = domains_batches[domain][0][cur_batch_id], domains_batches[domain][1][cur_batch_id]
                domains_batches_left_copy[domain] -= 1

                X, Y = X.squeeze(1).permute(1, 0, 2), Y.squeeze(1).permute(1, 0, 2)
                Y = Y[-dim_outputseqlen:]
                if X.size(1) == 1:
                    continue
                if scaling:
                    scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
                    X[:, :, -dim_lag:] /= scaleY
                    Y /= scaleY 
                else: 
                    scaleY = torch.tensor([1.0])
                X = X[:, :, 1:]
                X_emb = X[:, :, :dim_emb].long()
                X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
                X_lag = X[:window, :, -dim_lag:]

                X_emb, X_cov, X_lag, Y, scaleY = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device), scaleY.to(device)

                opt_inner.zero_grad()
                loc, scale = model_inner(X_emb, X_cov, X_lag, dim_outputseqlen) # (#out, #batch, 1), (#out, #batch, 1)
                distr = Normal(loc, scale)
                loss_batch = -distr.log_prob(Y).mean()
                loss_batch.backward()
                opt_inner.step()
                loss += loss_batch.item()
            opt_inner_pre = opt_inner.state_dict()
            # fish update
            meta_weights = fish_step(meta_weights=model.state_dict(),
                                inner_weights=model_inner.state_dict(),
                                meta_lr=meta_lr / meta_steps)
            model.reset_weights(meta_weights)
        
    end = time.time()
    print(f'Train loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    return model 


def loop_adarnn(model, data, optimizer, batch_size, id_samples, train, metrics, scaling, num_layers, epoch=None, pre_epoch=None, dist_old=None, weight_mat=None):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov, dim_lag = data.d_emb, data.d_cov, data.d_lag
    
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')
    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0
    n_samples_dist = 1000
    data_max = 1e4
    data_min = -1e4 if data.name.startswith('same') else 0
    dist_mat = torch.zeros(num_layers, dim_inputseqlen + dim_outputseqlen, device=device)

    start = time.time()
    for i, (X, Y) in enumerate(data_generator):
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2) # X torch.Size([150, 128, 15]) Y torch.Size([30, 128, 1])
        Y = Y[-dim_outputseqlen:]
        y_tot[:, i*batch_size:j] = Y.detach().numpy()
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY[:,[-1]]
        else:
            scaleY = torch.tensor([1.0])

        X_emb = X[:, :, :dim_emb].long()
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]
 
        X_emb, X_cov, X_lag, Y = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device)
        scaleY = scaleY.to(device)
        if train:
            if X_lag.size(1)<= 3:
                continue
            optimizer.zero_grad()
            if epoch < pre_epoch:
                loss_batch, loss_y, distr, mean, variance, out_weight_list = model.forward_pre_train(
                        X_emb, X_cov, X_lag, Y, dim_outputseqlen, len_win=window)
            else:
                loss_batch, loss_y, distr, mean, variance, dist, weight_mat = model.forward_Boosting(
                    X_emb, X_cov, X_lag, Y, dim_outputseqlen, weight_mat)
                dist_mat = dist_mat + dist
            distr = Normal(mean, variance) 
            loss_batch = -distr.log_prob(Y).mean()
            loss_batch.backward()
            optimizer.step()
        else:                        
            with torch.no_grad():
                mean_prev = X_lag[dim_inputseqlen, :, [-1]].clone().detach()
                for t in range(dim_outputseqlen):
                    X_lag[dim_inputseqlen + t, :, [-1]] = mean_prev
                    mean, variance = model.predict(X_emb, X_cov, X_lag[:dim_inputseqlen+t+1], Y, t + 1)
                    mean_prev = mean[-1].clone().detach().clamp(data_min, data_max)
                distr = Normal(mean, variance) 
                loss_batch = -distr.log_prob(Y).mean()
        if train:
            if epoch >= pre_epoch:
                if epoch > pre_epoch:
                    weight_mat = model.update_weight_Boosting(
                        weight_mat, dist_old, dist_mat)
            else:
                weight_mat = model.transform_type(out_weight_list)
                dist_mat = None
        else:
            weight_mat, dist_mat = None, None
        loss += loss_batch.item()
        yhat = distr.sample([n_samples_dist])
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q.detach().cpu().numpy()
        
    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    yhat_tot = np.clip(yhat_tot, data_min, data_max)

    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
            
    return model, loss, yhat_tot, y_tot, x_tot, df, weight_mat, dist_mat


def loop_groupdro(model, data, optimizer, batch_size, id_samples, train, metrics, scaling, q=None):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    data_subset = torchdata.Subset(data, id_samples)
    num_samples = len(id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size)
    # Quantile forecasting
    quantiles = torch.arange(1, 10, dtype=torch.float32, device=device) / 10
    num_forecasts = len(quantiles)
    # Initiate dimensions and book-keeping variables
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov, dim_lag = data.d_emb, data.d_cov, data.d_lag
    
    yhat_tot = np.zeros((num_forecasts, data.dim_outputseqlen, num_samples, dim_output), dtype='float32')
    y_tot = np.zeros((dim_outputseqlen, num_samples, dim_output), dtype='float32')        
    x_tot = np.zeros((window, num_samples, dim_input), dtype='float32')
    loss = 0
    n_samples_dist = 1000
    # Datamax
    data_max = 1e4
    data_min = -1e4 if data.name.startswith('same') else 0
    # Loop
    start = time.time()

    n_batches = len(data_generator)
    losses =  torch.zeros(n_batches).to(device) # TODO amy
    if q is None:
        q = torch.ones(n_batches).to(device)
    if data.name in ['samep_diffmv30','samepmv_difft30']:
        groupdro_eta = 1e-4
    elif data.name in ['samet_diffpmv30']:
        groupdro_eta = 1e-5
    else:
        groupdro_eta = 1e-2

    for i, (X, Y) in enumerate(data_generator):
        # print(X.shape,Y.shape,'X Y')
        j = np.min(((i + 1) * batch_size, len(id_samples)))
        # Permute to [seqlen x batch x feature] and transfer to device
        X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2) # X torch.Size([150, 128, 15]) Y torch.Size([30, 128, 1])
        Y = Y[-dim_outputseqlen:]
        # Fill bookkeeping variables
        y_tot[:, i*batch_size:j] = Y.detach().numpy()
        x_tot[:, i*batch_size:j] = X[:window].detach().numpy()            
        # Create lags and covariate tensors
        if scaling:
            scaleY = 1 + X[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
            X[:, :, -dim_lag:] /= scaleY
            Y /= scaleY[:,[-1]]
        else:
            scaleY = torch.tensor([1.0])

        X_emb = X[:, :, :dim_emb].long()
        X_cov = X[:, :, dim_emb:dim_emb + dim_cov]
        X_lag = X[:window, :, -dim_lag:]
 
        X_emb, X_cov, X_lag, Y = X_emb.to(device), X_cov.to(device), X_lag.to(device), Y.to(device)
        scaleY = scaleY.to(device)
        if train:
            # optimizer.zero_grad()
            mean, variance = model(X_emb, X_cov, X_lag, dim_outputseqlen) # (#out, #batch, 1), (#out, #batch, 1)
            distr = Normal(mean, variance) 
            loss_batch = -distr.log_prob(Y).mean()
            losses[i] = loss_batch
            q[i] *= (groupdro_eta * losses[i].data).exp()
            # loss_batch.backward()
            # optimizer.step()
        else:
            pass
            # with torch.no_grad():
            #     mean_prev = X_lag[dim_inputseqlen, :, [-1]].clone().detach()
            #     for t in range(dim_outputseqlen):
            #         X_lag[dim_inputseqlen + t, :, [-1]] = mean_prev
            #         mean, variance = model(X_emb, X_cov, X_lag[:dim_inputseqlen+t+1], t + 1)
            #         mean_prev = mean[-1].clone().detach().clamp(data_min, data_max)
            #     distr = Normal(mean, variance) 
            #     loss_batch = -distr.log_prob(Y).mean()

        # Append loss, calculate quantiles
        loss += loss_batch.item()
        yhat = distr.sample([n_samples_dist])
        yhat *= scaleY
        yhat_q = torch.quantile(yhat, quantiles, dim=0)
        yhat_tot[:, :, i*batch_size:j, :] = yhat_q.detach().cpu().numpy()
        
    if train:
        # print(q)
        q /= q.sum()
        loss_all = torch.dot(losses, q)
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

    end = time.time()
    print(f'{"  Train" if train else "  Validation/Test"} loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      
    yhat_tot = np.clip(yhat_tot, data_min, data_max)

    if metrics:
        output = 0
        y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]
        df = calc_metrics(yhat, y, quantiles.cpu().numpy())
    return model, loss, yhat_tot, y_tot, x_tot, df, q

'''
Model-Agnostic Meta-Learning
Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
'''

def split_meta_train_test(domains, X, Y, num_meta_test=2): 
    # print(X.shape,Y.shape,'X, Y')
    # batchify each domain
    # domains_batches_left = {}
    # domains_batches = {}
    domain_ids, inverse_indices = torch.unique(domains.long(), sorted=True, return_inverse=True)
    n_domains = len(domain_ids)
    perm = torch.randperm(n_domains).tolist()
    # print(perm)
    meta_train, meta_test = None, None
    X_by_domains = []
    Y_by_domains = []
    min_size = 1e10
    for i, did in enumerate(domain_ids):
        # print(i,'==')
        sel_idx = (inverse_indices==i).nonzero(as_tuple=False)
        sel_X = X[sel_idx]
        sel_Y = Y[sel_idx]
        min_size = min(min_size, sel_X.size(0))
        # print('sel_X',sel_X.shape,'sel_Y',sel_Y.shape)
        X_by_domains.append(sel_X)
        Y_by_domains.append(sel_Y)
    try:
        X_by_domains = torch.cat(X_by_domains,1)
        Y_by_domains = torch.cat(Y_by_domains,1)
    except:
        # print('here')
        # print('min_size',min_size)
        X_by_domains_new = []
        Y_by_domains_new = []
        for i in range(len(X_by_domains)):
            X_by_domains_new.append(X_by_domains[i][:min_size])
            Y_by_domains_new.append(Y_by_domains[i][:min_size])
        X_by_domains = torch.cat(X_by_domains_new,1)
        Y_by_domains = torch.cat(Y_by_domains_new,1)
    # print('X_by_domains',X_by_domains.shape,'Y_by_domains',Y_by_domains.shape)
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]
    # print('meta_train',meta_train)
    # print('meta_test',meta_test)
    for i,j in zip(meta_train, cycle(meta_test)):
        # print(i,j)
        xi, yi = X_by_domains[:,i], Y_by_domains[:,i]
        xj, yj = X_by_domains[:,j], Y_by_domains[:,j]
        # print(xi.shape,yi.shape,xj.shape,yj.shape)
        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs, n_domains 

def loop_mldg(model, data, optimizer, batch_size, id_samples, train, metrics, scaling, lr, mldg_beta):
    model = model.train() if train else model.eval()
    device = next(model.parameters()).device
    # data_subset = torchdata.Subset(data, id_samples)
    data_subset = torchdata.Subset(data, id_samples)
    data_generator = torchdata.DataLoader(data_subset, batch_size=len(id_samples))
 
    dim_input, dim_output, dim_inputseqlen, dim_outputseqlen, window = data.dim_input, data.dim_output, data.dim_inputseqlen, data.dim_outputseqlen, data.window
    dim_emb, dim_cov, dim_lag = data.d_emb, data.d_cov, data.d_lag
    
    loss = 0 
    start = time.time()

    num_meta_test = 2 
    n_batches = None
    objective = 0

    optimizer.zero_grad()
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

   
    for _, (all_X, all_Y) in enumerate(data_generator):
        # print(all_X.shape, all_Y.shape,'all_X.shape, all_Y.shape')
        pairs, n_batches = split_meta_train_test(all_X[:, 0, 0], all_X, all_Y, num_meta_test)
        for (xi, yi), (xj, yj) in pairs:
            Xi, Yi = xi.permute(1, 0, 2), yi.permute(1, 0, 2) # X torch.Size([150, 128, 15]) Y torch.Size([30, 128, 1])
            Yi = Yi[-dim_outputseqlen:]
            # print('Xi.shape, Yi.shape',Xi.shape, Yi.shape)
            Xj, Yj = xj.permute(1, 0, 2), yj.permute(1, 0, 2) # X torch.Size([150, 128, 15]) Y torch.Size([30, 128, 1])
            Yj = Yj[-dim_outputseqlen:]

            if scaling:
                scaleYi = 1 + Xi[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
                Xi[:, :, -dim_lag:] /= scaleYi
                Yi /= scaleYi[:,[-1]]
                scaleYj = 1 + Xj[:dim_inputseqlen, :, -dim_lag:].mean(dim = 0) # torch.Size([128, 8])
                Xj[:, :, -dim_lag:] /= scaleYj
                Yj /= scaleYj[:,[-1]]
            else:
                scaleYi = torch.tensor([1.0])
                scaleYj = torch.tensor([1.0])

            Xi_emb = Xi[:, :, :dim_emb].long()
            Xi_cov = Xi[:, :, dim_emb:dim_emb + dim_cov]
            Xi_lag = Xi[:window, :, -dim_lag:]

            Xj_emb = Xj[:, :, :dim_emb].long()
            Xj_cov = Xj[:, :, dim_emb:dim_emb + dim_cov]
            Xj_lag = Xj[:window, :, -dim_lag:]

            Xi_emb, Xi_cov, Xi_lag, Yi = Xi_emb.to(device), Xi_cov.to(device), Xi_lag.to(device), Yi.to(device)
            scaleYi = scaleYi.to(device)

            Xj_emb, Xj_cov, Xj_lag, Yj = Xj_emb.to(device), Xj_cov.to(device), Xj_lag.to(device), Yj.to(device)
            scaleYj = scaleYj.to(device)


            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(model)
            # model.train()

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=lr,
                # weight_decay=self.hparams['weight_decay']
            )

            # inner_obj = F.cross_entropy(inner_net(xi), yi)
            loc, scale = inner_net(Xi_emb, Xi_cov, Xi_lag, dim_outputseqlen) # (#out, #batch, 1), (#out, #batch, 1)
            distr = Normal(loc, scale)
            inner_obj = loss_batch = -distr.log_prob(Yi).mean()
            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(model.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / n_batches)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # loss_inner_j = F.cross_entropy(inner_net(xj), yj) # amy todo

            loc, scale = inner_net(Xj_emb, Xj_cov, Xj_lag, dim_outputseqlen) # (#out, #batch, 1), (#out, #batch, 1)
            distr = Normal(loc, scale)
            loss_inner_j = loss_batch = -distr.log_prob(Yj).mean()
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)
            
            # `objective` is populated for reporting purposes
            objective += (mldg_beta * loss_inner_j).item()

            for p, g_j in zip(model.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        mldg_beta * g_j.data / n_batches)
            
            objective /= n_batches
            optimizer.step()
 
        
    end = time.time()
    print(f'Train loss: {loss/len(data_generator):.4f} Time: {end-start:.5f}s')      

    return model
