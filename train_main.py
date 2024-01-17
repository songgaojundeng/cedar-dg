import time, os, sys
import datetime
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from lib.utils import * 
from lib.train_loop import loop_basic, loop_dann, loop_mmd, loop_vae, loop_fish, loop_adarnn, loop_groupdro, loop_mldg
from data import timeseries_dataset
torch.backends.cudnn.benchmark = False
num_cores = 2
torch.set_num_threads(2)

'''
e.g., python train_main.py traffic deepar.csv deepar
'''

try:
    dataset_name = sys.argv[1]
    setting_file = sys.argv[2]
    algorithm = sys.argv[3]
except:
    print ('Usage: python train_main.py <dataset_name> <setting_file> <algorithm>')
    exit()

print(f'dataset_name:{dataset_name}, setting_file:{setting_file}, algorithm:{algorithm}')


cuda_available = torch.cuda.is_available()
print(f'cuda:{cuda_available}')
cuda = 0

early_stopping_patience = 10
scaling = True
epochs = 150

os.makedirs("experiments", exist_ok=True) 
experiment_dir = 'experiments/'+dataset_name

if dataset_name.startswith(('favorita','stock')):
    dim_inputseqlen = 60
    dim_outputseqlen = 14
    dim_maxseqlen = dim_inputseqlen + dim_outputseqlen * 2
elif dataset_name in ['traffic']:
    dim_inputseqlen = 28
    dim_outputseqlen = 7
    dim_maxseqlen = dim_inputseqlen + dim_outputseqlen * 2
else:
    dim_inputseqlen = 30
    dim_outputseqlen = 7
    dim_maxseqlen = dim_inputseqlen + dim_outputseqlen * 2 

print(f"dim_inputseqlen:{dim_inputseqlen}, dim_outputseqlen:{dim_outputseqlen}")
file_experiments = experiment_dir + f'/{setting_file}'
table = read_table(file_experiments)
print(table)

d_emb = np.array([])
max_seed = 0 # check if we should train under more seeds or stop 

while table[table['in_progress'] == -1].isnull()['score'].sum() > 0:
    idx = table[table['in_progress'] == -1].isnull()['score'].idxmax()
    algorithm = table.loc[idx, 'algorithm']
    learning_rate = table.loc[idx, 'learning_rate']
    batch_size = int(table.loc[idx, 'batch_size'])
    d_hidden = int(table.loc[idx, 'd_hidden'])
    
    if not algorithm.startswith('deepar') and algorithm not in ['adarnn','vrnn']:
        kernel_size = int(table.loc[idx, 'kernel_size'])

    if algorithm.endswith('mmd') or algorithm.endswith('cedar'):
        gamma = float(table.loc[idx, 'gamma'])
        mmd_type = str(table.loc[idx, 'mmd'])

    if algorithm.endswith('cedar'):
        loss_dis = str(table.loc[idx, 'loss_dis'])
        reg = str(table.loc[idx, 'reg'])
    
    if algorithm.endswith('fish'):
        meta_lr = float(table.loc[idx, 'meta_lr'])
        pretrain_epochs = 20

    if algorithm == 'adarnn':
        pretrain_epochs = 20
        len_seq = dim_inputseqlen + dim_outputseqlen

    if algorithm.endswith('mldg'):
        mldg_beta = float(table.loc[idx, 'mldg_beta'])

    N = int(table.loc[idx, 'N'])
    dropout = table.loc[idx, 'dropout']
    seed = int(table.loc[idx, 'seed'])
    fix_seed(seed)

    max_seed = seed
    if cuda_available:
        device = torch.device("cuda:{}".format(cuda))
        table.loc[idx, 'in_progress'] = cuda
    else:
        device = torch.device("cpu")
        table.loc[idx, 'in_progress'] = 99

    table.to_csv(file_experiments, sep=';', index=False) # mark in_progress
    

    # Training loop
    print('---- Hyperparameter setting ----')
 
    filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed{seed}_hid{d_hidden}_lr{learning_rate}_bs{batch_size}_dp{dropout}_N{N}"
    os.makedirs(f"{experiment_dir}/{algorithm}", exist_ok=True) 

    if not algorithm.startswith('deepar') and algorithm not in ['adarnn','vrnn']:
        filename += f"_knl{kernel_size}"

    if algorithm.endswith('mmd') or algorithm.endswith('cedar'):
        filename += f"_gm{gamma}_mmd{mmd_type}"

    if algorithm.endswith('cedar'):
        filename += f"_ld{loss_dis}_rg{reg}"

    if algorithm.endswith('fish'):
        filename += f"_mlr{meta_lr}"
    
    if algorithm.endswith('mldg'):
        filename += f"_mlb{mldg_beta}"

    print(f'filename:{filename}')
    
    
    
    # init data
    dataset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, seed=seed, train_ratio=0.8)
    training_set = dataset.load('train')
    validation_set = dataset.load('validate')
    test_set = dataset.load('test')

    # Initialize sample sets
    id_samples_train = torch.randperm(len(training_set))
    id_samples_validate = torch.randperm(len(validation_set)) 
    id_samples_test = torch.randperm(len(test_set)) 
    print(f'num_samples_train:{len(training_set)}')
    print(f'num_samples_validate:{len(validation_set)}')
    print(f'num_samples_test:{len(test_set)}')


    n_batch_train = (len(id_samples_train) + batch_size - 1) // batch_size 
    n_batch_validate = (len(id_samples_validate) + batch_size - 1) // batch_size
    print(f'# batch_train:{n_batch_train}, #batch_validate:{n_batch_validate}')

    if 'model' in locals(): del model
    params = eval(table.loc[idx, 'params_train'])
    print('params=',params)
    fix_seed(seed)
    model = instantiate_model(algorithm)(*params).to(device) 
    # print(model) 
    print('#Params:{}'.format(count_parameters(model))) 

    start_training = time.time()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
 
    if algorithm.endswith('fish'):
        for epoch in range(pretrain_epochs):
            print(f'Pretrain Epoch {epoch + 1}/{pretrain_epochs}')
            model, _, _, _, _, _ = loop_basic(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
        
    loss_train = np.zeros((epochs))
    loss_validate = np.zeros((epochs))
    loss_validate_best = 1e12
    early_stopping_counter = 0
    best_epoch = 0
    if algorithm == 'adarnn':
        weight_mat, dist_mat = None, None
    if algorithm.endswith('groupdro'):
        q = None
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        if algorithm.endswith('dann'):
            model, loss_train[epoch], _, _, _, _ = loop_dann(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling, epoch=epoch, epochs=epochs)    
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_dann(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling, epoch=epoch, epochs=epochs)  
        elif algorithm.endswith('mmd') or algorithm.endswith('cedar'):
            model, loss_train[epoch], _, _, _, _ = loop_mmd(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_mmd(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)  
        elif algorithm == 'vrnn':
            model, loss_train[epoch], _, _, _, _ = loop_vae(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_vae(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)  
        elif algorithm.endswith('fish'):
            model = loop_fish(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling, meta_lr=meta_lr, params=params)    
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_basic(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)    
        elif algorithm == 'adarnn':
            model, loss_train[epoch], _, _, _, _, weight_mat, dist_mat = loop_adarnn(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling, num_layers=N, epoch=epoch, pre_epoch=pretrain_epochs, dist_old=dist_mat,weight_mat=weight_mat)
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate, _, _ = loop_adarnn(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling, num_layers=N)
        elif algorithm.endswith('groupdro'):
            model, loss_train[epoch], _, _, _, _, q = loop_groupdro(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling, q=q)    
            # _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate, q = loop_groupdro(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling, q=q) 
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_basic(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)    
        elif algorithm.endswith('mldg'):
            model = loop_mldg(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling, lr=learning_rate, mldg_beta=mldg_beta)    
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_basic(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)    
        
        else:
            model, loss_train[epoch], _, _, _, _ = loop_basic(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
            _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop_basic(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)    
        if loss_validate[epoch] < loss_validate_best:
            torch.save({'epoch':epoch, 
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()}, filename)
            df_validate.to_csv(filename + '_validate.csv')
            loss_validate_best = loss_validate[epoch]
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if (early_stopping_counter == early_stopping_patience) | (epoch == epochs - 1):
            loss_train = loss_train / n_batch_train
            loss_validate = loss_validate / n_batch_validate
            df_loss = pd.DataFrame({'Validation_loss':loss_validate,'Training_loss':loss_train})
            df_loss.to_csv(filename + '_loss.csv')
            break
    end_training = time.time()
    print(f'Training time: {end_training-start_training:.5f}s')
    table = read_table(file_experiments)
    table.loc[idx, 'score'] = loss_validate_best / n_batch_validate
    table.to_csv(file_experiments, sep=';', index=False)

    print('---- Training Done ----\n')

    # Test 
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer=None
    if algorithm.endswith('dann'):
        _, loss_test, yhat_tot, y_tot, x_tot, df_test = loop_dann(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling) 
    elif algorithm.endswith('mmd') or algorithm.endswith('cedar'):
        _, loss_test, yhat_tot, y_tot, x_tot, df_test = loop_mmd(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling) 
    elif algorithm == 'vrnn':
        _, loss_test, yhat_tot, y_tot, x_tot, df_test = loop_vae(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling) 
    elif algorithm == 'adarnn':
        _, loss_test, yhat_tot, y_tot, x_tot, df_test, _, _ = loop_adarnn(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling, num_layers=N)
    # elif algorithm == 'fish':
    else:
        _, loss_test, yhat_tot, y_tot, x_tot, df_test = loop_basic(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling)    
    
    df_test.to_csv(filename + '_test.csv')
    n_batch_test = (len(id_samples_test) + batch_size - 1) // batch_size
    table.loc[idx, 'test_score'] = loss_test / n_batch_test
    table.loc[idx, 'in_progress'] = -1
    table.to_csv(file_experiments, sep=';', index=False)
    print('---- Testing Done ----\n')
    print(f'filename:{filename} \n')


print('Training and testing completed')
print(datetime.datetime.now())
if algorithm.endswith('_cedar'):
    if max_seed == 0:
        table = read_table(file_experiments)
        check_train = table['score'].isnull().sum() == 0
        check_test =  table['test_score'].isnull().sum() == 0
        print(f"check_train {check_train}, check_test {check_test}")
        if check_train and check_test:
            idx_list = []
            for reg in ['dd','ddd']:
                for loss_dis in ['abs','sqaure']:
                    idx =  table[(table['score'] != '')&(table['loss_dis'] ==loss_dis)&(table['reg'] ==reg)&(table['seed'] ==0)]['score'].idxmin()
                    idx_list.append(idx)
                    print('best_params',table.loc[idx])
            for idx in idx_list:
                for i in range(1,5):
                    new_row = table.loc[idx].copy()
                    new_row['seed'] = i
                    new_row['score'] = ''
                    new_row['test_score'] = ''
                    table.loc[len(table.index)] = new_row.values
                table.to_csv(file_experiments, sep=';', index=False)
            print('Please retrain it for new settings...')
        else:
            print('!!! Something wrong/only partial settings were trained !!!')
else:
    if max_seed == 0:
        # Get the best parameters and create new settings
        table = read_table(file_experiments)
        check_train = table['score'].isnull().sum() == 0
        check_test =  table['test_score'].isnull().sum() == 0
        print(f"check_train {check_train}, check_test {check_test}")
        if check_train and check_test:
            idx = table[table['score'] != 0]['score'].idxmin()
            print('best_params',table.loc[idx])
            for i in range(1,5):
                new_row = table.loc[idx].copy()
                new_row['seed'] = i
                new_row['score'] = ''
                new_row['test_score'] = ''
                table.loc[len(table.index)] = new_row.values
            table.to_csv(file_experiments, sep=';', index=False)
            print('Please retrain it for new settings...')
        else:
            print('!!! Something wrong/only partial settings were trained !!!')
