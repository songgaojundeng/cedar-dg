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
e.g., python train_main_time_count.py traffic deepar.csv deepar
for large datasets
'''

try:
    dataset_name = sys.argv[1]
    # setting_file = sys.argv[2]
    algorithm = sys.argv[2]
    d_hidden = int(sys.argv[3])
    reg = str(sys.argv[4])
except:
    print ('Usage: python train_main_time_count.py <dataset_name> <algorithm> <d_hidden> <reg>')
    exit()

print('dataset_name:{}, algorithm:{}, d_hidden:{}, reg:{}'.format(dataset_name, algorithm, d_hidden, reg))


cuda_available = torch.cuda.is_available()
print(f'cuda:{cuda_available}')
cuda = 0

early_stopping_patience = 10
scaling = True
epochs = 3

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

print(f"dim_inputseqlen={dim_inputseqlen} dim_outputseqlen={dim_outputseqlen}")
# file_experiments = experiment_dir + f'/{setting_file}'
# table = read_table(file_experiments)
# print(table)

d_emb = np.array([])

learning_rate = 0.001
batch_size = 1024

if not algorithm.startswith('deepar'): 
    kernel_size = 9
    
if algorithm.endswith('cedar'):
    gamma = 0.1
    mmd_type = 'linear'
    loss_dis = 'abs'
    reg = reg

if algorithm.startswith('deepar'):
    N = 3
elif algorithm.startswith('wavenet'):
    N = 5

dropout = 0.3
seed = 5
fix_seed(seed)

if cuda_available:
    device = torch.device("cuda:{}".format(cuda))
else:
    device = torch.device("cpu")

# Training loop
print('---- Hyperparameter setting ----')
filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed{seed}_hid{d_hidden}_lr{learning_rate}_bs{batch_size}_dp{dropout}_N{N}_reg{reg}"
os.makedirs(f"{experiment_dir}/{algorithm}", exist_ok=True) 

os.makedirs(f"tmp/time_analysis/", exist_ok=True) 

if not algorithm.startswith('deepar'):
    filename += f"_knl{kernel_size}"

if algorithm.endswith('cedar'):
    filename += f"_gm{gamma}_mmd{mmd_type}_ld{loss_dis}"

print(f'filename:{filename}')

time_list = []
for train_ratio in [0.8]:
    
    # init data
    dataset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, seed=seed, train_ratio=0.8)
    training_set = dataset.load('train')

    # Initialize sample sets
    id_samples_train = torch.randperm(len(training_set))
    print(f'num_samples_train:{len(training_set)}')
    n_batch_train = (len(id_samples_train) + batch_size - 1) // batch_size 
    print('n_batch_train',n_batch_train)

    if algorithm == 'deepar':
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, dropout, N]"
    elif algorithm == 'wavenet':
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, N, kernel_size]"
    elif  algorithm in ['deepar_cedar']:
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, dropout, N, mmd_type, gamma, loss_dis, reg]"
    elif  algorithm in ['wavenet_cedar']:
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, N, kernel_size,mmd_type,gamma, loss_dis, reg]"


    if 'model' in locals(): del model
    params = eval(params_train)
    # print('params=',params)
    fix_seed(seed)
    model = instantiate_model(algorithm)(*params).to(device) 

    start_training = time.time()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        if algorithm.endswith('cedar'):
            model, _, _, _, _, _ = loop_mmd(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
        else:
            model, _, _, _, _, _ = loop_basic(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    

    end_training = time.time()
    print('train_ratio =',train_ratio)
    used_time_per_epoch = (end_training-start_training)/epochs
    print(f"Training time: {end_training-start_training:.5f}s. Average time per epoch: {used_time_per_epoch:.5f}")
    time_list.append(used_time_per_epoch)
    print('---- Training Done ----\n')

# time_list = np.array(time_list)
print(f"algorithm:{algorithm} d_hidden:{d_hidden} reg:{reg}")
print(time_list)
with open(f'tmp/time_analysis/{dataset_name}_{algorithm}_hid{d_hidden}_batch{batch_size}_reg{reg}.txt','w') as f:
    f.write(f"algorithm:{algorithm} d_hidden:{d_hidden} reg:{reg}\n")
    f.write(str(time_list))
    