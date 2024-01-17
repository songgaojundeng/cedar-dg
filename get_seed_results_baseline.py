import pandas as pd
import numpy as np
import os, sys
from lib.utils import * 


'''
E.g.: python get_seed_results_baseline.py traffic deepar.csv deepar
'''
try:
    dataset_name = sys.argv[1]
    setting_file = sys.argv[2]
    algorithm = sys.argv[3]
except:
    print ('Usage: python train.py <dataset_name> <setting_file> <algorithm>')
    exit()
print('dataset_name:{}, setting_file:{}, algorithm:{}'.format(dataset_name,setting_file,algorithm))

experiment_dir = 'experiments/'+dataset_name
file_experiments = experiment_dir + f'/{setting_file}'

df = pd.read_csv(file_experiments,sep=';')
best_param = df[df['seed']==2]
print('best_param', best_param)
learning_rate = best_param['learning_rate'].values[0]
d_hidden = int(best_param['d_hidden'].values[0])
batch_size = int(best_param['batch_size'].values[0])
dropout = best_param['dropout'].values[0]
N = int(best_param['N'].values[0])

if not algorithm.startswith('deepar') and not algorithm.startswith('vrnn') and algorithm != 'adarnn':
    kernel_size = int(best_param['kernel_size'].values[0])

if algorithm.endswith('mmd'):
    gamma = float(best_param['gamma'].values[0])
    mmd_type = str(best_param['mmd'].values[0])

if algorithm.endswith('fish'):
    meta_lr = float(best_param['meta_lr'].values[0])

if algorithm.endswith('mldg'):
    mldg_beta = float(best_param['mldg_beta'].values[0])
    
test_res_dict = {'RMSE':[],'NRMSE':[],'ND':[],'MAPE':[],
                 'sMAPE':[],'QuantileLoss5':[],'QuantileLoss9':[],
                 'QuantileLossMean':[]}

for i in range(5):
    seed = i
    filename_test = f"{experiment_dir}/{algorithm}/{algorithm}_seed{seed}_hid{d_hidden}_lr{learning_rate}_bs{batch_size}_dp{dropout}_N{N}"
    if not algorithm.startswith('deepar') and not algorithm.startswith('vrnn') and algorithm != 'adarnn':
        filename_test += f"_knl{kernel_size}"

    if algorithm.endswith('mmd'):
        filename_test += f"_gm{gamma}_mmd{mmd_type}"

    if algorithm.endswith('fish'):
        filename_test += f"_mlr{meta_lr}"

    if algorithm.endswith('mldg'):
        filename_test += f"_mlb{mldg_beta}"
        
    filename_test += "_test.csv"
    print('filename_test: ',filename_test,'exist',os.path.exists(filename_test))

    df_test = pd.read_csv(filename_test,sep=',',index_col=[0])

    for key in test_res_dict:
        if key == 'QuantileLoss5':
            test_res_dict[key].append(df_test[df_test.Quantile == 0.5]['QuantileLoss'].values)
        elif key == 'QuantileLoss9':
            test_res_dict[key].append(df_test[round(df_test.Quantile,1) == 0.9]['QuantileLoss'].values)
        elif  key == 'QuantileLossMean':
            test_res_dict[key].append(df_test['QuantileLoss'].mean())
        else:
            test_res_dict[key].append(df_test[df_test.Quantile == 0.5][key].values)
 
for key in test_res_dict:
    cur_val = np.array(test_res_dict[key])
    print(key,"{:.4f}, {:.4f}".format(cur_val.mean(), cur_val.std()))