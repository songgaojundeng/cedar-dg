import pandas as pd
import os, sys

'''
E.g.,
python gen_settings_from_base.py traffic deepar deepar_cedar deepar_cedar 64
python gen_settings_from_base.py traffic deepar deepar_mmd deepar_mmd 64

'''

try:
    dataset_name = sys.argv[1]
    base_model_file = sys.argv[2]
    target_model = sys.argv[3]
    outfile_name = sys.argv[4]
    new_batch_size = sys.argv[5]
except:
    print ('Usage: python train.py <dataset_name> <base_model_file `deepar/wavenet`> <target_model `deepar_cedar`> <outfile_name `deepar_cedar`> <new_batch_size `64`>')
    exit()

print(f'dataset_name:{dataset_name}, base_model_file:{base_model_file}, target_model:{target_model}, outfile_name:{outfile_name} new_batch_size:{new_batch_size}')


# read base model settings
base_model_file = f"experiments/{dataset_name}/{base_model_file}.csv"
print('base_model_file',base_model_file)
target_model_file = f"experiments/{dataset_name}/{outfile_name}.csv"
print('target_model_file',target_model_file)

assert os.path.exists(base_model_file), f"base model setting file not exist"
 
df = pd.read_csv(base_model_file,sep=';')
print(df)
best_param = df[df['seed']==2]
print('best_param', best_param)
base_model = best_param['algorithm'].values[0]
best_learning_rate = best_param['learning_rate'].values[0]
best_d_hidden = int(best_param['d_hidden'].values[0])
best_batch_size = int(best_param['batch_size'].values[0])
best_N = int(best_param['N'].values[0])
best_dropout = best_param['dropout'].values[0]
algorithm = target_model
seed = 0

if target_model.endswith('_mmd'):
    gamma = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    loss_dis = ['']
elif target_model.endswith('_cedar'):
    gamma = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    loss_dis = ['abs','sqaure'] 
    reg = ['dd','ddd']
else:
    print(f'{target_model} is not supported yet')

mmd = 'linear'
if base_model == 'deepar':
    if target_model.endswith('_mmd'):
        columns = "algorithm;seed;learning_rate;batch_size;d_hidden;N;mmd;gamma;dropout;score;in_progress;test_score;params_test;params_train"
        params_test = "[test_set.d_lag, test_set.d_cov, d_emb, test_set.dim_output, d_hidden, dropout, N, mmd_type, gamma]"
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, dropout, N, mmd_type, gamma]"
    else:
        columns = "algorithm;seed;learning_rate;batch_size;d_hidden;N;mmd;loss_dis;reg;gamma;dropout;score;in_progress;test_score;params_test;params_train"
        params_test = "[test_set.d_lag, test_set.d_cov, d_emb, test_set.dim_output, d_hidden, dropout, N, mmd_type, gamma, loss_dis, reg]"
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, dropout, N, mmd_type, gamma, loss_dis, reg]"

elif base_model == 'wavenet':
    if target_model.endswith('_mmd'):
        best_kernel_size = best_param['kernel_size'].values[0]
        columns = "algorithm;seed;learning_rate;batch_size;d_hidden;kernel_size;N;mmd;gamma;dropout;score;in_progress;test_score;params_test;params_train"
        params_test = "[test_set.d_lag, test_set.d_cov, d_emb, test_set.dim_output, d_hidden, N, kernel_size, mmd_type, gamma]"
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, N, kernel_size, mmd_type, gamma]"
    else:
        best_kernel_size = best_param['kernel_size'].values[0]
        columns = "algorithm;seed;learning_rate;batch_size;d_hidden;kernel_size;N;mmd;loss_dis;reg;gamma;dropout;score;in_progress;test_score;params_test;params_train"
        params_test = "[test_set.d_lag, test_set.d_cov, d_emb, test_set.dim_output, d_hidden, N, kernel_size, mmd_type, gamma, loss_dis, reg]"
        params_train = "[training_set.d_lag, training_set.d_cov, d_emb, training_set.dim_output, d_hidden, N, kernel_size, mmd_type, gamma, loss_dis, reg]"
else:
    print(f'{base_model} is not supported yet')     

target_model_settings = pd.DataFrame(columns = columns.split(';'))

if target_model.endswith('_mmd'):
    for gamma_v in gamma:
        new_row = {}
        new_row['algorithm'] = target_model
        new_row['seed'] = 0
        new_row['learning_rate'] = best_learning_rate
        new_row['batch_size'] = new_batch_size
        new_row['d_hidden'] = best_d_hidden
        new_row['N'] = best_N
        if base_model == 'wavenet':
                new_row['kernel_size'] = best_kernel_size
        new_row['mmd'] = mmd
        new_row['gamma'] = gamma_v
        new_row['dropout'] = best_dropout
        new_row['in_progress'] = -1
        new_row['params_test'] = params_test
        new_row['params_train'] = params_train
        target_model_settings = target_model_settings._append(new_row, ignore_index = True)
else:
    for reg_v in reg:
        for loss_dis_v in loss_dis:
            for gamma_v in gamma:
                new_row = {}
                new_row['algorithm'] = target_model
                new_row['seed'] = 0
                new_row['learning_rate'] = best_learning_rate
                new_row['batch_size'] = new_batch_size
                new_row['d_hidden'] = best_d_hidden
                new_row['N'] = best_N
                if base_model == 'wavenet':
                        new_row['kernel_size'] = best_kernel_size
                new_row['mmd'] = mmd
                new_row['loss_dis'] = loss_dis_v
                new_row['reg'] = reg_v
                new_row['gamma'] = gamma_v
                new_row['dropout'] = best_dropout
                new_row['in_progress'] = -1
                new_row['params_test'] = params_test
                new_row['params_train'] = params_train
                target_model_settings = target_model_settings._append(new_row, ignore_index = True)

target_model_settings.to_csv(target_model_file, sep=';', index=False)
print(target_model_settings)
print(f'{target_model_file} saved!')
