import time, sys, os
import torch
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
from lib.utils import fix_seed, calc_metrics
from data import timeseries_dataset
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel

num_cores = 2

try:
    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])
except:
    print ('Usage: python train.py <dataset_name> <seed>')
    exit()

print(f'dataset_name:{dataset_name} seed:{seed}')


cuda_available = torch.cuda.is_available()
print(f'cuda:{cuda_available}')
cuda = 0

early_stopping_patience = 10
scaling = True

os.makedirs("experiments", exist_ok=True) 
experiment_dir = 'experiments/'+dataset_name

#%%  Import data
if dataset_name.startswith(('favorita','stock')):
    dim_inputseqlen = 60
    dim_outputseqlen = 14
    dim_maxseqlen = dim_inputseqlen + dim_outputseqlen * 2
    frequency = 'D'
    seasonality = 7
elif dataset_name in ['traffic']:
    dim_inputseqlen = 28
    dim_outputseqlen = 7
    dim_maxseqlen = dim_inputseqlen + dim_outputseqlen * 2
    frequency = 'D'
    seasonality = 7
else:
    dim_inputseqlen = 30
    dim_outputseqlen = 7
    dim_maxseqlen = dim_inputseqlen + dim_outputseqlen * 2 
    frequency = 'D'
    seasonality = 10

print(f"dim_inputseqlen={dim_inputseqlen} dim_outputseqlen={dim_outputseqlen}") 

d_emb = np.array([])
fix_seed(seed)
dataset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, seed=seed)
test_set = dataset.load('test')
id_samples_test = torch.randperm(len(test_set)) 
num_samples_test = len(test_set)
print(f'num_samples_test:{len(test_set)}')
data_subset = torchdata.Subset(test_set, id_samples_test)
num_samples = len(id_samples_test)
data_generator = torchdata.DataLoader(data_subset)
data_max = 1e4
data_min = -1e4 if dataset_name.startswith('same') else 0
quantiles = np.arange(1, 10) / 10
yhat_tot_test = np.zeros((3, len(quantiles), dim_outputseqlen, num_samples_test), dtype='float32')
y_tot_test = np.zeros((dim_outputseqlen, num_samples_test))
alphas = [0.2, 0.4, 0.6, 0.8, 1]

t_ets, t_snaive, t_theta = 0, 0, 0
for i, (X, y) in enumerate(data_generator):
    y = y[:,-dim_outputseqlen:]

    y_train = X[0, :dim_inputseqlen + 1, -1].double().numpy()
    y_test =  X[0, dim_inputseqlen + 1:dim_inputseqlen + dim_outputseqlen + 1, -1]
    print('y_train',y_train.shape, 'y_test',y_test.shape)
    print('y_train',y_train, 'y_test',y_test)
 
    assert torch.allclose(y_test, y.squeeze()[:-1])
    y_tot_test[:, i] = y.squeeze().numpy()

    df_train = pd.Series(y_train, index=pd.date_range('2015-01-01', periods=len(y_train), freq=frequency))

    # ETS model
    a = time.perf_counter()
    ets_model = ETSModel(df_train,  error="add", trend="add", seasonal="add", damped_trend=True, seasonal_periods=seasonality)
    # ets_model = ETSModel(df_train,  error="add", trend="add", seasonal='add', damped_trend=True, seasonal_periods=None)
    fit_ets = ets_model.fit(maxiter=10000)
    b = time.perf_counter()
    t_ets += b - a
    simulated = fit_ets.simulate(anchor="end", nsimulations=dim_outputseqlen, repetitions=10000).clip(data_min, data_max)
    yhat_tot_test[0, :, :, i] = np.quantile(simulated, quantiles, axis=1)
    # Seasonal naive model
    a = time.perf_counter()
    yhat_snaive = X[0, dim_inputseqlen + 1 - dim_outputseqlen:dim_inputseqlen + 1, -1].numpy()
    yhat_tot_test[1, :, :, i] = yhat_snaive
    b = time.perf_counter()
    t_snaive += b-a    
    # Theta model
    a = time.perf_counter()
    theta_model = ThetaModel(df_train, period=seasonality)
    fit_theta = theta_model.fit()
    b = time.perf_counter()
    t_theta += b-a
    for j, alpha in enumerate(alphas):
        pi = fit_theta.prediction_intervals(dim_outputseqlen, alpha=alpha)
        yhat_tot_test[2, j, :, i] = pi.lower.clip(data_min, data_max)
        yhat_tot_test[2, -1 - j, :, i] = pi.upper.clip(data_min, data_max)
 
algorithms = ['ets', 'seasonalnaive', 'theta']
for a, algorithm in enumerate(algorithms):
    os.makedirs(f"{experiment_dir}/{algorithm}", exist_ok=True) 
    filename = f"{experiment_dir}/{algorithm}/{algorithm}"
    df_test = calc_metrics(yhat_tot_test[a], y_tot_test, quantiles)
    df_test.to_csv(f'{filename}_seed{seed}_test.csv')
timings = pd.DataFrame(np.array([t_ets, t_snaive, t_theta])).T
timings.columns = algorithms
timings.to_csv(f"{experiment_dir}/timings_traditional.csv")