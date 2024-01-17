import numpy as np
import random
import torch
import torch.utils.data as torchdata
import pandas as pd


class timeseries_dataset():
    def __init__(self, name, dim_inputseqlen, dim_outputseqlen, seed, train_ratio=0.8):
        self.name = name
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.seed = seed
        self.train_ratio = train_ratio
        
    def load(self, mode):
        if self.name.startswith('same'):
            output = syn_data(self.dim_inputseqlen, self.dim_outputseqlen, mode, self.name, seed=self.seed, filename=self.name, train_ratio=self.train_ratio)
        else:
            if self.name == 'favorita_family':
                filename = 'y15_s0_family26'
                folder = 'kaggle_favorita'
                key = 'favorita'
            elif self.name == 'favorita_family_store':
                filename = 'y15_s0_store45'
                folder = 'kaggle_favorita'
                key = 'favorita'
            elif self.name == 'traffic':
                filename = 'y22_ca_s19'
                folder = 'traffic'
                key = 'traffic'
            elif self.name == 'stock_vol':
                filename = 'y2021_stock12'
                folder = 'stock'
                key = 'stock'
            output = realworld_data(self.dim_inputseqlen, self.dim_outputseqlen, mode, self.name, seed=self.seed, filename=filename, folder=folder, key=key, train_ratio=self.train_ratio)
        return output


class syn_data(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, mode, name, seed=0, filename='',train_ratio=0.8):
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        # self.dim_maxseqlen = dim_maxseqlen
        self.mode = mode 
        self.name = name
        self.dataset_filename = filename
        # test large datasets
        if filename.endswith('1m'):
            self.train_maxdate = 540000  
            self.validate_maxdate = 720000
        elif filename.endswith('100k'):
            self.train_maxdate = 54000  
            self.validate_maxdate = 72000
        elif filename.endswith('10k'):
            self.train_maxdate = 5400  
            self.validate_maxdate = 7200
        elif filename.endswith('1k'):
            self.train_maxdate = 540  
            self.validate_maxdate = 720
        else:
            self.train_maxdate = 270  
            self.validate_maxdate = 360
        self.seed = seed
        self.links = None
        self.train_doms = None
        self.test_doms = None
        self.train_ratio = train_ratio
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx]:self.index[idx] + self.dim_outputseqlen + self.dim_inputseqlen]
        y = self.Y[self.index[idx]:self.index[idx] + self.dim_outputseqlen + self.dim_inputseqlen]
        return x, y  

    def get_data(self):
        df = pd.read_hdf('data/syn/df_{}.h5'.format(self.dataset_filename), key=self.dataset_filename)
        index = pd.read_hdf('data/syn/df_{}.h5'.format(self.dataset_filename), key='index') 

        df_Y = df[['x']] 
        df_X = df[['domain','x_lagged']]
        
        all_doms = df_X['domain'].unique()
        random.seed(self.seed)
        random.shuffle(all_doms)

        num_doms = len(all_doms)
        num_train_doms = int(num_doms*self.train_ratio) 
        train_doms = all_doms[:num_train_doms] 
        test_doms = all_doms[num_train_doms:]

        if self.mode == 'train':
            print(df_X.columns)
            print(f"domain: {df_X['domain'].unique().shape}")
            print(f'all_doms:{all_doms} train_doms:{train_doms} {len(train_doms)}  test_doms:{test_doms} {len(test_doms)}')

        self.train_doms = train_doms
        self.test_doms = test_doms

        X = df_X.to_numpy(dtype='float32')
        Y = df_Y.to_numpy(dtype='float32')
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # use training domain validation `IN SEARCH OF LOST DOMAIN GENERALIZATION`
        if self.mode == 'train':
            idx = index[(index['date'] <= self.train_maxdate) & (index['domain'].isin(train_doms))]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        elif self.mode == 'validate':
            idx = index[(index['date'] <= self.validate_maxdate) & (index['date'] > self.train_maxdate) & (index['domain'].isin(train_doms))]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        elif self.mode == 'test':
            idx = index[(index['date'] > self.validate_maxdate) & (index['domain'].isin(test_doms))]['index'].to_numpy()
            self.index = torch.from_numpy(idx)

        # TODO add comments
        self.d_emb = 0 # number of embedding categoricals in input
        self.d_cov = 0
        self.d_lag = 1 

        self.x_dim = 1
        self.y_dim = 1  
        self.d_dim = num_doms  # number of domains
        self.x_bin_dim = 0  
        self.x_con_dim = 1 
        return X, Y

     
class realworld_data(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, mode, name, seed=0, filename='', folder='', key='',train_ratio=0.8):
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        self.mode = mode 
        self.name = name
        self.dataset_filename = filename
        self.dataset_folder = folder
        self.dataset_key = key
        if 'family' in self.dataset_filename or 'store' in self.dataset_filename:  # 2015-03-01 to 2015-12-31
            self.train_maxdate = '2015-06-30'   # 122/46/51
            self.validate_maxdate = '2015-08-15'
        elif self.dataset_filename.endswith('ca_s19'): # 2022-04-01 to 2017-11-30
            self.train_maxdate = '2022-07-15'  
            self.validate_maxdate = '2022-08-20'
        elif self.dataset_filename.endswith(('stock12')): # 2020-01-01 to 2021-5-31
            self.train_maxdate = '2020-08-30'  
            self.validate_maxdate = '2020-11-30'
        self.seed = seed
        self.links = None
        self.train_doms = None
        self.test_doms = None
        self.train_ratio = train_ratio
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx]:self.index[idx] + self.dim_outputseqlen + self.dim_inputseqlen]
        y = self.Y[self.index[idx]:self.index[idx] + self.dim_outputseqlen + self.dim_inputseqlen]
        return x, y  

    def get_data(self):
        df = pd.read_hdf(f'data/{self.dataset_folder}/df_{self.dataset_filename}.h5', key=self.dataset_key)
        index = pd.read_hdf(f'data/{self.dataset_folder}/df_{self.dataset_filename}.h5', key='index') 
        if 'store' in self.dataset_filename:
            df['unit_sales'] = np.log(df['unit_sales']+1) 
            df['unit_sales_lagged'] = np.log(df['unit_sales_lagged']+1)
            df_Y = df[['unit_sales']] 
            df_X = df[['store_nbr','DayOfWeek_sin','DayOfWeek_cos','unit_sales_lagged']]
            domain_name = 'store_nbr'
        elif 'family' in self.dataset_filename:
            df['unit_sales'] = np.log(df['unit_sales']+1) 
            df['unit_sales_lagged'] = np.log(df['unit_sales_lagged']+1)
            df_Y = df[['unit_sales']] 
            df_X = df[['family','DayOfWeek_sin','DayOfWeek_cos','unit_sales_lagged']]
            domain_name = 'family'
        elif self.dataset_filename.endswith('ca_s19'):
            df_Y = df[['vol']] 
            df_X = df[['station_id','DayOfWeek_sin','DayOfWeek_cos','vol_lagged']]
            domain_name = 'station_id'
        elif self.dataset_filename.endswith('stock12'):
            df['vol'] = df['vol'] / 1e7
            df['vol_lagged'] = df['vol_lagged'] / 1e7
            df_Y = df[['vol']] 
            df_X = df[['Index','DayOfWeek_sin','DayOfWeek_cos','vol_lagged']]
            domain_name = 'Index'
        all_doms = df_X[domain_name].unique()
        random.seed(self.seed)
        random.shuffle(all_doms)

        num_doms = len(all_doms)
        num_train_doms = int(num_doms*self.train_ratio) 
        train_doms = all_doms[:num_train_doms] 
        test_doms = all_doms[num_train_doms:]

        if self.mode == 'train':
            print(df_X.columns)
            print(f"domain: {df_X[domain_name].unique().shape}")
            print(f'all_doms:{all_doms} train_doms:{train_doms} {len(train_doms)}  test_doms:{test_doms} {len(test_doms)}')

        self.train_doms = train_doms
        self.test_doms = test_doms

        X = df_X.to_numpy(dtype='float32')
        Y = df_Y.to_numpy(dtype='float32')
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # use training domain validation `IN SEARCH OF LOST DOMAIN GENERALIZATION`
        if self.mode == 'train':
            idx = index[(index['date'] <= self.train_maxdate) & (index[domain_name].isin(train_doms))]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        elif self.mode == 'validate':
            idx = index[(index['date'] <= self.validate_maxdate) & (index['date'] > self.train_maxdate) & (index[domain_name].isin(train_doms))]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        elif self.mode == 'test':
            idx = index[(index['date'] > self.validate_maxdate) & (index[domain_name].isin(test_doms))]['index'].to_numpy()
            self.index = torch.from_numpy(idx)

        self.d_emb = 0 # number of embedding categoricals in input
        self.d_cov = 2
        self.d_lag = 1 

        self.x_dim = 3
        self.y_dim = 1  
        self.d_dim = num_doms  # number of domains
        self.x_bin_dim = 0  
        self.x_con_dim = 3 
        return X, Y

    