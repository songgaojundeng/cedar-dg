# Cedar
Source code and supplementary materials for "Domain Generalization in Time Series Forecasting" appeared in TKDD

## Contents of this repository
* Source code and datasets.
* Pipelines about how to run and get results.
* Visualization of some datasets.


## Prerequisites
The code has been successfully tested in the following environment:
* Python 3.9.15
* PyTorch 1.7.1
* CUDA 10.2
* Numpy 1.23.5
* Pandas 1.5.3

## Folder structure
```sh
- cedar-dg
    - algorithms # model python files
	- data # dataset folders
		- kaggle_favorita
		- stock
		- traffic
        - syn # synthetic datasets
	- experiments # store experiment settings and results
        - base_settings 
            - deepar.csv
            - ... (other setting files)
    - lib # evaluation files, etc
    - preprocess # generate synthetic datasets
    - data.py # data loader file
	- train_main.py
    - ... (other python files)
```
## Getting Started
### Prepare your code
Clone this repo:
```bash
git clone https://github.com/songgaojundeng/cedar-dg
cd cedar-dg
```
### Create experiment folder and setting files
Choose one dataset from `traffic`, `favorita_family`, `favorita_family_store`, `stock_vol`, `samemv_diffp30`, `samep_diffmv30`, `samepmv_difft30`, `samet_diffpmv30`. Taking `traffic` as the dataset example, run the following commands:
```
cd experiments
mkdir traffic
cp base_settings/*.csv traffic
```

### Train baselines in `deepar`, `wavenet` (base), `adarnn`, `vrnn`, `[base]_dann`,  `[base]_groupdro`, `[base]_mldg`, `[base]_fish`
* Taking model `deepar` as the example, run the following command 2 times (at root directory). 
```
python train_main.py traffic deepar.csv deepar # run 2 times
```
The first time: train under one seed and find the best parameter. The second time: train again under other seeds.
### Train baseline `[base]_mmd`
* Step 1: Generate the optimal experimental settings from the base model `deepar` (64 is the batch size):
```
python gen_settings_from_base.py traffic deepar deepar_mmd deepar_mmd 64
```
* Step 2: Train the model `deepar_mmd` under different settings:
```
python train_main.py traffic deepar_mmd.csv deepar_mmd # run 2 times
```
### Train Cedar `[base]_cedar`
* Step 1: Generate the optimal experimental settings from the base model  `deepar` (64 is the batch size):
```
python gen_settings_from_base.py traffic deepar deepar_cedar deepar_cedar 64
```
* Step 2: Train the model `deepar_cedar` under different settings:
```
python train_main.py traffic deepar_cedar.csv deepar_cedar # run 2 times
```
### Read results
* for Cedar
```
python get_seed_results_cedar.py traffic deepar_cedar.csv deepar_cedar
```
* for all other baselines
```
python get_seed_results_baseline.py traffic deepar.csv deepar
```

### 6. Train traditional time series models
```
python train_traditional.py traffic 0
```


### Cite
Please cite our paper if you find this code useful for your research:

```
To update
```
