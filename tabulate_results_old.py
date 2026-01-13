# ============================================== IMPORTS ============================================== #

import pandas as pd
import numpy as np
import utils
import wandb
from concurrent.futures import ThreadPoolExecutor
from sys import argv
from tqdm import tqdm

# ============================================== GLOBAL VARIABLES ============================================== #

models = ['ResNet50', 'ResNet50-ImageNet', 'DINOv2', 'MPMAE', 'MPMAE-MMEarth']
splits = ['Random', 'Geographic']
adaptation_methods = ['lp', 'ft', 'llrd']

# ============================================== FUNCTIONS ============================================== #

def get_run_data(run, hyperparameters):
    if run.state != 'finished':
        print(f'{run.name} is not finished yet')
        exit()

    try:
        run_data = {'name': run.name,
                    'ID': run.id,
                    'Val RMSE': run.history(keys=['Val RMSE'])['Val RMSE'].min(),
                    'Random test RMSE': run.summary_metrics['Random test RMSE/dataloader_idx_0'],
                    'Random test R2': run.summary_metrics['Random test R2/dataloader_idx_0'],
                    'Random test MAE': run.summary_metrics['Random test MAE/dataloader_idx_0'],
                    'Random test ME': run.summary_metrics['Random test ME/dataloader_idx_0'],
                    'Geographic test RMSE': run.summary_metrics['Geographic test RMSE/dataloader_idx_1'],
                    'Geographic test R2': run.summary_metrics['Geographic test R2/dataloader_idx_1'],
                    'Geographic test MAE': run.summary_metrics['Geographic test MAE/dataloader_idx_1'],
                    'Geographic test ME': run.summary_metrics['Geographic test ME/dataloader_idx_1']
                    }
    except KeyError as e:
        print(f'KeyError: {e} in run {run.name}')
        exit()

    run_hyperparameter_data = {hyperparameter: run.config[hyperparameter] for hyperparameter in hyperparameters}
    run_data = {**run_data, **run_hyperparameter_data}

    return run_data

def get_best_run_in_sweep(name, data_dir_path):
    sweep_log_df = pd.read_csv(f'{data_dir_path}/experiments/sweep_log.csv')
    sweep_ID = sweep_log_df[sweep_log_df['name'] == name]['sweep_ID'].values[0]
    sweep = wandb.Api().sweep(sweep_ID)
    hyperparameters = sweep.config['parameters'].keys()
    all_run_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_run_data, run, hyperparameters) for run in sweep.runs]
        all_run_data = [future.result() for future in futures if future.result() is not None]

    df = pd.DataFrame(all_run_data).sort_values(by='Val RMSE')
    best_run = df.iloc[0]

    for hyperparameter in hyperparameters:
        print(f'{hyperparameter}: {best_run[hyperparameter]}')

    print(best_run['Random test RMSE'], best_run['Geographic test RMSE'])

    return best_run

def create_table(task):
    # metrics = ['RMSE', 'R2', 'MAE', 'ME']
    metrics = ['RMSE']
    results_arrays = {metric: np.full((len(models), len(splits) * len(adaptation_methods)), np.nan) for metric in metrics}

    for i, model in enumerate(tqdm(models, desc='Models', position=0)):
        for j, adaptation_method in enumerate(tqdm(adaptation_methods, desc='Adaptation methods', position=1, leave=False)):
            model_name = model.lower().replace('-', '_')
            run = get_best_run_in_sweep(name=f'{task}_{model_name}_{adaptation_method}')

            for metric in metrics:
                results_arrays[metric][i, j] = run[f'Random test {metric}']
                results_arrays[metric][i, j + len(adaptation_methods)] = run[f'Geographic test {metric}']

    columns = pd.MultiIndex.from_tuples([('Random', 'LP'), ('Random', 'FT'),
                                         ('Random', 'LLRD'), ('Geographic', 'LP'),
                                         ('Geographic', 'FT'), ('Geographic', 'LLRD')])
    dataframes = {metric: pd.DataFrame(results_arrays[metric], index=models, columns=columns) for metric in metrics}

    for metric in metrics:
        print(metric)
        dataframes[metric] = dataframes[metric].round(2)
        print(dataframes[metric])

if __name__ == '__main__':
    # create_table(task=argv[1])
    get_best_run_in_sweep(name='biomass_dinov2_ft')
