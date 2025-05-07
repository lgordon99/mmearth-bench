# ============================================== IMPORTS ============================================== #

from concurrent.futures import ThreadPoolExecutor
from sys import argv
import pandas as pd
import wandb

# ============================================== FUNCTIONS ============================================== #

def get_run_data(run, hyperparameters):
    if run.state != 'finished':
        print(f'Not all runs are finished for {sweep.name}')
        exit()

    run_data = {'name': run.name,
                'Val RMSE': round(run.history(keys=['Val RMSE'])['Val RMSE'].min(), 2),
                'Random test RMSE': round(run.summary_metrics['Random test RMSE/dataloader_idx_0'], 2),
                'Geographic test RMSE': round(run.summary_metrics['Geographic test RMSE/dataloader_idx_1'], 2),}
    run_hyperparameter_data = {hyperparameter: run.config[hyperparameter] for hyperparameter in hyperparameters}
    run_data = {**run_data, **run_hyperparameter_data}

    return run_data

def monitor_sweep(name):
    sweep_log_df = pd.read_csv('sweep_log.csv')
    sweep_ID = sweep_log_df[sweep_log_df['name'] == name]['sweep_ID'].values[0]
    sweep = wandb.Api().sweep(sweep_ID)
    hyperparameters = sweep.config['parameters'].keys()
    all_run_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_run_data, run, hyperparameters) for run in sweep.runs]
        all_run_data = [future.result() for future in futures if future.result() is not None]

    # for run in sweep.runs:
    #     if run.state != 'finished':
    #         print(f'Not all runs are finished for {sweep.name}')
    #         exit()

    #     run_data = {'name': run.name,
    #                 'Val RMSE': round(run.history(keys=['Val RMSE'])['Val RMSE'].min(), 2),
    #                 'Random test RMSE': round(run.summary_metrics['Random test RMSE/dataloader_idx_0'], 2),
    #                 'Geographic test RMSE': round(run.summary_metrics['Geographic test RMSE/dataloader_idx_1'], 2),}
    #     run_hyperparameter_data = {hyperparameter: run.config[hyperparameter] for hyperparameter in hyperparameters}
    #     run_data = {**run_data, **run_hyperparameter_data}
    #     all_run_data.append(run_data)

    df = pd.DataFrame(all_run_data).sort_values(by='Val RMSE')
    best_run = df.iloc[0]

    return best_run['Random test RMSE'], best_run['Geographic test RMSE']

if __name__ == '__main__':
    monitor_sweep(name=argv[1])
