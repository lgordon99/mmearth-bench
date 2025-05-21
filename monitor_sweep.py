# ============================================== IMPORTS ============================================== #

from concurrent.futures import ThreadPoolExecutor
from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import wandb

# ============================================== GLOBAL VARIABLES ============================================== #

num_decimals = 1

# ============================================== FUNCTIONS ============================================== #

def get_run_data(run, hyperparameters):
    if run.state != 'finished':
        print(f'{run.name} is not finished yet')
        exit()

    run_data = {'name': run.name,
                'Val RMSE': round(run.history(keys=['Val RMSE'])['Val RMSE'].min(), num_decimals),
                'Random test RMSE': round(run.summary_metrics['Random test RMSE/dataloader_idx_0'], num_decimals),
                'Geographic test RMSE': round(run.summary_metrics['Geographic test RMSE/dataloader_idx_1'], num_decimals)}
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

    df = pd.DataFrame(all_run_data).sort_values(by='Val RMSE')
    best_run = df.iloc[0]
    # print(best_run['model.max_lr'])
    # print(best_run['model.weight_decay'])
    # pivot = df.pivot_table(index='model.max_lr', columns='model.weight_decay', values='Val RMSE')
    # fig, ax = plt.subplots(dpi=300)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # c = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap='viridis')
    # ax.set_xlabel('Weight decay')
    # ax.set_ylabel('Max learning rate')
    # ax.set_title('Val RMSE Sensitivity to Hyperparameters')
    # fig.colorbar(c, label='Val RMSE')
    # plt.tight_layout()
    # plt.savefig(f'{name}_val_rmse_sensitivity.png')

    # fig, ax = plt.subplots(dpi=300)
    # scatterplot = ax.scatter(df['model.weight_decay'], df['model.max_lr'], c=df['Val RMSE'], cmap='viridis', s=50, edgecolor='k', alpha=0.8)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel('Weight decay')
    # ax.set_ylabel('Max learning rate')
    # ax.set_title(name.replace('_', ' ').capitalize().replace('mpmae mmearth', 'MPMAE-MMEarth').replace('ft', 'FT').replace('dinov2', 'DINOv2').replace('resnet50', 'ResNet50').replace('llrd', 'LLRD').replace('lp', 'LP').replace(' imagenet', '-ImageNet').replace('mpmae', 'MPMAE'))
    # cbar = fig.colorbar(scatterplot, ax=ax)
    # cbar.set_label('Val RMSE')

    # plt.tight_layout()
    # plt.savefig(f'{name}_val_rmse_sensitivity.png')

    return best_run['Random test RMSE'], best_run['Geographic test RMSE']

if __name__ == '__main__':
    monitor_sweep(name=argv[1])
