import numpy as np
import utils
import wandb

entity = utils.read_yaml('config-user.yml')['entity']
project = utils.read_yaml('config-user.yml')['project']
adaptation_mode = 'val-JT-TTT'
runs = wandb.Api().runs(f'{entity}/{project}', filters={'config.datamodule.adaptation_mode': adaptation_mode, 'config.datamodule.batch_size': 8})
architectures = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'Satlas', 'MPMAE', 'TerraMind', 'CopernicusFM']
tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']

def get_best_batch_size(task, architecture):
    name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_'
    candidate_runs = [run for run in runs if run.name.startswith(name)]
    metric = 'R2' if task != 'species' else 'MAP'
    val_metrics = [run.summary_metrics.get(f'Val {metric}') for run in candidate_runs]
    best_batch_size = candidate_runs[np.argmax(val_metrics)].config['datamodule']['batch_size']

    return best_batch_size

def get_best_lr(task, architecture):
    name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_'
    candidate_runs = [run for run in runs if run.name.startswith(name)]
    metric = 'R2' if task != 'species' else 'MAP'
    val_metrics = [run.summary_metrics.get(f'Val {metric}') for run in candidate_runs]
    best_lr = candidate_runs[np.argmax(val_metrics)].config['model']['inner_loop_lr']

    return best_lr

for architecture in architectures:
    # print(f'{architecture}: batch size {get_best_batch_size("biomass", architecture)}')
    print(f'{architecture}: lr {get_best_lr("biomass", architecture)}')
