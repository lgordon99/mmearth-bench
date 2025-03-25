# ============================================== IMPORTS ============================================== #

from datamodule import DataModule
from dataset import MMEarthBenchDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import Model
from sys import argv
import subprocess
import torch
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

# ============================================== FUNCTIONS ============================================== #

def run_model(task):
    torch.set_float32_matmul_precision('high')
    print(f'Task: {task}')
    split_type = 'random' # "world_random", "random", or "geographic"
    print(f'Split type: {split_type}')
    model = 'mmearth-pixelwise_regression'
    print(f'Model: {model}')
    print('Logging run to Wandb')
    logger = WandbLogger(project='mmearth-bench', name=f'{task}_{split_type}_{model}', log_model=True)
    print(f'Local log folder: {logger.experiment.dir}')
    monitor = 'Val loss'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=logger.experiment.dir, save_top_k=1, save_last=True)
    early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=100)
    trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                    #   fast_dev_run=False,
                      log_every_n_steps=1,
                      logger=logger,
                    #   min_epochs=1,
                      max_epochs=200,
                      num_sanity_val_steps=0)
    datamodule = DataModule(task=task, dataset_class=MMEarthBenchDataset, split_type=split_type, batch_size=64, h5_file_path=f'/scratch/{task}_h5.hdf5', num_workers=0)
    model = Model(task=task, model=model)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    if 'for' not in argv[1]: # python run_model.py TASK
        partitions = utils.read_yaml('config-user.yml')['gpu_partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        mem = 60 if argv[1] == 'biomass' else 20
        subprocess.run(['sbatch', '-t', '1-00:00:00', '-p', partitions, '--mem', f'{mem}G', '--gres', 'gpu:1', '--job-name', 'run', '-o', 'bash-outputs/run.out', '-e', 'bash-errors/run.err', 'job.sh', env_path, 'run_model.py', f'for_{argv[1]}'])
    else: # python run_model.py for_TASK
        task = argv[1].split('for_')[1]
        run_model(task)
