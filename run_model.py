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

def run_model():
    torch.set_float32_matmul_precision('high')
    task = 'soil_nitrogen'
    split_type = 'geographic' # "world_random", "random", or "geographic"
    logger = WandbLogger(project='mmearth-bench', name=f'{task}_{split_type}', log_model=True)
    print('Logging run to Wandb')
    monitor = 'Val loss'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=logger.experiment.dir, save_top_k=1, save_last=True)
    print(f'Local log folder: {logger.experiment.dir}')
    early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=100)
    trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                      fast_dev_run=False,
                      log_every_n_steps=1,
                      logger=logger,
                      min_epochs=1,
                      max_epochs=200,
                      num_sanity_val_steps=0)
    datamodule = DataModule(task=task, dataset_class=MMEarthBenchDataset, split_type=split_type, batch_size=64, num_workers=0)
    model = Model(model='resnet_rgb')
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    if len(argv) == 1: # python run_model.py
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        subprocess.run(['sbatch', '-t', '3-00:00:00', '-p', partitions, '--mem', '20G', '--gres', 'gpu:1', '--job-name', 'run', '-o', 'bash-outputs/run.out', '-e', 'bash-errors/run.err', 'job.sh', env_path, 'run_model.py', 'run'])
    else: # python run_model.py run
        run_model()
