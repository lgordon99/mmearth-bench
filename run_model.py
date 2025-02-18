# ============================================== IMPORTS ============================================== #

from datamodule import DataModule
from dataset import MMEarthBenchDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import Model
from sys import argv
import h5py
import json
import numpy as np
import random
import subprocess
import torch
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

# ============================================== FUNCTIONS ============================================== #

def run_model():
    torch.set_float32_matmul_precision('high')
    monitor = 'Val loss'
    patience = 100
    logger = WandbLogger(project='mmearth-bench', log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=logger.experiment.dir, save_top_k=1, save_last=True)
    print('Log dir: ', logger.experiment.dir)
    early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience)
    trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                    fast_dev_run=False,
                    log_every_n_steps=1,
                    logger=logger,
                    min_epochs=1,
                    max_epochs=200)
    model = Model(model='resnet_sentinel2')
    task = 'soil_nitrogen'

    # with h5py.File(f'{data_dir_path}/{task}/{task}_h5.hdf5', 'r') as h5_file:
    #     tile_count = len(h5_file['Sentinel2'])
    #     print(f'{task} tile count: {tile_count}')

    #     sentinel2 = h5_file['Sentinel2'][:]
    #     print(f'Sentinel-2: {sentinel2.shape}')

    #     tile_ids = h5_file['id'][:]
    #     print(tile_ids.shape)

    # random.seed(42)
    # random.shuffle(tile_ids)
    # end_train_ids = int(0.7 * tile_count)
    # end_val_ids = int(0.85 * tile_count)

    # train_ids = tile_ids[:end_train_ids].tolist()
    # val_ids = tile_ids[end_train_ids:end_val_ids].tolist()
    # test_ids = tile_ids[end_val_ids:].tolist()
    # print(f'Dataset lengths: Train = {len(train_ids)}, Val = {len(val_ids)}, Test = {len(test_ids)}')

    # train_indices = np.array([np.where(tile_ids == tile_id)[0][0] for tile_id in train_ids])
    # train_images = sentinel2[train_indices]
    # train_band_means = train_images.mean(axis=(0,2,3))[:, None, None].tolist()
    # train_band_stds = train_images.std(axis=(0,2,3))[:, None, None].tolist()
    # split_data = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids, 'train_band_means': train_band_means, 'train_band_stds': train_band_stds}

    # with open('split_data.json', 'w') as file:
    #     json.dump(split_data, file, indent=4)

    datamodule = DataModule(task=task, dataset_class=MMEarthBenchDataset, batch_size=128, num_workers=1)
    # datamodule = DataModule(task=task, dataset_class=MMEarthBenchDataset, split_data=split_data, batch_size=128, num_workers=0)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    if len(argv) == 1: # python run_model.py
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        subprocess.run(['sbatch', '-t', '3-00:00:00', '-p', partitions, '--mem', '20G', '--cpus-per-task', '4', '--gres', 'gpu:1', '--job-name', 'run', '-o', 'bash-outputs/run.out', '-e', 'bash-errors/run.err', 'job.sh', env_path, 'run_model.py', 'run'])
    else: # python run_model.py run
        run_model()
