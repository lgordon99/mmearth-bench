# ============================================== IMPORTS ============================================== #

from datamodule import DataModule
from dataset import MMEarthBenchDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import Model
import torch

torch.set_float32_matmul_precision('high')
monitor = 'Val loss'
patience = 10
logger = WandbLogger(project='termite-mound-detector', log_model=True)
checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=logger.experiment.dir, save_top_k=1, save_last=True)
print('Log dir: ', logger.experiment.dir)
early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience)
trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                  fast_dev_run=False,
                  log_every_n_steps=1,
                  logger=logger,
                  min_epochs=1,
                  max_epochs=50)
model = Model(model='resnet_rgb')
task = 'soil_nitrogen'
datamodule = DataModule(task=task, dataset_class=MMEarthBenchDataset, batch_size=128, num_workers=4)
trainer.fit(model, datamodule=datamodule)
