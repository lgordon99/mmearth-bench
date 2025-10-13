# ============================================== IMPORTS ============================================== #

from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
import hydra
import os
import shutil
import torch
import uuid
import wandb

# ============================================== GLOBAL ============================================== #

os.environ['JOB_ID'] = str(uuid.uuid4()).split('-')[0]
OmegaConf.register_new_resolver('env', lambda key: os.environ.get(key, ''))

# ============================================== FUNCTIONS ============================================== #

@hydra.main(config_path='.', config_name='train_config', version_base=None)
def train(cfg):
    print(f'Task: {cfg.task}')
    print(f'Encoder architecture: {cfg.architecture}')
    print(f'Adaptation mode: {cfg.adaptation_mode}')

    # set environment variables
    os.environ['DATA_DIR_PATH'] = cfg.data_dir_path
    os.environ['ENTITY'] = cfg.logger.entity
    os.environ['PROJECT'] = cfg.logger.project

    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.seed, workers=True)
    logger = hydra.utils.instantiate(cfg.logger)
    _ = logger.experiment # initializes the logger
    sweep_overridden_parameters = wandb.config
    print(f'Sweep overridden parameters: {sweep_overridden_parameters}')

    for parameter, value in sweep_overridden_parameters.items():
        key_1, key_2 = parameter.split('.')
        cfg[key_1][key_2] = value

    if cfg.task == 'species':
        cfg['trainer']['callbacks'][0]['monitor'] = 'Val MAP'

    if 'ttt' in cfg.adaptation_mode or '-10' in cfg.adaptation_mode or '-20' in cfg.adaptation_mode:
        cfg['datamodule']['batch_size'] = 1

    print(f'Config: {cfg}')

    trainer = hydra.utils.instantiate(cfg.trainer)
    torch.use_deterministic_algorithms(True, warn_only=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    num_train_batches = len(datamodule.train_dataloader())
    print(f'Number of training batches: {num_train_batches}')
    cfg['model']['num_train_batches'] = num_train_batches
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    model = hydra.utils.instantiate(cfg.model)

    if cfg.adaptation_mode == 'task_modality_decoder':
        trainer.fit(model, datamodule=datamodule)
    elif 'ttt' in cfg.adaptation_mode or '-10' in cfg.adaptation_mode or '-20' in cfg.adaptation_mode:
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)
        trainer.test(ckpt_path='best', datamodule=datamodule)

    wandb.finish()
    shutil.rmtree(os.getcwd()) # deletes the hydra-created working directory

if __name__ == '__main__':
    train()
