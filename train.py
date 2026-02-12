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
    print(f'Train percent: {cfg.train_percent}')

    # set environment variables
    os.environ['DATA_DIR_PATH'] = cfg.data_dir_path
    os.environ['ENTITY'] = cfg.logger.entity
    os.environ['PROJECT'] = cfg.logger.project
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.seed, workers=True)
    logger = hydra.utils.instantiate(cfg.logger)

    if os.environ['RANK'] == '0': # if the rank is 0
        _ = logger.experiment # initializes the logger

    if cfg.task == 'species':
        cfg['trainer']['callbacks'][0]['monitor'] = 'Val mAP'

    if 'TTT' in cfg.adaptation_mode:
        cfg['datamodule']['batch_size'] = 8
    elif 'AnySat' in cfg.architecture:
        cfg['datamodule']['batch_size'] = 8
        cfg['trainer']['accumulate_grad_batches'] = 4

    trainer = hydra.utils.instantiate(cfg.trainer)
    torch.use_deterministic_algorithms(True, warn_only=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    num_train_batches = len(datamodule.train_dataloader())
    cfg['model']['num_train_batches'] = num_train_batches

    if os.environ['RANK'] == '0':
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        print("\n======= CONFIG =======")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("======================\n")

    model = hydra.utils.instantiate(cfg.model)

    if 'TTT' in cfg.adaptation_mode:
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)
        trainer.test(ckpt_path='best', datamodule=datamodule)

    if trainer.is_global_zero:
        wandb.finish()
        shutil.rmtree(os.getcwd()) # deletes the hydra-created working directory

if __name__ == '__main__':
    train()
