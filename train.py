# ============================================== IMPORTS ============================================== #

from lightning.pytorch import seed_everything
import hydra
import torch

# ============================================== FUNCTIONS ============================================== #

@hydra.main(config_path='.', config_name='train_config', version_base=None)
def train(cfg):
    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.seed, workers=True)
    print(f'Task: {cfg.task}')
    print(f'Split type: {cfg.split_type}')
    print(f'Model type: {cfg.model_type}')
    print('Logging run to Wandb')
    logger = hydra.utils.instantiate(cfg.logger)
    print(f'Local log folder: {logger.experiment.dir}')
    trainer = hydra.utils.instantiate(cfg.trainer)
    print('Set up trainer')
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print('Set up datamodule')
    model = hydra.utils.instantiate(cfg.model)
    print('Training model')
    trainer.fit(model, datamodule=datamodule)
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('Testing model')
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    train()
