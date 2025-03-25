# ============================================== IMPORTS ============================================== #

from lightning.pytorch import seed_everything
import hydra
import torch

# ============================================== FUNCTIONS ============================================== #

@hydra.main(config_path='.', config_name='train_config', version_base=None)
def train(cfg):
    print(f'Task: {cfg.task}')
    print(f'Split type: {cfg.split_type}')
    print(f'Model type: {cfg.model_type}')

    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.seed, workers=True)

    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer)
    torch.use_deterministic_algorithms(True, warn_only=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    train()
