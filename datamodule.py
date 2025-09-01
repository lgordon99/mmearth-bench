'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from dataset import MMEarthBenchDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
import torch

# ============================================== CLASSES ============================================== #

class DataModule(LightningDataModule):
    def __init__(self, task, batch_size, num_workers, seed):
        super().__init__()

        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = MMEarthBenchDataset(task=task)

        self.setup('init')

    def setup(self, stage):
        if stage == 'init':
            splits = ['train']
        elif stage == 'fit':
            splits = ['val']
        elif stage == 'test':
            splits = ['random_test', 'geographic_test']

        for split in splits:
            setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_indices']))
            print(f'{split.capitalize().replace("_", " ")} dataset size: {len(getattr(self, f"{split}_dataset"))}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        random_test_dataloader = DataLoader(self.random_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        geographic_test_dataloader = DataLoader(self.geographic_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        return [random_test_dataloader, geographic_test_dataloader]
