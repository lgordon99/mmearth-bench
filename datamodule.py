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
    def __init__(self, task, adaptation_mode, batch_size, num_workers, seed):
        super().__init__()

        self.task = task
        self.adaptation_mode = adaptation_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = MMEarthBenchDataset(task=task, adaptation_mode=adaptation_mode)

        self.setup('init')

    def setup(self, stage):
        if stage == 'init':
            splits = ['train']
        elif stage in ['fit', 'validate']:
            splits = ['val']
        elif stage == 'test':
            splits = ['random_test', 'geographic_test']

        for split in splits:
            if self.adaptation_mode != 'task_modality_decoder':
                setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_indices']))
            else:
                setattr(self, f'{split}_dataset', self.dataset)

            print(f'{split.capitalize().replace("_", " ")} dataset size: {len(getattr(self, f"{split}_dataset"))}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        batch_size = 1 if self.adaptation_mode in ['mt3', 'maml', 'maml_input_embeddings', 'maml_encode', 'rna', 'rna_input_embeddings'] else self.batch_size

        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        batch_size = 1 if self.adaptation_mode in ['mt3', 'maml', 'maml_input_embeddings', 'maml_encode', 'rna', 'rna_input_embeddings'] else self.batch_size

        random_test_dataloader = DataLoader(self.random_test_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        geographic_test_dataloader = DataLoader(self.geographic_test_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

        return [random_test_dataloader, geographic_test_dataloader]
