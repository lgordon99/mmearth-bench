'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from dataset import MMEarthBenchDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Sampler, Subset
import numpy as np
import torch

# ============================================== CLASSES ============================================== #

class BalancedDomainSampler(Sampler):
    """
    Sampler that creates balanced batches with equal numbers of labeled source domain, unlabeled source domain, and target domain samples.
    This sampler yields indices in a pattern that ensures balanced batches when used with DataLoader.
    """

    def __init__(self, dataset, generator):
        self.dataset = dataset
        self.generator = generator
        self.labeled_source_domain_indices = np.array(dataset.split_data['train_100%_indices'])
        self.unlabeled_source_domain_indices = np.array(dataset.split_data['val_indices'] + dataset.split_data['random_test_indices'])
        self.target_domain_indices = np.array(dataset.split_data['geographic_test_indices'])

        print(f'{len(self.labeled_source_domain_indices)} labeled source domain indices')
        print(f'{len(self.unlabeled_source_domain_indices)} unlabeled source domain indices')
        print(f'{len(self.target_domain_indices)} target domain indices')

        assert len(self.labeled_source_domain_indices) + len(self.unlabeled_source_domain_indices) + len(self.target_domain_indices) == len(dataset)

    def __iter__(self):
        # shuffle indices for the epoch
        labeled_source_indices = self.labeled_source_domain_indices[torch.randperm(len(self.labeled_source_domain_indices), generator=self.generator).numpy()].tolist()
        unlabeled_source_indices = self.unlabeled_source_domain_indices[torch.randperm(len(self.unlabeled_source_domain_indices), generator=self.generator).numpy()].tolist()
        target_indices = self.target_domain_indices[torch.randperm(len(self.target_domain_indices), generator=self.generator).numpy()].tolist()

        domain_data = {'labeled_source': {'all_indices': labeled_source_indices, 'indices': labeled_source_indices.copy(), 'num_remaining': len(labeled_source_indices)},
                       'unlabeled_source': {'all_indices': unlabeled_source_indices, 'indices': unlabeled_source_indices.copy(), 'num_remaining': len(unlabeled_source_indices)},
                       'target': {'all_indices': target_indices, 'indices': target_indices.copy(), 'num_remaining': len(target_indices)}}
        indices = []

        while any(domain_data[domain]['num_remaining'] > 0 for domain in domain_data): # while some domain has unused indices
            for domain in domain_data:
                if len(domain_data[domain]['indices']) == 0: # if the list of indices is empty
                    domain_data[domain]['indices'] = domain_data[domain]['all_indices'].copy() # resets the list of indices to the list of all indices for the domain

                indices.append(domain_data[domain]['indices'].pop(0)) # moves the first index from the list of indices to the list of batch indices

                if domain_data[domain]['num_remaining'] > 0: # if the domain still has unused indices
                    domain_data[domain]['num_remaining'] -= 1 # decrements the number of remaining indices

        assert len(indices) == self.__len__()
        return iter(indices)

    def __len__(self):
        # length is max_domain_size * 3 because smaller domains get recycled
        max_domain_size = max(len(self.labeled_source_domain_indices), len(self.unlabeled_source_domain_indices), len(self.target_domain_indices))

        return max_domain_size * 3

class DataModule(LightningDataModule):
    def __init__(self, task, architecture, adaptation_mode, train_percent, batch_size, num_workers, seed):
        super().__init__()

        self.task = task
        self.adaptation_mode = adaptation_mode
        self.train_percent = train_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = MMEarthBenchDataset(task, architecture, adaptation_mode, train_percent)

        self.setup('init')

    def setup(self, stage):
        if stage == 'init':
            splits = ['train']
        elif stage in ['fit', 'validate']:
            splits = ['val']
        elif stage == 'test':
            splits = ['random_test', 'geographic_test']

        for split in splits:
            if split == 'train':
                if self.adaptation_mode in ['UDA-SS', 'task_modality_decoder']:
                    setattr(self, f'{split}_dataset', self.dataset) # uses the full dataset
                else:
                    setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_{self.train_percent}%_indices']))
            else:
                setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_indices']))

            print(f'{split.capitalize().replace("_", " ")} dataset size: {len(getattr(self, f"{split}_dataset"))}')

    def train_dataloader(self):
        if self.adaptation_mode == 'UDA-SS':
            balanced_domain_sampler = BalancedDomainSampler(dataset=self.train_dataset, generator=torch.Generator().manual_seed(self.seed))
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=balanced_domain_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        batch_size = 1 if self.adaptation_mode in ['mt3', 'multimodal_mt3', 'sln', 'multimodal_sln', 'maml_input_embeddings', 'maml_encode', 'rna', 'rna_input_embeddings'] else self.batch_size

        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        batch_size = 1 if self.adaptation_mode in ['mt3', 'multimodal_mt3', 'sln', 'multimodal_sln', 'maml_input_embeddings', 'maml_encode', 'rna', 'rna_input_embeddings'] else self.batch_size

        random_test_dataloader = DataLoader(self.random_test_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        geographic_test_dataloader = DataLoader(self.geographic_test_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

        return [random_test_dataloader, geographic_test_dataloader]
