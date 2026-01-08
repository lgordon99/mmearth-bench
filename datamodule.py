'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from dataset import MMEarthBenchDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader, Subset
import numpy as np
import torch

# ============================================== CLASSES ============================================== #

class TTTBatchSampler(BatchSampler):
    def __init__(self, dataset, split, batch_size, generator):
        self.indices = np.array(dataset.split_data[f'{split}_indices'])
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices), generator=self.generator).tolist()]

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i+self.batch_size]

            if len(indices) - i < 2 * self.batch_size:
                batch = indices[i:]
                print(f'Batch {i//self.batch_size} size: {len(batch)}')
                yield batch
                break

            print(f'Batch {i//self.batch_size} size: {len(batch)}')
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size

class GeographicBatchSampler(BatchSampler):
    def __init__(self, dataset, split, batch_size):
        self.indices = np.array(dataset.split_data[f'{split}_indices'])
        coordinates = dataset.geolocation[self.indices]
        points = np.column_stack((self.indices, coordinates))
        self.batch_size = batch_size
        self.batches = {}
        batch_id = 0

        def partition_recursive(points_subset, bounds):
            """
            Recursively partition points into balanced groups.
            bounds: [min_x, max_x, min_y, max_y]
            """
            nonlocal batch_id # allows batch_id to be accessed and modified in the outer scope
            n = len(points_subset) # number of points remaining to be batched

            # Base case: if points fit in one group (within tolerance)
            if n < 2 * batch_size:  # allows 20% tolerance
                self.batches[batch_id] = points_subset # adds points to cluster
                print(f'Created batch {batch_id} with {n} tiles')
                batch_id += 1 # increments cluster id
                return # stops recursion

            num_batches = max(1, (n + batch_size - 1) // batch_size) # calculates how many more batches we need

            # Decide split direction based on aspect ratio
            min_x, max_x, min_y, max_y = bounds
            width = max_x - min_x
            height = max_y - min_y

            # Split along longer dimension
            if width > height:
                axis = 1 # splits on x
            else:
                axis = 2  # splits on y

            # Sort points along chosen axis
            sorted_indices = np.argsort(points_subset[:, axis])
            sorted_points = points_subset[sorted_indices]

            # Calculate optimal split point
            num_batches_on_left = num_batches // 2 # calculates how many batches to put on the left
            split_idx = min(num_batches_on_left * batch_size, n - batch_size)
            split_idx = max(batch_size, split_idx)

            # Find the actual split value (between split_idx-1 and split_idx)
            split_value = (sorted_points[split_idx - 1, axis] + sorted_points[split_idx, axis]) / 2

            # Split the points
            left_points = sorted_points[:split_idx]
            right_points = sorted_points[split_idx:]

            # Create new bounds for each partition
            if axis == 1:  # Split on x
                left_bounds = [min_x, split_value, min_y, max_y]
                right_bounds = [split_value, max_x, min_y, max_y]
            else:  # Split on y
                left_bounds = [min_x, max_x, min_y, split_value]
                right_bounds = [min_x, max_x, split_value, max_y]

            # Recursively partition each half
            partition_recursive(left_points, left_bounds)
            partition_recursive(right_points, right_bounds)

        # get initial bounds of the points
        min_x, min_y = points[:, 1:].min(axis=0)
        max_x, max_y = points[:, 1:].max(axis=0)

        # Add small padding
        padding_x = (max_x - min_x) * 0.05
        padding_y = (max_y - min_y) * 0.05
        initial_bounds = [min_x - padding_x, max_x + padding_x,
                          min_y - padding_y, max_y + padding_y]

        partition_recursive(points, initial_bounds)

    def __iter__(self):
        batches = [batch[:, 0].astype(int) for batch in self.batches.values()]

        return iter(batches)

    def __len__(self):
        return len(self.batches)

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
                setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_{self.train_percent}%_indices']))
            else:
                setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_indices']))

            print(f'{split.capitalize().replace("_", " ")} dataset size: {len(getattr(self, f"{split}_dataset"))}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        if 'TTT-Geo' in self.adaptation_mode:
            val_geographic_sampler = GeographicBatchSampler(dataset=self.dataset, split='val', batch_size=self.batch_size)
            val_dataloader = DataLoader(self.dataset, batch_sampler=val_geographic_sampler, num_workers=self.num_workers, pin_memory=True)

            return val_dataloader
        elif 'TTT' in self.adaptation_mode:
            val_ttt_sampler = TTTBatchSampler(dataset=self.dataset, split='val', batch_size=self.batch_size, generator=torch.Generator().manual_seed(self.seed))
            val_dataloader = DataLoader(self.dataset, batch_sampler=val_ttt_sampler, num_workers=self.num_workers, pin_memory=True)

            return val_dataloader
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if 'TTT-Geo' in self.adaptation_mode:
            random_test_geographic_sampler = GeographicBatchSampler(dataset=self.dataset, split='random_test', batch_size=self.batch_size)
            random_test_dataloader = DataLoader(self.dataset, batch_sampler=random_test_geographic_sampler, num_workers=self.num_workers, pin_memory=True)

            geographic_test_geographic_sampler = GeographicBatchSampler(dataset=self.dataset, split='geographic_test', batch_size=self.batch_size)
            geographic_test_dataloader = DataLoader(self.dataset, batch_sampler=geographic_test_geographic_sampler, num_workers=self.num_workers, pin_memory=True)
        elif 'TTT' in self.adaptation_mode:
            random_test_ttt_sampler = TTTBatchSampler(dataset=self.dataset, split='random_test', batch_size=self.batch_size, generator=torch.Generator().manual_seed(self.seed))
            random_test_dataloader = DataLoader(self.dataset, batch_sampler=random_test_ttt_sampler, num_workers=self.num_workers, pin_memory=True)

            geographic_test_ttt_sampler = TTTBatchSampler(dataset=self.dataset, split='geographic_test', batch_size=self.batch_size, generator=torch.Generator().manual_seed(self.seed))
            geographic_test_dataloader = DataLoader(self.dataset, batch_sampler=geographic_test_ttt_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            random_test_dataloader = DataLoader(self.random_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
            geographic_test_dataloader = DataLoader(self.geographic_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        return [random_test_dataloader, geographic_test_dataloader]
