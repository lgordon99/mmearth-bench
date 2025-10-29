'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from dataset import MMEarthBenchDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
import numpy as np
import torch

# ============================================== CLASSES ============================================== #

class WithIndex(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        return idx, sample

class MT3Sampler(Sampler):
    def __init__(self, dataset, generator, mini_batch_size):
        self.dataset = dataset
        self.generator = generator
        self.mini_batch_size = mini_batch_size
        self.batch_size = 4
        self.indices = np.array(self.dataset.split_data[f'train_100%_indices'])
        self.index_to_slot = {}

        self._build_spatial_mini_batches()

    def _build_spatial_mini_batches(self):
        indices = np.array(self.dataset.split_data[f'train_100%_indices'])
        coordinates = self.dataset.geolocation[indices]
        points = np.column_stack((indices, coordinates))
        self.mini_batches = {}
        mini_batch_id = 0

        def partition_recursive(points_subset, bounds):
            """
            Recursively partition points into balanced groups.
            bounds: [min_x, max_x, min_y, max_y]
            """
            nonlocal mini_batch_id # allows batch_id to be accessed and modified in the outer scope
            n = len(points_subset)

            # Base case: if points fit in one group (within tolerance)
            if n <= self.mini_batch_size * 1.2:  # allows 20% tolerance
                self.mini_batches[mini_batch_id] = points_subset # adds points to cluster
                print(f'Created batch {mini_batch_id} with {n} tiles')
                mini_batch_id += 1 # increments cluster id
                return # stops recursion

            needed_groups = max(1, (n + self.mini_batch_size - 1) // self.mini_batch_size) # calculates how many groups we need

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
            left_groups = needed_groups // 2
            split_idx = min(left_groups * self.mini_batch_size, n - self.mini_batch_size)
            split_idx = max(self.mini_batch_size, split_idx)

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
        mini_batches = {mini_batch_id: mini_batch[:, 0][torch.randperm(len(mini_batch), generator=self.generator).numpy()].tolist() for mini_batch_id, mini_batch in self.mini_batches.items()}
        mini_batch_ids = list(mini_batches.keys())
        mini_batch_ids_perm = torch.randperm(len(mini_batch_ids), generator=self.generator).numpy()
        shuffled_mini_batch_ids = [mini_batch_ids[i] for i in mini_batch_ids_perm]
        shuffled_mini_batches = {mini_batch_id: mini_batches[mini_batch_id] for mini_batch_id in shuffled_mini_batch_ids}

        return iter([mini_batch for mini_batch in shuffled_mini_batches.values()])

    def __len__(self):
        return len(self.mini_batches)

class GeographicSampler(Sampler):
    def __init__(self, dataset, batch_size, split):
        self.indices = np.array(dataset.split_data[f'{split}_indices'])
        coordinates = dataset.geolocation[self.indices]
        points = np.column_stack((self.indices, coordinates))
        self.batches = {}
        batch_id = 0

        def partition_recursive(points_subset, bounds):
            """
            Recursively partition points into balanced groups.
            bounds: [min_x, max_x, min_y, max_y]
            """
            nonlocal batch_id # allows batch_id to be accessed and modified in the outer scope
            n = len(points_subset)

            # Base case: if points fit in one group (within tolerance)
            if n <= batch_size * 1.2:  # allows 20% tolerance
                self.batches[batch_id] = points_subset # adds points to cluster
                # self.batches.append(points_subset[:, 0]) # adds indices to cluster
                print(f'Created batch {batch_id} with {n} tiles')
                batch_id += 1 # increments cluster id
                return # stops recursion

            needed_groups = max(1, (n + batch_size - 1) // batch_size) # calculates how many groups we need

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
            left_groups = needed_groups // 2
            split_idx = min(left_groups * batch_size, n - batch_size)
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

        # polygons = {}

        # for batch_id, points in self.batches.items():
        #     points_array = np.array(points)

        #     # Get bounding box of the points
        #     min_x = points_array[:, 1].min()
        #     max_x = points_array[:, 1].max()
        #     min_y = points_array[:, 2].min()
        #     max_y = points_array[:, 2].max()

        #     # Create rectangle polygon
        #     polygons[batch_id] = [[min_x, min_y],
        #                           [max_x, min_y],
        #                           [max_x, max_y],
        #                           [min_x, max_y],
        #                           [min_x, min_y]]

        # features = []

        # for batch_id in sorted(self.batches.keys()):
        #     points = self.batches[batch_id]
        #     polygon_coordinates = polygons[batch_id]
        #     points_list = [[int(idx), float(lon), float(lat)] for idx, lon, lat in points] # converts points to list of floats
        #     polygon_coordinates_clean = [[float(x), float(y)] for x, y in polygon_coordinates] # converts polygon coordinates to list of floats

        #     feature = {
        #         "type": "Feature",
        #         "properties": {
        #             "batch_id": int(batch_id),
        #             "point_count": len(points),
        #             "points": points_list
        #         },
        #         "geometry": {
        #             "type": "Polygon",
        #             "coordinates": [polygon_coordinates_clean]
        #         }
        #     }
        #     features.append(feature)

        # output_file = f'/n/home01/luciagordon/mmearth-bench/{split}_batches.geojson'

        # with open(output_file, 'w') as f:
        #     import json
        #     json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

        # print(f"GeoJSON saved to: {output_file}")
        # exit()
    def __iter__(self):
        indices = [batch[:, 0] for batch in self.batches.values()]
        return iter([int(idx) for batch in indices for idx in batch])

    def __len__(self):
        return len(self.indices)

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
                if self.adaptation_mode in ['UDA-SS']:
                    setattr(self, f'{split}_dataset', self.dataset) # uses the full dataset
                else:
                    setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_{self.train_percent}%_indices']))
            else:
                setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_indices']))

            print(f'{split.capitalize().replace("_", " ")} dataset size: {len(getattr(self, f"{split}_dataset"))}')

    def train_dataloader(self):
        # if self.adaptation_mode == 'UDA-SS':
        #     balanced_domain_sampler = BalancedDomainSampler(dataset=self.train_dataset, generator=torch.Generator().manual_seed(self.seed))
        #     return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=balanced_domain_sampler, num_workers=self.num_workers, pin_memory=True)
        if self.adaptation_mode == 'MT3_metabatch':
            mt3_sampler = MT3Sampler(dataset=self.dataset, generator=torch.Generator().manual_seed(self.seed), mini_batch_size=round(self.batch_size/4))
            return DataLoader(self.train_dataset, batch_size=4, sampler=mt3_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        batch_size = 1 if self.adaptation_mode in ['mt3', 'MT3_metabatch', 'multimodal_mt3', 'sln', 'multimodal_sln', 'maml_input_embeddings', 'maml_encode', 'rna', 'rna_input_embeddings'] else self.batch_size

        if 'TTT-Geo' in self.adaptation_mode:
            val_geographic_sampler = GeographicSampler(dataset=self.dataset, batch_size=batch_size, split='val')
            val_dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_geographic_sampler, num_workers=self.num_workers, pin_memory=True)

            return val_dataloader
        else:
            return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        batch_size = 1 if self.adaptation_mode in ['mt3', 'MT3_metabatch', 'multimodal_mt3', 'sln', 'multimodal_sln', 'maml_input_embeddings', 'maml_encode', 'rna', 'rna_input_embeddings'] else self.batch_size

        if 'TTT-Geo' in self.adaptation_mode:
            random_test_geographic_sampler = GeographicSampler(dataset=self.dataset, batch_size=batch_size, split='random_test')
            random_test_dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=random_test_geographic_sampler, num_workers=self.num_workers, pin_memory=True)

            geographic_test_geographic_sampler = GeographicSampler(dataset=self.dataset, batch_size=batch_size, split='geographic_test')
            geographic_test_dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=geographic_test_geographic_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            random_test_dataloader = DataLoader(self.random_test_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
            geographic_test_dataloader = DataLoader(self.geographic_test_dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

        return [random_test_dataloader, geographic_test_dataloader]
