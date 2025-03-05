'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils

# ============================================== CLASSES ============================================== #

class DataModule(LightningDataModule):
    def __init__(self, task, dataset_class, split_type, batch_size, num_workers):
        super().__init__()

        self.task = task
        self.dataset_class = dataset_class
        self.split_type = split_type
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset = self.dataset_class(task=self.task, split_type=self.split_type)

        split_data = utils.read_json(f'{self.task}/{self.task}_{self.split_type}_split_data.json')

        self.train_dataset = Subset(dataset=self.dataset, indices=split_data['train_indices'])
        self.val_dataset = Subset(dataset=self.dataset, indices=split_data['val_indices'])
        self.test_dataset = Subset(dataset=self.dataset, indices=split_data['test_indices'])

        self._plot_task_distribution()
        self._calculate_test_rmse_with_train_mean()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, generator=torch.Generator().manual_seed(42), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def _plot_task_distribution(self):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True) # 3 rows, 1 column, shared x-axis
        max_value = max([np.max([task_value.squeeze() for _, task_value in dataset]) for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]])
        step = np.ceil(max_value / 10)
        bins = np.arange(0, max_value + step, step)

        for i, stage in enumerate(['train', 'val', 'test']):
            dataset = getattr(self, f'{stage}_dataset')
            task_values = [task_value.squeeze() for _, task_value in dataset]
            counts, bin_edges = np.histogram(task_values, bins=bins)
            percentages = counts / counts.sum() * 100

            axes[i].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), align='edge', edgecolor='black')
            axes[i].set_xticks(bin_edges)
            axes[i].set_ylabel('Percentage (%)')
            axes[i].set_title(stage.capitalize())

        fig.suptitle(self.task.replace("_", " ").capitalize(), fontweight='bold')
        axes[-1].set_xlabel(f'{self.task.replace("_", " ").capitalize()} value') # sets common x-label
        plt.tight_layout()
        plt.savefig(f'{self.task}/figures/{self.task}_{self.split_type}_distributions.png', dpi=300)
        plt.close()

    def _calculate_test_rmse_with_train_mean(self):
        train_values = [task_value.squeeze() for _, task_value in self.train_dataset]
        train_mean = np.mean(train_values)
        print(f'Mean of train values: {round(float(train_mean), 2)}')

        test_values = [task_value.squeeze() for _, task_value in self.test_dataset]
        test_rmse = np.sqrt(np.mean((np.array(test_values) - train_mean) ** 2))

        print(f'Test RMSE using the train mean as the prediction: {round(float(test_rmse), 2)}')
