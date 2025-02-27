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
        self.train_dataset = Subset(dataset=self.dataset, indices=split_data['train_ids'])
        self.val_dataset = Subset(dataset=self.dataset, indices=split_data['val_ids'])
        self.test_dataset = Subset(dataset=self.dataset, indices=split_data['test_ids'])
        self._plot_task_distribution()
        # if stage == 'fit':
        #     self.train_dataset = Subset(dataset=self.dataset, indices=split_data['train_ids'])
        #     self.val_dataset = Subset(dataset=self.dataset, indices=split_data['val_ids'])
        #     self._plot_task_distribution('train')
        #     self._plot_task_distribution('val')
        # elif stage == 'test':
        #     self.test_dataset = Subset(dataset=self.dataset, indices=split_data['test_ids'])
        #     self._plot_task_distribution('test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, generator=torch.Generator().manual_seed(42), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def _plot_task_distribution(self):
        # for stage in ['train', 'val', 'test']:
        #     dataset = getattr(self, f'{stage}_dataset')
        #     task_values = [task_value.squeeze() for _, task_value in dataset]

        #     if stage == 'train':
        #         print(f'Mean of {self.task} train values: {np.mean(task_values)}')

        #     max_value = np.max(task_values)
        #     bins = np.arange(0, max_value + 5, 5)
        #     counts, bin_edges = np.histogram(task_values, bins=bins)
        #     percentages = counts / counts.sum() * 100
        #     plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), align='edge', edgecolor='black')
        #     task_name = self.task.replace("_", " ").capitalize()
        #     plt.xlabel(f'{task_name} value')
        #     plt.ylabel('Percentage (%)')
        #     plt.title(f'{task_name}: Distribution of Task Values in {stage.capitalize()} Dataset')
        #     plt.tight_layout()
        #     plt.savefig(f'{self.task}/figures/{self.task}_{self.split_type}_{stage}.png', dpi=300)
        #     plt.close()

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True) # 3 rows, 1 column, shared x-axis

        for i, stage in enumerate(['train', 'val', 'test']):
            dataset = getattr(self, f'{stage}_dataset')
            task_values = [task_value.squeeze() for _, task_value in dataset]

            if stage == 'train':
                print(f'Mean of {self.task} train values: {np.mean(task_values)}')

            max_value = np.max(task_values)
            bins = np.arange(0, max_value + 5, 5)
            counts, bin_edges = np.histogram(task_values, bins=bins)
            percentages = counts / counts.sum() * 100

            axes[i].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), align='edge', edgecolor='black')
            axes[i].set_ylabel('Percentage (%)')
            axes[i].set_title(stage.capitalize())

        fig.suptitle(self.task.replace("_", " ").capitalize(), fontweight='bold')
        axes[-1].set_xlabel(f'{self.task.replace("_", " ").capitalize()} value') # sets common x-label
        plt.tight_layout()
        plt.savefig(f'{self.task}/figures/{self.task}_{self.split_type}_distributions.png', dpi=300)
        plt.close()
