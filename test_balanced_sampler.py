#!/usr/bin/env python3
"""
Test script to verify the balanced sampler creates balanced batches.
"""

import torch
from torch.utils.data import DataLoader
from dataset import MMEarthBenchDataset
from balanced_sampler import BalancedTargetDomainSampler

def test_balanced_sampler():
    """Test that the balanced sampler creates balanced batches."""

    # Create a small test dataset
    dataset = MMEarthBenchDataset('soil_nitrogen', 'ScaleMAE', 'FT', 100)

    # Create balanced sampler
    batch_size = 8
    sampler = BalancedTargetDomainSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False  # Disable shuffle for predictable testing
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Test first few batches
    for batch_idx, (input_data, task_modality_data, target, target_domain) in enumerate(dataloader):
        if batch_idx >= 3:  # Only test first 3 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Batch size: {len(target_domain)}")
        print(f"  Target domain True: {target_domain.sum().item()}")
        print(f"  Target domain False: {(~target_domain).sum().item()}")
        print(f"  Is balanced: {target_domain.sum().item() == len(target_domain) // 2}")

        # Check if batch is balanced (half True, half False)
        expected_true = len(target_domain) // 2
        actual_true = target_domain.sum().item()

        if actual_true == expected_true:
            print(f"  ✓ Batch is balanced!")
        else:
            print(f"  ✗ Batch is not balanced (expected {expected_true} True, got {actual_true})")

if __name__ == "__main__":
    test_balanced_sampler()
