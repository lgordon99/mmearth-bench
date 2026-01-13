#!/usr/bin/env python3
"""
Example script for Unsupervised Domain Adaptation (UDA) usage.

This script demonstrates how to use the UDA functionality to:
1. Load a pre-trained joint training model
2. Enable UDA by including "uda" in adaptation_mode
3. Run unsupervised domain adaptation on unlabeled test data for each split
4. Evaluate on the same splits after UDA
"""

import torch
from lightning.pytorch import Trainer
from model import Model
from datamodule import DataModule

def run_uda_example():
    """
    Example of how to use UDA mode with a joint training model.
    """

    # Configuration for UDA
    config = {
        'task': 'biomass',  # or 'species'
        'architecture': 'ScaleMAE',  # or other supported architectures
        'adaptation_mode': 'joint_training_uda',  # UDA automatically enabled when "uda" is in adaptation_mode
        'tuning_mode': 'full_finetuning',
        'pretrained': True,
        'max_lr': 1e-4,
        'weight_decay': 0.01,
        'warmup_epochs': 5,
        'num_train_batches': 1000,  # This will be updated by datamodule
        'min_lr': 1e-6,
        'epochs': 50,
        'inner_loop_lr': 1e-3
    }

    # Create datamodule
    datamodule = DataModule(
        task=config['task'],
        architecture=config['architecture'],
        adaptation_mode=config['adaptation_mode'],
        batch_size=8,  # Can be larger for TTT since we're not doing meta-learning
        num_workers=4,
        seed=42
    )

    # Update num_train_batches
    config['num_train_batches'] = len(datamodule.train_dataloader())

    # Create model - UDA mode automatically enabled by adaptation_mode
    model = Model(**config)

    # Create trainer
    trainer = Trainer(
        max_epochs=1,  # We only need 1 epoch for UDA
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,  # Disable logging for this example
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    print("Starting UDA evaluation...")
    print("This will:")
    print("1. Save original model state")
    print("2. Train encoder + task modality decoder on random_test split")
    print("3. Evaluate on random_test split")
    print("4. Automatically reset model to original state (via on_test_dataloader_start)")
    print("5. Train encoder + task modality decoder on geographic_test split")
    print("6. Evaluate on geographic_test split")
    print()

    # Run UDA evaluation
    # The test_step method will automatically handle UDA training for each split
    trainer.test(model, datamodule=datamodule)

    print("UDA evaluation completed!")

def run_uda_manual_control():
    """
    Example of manual UDA control with explicit resets between splits.
    """

    # Create model and datamodule (same as before)
    config = {
        'task': 'biomass',
        'architecture': 'ScaleMAE',
        'adaptation_mode': 'joint_training_uda',  # UDA automatically enabled
        'tuning_mode': 'full_finetuning',
        'pretrained': True,
        'max_lr': 1e-4,
        'weight_decay': 0.01,
        'warmup_epochs': 5,
        'num_train_batches': 1000,
        'min_lr': 1e-6,
        'epochs': 50,
        'inner_loop_lr': 1e-3
    }

    datamodule = DataModule(
        task=config['task'],
        architecture=config['architecture'],
        adaptation_mode=config['adaptation_mode'],
        batch_size=8,
        num_workers=4,
        seed=42
    )

    config['num_train_batches'] = len(datamodule.train_dataloader())
    model = Model(**config)

    # Manual UDA control
    print("Manual UDA control example:")
    print("Note: This is only needed if you're not using trainer.test()")
    print("The automatic reset via on_test_dataloader_start() handles this for you!")
    print()
    print("1. Random test split...")

    # Get random test dataloader
    random_test_loader = datamodule.random_test_dataloader()

    # Train on random test (unsupervised)
    model.train()
    # ... your training loop here ...
    model.eval()

    # Evaluate on random test
    # ... your evaluation here ...

    print("2. Manually resetting model for geographic test...")
    model.reset_for_new_split()  # Only needed for manual control

    print("3. Geographic test split...")

    # Get geographic test dataloader
    geographic_test_loader = datamodule.geographic_test_dataloader()

    # Train on geographic test (unsupervised)
    model.train()
    # ... your training loop here ...
    model.eval()

    # Evaluate on geographic test
    # ... your evaluation here ...

    print("Manual UDA completed!")

def run_uda_with_existing_model():
    """
    Example of how to enable UDA mode on an existing trained model.
    """

    # Load your existing model checkpoint
    checkpoint_path = "path/to/your/checkpoint.ckpt"

    # Load the model - UDA mode automatically enabled by adaptation_mode
    model = Model.load_from_checkpoint(
        checkpoint_path,
        adaptation_mode='joint_training_uda'  # UDA automatically enabled
    )

    # Create datamodule
    datamodule = DataModule(
        task='biomass',
        architecture='ScaleMAE',
        adaptation_mode='joint_training',
        batch_size=8,
        num_workers=4,
        seed=42
    )

    # Create trainer and run TTT
    trainer = Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    # Run the basic UDA example
    run_uda_example()

    # Uncomment to run with existing model
    # run_uda_with_existing_model()
