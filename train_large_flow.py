import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from mcts_single_flow import MCTSFlowSampler
from run_mcts_flow import calculate_metrics
import torchmetrics.image.fid as FID


def train_large_flow_model(
    num_epochs=1000,
    batch_size=128,
    device="cuda:0",
    save_interval=50,
    num_channels=256,
    learning_rate=5e-4,
    num_timesteps=50,
):
    """
    Train a larger flow matching model for an extended number of epochs.

    Args:
        num_epochs: Total number of epochs to train
        batch_size: Batch size for training
        device: Device to use for training
        save_interval: Save model every this many epochs
        num_channels: Number of channels in the UNet model
        learning_rate: Learning rate for optimizer
        num_timesteps: Number of timesteps for flow matching
    """
    print(
        f"Training large flow model with {num_channels} channels for {num_epochs} epochs"
    )
    print(f"Using {num_timesteps} timesteps for flow matching")

    # Setup CIFAR-10 dataset with appropriate transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize metrics
    fid = FID.FrechetInceptionDistance(normalize=True, reset_real_features=False).to(
        device
    )
    # Randomly sample real images
    indices = np.random.choice(len(train_dataset), 5000, replace=False)
    real_images = torch.stack([train_dataset[i][0] for i in indices]).to(device)

    # Process real images in batches
    real_batch_size = 100
    print("Processing real images...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid.update(batch, real=True)

    # Initialize sampler with larger UNet model
    sampler = MCTSFlowSampler(
        image_size=32,
        channels=3,
        device=device,
        num_timesteps=num_timesteps,
        num_classes=10,
        buffer_size=10,
        num_channels=num_channels,  # Use larger channel count
        learning_rate=learning_rate,
        load_models=False,
    )

    # Create directory for saving models
    os.makedirs("saved_models", exist_ok=True)

    # Check if a saved model exists and load it
    save_path = f"saved_models/large_flow_model.pt"
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        sampler.flow_model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("No existing model found. Starting training from scratch.")

    # Track metrics over training
    metrics_history = []

    # Train only the flow model
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train flow model for one epoch
        flow_loss = sampler.train_flow_matching(
            train_loader, desc=f"Flow training {epoch + 1}/{num_epochs}", use_tqdm=True
        )

        # Step the scheduler
        sampler.flow_scheduler.step()

        # Save model and evaluate metrics at intervals
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            save_path = f"saved_models/large_flow_model.pt"
            torch.save(sampler.flow_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

            print("Evaluating metrics...")
            # Test with regular flow matching (no branching)
            fid_score = calculate_metrics(
                sampler,
                num_branches=1,
                num_keep=1,
                device=device,
                n_samples=5000,
                sigma=0.0,
                fid=fid,
            )

            metrics_history.append(
                {
                    "epoch": epoch + 1,
                    "fid_score": fid_score.item(),
                }
            )

            print(f"Metrics at epoch {epoch + 1}:")
            print(f"   FID Score: {fid_score:.4f}")


if __name__ == "__main__":
    # Use the specified GPU device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_large_flow_model(
        num_epochs=1000,
        batch_size=128,
        device=device,
        save_interval=50,
        num_channels=256,
        num_timesteps=10,
    )
