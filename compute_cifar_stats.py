import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
import pickle
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from collections import defaultdict
import argparse


def compute_cifar10_statistics(feature_dim=64):
    """
    Compute and save per-class Inception feature statistics for CIFAR-10 dataset.

    Args:
        feature_dim (int): Dimension of features to extract. Options are:
            - 64: First layer features (block idx 0)
            - 192: Second layer features (block idx 1)
            - 768: Third layer features (block idx 2)
            - 2048: Final layer features (block idx 3)
    """
    # Map feature dimension to block index
    dim_to_block = {
        64: 0,  # First block
        192: 1,  # Second block
        768: 2,  # Third block
        2048: 3,  # Final block (default for standard FID)
    }

    if feature_dim not in dim_to_block:
        raise ValueError(
            f"Unsupported feature dimension: {feature_dim}. "
            f"Supported dimensions are: {list(dim_to_block.keys())}"
        )

    block_idx = dim_to_block[feature_dim]

    # Load inception model with specified feature dimension
    inception = InceptionV3([block_idx], normalize_input=True)
    inception.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = inception.to(device)

    # Transform for CIFAR-10
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    # Load both training and test sets
    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Dictionary to store features for each class
    class_features = defaultdict(list)

    print(f"Computing {feature_dim}-dim Inception features for CIFAR-10 by class...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            # Scale images to [-1, 1] range
            images = images * 2 - 1

            # Get features
            features = inception(images)[0]
            # Global average pooling if features have spatial dimensions
            if len(features.shape) > 2:
                features = features.mean([2, 3])
            features = features.cpu().numpy()

            # Store features by class
            for feat, label in zip(features, labels):
                class_features[label.item()].append(feat)

    # Compute statistics for each class
    stats = {}
    print("\nComputing statistics for each class...")

    for class_idx in range(10):
        features = np.stack(class_features[class_idx])
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        stats[f"class_{class_idx}_mu"] = mu
        stats[f"class_{class_idx}_sigma"] = sigma

        print(f"Class {class_idx}:")
        print(f"  Number of samples: {len(features)}")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Mean shape: {mu.shape}")
        print(f"  Covariance shape: {sigma.shape}")

    # Save statistics with dimension in filename
    output_file = f"cifar10_fid_stats_{feature_dim}dim.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nPer-class statistics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute CIFAR-10 statistics for FID calculation"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        choices=[64, 192, 768, 2048],
        help="Dimension of Inception features to extract (64, 192, 768, or 2048)",
    )
    args = parser.parse_args()

    compute_cifar10_statistics(feature_dim=args.dim)
