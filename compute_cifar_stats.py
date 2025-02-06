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


def compute_cifar10_statistics():
    """Compute and save per-class Inception feature statistics for CIFAR-10 dataset."""

    # Load inception model with 64-dim features
    inception = InceptionV3(
        [2], normalize_input=True
    )  # Block index 2 gives 64-dim features
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

    print("Computing 64-dim Inception features for CIFAR-10 by class...")

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

    # Save statistics
    with open("cifar10_fid_stats_64dim.pkl", "wb") as f:
        pickle.dump(stats, f)

    print(f"\nPer-class statistics saved to cifar10_fid_stats_64dim.pkl")


if __name__ == "__main__":
    compute_cifar10_statistics()
