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
from sklearn.decomposition import PCA


def compute_cifar10_statistics(feature_dim=64, pca_dim=None):
    """
    Compute and save Inception feature statistics for CIFAR-10 dataset.

    Args:
        feature_dim (int): Dimension of features to extract. Options are:
            - 64: First layer features (block idx 0)
            - 192: Second layer features (block idx 1)
            - 768: Third layer features (block idx 2)
            - 2048: Final layer features (block idx 3)
        pca_dim (int, optional): If provided, reduce feature dimension to this size using PCA.
                                Only applicable when feature_dim=2048.
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

    # Validate PCA dimension
    use_pca = pca_dim is not None
    if use_pca and feature_dim != 2048:
        print("Warning: PCA reduction is only supported for 2048-dimensional features.")
        print("Ignoring PCA dimension for non-2048 feature extraction.")
        use_pca = False

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
    # List to store all features for global statistics
    all_features_list = []

    print(f"Computing {feature_dim}-dim Inception features for CIFAR-10...")

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

            # Store features by class for per-class statistics
            for feat, label in zip(features, labels):
                class_features[label.item()].append(feat)

            # Store all features for global statistics
            all_features_list.extend(features)

    # Fit PCA if requested
    pca_model = None
    if use_pca:
        print(f"Fitting PCA to reduce from {feature_dim} to {pca_dim} dimensions...")
        # Collect all features for PCA fitting
        all_features = np.vstack(all_features_list)

        # Fit PCA
        pca_model = PCA(n_components=pca_dim, random_state=42)
        pca_model.fit(all_features)

        # Save PCA model
        pca_file = f"inception_pca_{feature_dim}to{pca_dim}.pkl"
        with open(pca_file, "wb") as f:
            pickle.dump(pca_model, f)
        print(f"Saved PCA model to {pca_file}")

        # Transform features using PCA
        for class_idx in range(10):
            if class_idx in class_features:
                class_features[class_idx] = [
                    pca_model.transform([feat])[0] for feat in class_features[class_idx]
                ]

        all_features_list = [
            pca_model.transform([feat])[0] for feat in all_features_list
        ]

    stats = {}

    # Compute per-class statistics
    print("\nComputing statistics for each class...")
    for class_idx in range(10):
        if class_idx in class_features and len(class_features[class_idx]) > 0:
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

    # Compute global statistics
    print("\nComputing global statistics...")
    all_features = np.stack(all_features_list)
    global_mu = np.mean(all_features, axis=0)
    global_sigma = np.cov(all_features, rowvar=False)

    stats["global_mu"] = global_mu
    stats["global_sigma"] = global_sigma

    print(f"Global statistics:")
    print(f"  Number of samples: {len(all_features)}")
    print(f"  Feature dimension: {all_features.shape[1]}")
    print(f"  Mean shape: {global_mu.shape}")
    print(f"  Covariance shape: {global_sigma.shape}")

    # Save statistics with dimension in filename
    if use_pca:
        output_file = f"cifar10_fid_stats_{feature_dim}to{pca_dim}dim.pkl"
    else:
        output_file = f"cifar10_fid_stats_{feature_dim}dim.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nStatistics saved to {output_file}")

    # Print what was saved in the file
    print("\nSaved statistics:")
    for key in stats.keys():
        print(f"  - {key}")


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
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Reduce 2048-dim features to this dimension using PCA (only applicable when --dim=2048)",
    )
    args = parser.parse_args()

    compute_cifar10_statistics(
        feature_dim=args.dim,
        pca_dim=args.pca_dim,
    )
