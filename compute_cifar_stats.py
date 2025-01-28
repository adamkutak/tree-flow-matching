import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
import pickle
from tqdm import tqdm


def compute_cifar100_statistics():
    """Compute and save Inception feature statistics for CIFAR-100 dataset."""

    # Load inception model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # Remove final classification layer
    inception_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_model.to(device)

    # Transform for CIFAR-100 (Inception v3 expects 299x299 images)
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load both training and test sets
    train_dataset = CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # Combine datasets and create dataloader
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    dataloader = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Collect features
    features = []
    print("Computing Inception features for CIFAR-100...")

    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            feat = inception_model(images).cpu().numpy()
            features.append(feat)

    features = np.concatenate(features, axis=0)

    # Compute statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    # Save statistics
    stats = {"mu_cifar100": mu, "sigma_cifar100": sigma}

    with open("cifar100_fid_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print(f"Statistics saved to cifar100_fid_stats.pkl")
    print(f"Feature shape: {features.shape}")
    print(f"Mean shape: {mu.shape}")
    print(f"Covariance shape: {sigma.shape}")


if __name__ == "__main__":
    compute_cifar100_statistics()
