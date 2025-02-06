import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from scipy.spatial.distance import mahalanobis
import pickle
from train_cifar_classifier import CIFAR100Classifier
from scipy.linalg import sqrtm


class FIDRewardNet(nn.Module):
    def __init__(self, dataset="cifar10", initial_batch_size=1000):
        super().__init__()
        print("Initializing FIDRewardNet...")

        # Load inception model for feature extraction
        self.inception_model = models.inception_v3(
            pretrained=True, transform_input=False
        )
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        print("Inception model loaded")

        # Load reference statistics for CIFAR-10
        with open("cifar10_fid_stats.pkl", "rb") as f:
            cifar_stats = pickle.load(f)
        self.ref_mu = cifar_stats["mu_cifar10"]
        self.ref_sigma = cifar_stats["sigma_cifar10"]
        print(
            f"Loaded CIFAR10 reference stats - mu shape: {self.ref_mu.shape}, sigma shape: {self.ref_sigma.shape}"
        )

        # Initialize running statistics with real CIFAR-10 images
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        cifar10 = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        # Randomly sample initial_batch_size images
        indices = np.random.choice(len(cifar10), initial_batch_size, replace=False)
        initial_images = torch.stack([cifar10[i][0] for i in indices])

        # Extract features for initial images
        print(
            f"Initializing running statistics with {initial_batch_size} real images..."
        )
        initial_features = self.extract_features(initial_images)
        self.running_features = list(initial_features)
        self.max_running_samples = initial_batch_size
        print(
            f"Running statistics initialized with {len(self.running_features)} features"
        )

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Define transforms
        self.inception_transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, images):
        """Extract inception features from images."""
        images = self.inception_transform(images)
        with torch.no_grad():
            features = self.inception_model(images).cpu().numpy()
        return features

    def compute_fid(self, features):
        """Compute FID between given features and reference statistics."""
        print(f"\nComputing FID - Input features shape: {features.shape}")
        print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")

        mu = np.mean(features, axis=0)
        print(f"Computed mean shape: {mu.shape}")
        print(f"Mean range: [{mu.min():.4f}, {mu.max():.4f}]")

        sigma = np.cov(features, rowvar=False)
        print(f"Computed covariance shape: {sigma.shape}")
        print(f"Covariance range: [{sigma.min():.4f}, {sigma.max():.4f}]")

        # Calculate FID
        diff = mu - self.ref_mu
        print(f"Mean difference range: [{diff.min():.4f}, {diff.max():.4f}]")

        print(
            f"Reference sigma range: [{self.ref_sigma.min():.4f}, {self.ref_sigma.max():.4f}]"
        )
        dot_product = sigma.dot(self.ref_sigma)
        print(f"Dot product range: [{dot_product.min():.4f}, {dot_product.max():.4f}]")

        covmean = sqrtm(dot_product)
        print(f"Covmean before real check - shape: {covmean.shape}")
        print(f"Covmean is complex: {np.iscomplexobj(covmean)}")
        if np.iscomplexobj(covmean):
            print(f"Complex component magnitude: {np.abs(covmean.imag).max():.4f}")
            covmean = covmean.real
        print(f"Covmean range: [{covmean.min():.4f}, {covmean.max():.4f}]")

        term1 = diff.dot(diff)
        term2 = np.trace(sigma)
        term3 = np.trace(self.ref_sigma)
        term4 = -2 * np.trace(covmean)

        print(f"FID components:")
        print(f"- Mean difference term: {term1:.4f}")
        print(f"- Sigma trace: {term2:.4f}")
        print(f"- Ref sigma trace: {term3:.4f}")
        print(f"- Covmean term: {term4:.4f}")

        fid = term1 + term2 + term3 + term4
        print(f"Final FID: {fid:.4f}")

        return float(fid)

    def compute_marginal_fid(self, image):
        """Compute how much this image changes the current FID."""
        # Extract features for new image
        print(f"\nInput image shape: {image.shape}")
        print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")

        features = self.extract_features(image)
        print(f"Extracted features shape: {features.shape}")
        print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")

        # Compute baseline FID with current running features
        print(f"\nCurrent running features count: {len(self.running_features)}")
        if len(self.running_features) > 0:
            current_features = np.array(self.running_features)
            print(f"Computing baseline FID...")
            baseline_fid = self.compute_fid(current_features)
            print(f"Baseline FID: {baseline_fid}")
        else:
            baseline_fid = float("inf")
            print("No baseline FID (empty running features)")

        # Add new features and compute new FID
        temp_features = self.running_features + [features[0]]
        print(f"\nComputing new FID with added feature...")
        new_fid = self.compute_fid(np.array(temp_features))
        print(f"New FID: {new_fid}")

        # Calculate FID change
        fid_change = new_fid - baseline_fid
        print(f"FID change: {fid_change}")

        # Update running features
        self.running_features.append(features[0])
        if len(self.running_features) > self.max_running_samples:
            self.running_features = self.running_features[-self.max_running_samples :]

        reward = -fid_change
        print(f"Final reward: {reward}")
        return torch.tensor([reward], device=image.device, dtype=torch.float32)

    def forward(self, image):
        """Compute reward for a single image."""
        return self.compute_marginal_fid(image)

    def reset_running_stats(self):
        """Reset running statistics."""
        self.running_features = []
