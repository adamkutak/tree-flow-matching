import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import mahalanobis
import pickle
from train_cifar_classifier import CIFAR100Classifier


class FIDRewardNet(nn.Module):
    def __init__(self, dataset="cifar10"):
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
        self.ref_mu = cifar_stats["mu"]  # Reference mean
        self.ref_sigma = cifar_stats["sigma"]  # Reference covariance
        print(
            f"Loaded CIFAR10 reference stats - mu shape: {self.ref_mu.shape}, sigma shape: {self.ref_sigma.shape}"
        )

        # Initialize running statistics
        self.running_features = []
        self.max_running_samples = 1000  # Keep track of last N samples

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
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        # Calculate FID
        diff = mu - self.ref_mu
        covmean = sqrtm(sigma.dot(self.ref_sigma))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma + self.ref_sigma - 2 * covmean)
        return float(fid)

    def compute_marginal_fid(self, image):
        """Compute how much this image changes the current FID."""
        # Extract features for new image
        features = self.extract_features(image)

        # Compute baseline FID with current running features
        if len(self.running_features) > 0:
            current_features = np.array(self.running_features)
            baseline_fid = self.compute_fid(current_features)
        else:
            baseline_fid = float("inf")

        # Add new features and compute new FID
        temp_features = self.running_features + [features[0]]
        if len(temp_features) > self.max_running_samples:
            temp_features = temp_features[-self.max_running_samples :]
        new_fid = self.compute_fid(np.array(temp_features))

        # Calculate FID change (negative means improvement)
        fid_change = new_fid - baseline_fid

        # Update running features
        self.running_features.append(features[0])
        if len(self.running_features) > self.max_running_samples:
            self.running_features = self.running_features[-self.max_running_samples :]

        # Convert to reward (negative FID change means improvement)
        reward = -fid_change
        return torch.tensor([reward], device=image.device, dtype=torch.float32)

    def forward(self, image):
        """Compute reward for a single image."""
        return self.compute_marginal_fid(image)

    def reset_running_stats(self):
        """Reset running statistics."""
        self.running_features = []
