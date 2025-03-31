from collections import deque
import torch
import torch.utils.data as data
import torchdiffeq
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ConditionalFlowMatcher,
)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from vector_mlps import MLPValue, MLPFlow
import torchvision.models as models
import pickle
from scipy.linalg import sqrtm
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from torchmetrics.image.fid import NoTrainInceptionV3


class TrajectoryBuffer:
    def __init__(self, max_size=500_000):
        self.max_size = max_size
        self.trajectories = []  # List of (state, t, label, final_score) tuples

    def add_trajectory(self, states, ts, labels, scores):
        """
        Add a complete trajectory and its final score
        states: list of tensors for each step
        ts: list of time points
        labels: tensor of target labels
        scores: final scores for each branch
        """
        for branch_idx in range(len(scores)):
            branch_states = [step_states[branch_idx] for step_states in states]
            for step, (state, t) in enumerate(zip(branch_states, ts)):
                self.trajectories.append(
                    (
                        state.cpu(),
                        t,
                        labels[branch_idx].cpu(),
                        scores[branch_idx].cpu(),
                    )
                )

        # Trim buffer if needed
        if len(self.trajectories) > self.max_size:
            self.trajectories = self.trajectories[-self.max_size :]

    def sample_batch(self, batch_size):
        """Sample a random batch of (state, t, label, score) tuples"""
        if len(self.trajectories) < batch_size:
            return None

        indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        batch = [self.trajectories[i] for i in indices]

        states = torch.stack([b[0] for b in batch])
        ts = torch.tensor([b[1] for b in batch])
        labels = torch.stack([b[2] for b in batch])
        scores = torch.tensor([b[3] for b in batch])

        return states, ts, labels, scores


class ValueModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_layer = torch.nn.Conv2d(self.out_channels, 1, 1)

    def forward(self, t, x, y):
        features = super().forward(t, x, y)
        return torch.sigmoid(self.final_layer(features).mean(dim=[1, 2, 3]))


class MCTSFlowSampler:
    def __init__(
        self,
        image_size=32,
        channels=3,
        hidden_dims=[256, 512, 256],
        device="cuda:0",
        num_timesteps=10,
        num_classes=100,
        buffer_size=1000,
        num_channels=128,
        learning_rate=5e-4,
        load_models=True,
        flow_model="single_flow_model.pt",
        value_model="single_value_model.pt",
        inception_layer=3,
        pca_dim=None,
    ):
        # Check if CUDA is available and set device
        if torch.cuda.is_available():
            self.device = torch.device(device)
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU")

        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(0, 1, num_timesteps, device=self.device)
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes

        self.flow_model = UNetModel(
            dim=(channels, image_size, image_size),
            num_channels=num_channels,
            num_res_blocks=2,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.0,
            num_classes=num_classes,
            class_cond=True,
        ).to(self.device)

        self.value_model = ValueModel(
            dim=(channels, image_size, image_size),
            num_channels=num_channels,
            num_res_blocks=2,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.0,
            num_classes=num_classes,
            class_cond=True,
        ).to(self.device)

        # Initialize optimizers
        self.flow_optimizer = torch.optim.Adam(
            self.flow_model.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=learning_rate
        )

        warmup_epochs = 100
        num_epochs = 1000
        initial_lr = 1e-8

        self.flow_optimizer = torch.optim.Adam(
            self.flow_model.parameters(), lr=initial_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=initial_lr
        )

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return initial_lr + (learning_rate - initial_lr) * (
                    epoch / warmup_epochs
                )
            else:
                decay_progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return learning_rate + (initial_lr - learning_rate) * decay_progress

        self.flow_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.flow_optimizer, lr_lambda=lambda epoch: lr_lambda(epoch) / initial_lr
        )

        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.value_optimizer, lr_lambda=lambda epoch: lr_lambda(epoch) / initial_lr
        )

        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.05)
        self.trajectory_buffer = TrajectoryBuffer()

        # Try to load pre-trained models
        if load_models:
            if self.load_models(
                flow_model=flow_model,
                value_model=value_model,
            ):
                print("Successfully loaded pre-trained flow and value models")
            else:
                print("No pre-trained models found, starting from scratch")

        layer_to_dim = {0: 64, 1: 192, 2: 768, 3: 2048}

        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(layer_to_dim[inception_layer])],
        ).to(device)
        self.inception.eval()

        if inception_layer not in layer_to_dim:
            raise ValueError(
                f"Unsupported inception layer: {inception_layer}. "
                f"Supported layers are: {list(layer_to_dim.keys())}"
            )

        self.feature_dim = layer_to_dim[inception_layer]
        self.inception_layer = inception_layer
        self.use_pca = (
            inception_layer == 3 and pca_dim is not None
        )  # Only use PCA for layer 3
        self.pca_dim = pca_dim if self.use_pca else self.feature_dim

        if self.use_pca:
            print(
                f"Using inception layer {inception_layer} with {self.feature_dim} feature dimensions"
            )
            print(f"Will reduce to {self.pca_dim} dimensions using PCA")
            self._load_or_fit_pca()
        else:
            print(
                f"Using inception layer {inception_layer} with {self.feature_dim} feature dimensions"
            )

        # Load reference statistics based on feature dimension and PCA if applicable
        if self.use_pca:
            stats_file = f"cifar10_fid_stats_{self.feature_dim}to{self.pca_dim}dim.pkl"
        else:
            stats_file = f"cifar10_fid_stats_{self.feature_dim}dim.pkl"

        try:
            with open(stats_file, "rb") as f:
                cifar_stats = pickle.load(f)
            print(f"Loaded CIFAR-10 statistics from {stats_file}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Statistics file {stats_file} not found. Please run compute_cifar_stats.py "
                f"with appropriate parameters to generate the required statistics."
            )

        # Initialize per-class FID metrics and buffers
        self.fids = {
            i: {"mu": None, "sigma": None, "features": deque(maxlen=buffer_size)}
            for i in range(num_classes)
        }

        # Initialize global FID metrics and buffer
        self.global_fid = {
            "mu": None,
            "sigma": None,
            "sigma_inv": None,
            "features": deque(maxlen=buffer_size),
            "baseline_fid": None,
        }

        # Load per-class statistics
        for class_idx in range(num_classes):
            self.fids[class_idx]["mu"] = cifar_stats[f"class_{class_idx}_mu"]
            self.fids[class_idx]["sigma"] = cifar_stats[f"class_{class_idx}_sigma"]
            self.fids[class_idx]["sigma_inv"] = np.linalg.inv(
                self.fids[class_idx]["sigma"]
            )

        # Load global statistics if available
        if "global_mu" in cifar_stats and "global_sigma" in cifar_stats:
            print("Global FID statistics found, loading...")
            self.global_fid["mu"] = cifar_stats["global_mu"]
            self.global_fid["sigma"] = cifar_stats["global_sigma"]
            self.global_fid["sigma_inv"] = np.linalg.inv(self.global_fid["sigma"])
            self.has_global_stats = True
        else:
            print("No global FID statistics found in the loaded file")
            self.has_global_stats = False

        print("Initializing per-class buffers...")
        self.initialize_class_buffers(buffer_size)

    def compute_mahalanobis_distance(self, features, class_idx):
        """
        Compute the Mahalanobis distance between a sample and a class distribution.

        Args:
            features: Feature vector of the sample (numpy array)
            class_idx: Class index to compare against

        Returns:
            Mahalanobis distance (lower is better, closer to the class distribution)
        """
        mu = self.fids[class_idx]["mu"]
        sigma_inv = self.fids[class_idx]["sigma_inv"]

        # Compute Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        diff = features - mu
        mahalanobis = np.sqrt(diff.dot(sigma_inv).dot(diff))

        return mahalanobis

    def compute_global_mahalanobis_distance(self, features):
        """
        Compute the Mahalanobis distance between a sample and the global distribution.

        Args:
            features: Feature vector of the sample (numpy array)

        Returns:
            Mahalanobis distance (lower is better, closer to the global distribution)
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        mu = self.global_fid["mu"]
        sigma_inv = self.global_fid["sigma_inv"]

        # Compute Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        diff = features - mu
        mahalanobis = np.sqrt(diff.dot(sigma_inv).dot(diff))

        return mahalanobis

    def batch_compute_mahalanobis_distance(self, images, class_indices):
        """
        Compute Mahalanobis distance for a batch of images.

        Args:
            images: Tensor of shape [batch_size, C, H, W]
            class_indices: Tensor or list of class indices for each image

        Returns:
            Tensor of negative Mahalanobis distances (higher is better)
        """
        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Convert class_indices to list if it's a tensor
        if torch.is_tensor(class_indices):
            class_indices = class_indices.cpu().tolist()

        # Calculate Mahalanobis distances for each image
        mahalanobis_distances = []

        for i, (feature, class_idx) in enumerate(zip(features, class_indices)):
            distance = self.compute_mahalanobis_distance(feature, class_idx)
            # Return negative distance so higher values are better (consistent with FID change)
            mahalanobis_distances.append(-distance)

        return torch.tensor(mahalanobis_distances, device=images.device)

    def compute_distribution_mahalanobis_distance(self, images, class_indices):
        """
        Compute Mahalanobis distance for a distribution of images.

        This calculates how far the distribution of generated samples is from
        the reference distribution, rather than averaging individual distances.

        Args:
            images: Tensor of shape [batch_size, C, H, W]
            class_indices: Tensor or list of class indices for each image

        Returns:
            Tensor of negative Mahalanobis distances (higher is better)
        """
        # Ensure images are on the same device as the inception model
        inception_device = next(self.inception.parameters()).device
        images = images.to(inception_device)

        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Convert class_indices to list if it's a tensor
        if torch.is_tensor(class_indices):
            class_indices = class_indices.cpu().tolist()

        # Group features by class
        class_features = {}
        for feature, class_idx in zip(features, class_indices):
            if class_idx not in class_features:
                class_features[class_idx] = []
            class_features[class_idx].append(feature)

        # Calculate distribution Mahalanobis distance for each class
        distribution_distances = []

        for class_idx, feat_list in class_features.items():
            # Calculate mean of generated features for this class
            features_array = np.stack(feat_list)
            generated_mean = np.mean(features_array, axis=0)

            # Get reference statistics for this class
            if class_idx in self.fids:
                mu = self.fids[class_idx]["mu"]
                sigma_inv = self.fids[class_idx]["sigma_inv"]

                # Calculate Mahalanobis distance between generated mean and reference distribution
                diff = generated_mean - mu
                mahalanobis_dist = np.sqrt(diff.dot(sigma_inv).dot(diff))

                # Return negative distance so higher values are better (consistent with FID change)
                distribution_distances.append(-mahalanobis_dist)
            else:
                # If no statistics for this class, use a placeholder value
                distribution_distances.append(float("-inf"))

        # Average the distances across classes
        if distribution_distances:
            avg_distance = sum(distribution_distances) / len(distribution_distances)
        else:
            avg_distance = float("-inf")

        return torch.tensor(avg_distance, device=images.device)

    def compute_global_distribution_mahalanobis_distance(self, images):
        """
        Compute global Mahalanobis distance for a distribution of images.

        This calculates how far the distribution of generated samples is from
        the global reference distribution, rather than averaging individual distances.

        Args:
            images: Tensor of shape [batch_size, C, H, W]

        Returns:
            Tensor of negative Mahalanobis distance (higher is better)
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        # Ensure images are on the same device as the inception model
        inception_device = next(self.inception.parameters()).device
        images = images.to(inception_device)

        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Calculate mean of generated features
        features_array = np.stack(features)
        generated_mean = np.mean(features_array, axis=0)

        # Get global reference statistics
        mu = self.global_fid["mu"]
        sigma_inv = self.global_fid["sigma_inv"]

        # Calculate Mahalanobis distance between generated mean and reference distribution
        diff = generated_mean - mu
        mahalanobis_dist = np.sqrt(diff.dot(sigma_inv).dot(diff))

        # Return negative distance so higher values are better (consistent with FID change)
        return torch.tensor(-mahalanobis_dist, device=images.device)

    def batch_compute_global_mahalanobis_distance(self, images):
        """
        Compute global Mahalanobis distance for a batch of images.

        Args:
            images: Tensor of shape [batch_size, C, H, W]

        Returns:
            Tensor of negative Mahalanobis distances (higher is better)
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Calculate Mahalanobis distances for each image
        mahalanobis_distances = []

        for feature in features:
            distance = self.compute_global_mahalanobis_distance(feature)
            # Return negative distance so higher values are better (consistent with FID change)
            mahalanobis_distances.append(-distance)

        return torch.tensor(mahalanobis_distances, device=images.device)

    def compute_global_mean_difference(self, features):
        """
        Compute the Euclidean distance between a sample and the global distribution mean.

        Args:
            features: Feature vector of the sample (numpy array)

        Returns:
            Euclidean distance to the mean (lower is better, closer to the global mean)
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        mu = self.global_fid["mu"]

        # Compute Euclidean distance: ||x-μ||
        diff = features - mu
        distance = np.sqrt(np.sum(diff**2))

        return distance

    def batch_compute_global_mean_difference(self, images):
        """
        Compute global mean difference for a batch of images.

        Args:
            images: Tensor of shape [batch_size, C, H, W]

        Returns:
            Tensor of negative mean differences (higher is better)
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Calculate mean differences for each image
        mean_differences = []

        for feature in features:
            distance = self.compute_global_mean_difference(feature)
            # Return negative distance so higher values are better (consistent with other metrics)
            mean_differences.append(-distance)

        return torch.tensor(mean_differences, device=images.device)

    def compute_mean_difference(self, features, class_idx):
        """
        Compute the Euclidean distance between a sample and a class distribution mean.

        Args:
            features: Feature vector of the sample (numpy array)
            class_idx: Class index to compare against

        Returns:
            Euclidean distance to the class mean (lower is better, closer to the class mean)
        """
        mu = self.fids[class_idx]["mu"]

        # Compute Euclidean distance: ||x-μ||
        diff = features - mu
        distance = np.sqrt(np.sum(diff**2))

        return distance

    def batch_compute_mean_difference(self, images, class_indices):
        """
        Compute mean difference for a batch of images per class.

        Args:
            images: Tensor of shape [batch_size, C, H, W]
            class_indices: Tensor or list of class indices for each image

        Returns:
            Tensor of negative mean differences (higher is better)
        """
        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Convert class_indices to list if it's a tensor
        if torch.is_tensor(class_indices):
            class_indices = class_indices.cpu().tolist()

        # Calculate mean differences for each image
        mean_differences = []

        for i, (feature, class_idx) in enumerate(zip(features, class_indices)):
            distance = self.compute_mean_difference(feature, class_idx)
            # Return negative distance so higher values are better (consistent with other metrics)
            mean_differences.append(-distance)

        return torch.tensor(mean_differences, device=images.device)

    def _load_or_fit_pca(self):
        """Load or fit a PCA model for dimensionality reduction."""
        import pickle
        from sklearn.decomposition import PCA

        pca_file = f"inception_pca_{self.feature_dim}to{self.pca_dim}.pkl"

        try:
            # Try to load existing PCA model
            with open(pca_file, "rb") as f:
                self.pca = pickle.load(f)
            print(f"Loaded PCA model from {pca_file}")
        except FileNotFoundError:
            print(f"PCA model not found at {pca_file}.")
            print(
                "Please run compute_cifar_stats.py with --dim 2048 --pca_dim {self.pca_dim} first."
            )
            raise FileNotFoundError(f"PCA model file {pca_file} not found.")

    def set_timesteps(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(0, 1, num_timesteps, device=self.device)

    def initialize_class_buffers(self, n_samples, batch_size=64):
        """Initialize buffers for each class with generated samples in batches to save memory."""
        self.flow_model.eval()
        with torch.no_grad():
            samples_per_class = n_samples
            all_global_features = []  # For global buffer

            for class_idx in range(self.num_classes):
                # Process in batches
                remaining = samples_per_class
                all_features = []

                while remaining > 0:
                    # Determine current batch size
                    current_batch_size = min(batch_size, remaining)

                    # Generate samples for this specific class
                    labels = torch.full(
                        (current_batch_size,), class_idx, device=self.device
                    )
                    x = torch.randn(
                        current_batch_size,
                        self.channels,
                        self.image_size,
                        self.image_size,
                        device=self.device,
                    )

                    # Generate samples using simple flow
                    for t_idx in range(len(self.timesteps) - 1):
                        t = self.timesteps[t_idx]
                        dt = self.timesteps[t_idx + 1] - t
                        t_batch = torch.full(
                            (current_batch_size,), t.item(), device=self.device
                        )
                        velocity = self.flow_model(t_batch, x, labels)
                        x = x + velocity * dt

                    # Extract and store features for this batch
                    features = self.extract_inception_features(x)
                    all_features.extend(list(features))

                    # Also add to global features if global stats are available
                    if self.has_global_stats:
                        all_global_features.extend(list(features))

                    # Update remaining count
                    remaining -= current_batch_size

                # Store all collected features for this class
                self.fids[class_idx]["features"].extend(all_features)
                print(
                    f"Class {class_idx} initialized with {len(self.fids[class_idx]['features'])} samples"
                )

            # Initialize global buffer if global stats are available
            if self.has_global_stats:
                # Take a random subset if we have too many features
                if len(all_global_features) > n_samples:
                    import random

                    all_global_features = random.sample(all_global_features, n_samples)

                self.global_fid["features"].extend(all_global_features)
                print(
                    f"Global buffer initialized with {len(self.global_fid['features'])} samples"
                )

        print("Computing baseline FID scores for each class...")
        for class_idx in range(self.num_classes):
            feats = np.array(list(self.fids[class_idx]["features"]))
            mu = np.mean(feats, axis=0)
            sigma = np.cov(feats, rowvar=False)
            baseline_fid = self.calculate_frechet_distance(
                mu, sigma, self.fids[class_idx]["mu"], self.fids[class_idx]["sigma"]
            )
            self.fids[class_idx]["baseline_fid"] = baseline_fid
            print(f"Class {class_idx} baseline FID: {baseline_fid:.4f}")

        # Compute global baseline FID if global stats are available
        if self.has_global_stats:
            print("Computing global baseline FID score...")
            global_feats = np.array(list(self.global_fid["features"]))
            global_mu = np.mean(global_feats, axis=0)
            global_sigma = np.cov(global_feats, rowvar=False)
            global_baseline_fid = self.calculate_frechet_distance(
                global_mu, global_sigma, self.global_fid["mu"], self.global_fid["sigma"]
            )
            self.global_fid["baseline_fid"] = global_baseline_fid
            print(f"Global baseline FID: {global_baseline_fid:.4f}")

    def extract_inception_features(self, images):
        """Extract inception features using torchmetrics InceptionV3 with optional PCA reduction."""
        # Convert to byte tensor in range [0, 255] as expected by torchmetrics implementation
        images = (images * 255).byte()

        with torch.no_grad():
            features = self.inception(images)

            # Apply PCA reduction if configured
            if self.use_pca:
                features_np = features.cpu().numpy()
                features_reduced = self.pca.transform(features_np)
                return features_reduced
            else:
                return features.cpu().numpy()

    def compute_fid_change(self, image, class_idx):
        """
        Compute how much an image would change the constant FID score for its class.

        This is computed by:
        1. Extracting the inception features of the new image.
        2. Forming a candidate set by replacing the last element of the constant buffer
            (for that class) with the new feature.
        3. Computing the FID for the candidate set and comparing it to the baseline FID
            (precomputed after buffer initialization).

        Returns:
            The negative change in FID times 100 as the reward.
        """
        # Extract features from the new image (assumes image is 4D, e.g. [1, C, H, W])
        features = self.extract_inception_features(image)  # shape: (1, feat_dim)
        class_fid = self.fids[class_idx]

        # Create a candidate set from the constant buffer (do not modify the actual buffer)
        new_features = list(class_fid["features"])
        # Replace the last feature with the new image's feature
        new_features[-1] = features[0]
        new_features = np.array(new_features)

        mu_new = np.mean(new_features, axis=0)
        sigma_new = np.cov(new_features, rowvar=False)
        new_fid = self.calculate_frechet_distance(
            mu_new, sigma_new, class_fid["mu"], class_fid["sigma"]
        )

        baseline_fid = class_fid["baseline_fid"]
        # The reward is the negative change in FID multiplied by a factor (here, 100)
        return -(new_fid - baseline_fid) * 100

    def compute_global_fid_change(self, image):
        """
        Compute how much an image would change the global FID score.

        Similar to compute_fid_change but uses global statistics.

        Returns:
            The negative change in global FID times 100 as the reward.
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        # Extract features from the new image
        features = self.extract_inception_features(image)  # shape: (1, feat_dim)

        # Create a candidate set from the global buffer
        new_features = list(self.global_fid["features"])
        # Replace the last feature with the new image's feature
        new_features[-1] = features[0]
        new_features = np.array(new_features)

        mu_new = np.mean(new_features, axis=0)
        sigma_new = np.cov(new_features, rowvar=False)
        new_fid = self.calculate_frechet_distance(
            mu_new, sigma_new, self.global_fid["mu"], self.global_fid["sigma"]
        )

        baseline_fid = self.global_fid["baseline_fid"]
        # The reward is the negative change in FID multiplied by a factor
        return -(new_fid - baseline_fid) * 100

    def batch_compute_fid_change(self, images, class_indices):
        """
        Compute FID change for a batch of images in a more efficient way.

        Args:
            images: Tensor of shape [batch_size, C, H, W]
            class_indices: Tensor or list of class indices for each image

        Returns:
            Tensor of FID change scores (higher is better)
        """
        # Extract features for all images in one batch
        features = self.extract_inception_features(
            images
        )  # shape: (batch_size, feat_dim)

        # Convert class_indices to list if it's a tensor
        if torch.is_tensor(class_indices):
            class_indices = class_indices.cpu().tolist()

        # Calculate FID changes for each image
        fid_changes = []

        for i, (feature, class_idx) in enumerate(zip(features, class_indices)):
            class_fid = self.fids[class_idx]

            # Create a candidate set from the constant buffer
            new_features = list(class_fid["features"])
            # Replace the last feature with the new image's feature
            new_features[-1] = feature
            new_features = np.array(new_features)

            mu_new = np.mean(new_features, axis=0)
            sigma_new = np.cov(new_features, rowvar=False)
            new_fid = self.calculate_frechet_distance(
                mu_new, sigma_new, class_fid["mu"], class_fid["sigma"]
            )

            baseline_fid = class_fid["baseline_fid"]
            # The reward is the negative change in FID multiplied by a factor
            fid_changes.append(-100 * (new_fid - baseline_fid))

        return torch.tensor(fid_changes, device=images.device)

    def batch_compute_global_fid_change(self, images):
        """
        Compute global FID change for a batch of images.

        Args:
            images: Tensor of shape [batch_size, C, H, W]

        Returns:
            Tensor of global FID change scores (higher is better)
        """
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        # Extract features for all images in one batch
        features = self.extract_inception_features(images)

        # Calculate global FID changes for each image
        fid_changes = []

        for feature in features:
            # Create a candidate set from the global buffer
            new_features = list(self.global_fid["features"])
            # Replace the last feature with the new image's feature
            new_features[-1] = feature
            new_features = np.array(new_features)

            mu_new = np.mean(new_features, axis=0)
            sigma_new = np.cov(new_features, rowvar=False)
            new_fid = self.calculate_frechet_distance(
                mu_new, sigma_new, self.global_fid["mu"], self.global_fid["sigma"]
            )

            baseline_fid = self.global_fid["baseline_fid"]
            # The reward is the negative change in FID multiplied by a factor
            fid_changes.append(-(new_fid - baseline_fid) * 100)

        return torch.tensor(fid_changes, device=images.device)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Calculate the Frechet distance between two distributions."""
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute_sample_quality(self, samples, target_labels):
        """Compute quality score using class-specific FID change."""
        with torch.no_grad():
            rewards = []
            for sample, label in zip(samples, target_labels):
                reward = self.compute_fid_change(sample.unsqueeze(0), label.item())
                rewards.append(reward)
            return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def compute_global_sample_quality(self, samples):
        """Compute quality score using global FID change."""
        if not self.has_global_stats:
            raise ValueError("Global statistics not available")

        with torch.no_grad():
            rewards = []
            for sample in samples:
                reward = self.compute_global_fid_change(sample.unsqueeze(0))
                rewards.append(reward)
            return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def load_classifier(self, path="saved_models/mnist_classifier.pt"):
        """Load pre-trained MNIST classifier."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No pre-trained classifier found at {path}. "
                f"Please run train_mnist_classifier.py first."
            )
        self.classifier.load_state_dict(
            torch.load(path, weights_only=True, map_location=self.device)
        )
        print(f"Loaded pre-trained classifier from {path}")

    def generate_training_trajectory(self, y, noise_scale=0.1, upscale_factor=1):
        """Generate complete trajectories for training the value model using batch processing.

        An optional upscale_factor allows inserting additional timepoints by linearly interpolating
        between the computed trajectory points. For example, if upscale_factor is set to 5 and the
        original trajectory has 10 timepoints, the resulting trajectory will have approximately 50 timepoints.
        """
        trajectories = []
        ts = []
        batch_size = len(y)

        with torch.no_grad():
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )

            trajectories.append(current_samples.clone())
            ts.append(0.0)

            # Generate trajectory for all samples in batch
            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full((batch_size,), t.item(), device=self.device)

                # Flow model step
                velocity = self.flow_model(t_batch, current_samples, y)
                current_samples = current_samples + velocity * dt

                # Add small noise at each step
                if noise_scale > 0:
                    noise = (
                        torch.randn_like(current_samples) * noise_scale * (1 - float(t))
                    )
                    current_samples = current_samples + noise

                trajectories.append(current_samples.clone())
                ts.append(float(self.timesteps[step + 1]))

            # Compute final quality scores based on the final samples
            final_scores = self.compute_sample_quality(current_samples, y)

        # upscaling
        if upscale_factor > 1.0:
            T_orig = len(ts)
            T_new = int(T_orig * upscale_factor)
            new_ts = np.linspace(ts[0], ts[-1], T_new).tolist()
            new_trajectories = []
            for t_new in new_ts:
                if t_new >= ts[-1]:
                    new_trajectories.append(trajectories[-1])
                else:
                    for i in range(T_orig - 1):
                        if ts[i] <= t_new < ts[i + 1]:
                            alpha = (t_new - ts[i]) / (ts[i + 1] - ts[i])
                            new_img = torch.lerp(
                                trajectories[i], trajectories[i + 1], alpha
                            )
                            new_trajectories.append(new_img)
                            break
            trajectories = new_trajectories
            ts = new_ts

        return trajectories, ts, y, final_scores

    def train(
        self,
        train_loader,
        n_epochs=3,
        initial_flow_epochs=10,
        value_epochs=10,
        flow_epochs=10,
        use_tqdm=True,
    ):
        """Train both flow and value models."""
        # Initial flow model training
        print("\nInitial flow model training...")
        for epoch in range(initial_flow_epochs):
            print(f"\nInitial flow epoch {epoch + 1}/{initial_flow_epochs}")
            flow_loss = self.train_flow_matching(
                train_loader,
                desc=f"Initial flow training {epoch + 1}/{initial_flow_epochs}",
                use_tqdm=use_tqdm,
            )
            self.save_models()
            self.flow_scheduler.step()

        # Main training loop
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Train flow model for flow_epochs epochs
            for flow_epoch in range(flow_epochs):
                flow_loss = self.train_flow_matching(train_loader, use_tqdm=use_tqdm)
                self.flow_scheduler.step()

            # Generate trajectories
            print("Generating trajectories for value training...")
            self.flow_model.eval()  # Set to eval mode for trajectory generation

            # Get the batch size from first batch without wrapping in tqdm yet
            first_batch = next(iter(train_loader))[1]
            batch_size = len(first_batch)
            max_calls = int(10000 // batch_size)
            call_count = 0

            # Now create the iterator with tqdm
            iterator = (
                tqdm(train_loader, desc="Generating trajectories")
                if use_tqdm
                else train_loader
            )

            with torch.no_grad():
                for batch_idx, (_, y) in enumerate(iterator):
                    if call_count >= max_calls:
                        break

                    y = y.to(self.device)
                    trajectories, ts, labels, scores = (
                        self.generate_training_trajectory(
                            y,
                            upscale_factor=1.0,
                            noise_scale=0.0,
                        )
                    )
                    self.trajectory_buffer.add_trajectory(
                        trajectories, ts, labels, scores
                    )
                    call_count += 1

            # Train value model for multiple epochs
            print(f"Training value model for {value_epochs} epochs...")
            self.train_value_model(
                n_epochs=value_epochs,
                batch_size=128,
                use_tqdm=use_tqdm,
            )

            # Save after each main epoch
            self.save_models()

    def train_flow_matching(self, train_loader, desc="Training flow", use_tqdm=True):
        """Train flow model for one epoch."""
        self.flow_model.train()
        iterator = tqdm(train_loader, desc=desc) if use_tqdm else train_loader
        final_loss = 0

        for batch_idx, (x1, y) in enumerate(iterator):
            x1, y = x1.to(self.device), y.to(self.device)
            x0 = torch.randn_like(x1)

            # Train flow matching
            self.flow_optimizer.zero_grad()
            t, xt, ut, y0, y1 = self.FM.guided_sample_location_and_conditional_flow(
                x0, x1, y0=y, y1=y
            )
            vt = self.flow_model(t, xt, y1)
            flow_loss = torch.mean((vt - ut) ** 2)
            flow_loss.backward()
            self.flow_optimizer.step()

            final_loss = flow_loss.item()
            if use_tqdm:
                iterator.set_postfix({"flow_loss": f"{final_loss:.4f}"})

        print(f"{desc} - Flow Loss: {final_loss:.4f}")
        return final_loss

    def train_value_model(self, n_epochs=1, batch_size=64, use_tqdm=True):
        """Train value model on collected trajectories."""
        self.value_model.train()

        for epoch in range(n_epochs):
            n_batches = len(self.trajectory_buffer.trajectories) // batch_size
            iterator = (
                tqdm(
                    range(n_batches), desc=f"Value training(Epoch {epoch+1}/{n_epochs})"
                )
                if use_tqdm
                else range(n_batches)
            )

            total_loss = 0
            for batch_idx in iterator:
                batch = self.trajectory_buffer.sample_batch(batch_size)
                if batch is None:
                    continue

                states, ts, labels, scores = batch
                states = states.to(self.device)
                ts = ts.to(self.device)
                labels = labels.to(self.device)
                scores = scores.to(self.device)

                self.value_optimizer.zero_grad()
                value_pred = self.value_model(ts, states, labels)
                value_loss = F.mse_loss(value_pred, scores)
                value_loss.backward()
                self.value_optimizer.step()

                total_loss += value_loss.item()
                if use_tqdm:
                    iterator.set_postfix({"value_loss": f"{value_loss.item():.4f}"})

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"Average value loss: {avg_loss:.4f}")

    def batch_sample(
        self, class_label, batch_size=16, num_branches=4, num_keep=2, sigma=0.1
    ):
        """
        Efficient batched sampling method that maintains constant number of samples per batch element.
        Args:
            class_label: Target class to generate
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element (constant throughout)
            num_keep: Number of samples to keep before expansion
            sigma: Noise scale for branching
        Returns:
            Tensor of shape [batch_size, C, H, W]
        """
        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        expansion_factor = num_branches // num_keep

        self.flow_model.eval()
        self.value_model.eval()

        with torch.no_grad():
            # Initialize with num_branches samples per batch element
            current_samples = torch.randn(
                batch_size * num_branches,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_label = torch.full(
                (batch_size * num_branches,), class_label, device=self.device
            )

            # Track which batch element each sample belongs to
            batch_indices = torch.arange(
                batch_size, device=self.device
            ).repeat_interleave(num_branches)

            # Generate samples with branching
            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full(
                    (len(current_samples),), t.item(), device=self.device
                )

                # Flow step - process all samples in one batch
                velocity = self.flow_model(t_batch, current_samples, current_label)
                generated = current_samples + velocity * dt

                # Get value predictions for all samples
                value_scores = self.value_model(t_batch, generated, current_label)

                # Select top num_keep samples for each batch element
                selected_samples = []

                for batch_idx in range(batch_size):
                    # Get samples for this batch element
                    batch_mask = batch_indices == batch_idx
                    batch_samples = generated[batch_mask]
                    batch_scores = value_scores[batch_mask]

                    # Select top num_keep samples
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=num_keep, dim=0
                    )
                    selected_samples.append(batch_samples[top_k_indices])

                # Stack selected samples
                current_samples = torch.cat(
                    selected_samples, dim=0
                )  # shape: [batch_size * num_keep, C, H, W]

                # Expand each kept sample into expansion_factor new samples
                current_samples = current_samples.repeat_interleave(
                    expansion_factor, dim=0
                )
                current_label = torch.full(
                    (batch_size * num_branches,), class_label, device=self.device
                )
                batch_indices = torch.arange(
                    batch_size, device=self.device
                ).repeat_interleave(num_branches)
                # Add noise to create branches
                noise_scale = sigma * (1 - float(t))
                perturbations = torch.randn_like(current_samples) * noise_scale
                current_samples = current_samples + perturbations

            # Final selection - take best sample from each batch element's num_branches samples
            final_samples = []
            for batch_idx in range(batch_size):
                batch_mask = batch_indices == batch_idx
                batch_samples = current_samples[batch_mask]
                batch_scores = value_scores[batch_mask]
                best_idx = torch.argmax(batch_scores)
                final_samples.append(batch_samples[best_idx])

            return torch.stack(final_samples)  # shape: [batch_size, C, H, W]

    def batch_sample_wdt_with_selector(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        dt_std=0.1,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
        branch_dt=None,  # New parameter
    ):
        """
        Enhanced sampling method with configurable selection criteria and delayed branching.

        Args:
            class_label: Target class to generate
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element (constant throughout)
            num_keep: Number of samples to keep before expansion
            dt_std: Standard deviation for sampling different dt values
            selector: Selection criteria - one of ["fid", "mahalanobis", "mean"]
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
            branch_dt: Step size to use after branching begins (if None, uses base_dt)
        Returns:
            Tensor of shape [batch_size, C, H, W]
        """
        if num_branches == 1 and num_keep == 1:
            return self.regular_batch_sample(class_label, batch_size)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"

        # Select the appropriate scoring function based on parameters
        if selector == "fid":
            score_fn = (
                self.batch_compute_global_fid_change
                if use_global
                else lambda x, y: self.batch_compute_fid_change(x, y)
            )
        elif selector == "mahalanobis":
            score_fn = (
                self.batch_compute_global_mahalanobis_distance
                if use_global
                else lambda x, y: self.batch_compute_mahalanobis_distance(x, y)
            )
        elif selector == "mean":
            score_fn = (
                self.batch_compute_global_mean_difference
                if use_global
                else lambda x, y: self.batch_compute_mean_difference(x, y)
            )
        else:
            raise ValueError(f"Unknown selector: {selector}")

        expansion_factor = num_branches // num_keep
        self.flow_model.eval()

        with torch.no_grad():
            # Start with just batch_size samples until branching time
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_label = torch.full((batch_size,), class_label, device=self.device)
            current_times = torch.zeros(batch_size, device=self.device)

            base_dt = 1 / self.num_timesteps
            # Use provided branch_dt or default to base_dt
            branch_dt = branch_dt if branch_dt is not None else base_dt

            # Regular flow matching until branch_start_time
            while torch.all(current_times < branch_start_time):
                velocity = self.flow_model(
                    current_times, current_samples, current_label
                )

                # Use fixed dt until branching starts
                dt = min(base_dt, branch_start_time - current_times[0].item())
                current_samples = current_samples + velocity * dt
                current_times += dt

            # Expand to full number of branches when we reach branching time
            if current_times[0].item() >= branch_start_time:
                current_samples = current_samples.repeat_interleave(num_branches, dim=0)
                current_label = current_label.repeat_interleave(num_branches)
                current_times = current_times.repeat_interleave(num_branches)
                batch_indices = torch.arange(
                    batch_size, device=self.device
                ).repeat_interleave(num_branches)

            # Continue with branching until the end
            while torch.any(current_times < 1.0):
                # STEP 1: Take a random step to create branches
                velocity = self.flow_model(
                    current_times, current_samples, current_label
                )

                # Sample different dt values for each sample, using branch_dt as mean
                dts = torch.normal(
                    mean=branch_dt,
                    std=dt_std * branch_dt,
                    size=(len(current_samples),),
                    device=self.device,
                )
                dts = torch.clamp(
                    dts,
                    min=torch.tensor(0.0, device=self.device),
                    max=(1.0 - current_times) / 2,  # Ensure room for alignment
                )

                # Apply different step sizes to create branches
                branched_samples = current_samples + velocity * dts.view(-1, 1, 1, 1)
                branched_times = current_times + dts

                # STEP 2: Align all branches to the same timepoint
                target_time = (
                    torch.min(current_times) + 2 * branch_dt
                )  # Use branch_dt here too
                target_time = torch.min(
                    target_time, torch.tensor(1.0, device=self.device)
                )

                # Calculate dt to reach the target time
                dt_to_target = target_time - branched_times

                # Get velocity for alignment step
                velocity = self.flow_model(
                    branched_times, branched_samples, current_label
                )

                # Apply the step to align all branches
                aligned_samples = branched_samples + velocity * dt_to_target.view(
                    -1, 1, 1, 1
                )
                aligned_times = torch.full(
                    (len(branched_times),), target_time.item(), device=self.device
                )

                # Verify all branches are at the same time
                time_diff = torch.max(torch.abs(aligned_times - target_time))
                if time_diff > 1e-8:
                    print(
                        f"WARNING: Branches not at same time. Max difference: {time_diff.item():.8f}"
                    )

                # Calculate scores using the selected scoring function
                if use_global:
                    scores = score_fn(aligned_samples)
                else:
                    scores = score_fn(aligned_samples, current_label)

                # Select top num_keep samples for each batch element
                selected_samples = []
                selected_times = []

                for batch_idx in range(batch_size):
                    # Get samples for this batch element
                    batch_mask = batch_indices == batch_idx
                    batch_samples = aligned_samples[batch_mask]
                    batch_scores = scores[batch_mask]
                    batch_times = aligned_times[batch_mask]

                    # Select top num_keep samples
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])

                # Stack selected samples and times
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)

                # Expand each kept sample into expansion_factor new samples
                current_samples = current_samples.repeat_interleave(
                    expansion_factor, dim=0
                )
                current_times = current_times.repeat_interleave(expansion_factor)
                current_label = torch.full(
                    (batch_size * num_branches,), class_label, device=self.device
                )
                batch_indices = torch.arange(
                    batch_size, device=self.device
                ).repeat_interleave(num_branches)

                # Break if all samples have reached t=1
                if torch.all(current_times >= 1.0):
                    break

            # Final selection - take best sample from each batch element
            final_samples = []

            if not torch.all(torch.abs(current_times - 1.0) < 1e-8):
                print("WARNING: Not all samples reached t=1 after simulation")
                print(f"Current times: {current_times}")
                print(
                    f"Min time: {current_times.min().item():.6f}, Max time: {current_times.max().item():.6f}"
                )
                print(
                    f"Number of samples not at 1.0: {torch.sum(torch.abs(current_times - 1.0) >= 1e-8).item()}"
                )

            # Calculate final scores for all samples
            if use_global:
                final_scores = score_fn(current_samples)
            else:
                final_scores = score_fn(current_samples, current_label)

            for batch_idx in range(batch_size):
                batch_mask = batch_indices == batch_idx
                batch_samples = current_samples[batch_mask]
                batch_scores = final_scores[batch_mask]
                best_idx = torch.argmax(batch_scores)
                final_samples.append(batch_samples[best_idx])

            return torch.stack(final_samples)  # shape: [batch_size, C, H, W]

    def batch_sample_with_path_exploration(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        dt_std=0.1,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
        branch_dt=None,
    ):
        """
        Enhanced sampling method that explores complete paths before selection.
        At each branching step:
        1. Creates branches from current state
        2. Simulates each branch to completion (t=1)
        3. Selects best branches based on final outcomes
        4. Continues from selected branches at the next timestep

        Args:
            class_label: Target class to generate
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            dt_std: Standard deviation for sampling different dt values
            selector: Selection criteria - one of ["fid", "mahalanobis", "mean"]
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
            branch_dt: Step size to use after branching begins (if None, uses base_dt)
        """
        if num_branches == 1 and num_keep == 1:
            return self.regular_batch_sample(class_label, batch_size)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"

        # Select scoring function
        if selector == "fid":
            score_fn = (
                self.batch_compute_global_fid_change
                if use_global
                else lambda x, y: self.batch_compute_fid_change(x, y)
            )
        elif selector == "mahalanobis":
            score_fn = (
                self.batch_compute_global_mahalanobis_distance
                if use_global
                else lambda x, y: self.batch_compute_mahalanobis_distance(x, y)
            )
        elif selector == "mean":
            score_fn = (
                self.batch_compute_global_mean_difference
                if use_global
                else lambda x, y: self.batch_compute_mean_difference(x, y)
            )
        else:
            raise ValueError(f"Unknown selector: {selector}")

        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps
        branch_dt = branch_dt if branch_dt is not None else base_dt

        with torch.no_grad():
            # Initialize with batch_size samples
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_times = torch.zeros(batch_size, device=self.device)
            current_label = torch.full((batch_size,), class_label, device=self.device)

            # Regular flow until branch_start_time
            while torch.all(current_times < branch_start_time):
                velocity = self.flow_model(
                    current_times, current_samples, current_label
                )
                dt = min(base_dt, branch_start_time - current_times[0].item())
                current_samples = current_samples + velocity * dt
                current_times += dt

            # Main loop - continue until all samples reach t=1
            while torch.any(current_times < 1.0):
                # Create branches from current state
                branched_samples = current_samples.repeat_interleave(
                    num_branches, dim=0
                )
                branched_times = current_times.repeat_interleave(num_branches)
                branched_label = current_label.repeat_interleave(num_branches)
                batch_indices = torch.arange(
                    len(current_samples), device=self.device
                ).repeat_interleave(num_branches)

                # Take one branching step with different dt values
                velocity = self.flow_model(
                    branched_times, branched_samples, branched_label
                )

                # Sample different dt values for each branch
                dts = torch.normal(
                    mean=branch_dt,
                    std=dt_std * branch_dt,
                    size=(len(branched_samples),),
                    device=self.device,
                )
                dts = torch.clamp(
                    dts,
                    min=torch.tensor(0.0, device=self.device),
                    max=1.0 - branched_times,
                )

                # Apply the branching step
                branched_samples = branched_samples + velocity * dts.view(-1, 1, 1, 1)
                branched_times = branched_times + dts

                # Simulate each branch to completion (t=1) without further branching
                simulated_samples = branched_samples.clone()
                simulated_times = branched_times.clone()

                while torch.any(simulated_times < 1.0):
                    # Only update samples that haven't reached t=1
                    active_mask = simulated_times < 1.0
                    if not torch.any(active_mask):
                        break

                    velocity = self.flow_model(
                        simulated_times[active_mask],
                        simulated_samples[active_mask],
                        branched_label[active_mask],
                    )

                    dt = torch.min(
                        base_dt * torch.ones_like(simulated_times[active_mask]),
                        1.0 - simulated_times[active_mask],
                    )
                    simulated_samples[active_mask] = simulated_samples[
                        active_mask
                    ] + velocity * dt.view(-1, 1, 1, 1)
                    simulated_times[active_mask] = simulated_times[active_mask] + dt

                # Evaluate final samples
                if use_global:
                    final_scores = score_fn(simulated_samples)
                else:
                    final_scores = score_fn(simulated_samples, branched_label)

                # Select best branches for each batch element
                selected_samples = []
                selected_times = []
                selected_indices = []

                for idx in range(len(current_samples)):
                    # Get branches for this batch element
                    batch_mask = batch_indices == idx
                    batch_samples = branched_samples[
                        batch_mask
                    ]  # Use branched, not simulated
                    batch_times = branched_times[batch_mask]
                    batch_scores = final_scores[batch_mask]

                    # Select top num_keep branches based on final scores
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])
                    selected_indices.extend(
                        torch.where(batch_mask)[0][top_k_indices].tolist()
                    )

                # Update current state with selected branches
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)
                current_label = torch.full(
                    (len(current_samples),), class_label, device=self.device
                )

                # Break if all samples have reached t=1
                if torch.all(current_times >= 1.0):
                    break

            # Final selection - take best sample from each batch element
            final_samples = []

            # Evaluate final samples one last time
            if use_global:
                final_scores = score_fn(current_samples)
            else:
                final_scores = score_fn(current_samples, current_label)

            batch_indices = torch.arange(
                batch_size, device=self.device
            ).repeat_interleave(num_keep)

            # Group by original batch index
            samples_by_batch = {}
            for i in range(batch_size):
                batch_mask = batch_indices == i
                samples_by_batch[i] = {
                    "samples": current_samples[batch_mask],
                    "scores": final_scores[batch_mask],
                }

            # Select best sample for each batch element
            for i in range(batch_size):
                batch_data = samples_by_batch[i]
                best_idx = torch.argmax(batch_data["scores"])
                final_samples.append(batch_data["samples"][best_idx])

            return torch.stack(final_samples)

    def batch_sample_with_path_exploration_timewarp(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        warp_scale=0.5,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
        branch_dt=None,
        sqrt_epsilon=1e-8,
    ):
        """
        Enhanced sampling method using time warping path exploration.
        Supports num_branches > 4.
        Applies chain rule for time warping: dx/dt = v(x, f(t)) * f'(t).
        """
        if num_branches == 1 and num_keep == 1:
            return self.regular_batch_sample(class_label, batch_size)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"

        # --- Get warping functions and derivatives ---
        warp_fns, warp_deriv_fns = self._get_warp_functions(
            num_branches, self.device, sqrt_epsilon
        )
        # --- End Warp Function Definitions ---

        # Select scoring function
        if selector == "fid":
            score_fn = (
                self.batch_compute_global_fid_change
                if use_global
                else lambda x, y: self.batch_compute_fid_change(x, y)
            )
        elif selector == "mahalanobis":
            score_fn = (
                self.batch_compute_global_mahalanobis_distance
                if use_global
                else lambda x, y: self.batch_compute_mahalanobis_distance(x, y)
            )
        elif selector == "mean":
            score_fn = (
                self.batch_compute_global_mean_difference
                if use_global
                else lambda x, y: self.batch_compute_mean_difference(x, y)
            )
        else:
            raise ValueError(f"Unknown selector: {selector}")

        self.flow_model.eval()
        actual_dt_step = (
            branch_dt if branch_dt is not None else (1 / self.num_timesteps)
        )

        with torch.no_grad():
            # Initialize (same as before)
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_times = torch.zeros(batch_size, device=self.device)
            current_label = torch.full((batch_size,), class_label, device=self.device)

            # Regular flow until branch_start_time (same as before)
            base_dt = 1 / self.num_timesteps  # Keep base_dt for this initial phase
            while torch.any(current_times < branch_start_time):
                # Ensure we only step up to branch_start_time
                max_dt = torch.clamp(branch_start_time - current_times, min=0.0)
                dt = torch.minimum(base_dt * torch.ones_like(current_times), max_dt)
                active_mask = dt > 0  # Only compute velocity for active samples

                if not torch.any(active_mask):
                    break  # Exit if all reached start time

                velocity = self.flow_model(
                    current_times[active_mask],
                    current_samples[active_mask],
                    current_label[active_mask],
                )
                current_samples[active_mask] = current_samples[
                    active_mask
                ] + velocity * dt[active_mask].view(-1, 1, 1, 1)
                current_times[active_mask] = (
                    current_times[active_mask] + dt[active_mask]
                )

            # --- Main Path Exploration Loop ---
            while torch.any(current_times < 1.0):
                active_batch_mask = current_times < 1.0
                if not torch.any(active_batch_mask):
                    break

                active_samples = current_samples[active_batch_mask]
                active_times = current_times[active_batch_mask]
                active_label = current_label[active_batch_mask]
                num_active = len(active_samples)

                # --- 1. Create Branches ---
                branched_samples = active_samples.repeat_interleave(num_branches, dim=0)
                branched_times = active_times.repeat_interleave(num_branches)
                branched_label = active_label.repeat_interleave(num_branches)
                branch_indices_map = torch.arange(
                    num_active, device=self.device
                ).repeat_interleave(num_branches)
                warp_indices = (
                    torch.arange(len(branched_samples), device=self.device)
                    % num_branches
                )

                # --- 2. Take ONE Branching Step with Warping ---
                dt_step = torch.minimum(
                    actual_dt_step * torch.ones_like(branched_times),
                    1.0 - branched_times,
                )

                warped_times_step = torch.zeros_like(branched_times)
                warp_derivs_step = torch.zeros_like(branched_times)
                # Use the fetched functions
                for i in range(num_branches):
                    mask = warp_indices == i
                    if torch.any(mask):
                        orig_t = branched_times[mask]
                        warped_t = warp_fns[i](orig_t)  # Use warp_fns list
                        final_warped_t = (
                            1 - warp_scale
                        ) * orig_t + warp_scale * warped_t
                        warped_times_step[mask] = final_warped_t

                        orig_deriv = warp_deriv_fns[i](
                            orig_t
                        )  # Use warp_deriv_fns list
                        final_warp_deriv = (1 - warp_scale) * torch.ones_like(
                            orig_deriv
                        ) + warp_scale * orig_deriv
                        warp_derivs_step[mask] = final_warp_deriv

                velocity_step = self.flow_model(
                    warped_times_step, branched_samples, branched_label
                )
                branched_samples = (
                    branched_samples
                    + velocity_step
                    * warp_derivs_step.view(-1, 1, 1, 1)
                    * dt_step.view(-1, 1, 1, 1)
                )
                branched_times = branched_times + dt_step

                # --- 3. Simulate Each Branch to Completion (for Scoring) ---
                simulated_samples = branched_samples.clone()
                simulated_times = branched_times.clone()
                sim_warp_indices = warp_indices.clone()

                while torch.any(simulated_times < 1.0):
                    sim_active_mask = simulated_times < 1.0
                    if not torch.any(sim_active_mask):
                        break

                    active_sim_samples = simulated_samples[sim_active_mask]
                    active_sim_times = simulated_times[sim_active_mask]
                    active_sim_label = branched_label[sim_active_mask]
                    active_sim_warp_indices = sim_warp_indices[sim_active_mask]

                    dt_sim = torch.minimum(
                        actual_dt_step * torch.ones_like(active_sim_times),
                        1.0 - active_sim_times,
                    )

                    warped_times_sim = torch.zeros_like(active_sim_times)
                    warp_derivs_sim = torch.zeros_like(active_sim_times)
                    # Use the fetched functions
                    for i in range(num_branches):
                        mask = active_sim_warp_indices == i
                        if torch.any(mask):
                            orig_t = active_sim_times[mask]
                            warped_t = warp_fns[i](orig_t)  # Use warp_fns list
                            final_warped_t = (
                                1 - warp_scale
                            ) * orig_t + warp_scale * warped_t
                            warped_times_sim[mask] = final_warped_t

                            orig_deriv = warp_deriv_fns[i](
                                orig_t
                            )  # Use warp_deriv_fns list
                            final_warp_deriv = (1 - warp_scale) * torch.ones_like(
                                orig_deriv
                            ) + warp_scale * orig_deriv
                            warp_derivs_sim[mask] = final_warp_deriv

                    velocity_sim = self.flow_model(
                        warped_times_sim, active_sim_samples, active_sim_label
                    )
                    update = (
                        velocity_sim
                        * warp_derivs_sim.view(-1, 1, 1, 1)
                        * dt_sim.view(-1, 1, 1, 1)
                    )
                    simulated_samples[sim_active_mask] = active_sim_samples + update
                    simulated_times[sim_active_mask] = active_sim_times + dt_sim

                # --- 4. Score Simulated Samples ---
                if use_global:
                    final_scores = score_fn(simulated_samples)
                else:
                    # Ensure labels match simulated samples
                    final_scores = score_fn(simulated_samples, branched_label)

                breakpoint()

                # --- 5. Select Best Branches ---
                selected_samples_list = []
                selected_times_list = []

                for idx in range(num_active):  # Iterate through original active samples
                    # Find all branches originating from the idx-th active sample
                    mask = branch_indices_map == idx
                    # Get the states AFTER THE FIRST BRANCHING STEP for these branches
                    batch_branched_samples = branched_samples[mask]
                    batch_branched_times = branched_times[mask]
                    # Get the scores from the SIMULATED results for these branches
                    batch_scores = final_scores[mask]

                    # Select top num_keep based on scores
                    num_to_keep = min(
                        num_keep, len(batch_scores)
                    )  # Handle cases with fewer branches than num_keep
                    if num_to_keep > 0:
                        top_k_values, top_k_indices = torch.topk(
                            batch_scores, k=num_to_keep, dim=0
                        )
                        # Keep the state from *after the branching step*
                        selected_samples_list.append(
                            batch_branched_samples[top_k_indices]
                        )
                        selected_times_list.append(batch_branched_times[top_k_indices])

                # --- 6. Update Current State ---
                if not selected_samples_list:  # Handle empty case
                    break  # No active samples left to process

                # Update the state for the *next iteration* using the selected branches
                next_samples = torch.cat(selected_samples_list, dim=0)
                next_times = torch.cat(selected_times_list, dim=0)
                next_label = torch.full(
                    (len(next_samples),), class_label, device=self.device
                )

                # Replace the finished/processed samples with the new selected ones
                new_current_samples = torch.zeros_like(current_samples)
                new_current_times = torch.zeros_like(current_times)
                new_current_label = torch.zeros_like(current_label)

                # Keep samples that were already >= 1.0
                finished_mask = ~active_batch_mask
                if torch.any(finished_mask):
                    new_current_samples[finished_mask] = current_samples[finished_mask]
                    new_current_times[finished_mask] = current_times[finished_mask]
                    new_current_label[finished_mask] = current_label[finished_mask]

                # Add the newly selected samples (results of the branching)
                new_current_samples[active_batch_mask] = next_samples
                new_current_times[active_batch_mask] = next_times
                new_current_label[active_batch_mask] = (
                    next_label  # Assuming label stays same
                )

                current_samples = new_current_samples
                current_times = new_current_times
                current_label = new_current_label

            # --- Final Selection (after loop finishes) ---
            if use_global:
                final_scores = score_fn(current_samples)
            else:
                final_scores = score_fn(current_samples, current_label)

            final_samples = []
            samples_per_original = (
                current_samples.shape[0] // batch_size
            )  # Should be num_keep
            # assert samples_per_original == num_keep # This might fail if some finished early and weren't replaced

            # Correct way to handle potentially varying number of samples per batch item
            current_batch_indices = torch.arange(
                batch_size, device=self.device
            ).repeat_interleave(
                samples_per_original
            )  # Map current samples back to original batch index

            for i in range(batch_size):
                batch_mask = current_batch_indices == i
                batch_final_samples = current_samples[batch_mask]
                batch_final_scores = final_scores[batch_mask]

                if len(batch_final_scores) > 0:
                    best_idx_in_batch = torch.argmax(batch_final_scores)
                    final_samples.append(batch_final_samples[best_idx_in_batch])
                else:
                    # Handle case where a batch item might have finished very early
                    # and has no samples left in current_samples. Need a fallback.
                    # Option 1: Raise error
                    # Option 2: Return fewer than batch_size samples
                    # Option 3: Generate a fallback sample (e.g., random noise or rerun?)
                    print(
                        f"Warning: No final samples found for original batch index {i}. Skipping."
                    )

            if not final_samples:
                raise RuntimeError("No final samples generated.")

            return torch.stack(final_samples)

    def batch_sample_with_random_search(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,  # Unused but kept for compatibility
        dt_std=0.1,  # Unused but kept for compatibility
        selector="fid",
        use_global=False,
        branch_start_time=0.0,  # Unused but kept for compatibility
        branch_dt=None,  # Unused but kept for compatibility
    ):
        """
        Simple random search sampling method that:
        1. Runs num_branches independent flow matching batches
        2. Evaluates all samples using Mahalanobis distance
        3. Returns the best batch_size samples

        Args are kept the same as batch_sample_with_path_exploration for compatibility,
        though many are unused in this simpler implementation.
        """
        # Select scoring function (kept for compatibility)
        if selector == "fid":
            score_fn = (
                self.batch_compute_global_fid_change
                if use_global
                else lambda x, y: self.batch_compute_fid_change(x, y)
            )
        elif selector == "mahalanobis":
            score_fn = (
                self.batch_compute_global_mahalanobis_distance
                if use_global
                else lambda x, y: self.batch_compute_mahalanobis_distance(x, y)
            )
        elif selector == "mean":
            score_fn = (
                self.batch_compute_global_mean_difference
                if use_global
                else lambda x, y: self.batch_compute_mean_difference(x, y)
            )
        else:
            raise ValueError(f"Unknown selector: {selector}")

        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps

        with torch.no_grad():
            # Generate num_branches batches of samples
            all_samples = []

            for _ in range(num_branches):
                # Initialize one batch of samples
                current_samples = torch.randn(
                    batch_size,
                    self.channels,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )
                current_label = torch.full(
                    (batch_size,), class_label, device=self.device
                )

                # Regular flow matching for this batch
                for step, t in enumerate(self.timesteps[:-1]):
                    dt = self.timesteps[step + 1] - t
                    t_batch = torch.full((batch_size,), t.item(), device=self.device)

                    # Flow step
                    velocity = self.flow_model(t_batch, current_samples, current_label)
                    current_samples = current_samples + velocity * dt

                all_samples.append(current_samples)

            # Stack all batches
            all_samples = torch.cat(
                all_samples, dim=0
            )  # shape: [batch_size * num_branches, C, H, W]
            all_labels = torch.full(
                (batch_size * num_branches,), class_label, device=self.device
            )

            # Score all samples
            if use_global:
                scores = score_fn(all_samples)
            else:
                scores = score_fn(all_samples, all_labels)

            # Select the best batch_size samples
            top_k_values, top_k_indices = torch.topk(scores, k=batch_size, dim=0)
            final_samples = all_samples[top_k_indices]

            return final_samples

    def regular_batch_sample(self, class_label, batch_size=16):
        """
        Regular flow matching sampling without branching.
        """
        self.flow_model.eval()

        with torch.no_grad():
            # Initialize samples
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_label = torch.full((batch_size,), class_label, device=self.device)

            # Generate samples using timesteps
            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full((batch_size,), t.item(), device=self.device)

                # Flow step
                velocity = self.flow_model(t_batch, current_samples, current_label)
                current_samples = current_samples + velocity * dt

            return current_samples

    def save_models(self, path="saved_models"):
        """Save flow and value models separately."""
        os.makedirs(path, exist_ok=True)

        # Save flow model
        flow_path = f"{path}/single_flow_model.pt"
        torch.save(
            {
                "model": self.flow_model.state_dict(),
            },
            flow_path,
        )
        print(f"Flow model saved to {flow_path}")

        # Save value model
        value_path = f"{path}/single_value_model.pt"
        torch.save(
            {
                "model": self.value_model.state_dict(),
            },
            value_path,
        )
        print(f"Value model saved to {value_path}")

    def load_models(
        self,
        path="saved_models",
        flow_model="single_flow_model.pt",
        value_model="single_value_model.pt",
    ):
        """Load flow and value models if they exist."""
        flow_path = f"{path}/{flow_model}"
        value_path = f"{path}/{value_model}"

        flow_exists = os.path.exists(flow_path)
        value_exists = os.path.exists(value_path)

        if flow_exists:
            try:
                # First try loading as a checkpoint dictionary
                checkpoint = torch.load(
                    flow_path, map_location=self.device, weights_only=True
                )
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.flow_model.load_state_dict(checkpoint["model"])
                else:
                    # If not a checkpoint dict, assume it's a direct state_dict
                    self.flow_model.load_state_dict(checkpoint)
                print(f"Flow model loaded from {flow_path}")
            except Exception as e:
                print(f"Error loading flow model: {e}")
                flow_exists = False

        if value_exists:
            try:
                # First try loading as a checkpoint dictionary
                checkpoint = torch.load(
                    value_path, map_location=self.device, weights_only=True
                )
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.value_model.load_state_dict(checkpoint["model"])
                else:
                    # If not a checkpoint dict, assume it's a direct state_dict
                    self.value_model.load_state_dict(checkpoint)
                print(f"Value model loaded from {value_path}")
            except Exception as e:
                print(f"Error loading value model: {e}")
                value_exists = False

        return flow_exists and value_exists

    # --- Helper Function to Generate Warp Functions ---
    def _get_warp_functions(self, n, device, sqrt_epsilon=1e-4):
        """Generates n warp functions and their derivatives."""
        warp_fns = []
        warp_deriv_fns = []

        # Base functions
        # 1. Linear
        def linear_warp(t):
            return t

        def linear_warp_deriv(t):
            return torch.ones_like(t)

        warp_fns.append(linear_warp)
        warp_deriv_fns.append(linear_warp_deriv)

        # 2. Square
        def square_warp(t):
            return t**2

        def square_warp_deriv(t):
            return 2 * t

        warp_fns.append(square_warp)
        warp_deriv_fns.append(square_warp_deriv)

        # 3. Modified Sqrt
        def sqrt_warp(t):
            return torch.sqrt(t + sqrt_epsilon)

        def sqrt_warp_deriv(t):
            return 0.5 / torch.sqrt(t + sqrt_epsilon)

        warp_fns.append(sqrt_warp)
        warp_deriv_fns.append(sqrt_warp_deriv)

        # 4. Sigmoid (Standard)
        def sigmoid_warp_k12(t):
            t_scaled = 12 * t - 6
            return torch.sigmoid(t_scaled)

        def sigmoid_warp_deriv_k12(t):
            sig_t = sigmoid_warp_k12(t)
            return 12 * sig_t * (1 - sig_t)

        warp_fns.append(sigmoid_warp_k12)
        warp_deriv_fns.append(sigmoid_warp_deriv_k12)

        # Add more if needed (cyclically or by adding new types)
        current_len = 4
        while current_len < n:
            next_fn_idx = (
                current_len % 4
            )  # Cycle through base types for now, can add more complex logic
            if next_fn_idx == 0:  # Add Cubic
                p = 3 + (current_len // 4)  # Power increases: 3, 4, 5...

                def cubic_warp(t, p=p):  # Need to capture p
                    return t**p

                def cubic_warp_deriv(t, p=p):
                    return p * (t ** (p - 1))

                warp_fns.append(cubic_warp)
                warp_deriv_fns.append(cubic_warp_deriv)
            elif next_fn_idx == 1:  # Add Wider Sigmoid
                k = 6 * (
                    1 + current_len // 4
                )  # Steepness decreases: 6, 3, ...? Let's increase: 6, 18, 24
                k = 6 * (2 + current_len // 4)  # k = 18, 24, 30...

                def sigmoid_warp_k_wide(t, k=k):
                    t_scaled = k * t - k / 2  # Center at 0.5
                    return torch.sigmoid(t_scaled)

                def sigmoid_warp_deriv_k_wide(t, k=k):
                    sig_t = sigmoid_warp_k_wide(t, k=k)
                    return k * sig_t * (1 - sig_t)

                warp_fns.append(sigmoid_warp_k_wide)
                warp_deriv_fns.append(sigmoid_warp_deriv_k_wide)
            elif (
                next_fn_idx == 2
            ):  # Add Higher Root (like cube root) - also has derivative issues near 0
                p_inv = 3 + (current_len // 4)  # Root increases: 3, 4, 5...

                def root_warp(t, p_inv=p_inv, eps=sqrt_epsilon):
                    return (t + eps) ** (1.0 / p_inv)  # Add epsilon

                def root_warp_deriv(t, p_inv=p_inv, eps=sqrt_epsilon):
                    return (1.0 / p_inv) * (t + eps) ** ((1.0 / p_inv) - 1.0)

                warp_fns.append(root_warp)
                warp_deriv_fns.append(root_warp_deriv)
            elif next_fn_idx == 3:  # Add Narrower Sigmoid
                k = 18 * (1 + current_len // 4)  # Steepness increases: 18, 36, ...

                def sigmoid_warp_k_narrow(t, k=k):
                    t_scaled = k * t - k / 2  # Center at 0.5
                    return torch.sigmoid(t_scaled)

                def sigmoid_warp_deriv_k_narrow(t, k=k):
                    sig_t = sigmoid_warp_k_narrow(t, k=k)
                    return k * sig_t * (1 - sig_t)

                warp_fns.append(sigmoid_warp_k_narrow)
                warp_deriv_fns.append(sigmoid_warp_deriv_k_narrow)

            current_len += 1

        # Ensure we only return n functions if the loop overshoots (shouldn't happen with while)
        return warp_fns[:n], warp_deriv_fns[:n]

    # --- End Helper Function ---
