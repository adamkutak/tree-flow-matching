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
import torchvision.models as models
import pickle
from scipy.linalg import sqrtm
from pytorch_fid import fid_score
from torchmetrics.image.fid import NoTrainInceptionV3
from dino_v2_classifier_training import DINOv2Classifier


class MCTSFlowSampler:
    def __init__(
        self,
        image_size=32,
        channels=3,
        device="cuda:0",
        num_timesteps=10,
        num_classes=100,
        buffer_size=1000,
        num_channels=128,
        learning_rate=5e-4,
        load_models=True,
        flow_model="flow_model_imagenet32.pt",
        inception_layer=3,
        pca_dim=None,
        dataset="cifar10",
        flow_model_config=None,
        load_dino=True,
        dino_classifier_path="saved_models/dino_imagenet32_best.pt",
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
        self.dataset = dataset.lower()
        self.flow_model_config = flow_model_config or {}

        # For ImageNet256, we'll use the SiT model
        if self.dataset == "imagenet256":
            from third_party.SiT.models import SiT_models
            from diffusers.models import AutoencoderKL
            import pathlib
            import urllib.request

            print(f"Using SiT model for ImageNet256 with image size {image_size}")
            self.latent_hw = 32  # SD-VAE latent size

            # Create SiTWrapper to maintain interface compatibility
            class SiTWrapper(torch.nn.Module):
                def __init__(self, sit_model, device):
                    super().__init__()
                    self.model = sit_model
                    self.device = device

                def forward(self, t, x, y):
                    return self.model(x, t, y)

            # Load the base SiT-XL/2 model
            base_model = SiT_models["SiT-XL/2"](
                input_size=self.latent_hw, num_classes=num_classes, learn_sigma=True
            ).to(self.device)

            # Wrap the model to maintain interface compatibility
            self.flow_model = SiTWrapper(base_model, self.device)

            # Load weights following the provided approach
            ckpt_path = pathlib.Path("saved_models/SiT-XL-2-256.pt")
            # Load the weights into the base model
            self.flow_model.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )
            print(f"Loaded SiT-XL model weights from {ckpt_path}")

            # Load VAE for latent encoding/decoding
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
            ).to(self.device)
            self.vae.eval()

            # Flag to indicate we're using latent space for ImageNet256
            self.use_latent_space = True

        else:
            self.use_latent_space = False
            # Default UNet parameters
            default_config = {
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 2, 2],
                "attention_resolutions": "16",
                "num_heads": 4,
                "num_head_channels": 64,
            }
            model_params = {**default_config, **self.flow_model_config}

            self.flow_model = UNetModel(
                dim=(channels, image_size, image_size),
                num_channels=num_channels,
                num_res_blocks=model_params["num_res_blocks"],
                channel_mult=model_params["channel_mult"],
                num_heads=model_params["num_heads"],
                num_head_channels=model_params["num_head_channels"],
                attention_resolutions=model_params["attention_resolutions"],
                dropout=0.0,
                num_classes=num_classes,
                class_cond=True,
            ).to(self.device)

        warmup_epochs = 3
        num_epochs = 150
        initial_lr = 1e-8

        self.flow_optimizer = torch.optim.Adam(
            self.flow_model.parameters(), lr=initial_lr
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

        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.05)

        # Try to load pre-trained models
        if load_models and self.dataset != "imagenet256":
            if self.load_models(
                flow_model=flow_model,
            ):
                print("Successfully loaded pre-trained flow model")
            else:
                print("No pre-trained models found, starting from scratch")

        layer_to_dim = {0: 64, 1: 192, 2: 768, 3: 2048}

        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(layer_to_dim[inception_layer])],
        ).to(device)
        self.inception.eval()

        # 2. For logits (used for Inception Score calculation)
        self.inception_logits_model = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=["logits"]
        ).to(device)
        self.inception_logits_model.eval()

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
            stats_file = (
                f"{self.dataset}_fid_stats_{self.feature_dim}to{self.pca_dim}dim.pkl"
            )
        else:
            stats_file = f"{self.dataset}_fid_stats_{self.feature_dim}dim.pkl"
        try:
            with open(stats_file, "rb") as f:
                stats = pickle.load(f)
            print(f"Loaded {self.dataset.upper()} statistics from {stats_file}")
        except FileNotFoundError:
            print(f"Warning: Statistics file {stats_file} not found for {self.dataset}")
            stats = {}

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

        # Load per-class statistics if available
        for class_idx in range(num_classes):
            if f"class_{class_idx}_mu" in stats and f"class_{class_idx}_sigma" in stats:
                self.fids[class_idx]["mu"] = stats[f"class_{class_idx}_mu"]
                self.fids[class_idx]["sigma"] = stats[f"class_{class_idx}_sigma"]
                self.fids[class_idx]["sigma_inv"] = np.linalg.inv(
                    self.fids[class_idx]["sigma"]
                )
        # Load global statistics if available
        if "global_mu" in stats and "global_sigma" in stats:
            print("Global FID statistics found, loading...")
            self.global_fid["mu"] = stats["global_mu"]
            self.global_fid["sigma"] = stats["global_sigma"]
            self.global_fid["sigma_inv"] = np.linalg.inv(self.global_fid["sigma"])
            self.has_global_stats = True
        else:
            print("No global FID statistics found in the loaded file")
            self.has_global_stats = False

        if load_dino:
            try:
                print("Loading pretrained DINO classifier...")
                # Load the pretrained model with linear classifier (_lc suffix)
                model_name = "dinov2_vitb14_lc"
                self.dino_model = torch.hub.load(
                    "facebookresearch/dinov2", model_name
                ).to(device)
                self.dino_model.eval()
                print(f"Successfully loaded pretrained DINO classifier: {model_name}")
            except Exception as e:
                print(f"Error loading pretrained DINO classifier: {e}")

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

    def batch_compute_inception_score(self, images, class_labels=None):
        """
        Compute the class confidence score for each image.
        Uses either the probability of the highest predicted class (if class_labels=None)
        or the probability of the specified target class.

        Args:
            images: Tensor of shape [batch_size, C, H, W]
            class_labels: Optional tensor of class indices. If provided, returns the
                        confidence for those specific classes.

        Returns:
            Tensor of scores (higher is better for optimization)
        """
        import torch.nn.functional as F

        with torch.no_grad():
            images = (self.unnormalize_images(images) * 255).byte()

            # Get predictions and apply softmax
            logits = self.inception_logits_model(images)
            probs = F.softmax(logits, dim=1)

            if class_labels is not None:
                # Get the probability for the specific target class for each image
                batch_indices = torch.arange(len(images), device=self.device)
                confidence_scores = probs[batch_indices, class_labels]
            else:
                # Get the maximum probability (confidence) for each image
                confidence_scores, _ = torch.max(probs, dim=1)

            # Higher confidence is better
            return confidence_scores

    def batch_compute_dino_score(self, images, class_labels):
        """
        Compute scores using the pretrained DINO classifier.

        Args:
            images: Tensor of shape [batch_size, C, H, W]
            class_labels: Tensor of class indices

        Returns:
            Tensor of scores (higher is better for optimization)
        """

        with torch.no_grad():
            images = self.unnormalize_images(images)

            # now normalize the images to be imagenet normalized
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
                1, 3, 1, 1
            )
            images = (images - mean) / std

            if images.shape[-1] != 224:
                resized_images = torch.nn.functional.interpolate(
                    images, size=(224, 224), mode="bilinear", align_corners=False
                )
            else:
                resized_images = images

            # Forward pass through the pretrained DINO classifier
            logits = self.dino_model(resized_images)
            self.debug_mode = True

            # Optional: Calculate and print accuracy for monitoring
            if getattr(self, "debug_mode", False):
                # Top-1 accuracy
                predicted_classes = torch.argmax(logits, dim=1)
                correct_predictions = predicted_classes == class_labels
                num_correct = correct_predictions.sum().item()
                accuracy = num_correct / len(images) * 100

                # Top-5 accuracy
                _, top5_preds = logits.topk(5, 1, True, True)
                top5_correct = torch.zeros_like(class_labels, dtype=torch.bool)
                for i in range(5):
                    top5_correct = top5_correct | (top5_preds[:, i] == class_labels)
                num_top5_correct = top5_correct.sum().item()
                top5_accuracy = num_top5_correct / len(images) * 100

                print(
                    f"DINO Accuracy: Top-1: {num_correct}/{len(images)} ({accuracy:.2f}%), "
                    f"Top-5: {num_top5_correct}/{len(images)} ({top5_accuracy:.2f}%)"
                )

            # Get scores for the specified class labels
            batch_indices = torch.arange(len(images), device=self.device)
            scores = logits[batch_indices, class_labels]

            return scores

    def _load_or_fit_pca(self):
        """Load or fit a PCA model for dimensionality reduction."""
        import pickle

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
        images = (self.unnormalize_images(images) * 255).byte()

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

    def compute_batch_fid(self, image_batch, class_label, use_global=True):
        """
        Computes the FID score for a given batch of images against pre-computed
        statistics of the target distribution (real data). Lower is better.

        Args:
            image_batch: Tensor of shape [batch_size, C, H, W].
            class_label: The target class index.
            use_global: If True, compare against global stats, otherwise class-specific.

        Returns:
            A single float value representing the FID score. Returns infinity if
            batch size is too small or stats are missing.
        """
        batch_size = image_batch.shape[0]

        features = self.extract_inception_features(image_batch)

        mu_gen = np.mean(features, axis=0)
        # Add epsilon to prevent singular covariance
        sigma_gen = np.cov(features, rowvar=False)
        fid = self.calculate_frechet_distance(
            mu_gen, sigma_gen, self.global_fid["mu"], self.global_fid["sigma"]
        )
        return fid

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Calculate the Frechet distance between two distributions."""
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

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

    def _get_score_function(self, selector, use_global=False):
        """
        Helper function to get the appropriate scoring function based on selector
        and global flag. Returns the scoring function and updated use_global flag.

        Args:
            selector (str): Selection criteria - one of ["fid", "mahalanobis", "mean",
                            "inception_score", "dino_score"]
            use_global (bool): Whether to use global statistics instead of class-specific ones

        Returns:
            tuple: (score_function, use_global) - The scoring function and possibly updated use_global flag
        """
        if selector == "fid":
            if use_global:
                return self.batch_compute_global_fid_change, True
            else:
                return lambda x, y: self.batch_compute_fid_change(x, y), False
        elif selector == "mahalanobis":
            if use_global:
                return self.batch_compute_global_mahalanobis_distance, True
            else:
                return lambda x, y: self.batch_compute_mahalanobis_distance(x, y), False
        elif selector == "mean":
            if use_global:
                return self.batch_compute_global_mean_difference, True
            else:
                return lambda x, y: self.batch_compute_mean_difference(x, y), False
        elif selector == "inception_score":
            # Inception score is always global
            return self.batch_compute_inception_score, True
        elif selector == "dino_score":
            # DINO score is always not global (we use the class labels)
            return lambda x, y: self.batch_compute_dino_score(x, y), False
        else:
            raise ValueError(f"Unknown selector: {selector}")

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
            class_label: Tensor of class labels for each sample
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            dt_std: Standard deviation for sampling different dt values
            selector: Selection criteria - options include "fid", "mahalanobis", "mean", "inception_score", "dino_score"
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

        score_fn, use_global = self._get_score_function(selector, use_global)

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
            # Class label is already a tensor of batch_size
            current_label = class_label

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
                # Repeat each label num_branches times
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
                selected_labels = []  # Need to track labels too

                for idx in range(len(current_samples)):
                    # Get branches for this batch element
                    batch_mask = batch_indices == idx
                    batch_samples = branched_samples[
                        batch_mask
                    ]  # Use branched, not simulated
                    batch_times = branched_times[batch_mask]
                    batch_labels = branched_label[batch_mask]  # Keep track of labels
                    batch_scores = final_scores[batch_mask]

                    # Select top num_keep branches based on final scores
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])
                    selected_labels.append(
                        batch_labels[top_k_indices]
                    )  # Keep selected labels

                # Update current state with selected branches
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)
                current_label = torch.cat(
                    selected_labels, dim=0
                )  # Update with selected labels

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

            # Need to track both original batch indices and corresponding labels
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
                    "labels": current_label[batch_mask],  # Keep track of labels
                }

            # Select best sample for each batch element
            for i in range(batch_size):
                batch_data = samples_by_batch[i]
                best_idx = torch.argmax(batch_data["scores"])
                final_samples.append(batch_data["samples"][best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

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

        Args:
            class_label: Tensor of class labels for each sample
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            warp_scale: Scale factor for time warping
            selector: Selection criteria - options include "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
            branch_dt: Step size to use after branching begins (if None, uses base_dt)
            sqrt_epsilon: Small value for numerical stability
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

        score_fn, use_global = self._get_score_function(selector, use_global)

        self.flow_model.eval()
        actual_dt_step = (
            branch_dt if branch_dt is not None else (1 / self.num_timesteps)
        )

        with torch.no_grad():
            # Initialize with tensor class labels
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_times = torch.zeros(batch_size, device=self.device)
            # Class label is already a tensor of batch_size
            current_label = class_label

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
                active_label = current_label[active_batch_mask]  # Get active labels
                num_active = len(active_samples)

                # --- 1. Create Branches ---
                branched_samples = active_samples.repeat_interleave(num_branches, dim=0)
                branched_times = active_times.repeat_interleave(num_branches)
                branched_label = active_label.repeat_interleave(
                    num_branches
                )  # Repeat active labels
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
                    final_scores = score_fn(simulated_samples, branched_label)

                # --- 5. Select Best Branches ---
                selected_samples_list = []
                selected_times_list = []
                selected_labels_list = []  # Track labels

                for idx in range(num_active):  # Iterate through original active samples
                    # Find all branches originating from the idx-th active sample
                    mask = branch_indices_map == idx
                    # Get the states AFTER THE FIRST BRANCHING STEP for these branches
                    batch_branched_samples = branched_samples[mask]
                    batch_branched_times = branched_times[mask]
                    batch_branched_labels = branched_label[mask]  # Keep labels
                    # Get the scores from the SIMULATED results for these branches
                    batch_scores = final_scores[mask]

                    # Select top num_keep based on scores
                    num_to_keep = min(
                        num_keep, len(batch_scores)
                    )  # Handle cases with fewer branches than num_keep
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=num_to_keep, dim=0
                    )
                    # Keep the state from *after the branching step*
                    selected_samples_list.append(batch_branched_samples[top_k_indices])
                    selected_times_list.append(batch_branched_times[top_k_indices])
                    selected_labels_list.append(
                        batch_branched_labels[top_k_indices]
                    )  # Keep labels

                # --- 6. Update Current State ---
                if not selected_samples_list:  # Handle empty case
                    break  # No active samples left to process

                # Update the state for the *next iteration* using the selected branches
                next_samples = torch.cat(selected_samples_list, dim=0)
                next_times = torch.cat(selected_times_list, dim=0)
                next_label = torch.cat(
                    selected_labels_list, dim=0
                )  # Use concat for labels

                # Replace the finished/processed samples with the new selected ones
                new_current_samples = torch.zeros_like(current_samples)
                new_current_times = torch.zeros_like(current_times)
                new_current_label = torch.zeros_like(
                    current_label
                )  # Keep label tensor shape

                # Keep samples that were already >= 1.0
                finished_mask = ~active_batch_mask
                if torch.any(finished_mask):
                    new_current_samples[finished_mask] = current_samples[finished_mask]
                    new_current_times[finished_mask] = current_times[finished_mask]
                    new_current_label[finished_mask] = current_label[finished_mask]

                # Add the newly selected samples (results of the branching)
                new_current_samples[active_batch_mask] = next_samples
                new_current_times[active_batch_mask] = next_times
                new_current_label[active_batch_mask] = next_label

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

                best_idx_in_batch = torch.argmax(batch_final_scores)
                final_samples.append(batch_final_samples[best_idx_in_batch])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_with_path_exploration_timewarp_shifted(
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

        Args:
            class_label: Tensor of class labels for each sample
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            warp_scale: Scale factor for time warping
            selector: Selection criteria - options include "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
            branch_dt: Step size to use after branching begins (if None, uses base_dt)
            sqrt_epsilon: Small value for numerical stability
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

        score_fn, use_global = self._get_score_function(selector, use_global)

        self.flow_model.eval()
        actual_dt_step = (
            branch_dt if branch_dt is not None else (1 / self.num_timesteps)
        )

        with torch.no_grad():
            # Initialize with tensor class labels
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            current_times = torch.zeros(batch_size, device=self.device)
            # Class label is already a tensor of batch_size
            current_label = class_label

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

            current_warped_times = current_times.clone()
            # --- Main Path Exploration Loop ---
            while torch.any(current_times < 1.0):
                active_batch_mask = current_times < 1.0
                if not torch.any(active_batch_mask):
                    break

                active_samples = current_samples[active_batch_mask]
                active_times = current_times[active_batch_mask]
                active_warped_times = current_warped_times[
                    active_batch_mask
                ]  # Get current warped times
                active_label = current_label[active_batch_mask]
                num_active = len(active_samples)

                # --- 1. Create Branches ---
                branched_samples = active_samples.repeat_interleave(num_branches, dim=0)
                branched_times = active_times.repeat_interleave(num_branches)
                branched_warped_times = active_warped_times.repeat_interleave(
                    num_branches
                )  # Include warped times
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

                # Create all warping functions for each branch directly
                all_warp_fns = []
                all_warp_deriv_fns = []

                # For each branch, create continuous warping functions at its current time
                for i in range(len(branched_samples)):
                    branch_time = branched_times[i].item()
                    branch_warped_time = branched_warped_times[
                        i
                    ].item()  # Use the current warped time
                    warp_idx = warp_indices[i].item()

                    # Generate continuous warping functions at this branch's time and warped time
                    warp_fns_i, warp_deriv_fns_i = self._get_continuous_warp_functions(
                        num_branches,
                        self.device,
                        branch_time,
                        current_warped_time=branch_warped_time,  # Pass the current warped time
                        sqrt_epsilon=sqrt_epsilon,
                        clamp_time=True,
                    )

                    # Store the specific function needed for this branch
                    all_warp_fns.append(warp_fns_i[warp_idx])
                    all_warp_deriv_fns.append(warp_deriv_fns_i[warp_idx])

                # Apply warping to each branch
                warped_times_step = torch.zeros_like(branched_times)
                warp_derivs_step = torch.zeros_like(branched_times)

                for i in range(len(branched_samples)):
                    t = branched_times[i]
                    warped_t = all_warp_fns[i](t.unsqueeze(0)).squeeze()
                    final_warped_t = (1 - warp_scale) * t + warp_scale * warped_t
                    warped_times_step[i] = final_warped_t

                    warp_deriv = all_warp_deriv_fns[i](t.unsqueeze(0)).squeeze()
                    final_warp_deriv = (1 - warp_scale) * torch.ones_like(
                        warp_deriv
                    ) + warp_scale * warp_deriv
                    warp_derivs_step[i] = final_warp_deriv

                # Compute velocity using the warped times
                velocity_step = self.flow_model(
                    warped_times_step, branched_samples, branched_label
                )

                # Update samples using the velocity and warp derivatives
                branched_samples = (
                    branched_samples
                    + velocity_step
                    * warp_derivs_step.view(-1, 1, 1, 1)
                    * dt_step.view(-1, 1, 1, 1)
                )
                branched_times = branched_times + dt_step
                next_branched_warped_times = (
                    warped_times_step.clone()
                )  # Store for next iteration

                # --- 3. Simulate Each Branch to Completion (for Scoring) ---
                simulated_samples = branched_samples.clone()
                simulated_times = branched_times.clone()
                simulated_warped_times = (
                    next_branched_warped_times.clone()
                )  # Track warped times during simulation

                # Clone the warping functions for simulation
                sim_warp_fns = all_warp_fns.copy()
                sim_warp_deriv_fns = all_warp_deriv_fns.copy()

                while torch.any(simulated_times < 1.0):
                    sim_active_mask = simulated_times < 1.0
                    if not torch.any(sim_active_mask):
                        break

                    active_indices = torch.where(sim_active_mask)[0]
                    active_sim_samples = simulated_samples[sim_active_mask]
                    active_sim_times = simulated_times[sim_active_mask]
                    active_sim_warped_times = simulated_warped_times[sim_active_mask]
                    active_sim_label = branched_label[sim_active_mask]

                    dt_sim = torch.minimum(
                        actual_dt_step * torch.ones_like(active_sim_times),
                        1.0 - active_sim_times,
                    )

                    warped_times_sim = torch.zeros_like(active_sim_times)
                    warp_derivs_sim = torch.zeros_like(active_sim_times)

                    # Apply warping functions to each active branch
                    for idx, i in enumerate(active_indices):
                        t = active_sim_times[idx]
                        warped_t = sim_warp_fns[i](t.unsqueeze(0)).squeeze()
                        final_warped_t = (1 - warp_scale) * t + warp_scale * warped_t
                        warped_times_sim[idx] = final_warped_t

                        warp_deriv = sim_warp_deriv_fns[i](t.unsqueeze(0)).squeeze()
                        final_warp_deriv = (1 - warp_scale) * torch.ones_like(
                            warp_deriv
                        ) + warp_scale * warp_deriv
                        warp_derivs_sim[idx] = final_warp_deriv

                    # Compute velocity and update samples
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
                    simulated_warped_times[sim_active_mask] = (
                        warped_times_sim  # Update warped times for next step
                    )

                # After simulation, score samples as before
                if use_global:
                    final_scores = score_fn(simulated_samples)
                else:
                    final_scores = score_fn(simulated_samples, branched_label)

                # --- 5. Select Best Branches and update ---
                selected_samples_list = []
                selected_times_list = []
                selected_warped_times_list = []  # Also track selected warped times
                selected_labels_list = []

                for idx in range(num_active):
                    mask = branch_indices_map == idx
                    batch_branched_samples = branched_samples[mask]
                    batch_branched_times = branched_times[mask]
                    batch_branched_warped_times = next_branched_warped_times[
                        mask
                    ]  # Get warped times
                    batch_branched_labels = branched_label[mask]
                    batch_scores = final_scores[mask]

                    num_to_keep = min(num_keep, len(batch_scores))
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=num_to_keep, dim=0
                    )

                    selected_samples_list.append(batch_branched_samples[top_k_indices])
                    selected_times_list.append(batch_branched_times[top_k_indices])
                    selected_warped_times_list.append(
                        batch_branched_warped_times[top_k_indices]
                    )  # Keep warped times
                    selected_labels_list.append(batch_branched_labels[top_k_indices])

                # --- 6. Update Current State ---
                if not selected_samples_list:
                    break

                next_samples = torch.cat(selected_samples_list, dim=0)
                next_times = torch.cat(selected_times_list, dim=0)
                next_warped_times = torch.cat(
                    selected_warped_times_list, dim=0
                )  # Update warped times
                next_label = torch.cat(selected_labels_list, dim=0)

                # Replace the finished/processed samples with the new selected ones
                new_current_samples = torch.zeros_like(current_samples)
                new_current_times = torch.zeros_like(current_times)
                new_current_warped_times = torch.zeros_like(
                    current_warped_times
                )  # For warped times
                new_current_label = torch.zeros_like(current_label)

                # Keep samples that were already >= 1.0
                finished_mask = ~active_batch_mask
                if torch.any(finished_mask):
                    new_current_samples[finished_mask] = current_samples[finished_mask]
                    new_current_times[finished_mask] = current_times[finished_mask]
                    new_current_warped_times[finished_mask] = current_warped_times[
                        finished_mask
                    ]
                    new_current_label[finished_mask] = current_label[finished_mask]

                # Add the newly selected samples (results of the branching)
                new_current_samples[active_batch_mask] = next_samples
                new_current_times[active_batch_mask] = next_times
                new_current_warped_times[active_batch_mask] = next_warped_times
                new_current_label[active_batch_mask] = next_label

                current_samples = new_current_samples
                current_times = new_current_times
                current_warped_times = new_current_warped_times
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

                best_idx_in_batch = torch.argmax(batch_final_scores)
                final_samples.append(batch_final_samples[best_idx_in_batch])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_with_random_search(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        selector="fid",
        use_global=False,
    ):
        """
        Simple random search sampling method that:
        1. Runs num_branches independent flow matching batches
        2. Evaluates all samples using selected metric
        3. Returns the best sample for each class position

        Args:
            class_label: Tensor of class labels for the batch
            batch_size: Number of final samples to generate
            num_branches: Number of branches to evaluate per sample
            num_keep: Unused but kept for compatibility
            dt_std: Unused but kept for compatibility
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", or "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Unused but kept for compatibility
            branch_dt: Unused but kept for compatibility
        """
        assert (
            len(class_label) == batch_size
        ), "class_label tensor length must match batch_size"

        if num_branches == 1:
            return self.regular_batch_sample(class_label, batch_size)

        score_fn, use_global = self._get_score_function(selector, use_global)

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

                # Regular flow matching for this batch
                for step, t in enumerate(self.timesteps[:-1]):
                    t_batch = torch.full((batch_size,), t.item(), device=self.device)

                    # Flow step
                    velocity = self.flow_model(t_batch, current_samples, class_label)
                    current_samples = current_samples + velocity * base_dt

                all_samples.append(current_samples)

            # Calculate scores for each complete batch
            all_scores = []
            for branch_samples in all_samples:
                if use_global:
                    scores = score_fn(branch_samples)
                else:
                    scores = score_fn(branch_samples, class_label)
                all_scores.append(scores)

            # Stack scores for each batch [num_branches, batch_size]
            all_scores = torch.stack(all_scores, dim=0)

            # Find best branch for each position
            best_branch_indices = torch.argmax(all_scores, dim=0)  # Shape: [batch_size]

            # Construct final batch by selecting best sample for each position
            final_samples = torch.zeros(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )

            for i in range(batch_size):
                best_branch = best_branch_indices[i]
                final_samples[i] = all_samples[best_branch][i]

            return self.unnormalize_images(final_samples)

    # ---------------------------------------------------------
    # Section for minimizing FID over the entire set of samples using iterative refinement
    # ---------------------------------------------------------

    def batch_sample_refine_global_fid_random(
        self,
        n_samples: int,  # Total samples in the final dataset
        refinement_batch_size: int,  # Size of batches to swap
        num_branches: int,  # Candidates generated per swap slot
        num_batches: int = 10,  # Number of random batches to evaluate
        num_iterations: int = 1,  # Number of FULL refinement passes over the data
        use_global: bool = True,  # Use global or class-specific target FID stats
    ):
        """
        Generates a full dataset and iteratively refines it to minimize global FID
        using random search candidate generation. Uses random uniform class sampling
        to be compatible with large-class datasets like ImageNet.

        Args:
            n_samples: Total number of samples in the final dataset.
            refinement_batch_size: Size of the batches to consider swapping out.
            num_branches: Multiplier for batch_size to determine candidate pool size per attempt.
            num_batches: Number of random batches to evaluate for improvement.
            num_iterations: Number of full passes over the dataset for refinement (default: 1).
            use_global: Whether to use global target stats for FID calculation.

        Returns:
            Tensor of [n_samples, C, H, W] representing the final refined dataset.
        """

        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps

        # --- 1. Initialization: Generate Initial Pool ---
        initial_pool_samples = torch.zeros(
            (n_samples, self.channels, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.device,
        )
        initial_pool_labels = torch.zeros(
            n_samples, dtype=torch.long, device=self.device
        )
        current_idx = 0
        generation_chunk_size = refinement_batch_size

        # Generate initial pool with random class labels
        while current_idx < n_samples:
            chunk_size = min(generation_chunk_size, n_samples - current_idx)
            if chunk_size <= 0:
                break

            # Generate random class labels
            chunk_label = torch.randint(
                0, self.num_classes, (chunk_size,), device=self.device
            )

            # Generate a chunk using standard flow matching
            chunk_samples = torch.randn(
                chunk_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )

            # Standard Euler integration
            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full((chunk_size,), t.item(), device=self.device)
                with torch.no_grad():
                    velocity = self.flow_model(t_batch, chunk_samples, chunk_label)
                chunk_samples = chunk_samples + velocity * dt

            # Add to pool
            indices = slice(current_idx, current_idx + chunk_size)
            initial_pool_samples[indices] = chunk_samples
            initial_pool_labels[indices] = chunk_label
            current_idx += chunk_size

        # --- Pre-compute Initial Features and FID (Only if refining) ---
        with torch.no_grad():
            all_features_list = []
            feature_batch_size = 128
            for i in range(0, n_samples, feature_batch_size):
                batch = initial_pool_samples[i : i + feature_batch_size]
                features = self.extract_inception_features(batch)
                all_features_list.append(features)
            current_pool_features = np.concatenate(all_features_list, axis=0)

        # Get target statistics
        if use_global:
            target_mu = self.global_fid["mu"]
            target_sigma = self.global_fid["sigma"]
        else:
            raise NotImplementedError(
                "Class-specific FID refinement target not implemented yet."
            )

        current_global_fid = self.compute_fid_from_features(
            current_pool_features, target_mu, target_sigma
        )
        print(f"Initial Global FID: {current_global_fid:.4f}")

        # --- Handle num_branches == 1 Case ---
        if num_branches == 1:
            return initial_pool_samples

        # --- 2. Refinement Loop ---
        for pass_num in range(num_iterations):
            print(f"\n--- Refinement Pass {pass_num + 1}/{num_iterations} ---")
            num_swaps_this_pass = 0

            # Iterate through all batches without class segregation
            for batch_start_idx in range(0, n_samples, refinement_batch_size):
                batch_end_idx = min(batch_start_idx + refinement_batch_size, n_samples)
                actual_refinement_size = batch_end_idx - batch_start_idx
                if actual_refinement_size == 0:
                    continue

                # Get the indices to replace
                indices_to_replace = np.arange(batch_start_idx, batch_end_idx)

                # Get class labels of current batch for replacement generation
                batch_labels = initial_pool_labels[indices_to_replace].cpu()

                # --- Generate Candidate Replacements ---
                num_candidates = actual_refinement_size * num_branches
                candidate_samples = torch.zeros(
                    (
                        num_candidates,
                        self.channels,
                        self.image_size,
                        self.image_size,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )

                # Generate random class labels for candidates
                candidate_label = torch.randint(
                    0, self.num_classes, (num_candidates,), device=self.device
                )

                generated_count = 0
                # Generate candidates in chunks
                while generated_count < num_candidates:
                    chunk_size = min(
                        generation_chunk_size, num_candidates - generated_count
                    )
                    if chunk_size <= 0:
                        break

                    chunk_cand_samples = torch.randn(
                        chunk_size,
                        self.channels,
                        self.image_size,
                        self.image_size,
                        device=self.device,
                    )

                    chunk_cand_label = candidate_label[
                        generated_count : generated_count + chunk_size
                    ]

                    # Standard Euler integration
                    for step, t in enumerate(self.timesteps[:-1]):
                        dt = self.timesteps[step + 1] - t
                        t_batch = torch.full(
                            (chunk_size,), t.item(), device=self.device
                        )
                        with torch.no_grad():
                            velocity = self.flow_model(
                                t_batch, chunk_cand_samples, chunk_cand_label
                            )
                        chunk_cand_samples = chunk_cand_samples + velocity * dt

                    candidate_samples[
                        generated_count : generated_count + chunk_size
                    ] = chunk_cand_samples
                    generated_count += chunk_size

                # --- Evaluate Potential Swaps ---
                best_hypothetical_fid = (
                    current_global_fid  # Start assuming no improvement
                )
                best_candidate_batch_indices = None

                # Extract features for all candidates
                with torch.no_grad():
                    cand_features_list_temp = []
                    for i in range(0, num_candidates, feature_batch_size):
                        batch = candidate_samples[i : i + feature_batch_size]
                        features = self.extract_inception_features(batch)
                        cand_features_list_temp.append(features)

                    all_candidate_features = (
                        np.concatenate(cand_features_list_temp, axis=0)
                        if cand_features_list_temp
                        else np.empty(
                            (0, current_pool_features.shape[1]),
                            dtype=current_pool_features.dtype,
                        )
                    )

                # Prepare the feature pool *without* the samples being replaced
                pool_indices_mask = np.ones(n_samples, dtype=bool)
                pool_indices_mask[indices_to_replace] = False
                features_pool_without_replaced = current_pool_features[
                    pool_indices_mask
                ]

                # Evaluate num_batches random candidate batches
                for i in range(num_batches):
                    # Randomly select a batch of samples from the candidates
                    if num_candidates <= actual_refinement_size:
                        # If we have fewer candidates than batch size, use all of them
                        batch_indices = np.arange(num_candidates)
                    else:
                        # Randomly select samples
                        batch_indices = np.random.choice(
                            num_candidates,
                            size=actual_refinement_size,
                            replace=False,
                        )

                    candidate_batch_features = all_candidate_features[batch_indices]

                    # Combine features for hypothetical pool
                    hypothetical_features = np.concatenate(
                        (features_pool_without_replaced, candidate_batch_features),
                        axis=0,
                    )

                    # Calculate hypothetical FID
                    hypothetical_fid = self.compute_fid_from_features(
                        hypothetical_features, target_mu, target_sigma
                    )

                    if hypothetical_fid < best_hypothetical_fid:
                        best_hypothetical_fid = hypothetical_fid
                        best_candidate_batch_indices = batch_indices

                # --- Perform Swap if Improvement Found ---
                if (
                    best_candidate_batch_indices is not None
                    and best_hypothetical_fid < current_global_fid
                ):
                    print(
                        f"      Swapping batch {batch_start_idx//refinement_batch_size + 1}. New best FID: {best_hypothetical_fid:.4f}"
                    )
                    num_swaps_this_pass += 1
                    # Get the winning samples and features
                    winning_candidate_samples = candidate_samples[
                        best_candidate_batch_indices
                    ]
                    winning_candidate_features = all_candidate_features[
                        best_candidate_batch_indices
                    ]
                    winning_candidate_labels = candidate_label[
                        best_candidate_batch_indices
                    ]

                    # Update the main pool samples and labels
                    initial_pool_samples[indices_to_replace] = winning_candidate_samples
                    initial_pool_labels[indices_to_replace] = winning_candidate_labels
                    # Update the main pool features efficiently
                    current_pool_features[indices_to_replace] = (
                        winning_candidate_features
                    )
                    # Update the official current FID
                    current_global_fid = best_hypothetical_fid

                # Clean up memory for this batch attempt
                del candidate_samples, all_candidate_features
                if "hypothetical_features" in locals():
                    del hypothetical_features
                torch.cuda.empty_cache()

            print(
                f"--- End Pass {pass_num + 1}: Made {num_swaps_this_pass} swaps. Current FID: {current_global_fid:.4f} ---"
            )

        return self.unnormalize_images(initial_pool_samples)

    def batch_sample_refine_global_fid_path_explore(
        self,
        n_samples: int,  # Total samples in the final dataset
        refinement_batch_size: int,  # Size of batches to swap
        num_branches: int,  # Branches per sample at each step
        num_batches: int = 10,  # Number of random batches to evaluate
        dt_std: float = 0.1,  # Standard deviation for dt sampling
        num_iterations: int = 1,  # Number of FULL refinement passes over the data
        use_global: bool = True,  # Use global or class-specific target FID stats
        branch_start_time: float = 0.0,  # When to start branching
        branch_dt: float = None,  # Step size during branching phase
    ):
        """
        Generates a full dataset and iteratively refines it to minimize global FID
        using path exploration with dt sampling. Systematically attempts to replace
        each batch num_iterations times, starting from initial noise states at
        branch_start_time and exploring different paths from there.

        Args:
            n_samples: Total number of samples in the final dataset.
            refinement_batch_size: Size of the batches to consider swapping out.
            num_branches: Branches per sample at each step.
            num_batches: Number of random batches to evaluate for improvement.
            dt_std: Standard deviation for sampling different dt values.
            num_iterations: Number of full passes over the dataset for refinement.
            use_global: Whether to use global target stats for FID calculation.
            branch_start_time: Time point at which to start branching (0.0 to 1.0).
            branch_dt: Step size to use after branching begins (if None, uses base_dt).

        Returns:
            Tensor of [n_samples, C, H, W] representing the final refined dataset.
        """

        self.flow_model.eval()
        samples_per_class = n_samples // self.num_classes
        base_dt = 1 / self.num_timesteps
        branch_dt = branch_dt if branch_dt is not None else base_dt

        # --- 1. Initialization: Generate Initial Noise and Partial Integration ---
        # We store both initial noise and integrated samples at branch_start_time
        initial_pool_noise = torch.randn(
            (n_samples, self.channels, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.device,
        )
        initial_pool_samples = initial_pool_noise.clone()
        initial_pool_labels = torch.zeros(
            n_samples, dtype=torch.long, device=self.device
        )

        # Store samples at branch_start_time for later refinement
        branched_pool_samples = initial_pool_noise.clone()

        current_idx = 0
        feature_batch_size = 128

        # Generate initial samples by class and integrate to branch_start_time
        for class_label in range(self.num_classes):
            generated_for_class = 0
            # Calculate target number ensuring the last class gets the remainder
            target_for_class = (
                samples_per_class
                if class_label < self.num_classes - 1
                else n_samples - current_idx
            )

            while generated_for_class < target_for_class:
                chunk_size = min(
                    refinement_batch_size, target_for_class - generated_for_class
                )
                if chunk_size <= 0:
                    break

                # Get the slice of initial noise
                chunk_start = current_idx
                chunk_end = chunk_start + chunk_size
                chunk_samples = initial_pool_noise[chunk_start:chunk_end].clone()
                chunk_label = torch.full((chunk_size,), class_label, device=self.device)

                # Set labels in the main pool
                initial_pool_labels[chunk_start:chunk_end] = chunk_label

                # Integrate to branch_start_time
                if branch_start_time > 0:
                    current_t = 0.0
                    while current_t < branch_start_time:
                        dt = min(base_dt, branch_start_time - current_t)
                        t_batch = torch.full(
                            (chunk_size,), current_t, device=self.device
                        )
                        with torch.no_grad():
                            velocity = self.flow_model(
                                t_batch, chunk_samples, chunk_label
                            )
                        chunk_samples = chunk_samples + velocity * dt
                        current_t += dt

                # Store partially integrated samples at branch_start_time for refinement later
                branched_pool_samples[chunk_start:chunk_end] = chunk_samples

                # Continue integration to t=1.0 for initial pool
                current_t = branch_start_time
                while current_t < 1.0:
                    dt = min(base_dt, 1.0 - current_t)
                    t_batch = torch.full((chunk_size,), current_t, device=self.device)
                    with torch.no_grad():
                        velocity = self.flow_model(t_batch, chunk_samples, chunk_label)
                    chunk_samples = chunk_samples + velocity * dt
                    current_t += dt

                # Store final integrated samples
                initial_pool_samples[chunk_start:chunk_end] = chunk_samples

                # Update counters
                current_idx += chunk_size
                generated_for_class += chunk_size

        # --- Handle num_branches == 1 Case ---
        if num_branches == 1:
            return initial_pool_samples

        # --- Pre-compute Initial Features and FID ---
        with torch.no_grad():
            all_features_list = []
            for i in range(0, n_samples, feature_batch_size):
                batch = initial_pool_samples[i : i + feature_batch_size]
                features = self.extract_inception_features(batch)
                all_features_list.append(features)
            current_pool_features = np.concatenate(all_features_list, axis=0)

        # Get target statistics
        if use_global:
            target_mu = self.global_fid["mu"]
            target_sigma = self.global_fid["sigma"]
        else:
            raise NotImplementedError(
                "Class-specific FID refinement target not implemented yet."
            )

        current_global_fid = self.compute_fid_from_features(
            current_pool_features, target_mu, target_sigma
        )
        print(f"Initial Global FID: {current_global_fid:.4f}")

        # --- 2. Refinement Loop ---
        for pass_num in range(num_iterations):
            print(f"\n--- Refinement Pass {pass_num + 1}/{num_iterations} ---")
            num_swaps_this_pass = 0

            # Iterate systematically through classes and batches within classes
            for target_class in range(self.num_classes):
                class_indices = torch.where(initial_pool_labels == target_class)[0]
                num_samples_in_class = len(class_indices)
                if num_samples_in_class == 0:
                    continue

                # Iterate through batches within the class
                for batch_start_idx in range(
                    0, num_samples_in_class, refinement_batch_size
                ):
                    batch_end_idx = min(
                        batch_start_idx + refinement_batch_size, num_samples_in_class
                    )
                    actual_refinement_size = batch_end_idx - batch_start_idx
                    if actual_refinement_size == 0:
                        continue

                    # Get the indices within the GLOBAL pool for this batch
                    indices_to_replace = class_indices[
                        batch_start_idx:batch_end_idx
                    ].cpu()

                    # Start with the partially integrated samples at branch_start_time
                    current_batch_samples = branched_pool_samples[
                        indices_to_replace
                    ].clone()
                    current_batch_times = torch.full(
                        (actual_refinement_size,), branch_start_time, device=self.device
                    )
                    current_batch_label = torch.full(
                        (actual_refinement_size,), target_class, device=self.device
                    )

                    # Prepare feature pool without the batch we're replacing
                    pool_indices_mask = np.ones(n_samples, dtype=bool)
                    pool_indices_mask[indices_to_replace] = False
                    features_pool_without_replaced = current_pool_features[
                        pool_indices_mask
                    ]

                    # Path exploration with continuous checking
                    with torch.no_grad():
                        current_time = branch_start_time

                        while current_time < 1.0:
                            # Create branches for the current batch
                            branched_samples = current_batch_samples.repeat_interleave(
                                num_branches, dim=0
                            )
                            branched_times = current_batch_times.repeat_interleave(
                                num_branches
                            )
                            branched_label = current_batch_label.repeat_interleave(
                                num_branches
                            )

                            # Track which original sample each branch belongs to
                            branch_indices = torch.arange(
                                len(current_batch_samples), device=self.device
                            ).repeat_interleave(num_branches)

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

                            # Take the branching step
                            velocity = self.flow_model(
                                branched_times, branched_samples, branched_label
                            )
                            branched_samples = branched_samples + velocity * dts.view(
                                -1, 1, 1, 1
                            )
                            branched_times = branched_times + dts

                            # Simulate forward to completion (t=1.0)
                            simulated_samples = branched_samples.clone()
                            simulated_times = branched_times.clone()

                            while torch.any(simulated_times < 1.0):
                                active_mask = simulated_times < 1.0
                                if not torch.any(active_mask):
                                    break

                                sim_velocity = self.flow_model(
                                    simulated_times[active_mask],
                                    simulated_samples[active_mask],
                                    branched_label[active_mask],
                                )

                                sim_dt = torch.min(
                                    base_dt
                                    * torch.ones_like(simulated_times[active_mask]),
                                    1.0 - simulated_times[active_mask],
                                )
                                simulated_samples[active_mask] = simulated_samples[
                                    active_mask
                                ] + sim_velocity * sim_dt.view(-1, 1, 1, 1)
                                simulated_times[active_mask] = (
                                    simulated_times[active_mask] + sim_dt
                                )

                            # Extract features for all completed samples
                            all_simulated_features = []
                            for i in range(
                                0, len(simulated_samples), feature_batch_size
                            ):
                                batch = simulated_samples[i : i + feature_batch_size]
                                features = self.extract_inception_features(batch)
                                all_simulated_features.append(features)
                            all_simulated_features = np.concatenate(
                                all_simulated_features, axis=0
                            )

                            # Evaluate potential batches
                            best_hypothetical_fid = float("inf")
                            best_batch_indices = None
                            best_original_indices = None

                            # Create and evaluate num_batches random batches, ensuring diversity
                            for batch_idx in range(num_batches):
                                # Select exactly one branch from each original sample
                                batch_indices = []

                                # For each original sample index
                                for orig_idx in range(actual_refinement_size):
                                    # Find all branches from this original sample
                                    branch_mask = branch_indices == orig_idx
                                    candidate_indices = torch.where(branch_mask)[0]

                                    # Randomly select one branch
                                    if len(candidate_indices) > 0:
                                        selected_idx = candidate_indices[
                                            torch.randint(
                                                len(candidate_indices),
                                                (1,),
                                                device=self.device,
                                            )
                                        ].item()
                                        batch_indices.append(selected_idx)

                                # Skip if we couldn't form a complete batch
                                if len(batch_indices) != actual_refinement_size:
                                    continue

                                # Get features for this batch
                                batch_features = all_simulated_features[batch_indices]

                                # Calculate hypothetical FID
                                hypothetical_features = np.concatenate(
                                    (features_pool_without_replaced, batch_features),
                                    axis=0,
                                )
                                hypothetical_fid = self.compute_fid_from_features(
                                    hypothetical_features, target_mu, target_sigma
                                )

                                # Update best if improved
                                if hypothetical_fid < best_hypothetical_fid:
                                    best_hypothetical_fid = hypothetical_fid
                                    best_batch_indices = batch_indices
                                    best_original_indices = np.array(
                                        [
                                            branch_indices[idx].item()
                                            for idx in batch_indices
                                        ]
                                    )

                            # Update the pool samples ONLY if the best batch improves global FID
                            if best_hypothetical_fid < current_global_fid:
                                print(
                                    f"      Found better batch at t={current_time:.4f}. New FID: {best_hypothetical_fid:.4f}"
                                )
                                num_swaps_this_pass += 1

                                # Update the final samples in the main pool
                                initial_pool_samples[indices_to_replace] = (
                                    simulated_samples[best_batch_indices]
                                )

                                # Update features
                                current_pool_features[indices_to_replace] = (
                                    all_simulated_features[best_batch_indices]
                                )

                                # Update the current FID
                                current_global_fid = best_hypothetical_fid

                            # For path exploration, select the best branches
                            # We need exactly one branch per original sample
                            next_batch_samples = []
                            next_batch_times = []

                            # Use the original indices from the best batch to select which samples to continue with
                            for orig_idx in range(actual_refinement_size):
                                # Find position in the best batch where this original index appears
                                positions = np.where(best_original_indices == orig_idx)[
                                    0
                                ]

                                if len(positions) > 0:
                                    # If present in best batch, use the corresponding branch
                                    branch_idx = best_batch_indices[positions[0]]
                                    next_batch_samples.append(
                                        branched_samples[branch_idx]
                                    )
                                    next_batch_times.append(branched_times[branch_idx])
                                else:
                                    # If not in best batch, use a random branch for this original
                                    branch_mask = branch_indices == orig_idx
                                    candidate_indices = torch.where(branch_mask)[0]
                                    if len(candidate_indices) > 0:
                                        random_idx = candidate_indices[
                                            torch.randint(
                                                len(candidate_indices),
                                                (1,),
                                                device=self.device,
                                            )
                                        ].item()
                                        next_batch_samples.append(
                                            branched_samples[random_idx]
                                        )
                                        next_batch_times.append(
                                            branched_times[random_idx]
                                        )
                                    else:
                                        # This case shouldn't happen with proper branching
                                        pass

                            # Update current batch for next iteration
                            if len(next_batch_samples) == actual_refinement_size:
                                current_batch_samples = torch.stack(next_batch_samples)
                                current_batch_times = torch.stack(next_batch_times)

                                # Update current time to minimum of current batch times
                                current_time = current_batch_times.min().item()
                            else:
                                # If we couldn't form a complete batch, end path exploration
                                break

                            # Clean up to save memory
                            del (
                                branched_samples,
                                branched_times,
                                simulated_samples,
                                simulated_times,
                            )
                            del all_simulated_features
                            if "hypothetical_features" in locals():
                                del hypothetical_features
                            torch.cuda.empty_cache()

            print(
                f"--- End Pass {pass_num + 1}: Made {num_swaps_this_pass} swaps. Current FID: {current_global_fid:.4f} ---"
            )

        return self.unnormalize_images(initial_pool_samples)

    def batch_sample_refine_global_fid_timewarp(
        self,
        n_samples: int,  # Total samples in the final dataset
        refinement_batch_size: int,  # Size of batches to swap
        num_branches: int,  # Branches per sample at each step
        num_batches: int = 10,  # Number of random batches to evaluate
        warp_scale: float = 0.5,  # Scale of time warping (0-1)
        num_iterations: int = 1,  # Number of FULL refinement passes over the data
        use_global: bool = True,  # Use global or class-specific target FID stats
        branch_start_time: float = 0.0,  # When to start branching
        branch_dt: float = None,  # Step size during branching phase
        sqrt_epsilon: float = 1e-8,  # Small epsilon for safe sqrt operations
    ):
        """
        Generates a full dataset and iteratively refines it to minimize global FID
        using time warping path exploration. Systematically attempts to replace
        each batch num_iterations times, starting from initial noise states at
        branch_start_time and exploring different paths from there using time warping.

        Args:
            n_samples: Total number of samples in the final dataset.
            refinement_batch_size: Size of the batches to consider swapping out.
            num_branches: Branches per sample at each step.
            num_batches: Number of random batches to evaluate for improvement.
            warp_scale: Scale of time warping effect (0-1).
            num_iterations: Number of full passes over the dataset for refinement.
            use_global: Whether to use global target stats for FID calculation.
            branch_start_time: Time point at which to start branching (0.0 to 1.0).
            branch_dt: Step size to use after branching begins (if None, uses base_dt).
            sqrt_epsilon: Small constant to ensure numerical stability in warping functions.

        Returns:
            Tensor of [n_samples, C, H, W] representing the final refined dataset.
        """

        self.flow_model.eval()
        samples_per_class = n_samples // self.num_classes
        base_dt = 1 / self.num_timesteps
        branch_dt = branch_dt if branch_dt is not None else base_dt

        # --- Get time warping functions and derivatives ---
        warp_fns, warp_deriv_fns = self._get_warp_functions(
            num_branches, self.device, sqrt_epsilon
        )

        # --- 1. Initialization: Generate Initial Noise and Partial Integration ---
        # We store both initial noise and integrated samples at branch_start_time
        initial_pool_noise = torch.randn(
            (n_samples, self.channels, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.device,
        )
        initial_pool_samples = initial_pool_noise.clone()
        initial_pool_labels = torch.zeros(
            n_samples, dtype=torch.long, device=self.device
        )

        # Store samples at branch_start_time for later refinement
        branched_pool_samples = initial_pool_noise.clone()

        current_idx = 0
        feature_batch_size = 128

        # Generate initial samples by class and integrate to branch_start_time
        for class_label in range(self.num_classes):
            generated_for_class = 0
            # Calculate target number ensuring the last class gets the remainder
            target_for_class = (
                samples_per_class
                if class_label < self.num_classes - 1
                else n_samples - current_idx
            )

            while generated_for_class < target_for_class:
                chunk_size = min(
                    refinement_batch_size, target_for_class - generated_for_class
                )
                if chunk_size <= 0:
                    break

                # Get the slice of initial noise
                chunk_start = current_idx
                chunk_end = chunk_start + chunk_size
                chunk_samples = initial_pool_noise[chunk_start:chunk_end].clone()
                chunk_label = torch.full((chunk_size,), class_label, device=self.device)

                # Set labels in the main pool
                initial_pool_labels[chunk_start:chunk_end] = chunk_label

                # Integrate to branch_start_time
                if branch_start_time > 0:
                    current_t = 0.0
                    while current_t < branch_start_time:
                        dt = min(base_dt, branch_start_time - current_t)
                        t_batch = torch.full(
                            (chunk_size,), current_t, device=self.device
                        )
                        with torch.no_grad():
                            velocity = self.flow_model(
                                t_batch, chunk_samples, chunk_label
                            )
                        chunk_samples = chunk_samples + velocity * dt
                        current_t += dt

                # Store partially integrated samples at branch_start_time for refinement later
                branched_pool_samples[chunk_start:chunk_end] = chunk_samples

                # Continue integration to t=1.0 for initial pool
                current_t = branch_start_time
                while current_t < 1.0:
                    dt = min(base_dt, 1.0 - current_t)
                    t_batch = torch.full((chunk_size,), current_t, device=self.device)
                    with torch.no_grad():
                        velocity = self.flow_model(t_batch, chunk_samples, chunk_label)
                    chunk_samples = chunk_samples + velocity * dt
                    current_t += dt

                # Store final integrated samples
                initial_pool_samples[chunk_start:chunk_end] = chunk_samples

                # Update counters
                current_idx += chunk_size
                generated_for_class += chunk_size

        # --- Handle num_branches == 1 Case ---
        if num_branches == 1:
            return initial_pool_samples

        # --- Pre-compute Initial Features and FID ---
        with torch.no_grad():
            all_features_list = []
            for i in range(0, n_samples, feature_batch_size):
                batch = initial_pool_samples[i : i + feature_batch_size]
                features = self.extract_inception_features(batch)
                all_features_list.append(features)
            current_pool_features = np.concatenate(all_features_list, axis=0)

        # Get target statistics
        if use_global:
            target_mu = self.global_fid["mu"]
            target_sigma = self.global_fid["sigma"]
        else:
            raise NotImplementedError(
                "Class-specific FID refinement target not implemented yet."
            )

        current_global_fid = self.compute_fid_from_features(
            current_pool_features, target_mu, target_sigma
        )
        print(f"Initial Global FID: {current_global_fid:.4f}")

        # --- 2. Refinement Loop ---
        for pass_num in range(num_iterations):
            print(f"\n--- Refinement Pass {pass_num + 1}/{num_iterations} ---")
            num_swaps_this_pass = 0

            # Iterate systematically through classes and batches within classes
            for target_class in range(self.num_classes):
                class_indices = torch.where(initial_pool_labels == target_class)[0]
                num_samples_in_class = len(class_indices)
                if num_samples_in_class == 0:
                    continue

                # Iterate through batches within the class
                for batch_start_idx in range(
                    0, num_samples_in_class, refinement_batch_size
                ):
                    batch_end_idx = min(
                        batch_start_idx + refinement_batch_size, num_samples_in_class
                    )
                    actual_refinement_size = batch_end_idx - batch_start_idx
                    if actual_refinement_size == 0:
                        continue

                    # Get the indices within the GLOBAL pool for this batch
                    indices_to_replace = class_indices[
                        batch_start_idx:batch_end_idx
                    ].cpu()

                    # Start with the partially integrated samples at branch_start_time
                    current_batch_samples = branched_pool_samples[
                        indices_to_replace
                    ].clone()
                    current_batch_times = torch.full(
                        (actual_refinement_size,), branch_start_time, device=self.device
                    )
                    current_batch_label = torch.full(
                        (actual_refinement_size,), target_class, device=self.device
                    )

                    # Prepare feature pool without the batch we're replacing
                    pool_indices_mask = np.ones(n_samples, dtype=bool)
                    pool_indices_mask[indices_to_replace] = False
                    features_pool_without_replaced = current_pool_features[
                        pool_indices_mask
                    ]

                    # Path exploration with continuous checking
                    with torch.no_grad():
                        current_time = branch_start_time

                        while current_time < 1.0:
                            # Create branches for the current batch
                            branched_samples = current_batch_samples.repeat_interleave(
                                num_branches, dim=0
                            )
                            branched_times = current_batch_times.repeat_interleave(
                                num_branches
                            )
                            branched_label = current_batch_label.repeat_interleave(
                                num_branches
                            )

                            # Track which original sample each branch belongs to
                            branch_indices = torch.arange(
                                len(current_batch_samples), device=self.device
                            ).repeat_interleave(num_branches)

                            # Track which warping function to use for each sample
                            warp_indices = (
                                torch.arange(len(branched_samples), device=self.device)
                                % num_branches
                            )

                            # Take ONE branching step with time warping
                            dt_step = torch.minimum(
                                branch_dt * torch.ones_like(branched_times),
                                1.0 - branched_times,
                            )

                            # Apply time warping
                            warped_times_step = torch.zeros_like(branched_times)
                            warp_derivs_step = torch.zeros_like(branched_times)

                            # Use the fetched warping functions
                            for i in range(num_branches):
                                mask = warp_indices == i
                                if torch.any(mask):
                                    orig_t = branched_times[mask]
                                    warped_t = warp_fns[i](orig_t)
                                    final_warped_t = (
                                        1 - warp_scale
                                    ) * orig_t + warp_scale * warped_t
                                    warped_times_step[mask] = final_warped_t

                                    orig_deriv = warp_deriv_fns[i](orig_t)
                                    final_warp_deriv = (
                                        1 - warp_scale
                                    ) * torch.ones_like(
                                        orig_deriv
                                    ) + warp_scale * orig_deriv
                                    warp_derivs_step[mask] = final_warp_deriv

                            # Calculate velocity using warped times
                            velocity_step = self.flow_model(
                                warped_times_step, branched_samples, branched_label
                            )

                            # Apply the step with chain rule: dx/dt = v(x, f(t)) * f'(t)
                            branched_samples = (
                                branched_samples
                                + velocity_step
                                * warp_derivs_step.view(-1, 1, 1, 1)
                                * dt_step.view(-1, 1, 1, 1)
                            )
                            branched_times = branched_times + dt_step

                            # Simulate forward to completion (t=1.0)
                            simulated_samples = branched_samples.clone()
                            simulated_times = branched_times.clone()
                            sim_warp_indices = warp_indices.clone()

                            while torch.any(simulated_times < 1.0):
                                active_mask = simulated_times < 1.0
                                if not torch.any(active_mask):
                                    break

                                active_sim_samples = simulated_samples[active_mask]
                                active_sim_times = simulated_times[active_mask]
                                active_sim_label = branched_label[active_mask]
                                active_sim_warp_indices = sim_warp_indices[active_mask]

                                dt_sim = torch.minimum(
                                    branch_dt * torch.ones_like(active_sim_times),
                                    1.0 - active_sim_times,
                                )

                                # Apply time warping for simulation
                                warped_times_sim = torch.zeros_like(active_sim_times)
                                warp_derivs_sim = torch.zeros_like(active_sim_times)

                                for i in range(num_branches):
                                    mask = active_sim_warp_indices == i
                                    if torch.any(mask):
                                        orig_t = active_sim_times[mask]
                                        warped_t = warp_fns[i](orig_t)
                                        final_warped_t = (
                                            1 - warp_scale
                                        ) * orig_t + warp_scale * warped_t
                                        warped_times_sim[mask] = final_warped_t

                                        orig_deriv = warp_deriv_fns[i](orig_t)
                                        final_warp_deriv = (
                                            1 - warp_scale
                                        ) * torch.ones_like(
                                            orig_deriv
                                        ) + warp_scale * orig_deriv
                                        warp_derivs_sim[mask] = final_warp_deriv

                                # Calculate velocity using warped times
                                velocity_sim = self.flow_model(
                                    warped_times_sim,
                                    active_sim_samples,
                                    active_sim_label,
                                )

                                # Apply the step with chain rule
                                update = (
                                    velocity_sim
                                    * warp_derivs_sim.view(-1, 1, 1, 1)
                                    * dt_sim.view(-1, 1, 1, 1)
                                )
                                simulated_samples[active_mask] = (
                                    active_sim_samples + update
                                )
                                simulated_times[active_mask] = active_sim_times + dt_sim

                            # Extract features for all completed samples
                            all_simulated_features = []
                            for i in range(
                                0, len(simulated_samples), feature_batch_size
                            ):
                                batch = simulated_samples[i : i + feature_batch_size]
                                features = self.extract_inception_features(batch)
                                all_simulated_features.append(features)
                            all_simulated_features = np.concatenate(
                                all_simulated_features, axis=0
                            )

                            # Evaluate potential batches
                            best_hypothetical_fid = float("inf")
                            best_batch_indices = None
                            best_original_indices = None

                            # Create and evaluate num_batches random batches, ensuring diversity
                            for batch_idx in range(num_batches):
                                # Select exactly one branch from each original sample
                                batch_indices = []

                                # For each original sample index
                                for orig_idx in range(actual_refinement_size):
                                    # Find all branches from this original sample
                                    branch_mask = branch_indices == orig_idx
                                    candidate_indices = torch.where(branch_mask)[0]

                                    # Randomly select one branch
                                    if len(candidate_indices) > 0:
                                        selected_idx = candidate_indices[
                                            torch.randint(
                                                len(candidate_indices),
                                                (1,),
                                                device=self.device,
                                            )
                                        ].item()
                                        batch_indices.append(selected_idx)

                                # Skip if we couldn't form a complete batch
                                if len(batch_indices) != actual_refinement_size:
                                    continue

                                # Get features for this batch
                                batch_features = all_simulated_features[batch_indices]

                                # Calculate hypothetical FID
                                hypothetical_features = np.concatenate(
                                    (features_pool_without_replaced, batch_features),
                                    axis=0,
                                )
                                hypothetical_fid = self.compute_fid_from_features(
                                    hypothetical_features, target_mu, target_sigma
                                )

                                # Update best if improved
                                if hypothetical_fid < best_hypothetical_fid:
                                    best_hypothetical_fid = hypothetical_fid
                                    best_batch_indices = batch_indices
                                    best_original_indices = np.array(
                                        [
                                            branch_indices[idx].item()
                                            for idx in batch_indices
                                        ]
                                    )

                            # Update the pool samples ONLY if the best batch improves global FID
                            if best_hypothetical_fid < current_global_fid:
                                print(
                                    f"      Found better batch at t={current_time:.4f}. New FID: {best_hypothetical_fid:.4f}"
                                )
                                num_swaps_this_pass += 1

                                # Update the final samples in the main pool
                                initial_pool_samples[indices_to_replace] = (
                                    simulated_samples[best_batch_indices]
                                )

                                # Update features
                                current_pool_features[indices_to_replace] = (
                                    all_simulated_features[best_batch_indices]
                                )

                                # Update the current FID
                                current_global_fid = best_hypothetical_fid

                            # For path exploration, select the best branches
                            # We need exactly one branch per original sample
                            next_batch_samples = []
                            next_batch_times = []
                            next_batch_warp_indices = []

                            # Use the original indices from the best batch to select which samples to continue with
                            for orig_idx in range(actual_refinement_size):
                                # Find position in the best batch where this original index appears
                                positions = np.where(best_original_indices == orig_idx)[
                                    0
                                ]

                                if len(positions) > 0:
                                    # If present in best batch, use the corresponding branch
                                    branch_idx = best_batch_indices[positions[0]]
                                    next_batch_samples.append(
                                        branched_samples[branch_idx]
                                    )
                                    next_batch_times.append(branched_times[branch_idx])
                                    next_batch_warp_indices.append(
                                        warp_indices[branch_idx].item()
                                    )
                                else:
                                    # If not in best batch, use a random branch for this original
                                    branch_mask = branch_indices == orig_idx
                                    candidate_indices = torch.where(branch_mask)[0]
                                    if len(candidate_indices) > 0:
                                        random_idx = candidate_indices[
                                            torch.randint(
                                                len(candidate_indices),
                                                (1,),
                                                device=self.device,
                                            )
                                        ].item()
                                        next_batch_samples.append(
                                            branched_samples[random_idx]
                                        )
                                        next_batch_times.append(
                                            branched_times[random_idx]
                                        )
                                        next_batch_warp_indices.append(
                                            warp_indices[random_idx].item()
                                        )

                            # Update current batch for next iteration
                            if len(next_batch_samples) == actual_refinement_size:
                                current_batch_samples = torch.stack(next_batch_samples)
                                current_batch_times = torch.stack(next_batch_times)

                                # Update current time to minimum of current batch times
                                current_time = current_batch_times.min().item()
                            else:
                                # If we couldn't form a complete batch, end path exploration
                                break

                            # Clean up to save memory
                            del (
                                branched_samples,
                                branched_times,
                                simulated_samples,
                                simulated_times,
                            )
                            del all_simulated_features
                            if "hypothetical_features" in locals():
                                del hypothetical_features
                            torch.cuda.empty_cache()

            print(
                f"--- End Pass {pass_num + 1}: Made {num_swaps_this_pass} swaps. Current FID: {current_global_fid:.4f} ---"
            )

        return self.unnormalize_images(initial_pool_samples)

    def compute_fid_from_features(
        self, features_gen, target_mu, target_sigma, eps=1e-6
    ):
        """Helper to compute FID directly from generated features and target stats."""
        if features_gen.shape[0] < 2:
            # Cannot compute covariance
            return float("inf")

        mu_gen = np.mean(features_gen, axis=0)
        sigma_gen = (
            np.cov(features_gen, rowvar=False) + np.eye(features_gen.shape[1]) * eps
        )

        # Ensure target sigma is non-singular
        target_sigma = target_sigma + np.eye(target_sigma.shape[0]) * eps

        fid_value = self.calculate_frechet_distance(
            mu_gen, sigma_gen, target_mu, target_sigma
        )
        return max(0, fid_value)  # Clamp negative FID

    def unnormalize_images(self, images):
        """
        Unnormalize images from ImageNet stats back to [0,1] range.

        Args:
            images: Tensor of shape [batch_size, C, H, W] normalized with ImageNet stats

        Returns:
            Unnormalized images in [0,1] range

        """

        if self.dataset == "imagenet256" and images.shape[1] == 3:
            return images
        elif (
            self.dataset != "imagenet256"
            and images.max() <= 1.0
            and images.min() >= 0.0
        ):
            return images

        if self.dataset == "imagenet256":
            return self.decode_latents(images)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        unnormalized = images * std + mean
        return torch.clamp(unnormalized, 0.0, 1.0)

    def decode_latents(self, latents):
        """
        Decode latents to images using VAE.
        """
        # Convert latents to half precision to match VAE parameters
        latents_half = latents.half()
        images = self.vae.decode(latents_half / 0.18215).sample
        return torch.clamp((images + 1) / 2, 0.0, 1.0).float()

    def regular_batch_sample(self, class_label, batch_size=16):
        """
        Regular flow matching sampling without branching.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
        """
        # Check if class_label is a tensor or a single integer
        is_tensor = torch.is_tensor(class_label)

        self.flow_model.eval()

        with torch.no_grad():
            current_samples = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )

            # Handle both tensor and single class label cases
            if is_tensor:
                current_label = class_label
            else:
                current_label = torch.full(
                    (batch_size,), class_label, device=self.device
                )

            # Generate samples using timesteps
            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full((batch_size,), t.item(), device=self.device)

                # Flow step
                velocity = self.flow_model(t_batch, current_samples, current_label)
                current_samples = current_samples + velocity * dt

            return self.unnormalize_images(current_samples)

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

    def load_models(
        self,
        path="saved_models",
        flow_model="single_flow_model.pt",
    ):
        """Load flow and value models if they exist."""
        flow_path = f"{path}/{flow_model}"

        flow_exists = os.path.exists(flow_path)

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

        return flow_exists

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

    def _get_continuous_warp_functions(
        self,
        n,
        device,
        current_time,
        current_warped_time=None,
        sqrt_epsilon=1e-4,
        clamp_time=True,
    ):
        """
        Generates n warp functions and their derivatives that pass through
        the point (current_time, current_warped_time).

        Args:
            n: Number of warping functions to generate
            device: Device to run calculations on
            current_time: The time point at which continuity must be preserved
            current_warped_time: The warped time value at current_time (if None, uses current_time)
            sqrt_epsilon: Small value for numerical stability
            enforce_endpoint: If True, ensures f(1) = 1 for all functions
        """
        warp_fns = []
        warp_deriv_fns = []

        # Generate base warping functions (unadjusted)
        base_warp_fns = []
        base_warp_deriv_fns = []

        # 1. Linear
        def base_linear_warp(t):
            return t

        def base_linear_warp_deriv(t):
            return torch.ones_like(t)

        base_warp_fns.append(base_linear_warp)
        base_warp_deriv_fns.append(base_linear_warp_deriv)

        # 2. Square
        def base_square_warp(t):
            return t**2

        def base_square_warp_deriv(t):
            return 2 * t

        base_warp_fns.append(base_square_warp)
        base_warp_deriv_fns.append(base_square_warp_deriv)

        # 3. Sqrt
        def base_sqrt_warp(t):
            return torch.sqrt(t + sqrt_epsilon)

        def base_sqrt_warp_deriv(t):
            return 0.5 / torch.sqrt(t + sqrt_epsilon)

        base_warp_fns.append(base_sqrt_warp)
        base_warp_deriv_fns.append(base_sqrt_warp_deriv)

        # 4. Sigmoid
        def base_sigmoid_warp_k12(t):
            t_scaled = 12 * t - 6
            return torch.sigmoid(t_scaled)

        def base_sigmoid_warp_deriv_k12(t):
            sig_t = base_sigmoid_warp_k12(t)
            return 12 * sig_t * (1 - sig_t)

        base_warp_fns.append(base_sigmoid_warp_k12)
        base_warp_deriv_fns.append(base_sigmoid_warp_deriv_k12)

        # Add additional base functions if needed
        current_len = 4
        while current_len < n:
            next_fn_idx = current_len % 4
            if next_fn_idx == 0:  # Cubic
                p = 3 + (current_len // 4)

                def base_cubic_warp(t, p=p):
                    return t**p

                def base_cubic_warp_deriv(t, p=p):
                    return p * (t ** (p - 1))

                base_warp_fns.append(base_cubic_warp)
                base_warp_deriv_fns.append(base_cubic_warp_deriv)

            elif next_fn_idx == 1:  # Wider Sigmoid
                k = 6 * (2 + current_len // 4)

                def base_sigmoid_warp_k_wide(t, k=k):
                    t_scaled = k * t - k / 2
                    return torch.sigmoid(t_scaled)

                def base_sigmoid_warp_deriv_k_wide(t, k=k):
                    sig_t = base_sigmoid_warp_k_wide(t, k=k)
                    return k * sig_t * (1 - sig_t)

                base_warp_fns.append(base_sigmoid_warp_k_wide)
                base_warp_deriv_fns.append(base_sigmoid_warp_deriv_k_wide)

            elif next_fn_idx == 2:  # Higher Root
                p_inv = 3 + (current_len // 4)

                def base_root_warp(t, p_inv=p_inv, eps=sqrt_epsilon):
                    return (t + eps) ** (1.0 / p_inv)

                def base_root_warp_deriv(t, p_inv=p_inv, eps=sqrt_epsilon):
                    return (1.0 / p_inv) * (t + eps) ** ((1.0 / p_inv) - 1.0)

                base_warp_fns.append(base_root_warp)
                base_warp_deriv_fns.append(base_root_warp_deriv)

            elif next_fn_idx == 3:  # Narrower Sigmoid
                k = 18 * (1 + current_len // 4)

                def base_sigmoid_warp_k_narrow(t, k=k):
                    t_scaled = k * t - k / 2
                    return torch.sigmoid(t_scaled)

                def base_sigmoid_warp_deriv_k_narrow(t, k=k):
                    sig_t = base_sigmoid_warp_k_narrow(t, k=k)
                    return k * sig_t * (1 - sig_t)

                base_warp_fns.append(base_sigmoid_warp_k_narrow)
                base_warp_deriv_fns.append(base_sigmoid_warp_deriv_k_narrow)

            current_len += 1

        # If current_warped_time is None, use current_time as the reference value
        if current_warped_time is None:
            current_warped_time = current_time

        # Special case: If we're at t=0, just use the original functions
        if current_time == 0.0 and current_warped_time == 0.0:
            return base_warp_fns[:n], base_warp_deriv_fns[:n]

        # Create adjusted warping functions
        for i in range(min(n, len(base_warp_fns))):
            base_fn = base_warp_fns[i]
            base_deriv_fn = base_warp_deriv_fns[i]

            # Get base function value at current_time
            base_value = base_fn(torch.tensor([current_time], device=device))[0].item()

            # The shift ensures f(current_time) = current_warped_time
            shift = current_warped_time - base_value

            # Create adjusted warp function that passes through the desired point
            def make_adjusted_warp(base_fn, shift, clamp=clamp_time):
                def adjusted_warp(t):
                    result = base_fn(t) + shift
                    if clamp:
                        result = torch.clamp(result, 0.0, 0.99)
                    return result

                return adjusted_warp

            # The derivative is unchanged
            def make_adjusted_deriv(base_deriv_fn):
                def adjusted_deriv(t):
                    return base_deriv_fn(t)

                return adjusted_deriv

            warp_fns.append(make_adjusted_warp(base_fn, shift))
            warp_deriv_fns.append(make_adjusted_deriv(base_deriv_fn))

        return warp_fns[:n], warp_deriv_fns[:n]

    # def batch_sample_with_path_exploration_timewarp_batch_fid(
    #     self,
    #     class_label,
    #     batch_size=16,
    #     num_branches=4,
    #     num_scoring_batches=10,  # How many random batches to score
    #     warp_scale=0.5,
    #     use_global=False,
    #     branch_start_time=0.0,
    #     branch_dt=None,
    #     sqrt_epsilon=1e-8,
    # ):
    #     """
    #     Samples using time warping path exploration, selecting based on BATCH FID score.

    #     Args:
    #         class_label: Target class label.
    #         batch_size: Final number of samples and size of batches for FID scoring.
    #         num_branches: Number of branches per sample at each step.
    #         num_scoring_batches: Number of random batches (size batch_size) to evaluate with FID.
    #         warp_scale: Scaling factor for the warp effect (0=no warp, 1=full warp).
    #         use_global: Use global FID statistics instead of class-specific.
    #         branch_start_time: Time to start the branching exploration.
    #         branch_dt: Timestep size during exploration.
    #         sqrt_epsilon: Epsilon for sqrt warp derivative stability.

    #     Returns:
    #         Tensor of [batch_size, C, H, W] generated samples.
    #     """
    #     if num_branches == 1:
    #         return self.regular_batch_sample(class_label, batch_size)

    #     assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"
    #     assert num_scoring_batches > 0, "num_scoring_batches must be positive"

    #     # Get warping functions
    #     warp_fns, warp_deriv_fns = self._get_warp_functions(
    #         num_branches, self.device, sqrt_epsilon
    #     )

    #     self.flow_model.eval()
    #     actual_dt_step = (
    #         branch_dt if branch_dt is not None else (1 / self.num_timesteps)
    #     )
    #     base_dt = 1 / self.num_timesteps  # For initial phase and potentially simulation

    #     with torch.no_grad():
    #         # Initialize with batch_size samples
    #         current_samples = torch.randn(
    #             batch_size,
    #             self.channels,
    #             self.image_size,
    #             self.image_size,
    #             device=self.device,
    #         )
    #         current_times = torch.zeros(batch_size, device=self.device)
    #         current_label = torch.full((batch_size,), class_label, device=self.device)

    #         # --- Regular flow until branch_start_time ---
    #         while torch.any(current_times < branch_start_time):
    #             max_dt = torch.clamp(branch_start_time - current_times, min=0.0)
    #             dt = torch.minimum(base_dt * torch.ones_like(current_times), max_dt)
    #             active_mask = dt > 0
    #             if not torch.any(active_mask):
    #                 break

    #             velocity = self.flow_model(
    #                 current_times[active_mask],
    #                 current_samples[active_mask],
    #                 current_label[active_mask],
    #             )
    #             current_samples[active_mask] = current_samples[
    #                 active_mask
    #             ] + velocity * dt[active_mask].view(-1, 1, 1, 1)
    #             current_times[active_mask] = (
    #                 current_times[active_mask] + dt[active_mask]
    #             )
    #             # Ensure times do not exceed branch_start_time due to float precision
    #             current_times = torch.clamp(current_times, max=branch_start_time)

    #         # --- Main Path Exploration Loop ---
    #         loop_count = 0
    #         max_loops = int(1.0 / actual_dt_step) + 50  # Safety break
    #         while torch.any(current_times < 1.0):
    #             loop_count += 1
    #             if loop_count > max_loops:
    #                 print("Warning: Max loops reached in timewarp_batch_fid. Breaking.")
    #                 break

    #             # If all samples are done, exit
    #             active_batch_mask = current_times < 1.0
    #             if not torch.any(active_batch_mask):
    #                 print("All samples finished.")
    #                 break

    #             # We operate on the full batch_size set of samples
    #             active_samples = current_samples
    #             active_times = current_times
    #             active_label = current_label
    #             num_active = len(active_samples)  # Should remain batch_size

    #             # --- 1. Create Branches ---
    #             # Result size: [batch_size * num_branches, C, H, W]
    #             branched_samples = active_samples.repeat_interleave(num_branches, dim=0)
    #             branched_times = active_times.repeat_interleave(num_branches)
    #             branched_label = active_label.repeat_interleave(num_branches)
    #             # Map back to original sample index: [0,0,0,0, 1,1,1,1, ...] Needed? Not directly for selection.
    #             # Index of the warp function applied: [0,1,2,3, 0,1,2,3, ...]
    #             warp_indices = (
    #                 torch.arange(len(branched_samples), device=self.device)
    #                 % num_branches
    #             )

    #             # --- 2. Take ONE Branching Step with Warping ---
    #             # Calculate dt, ensuring it doesn't step past 1.0
    #             dt_step = torch.minimum(
    #                 actual_dt_step * torch.ones_like(branched_times),
    #                 1.0 - branched_times,
    #             )
    #             # Zero out dt for samples already at t=1
    #             dt_step = torch.where(
    #                 branched_times >= 1.0, torch.zeros_like(dt_step), dt_step
    #             )

    #             post_branch_samples = branched_samples.clone()  # State AFTER the step
    #             post_branch_times = branched_times.clone()  # Time AFTER the step

    #             active_step_mask = (
    #                 dt_step > 0
    #             )  # Only compute velocity for steps that actually move
    #             if torch.any(active_step_mask):
    #                 # Select only the branches that will take a step
    #                 active_branched_samples_step = branched_samples[active_step_mask]
    #                 active_branched_times_step = branched_times[active_step_mask]
    #                 active_branched_label_step = branched_label[active_step_mask]
    #                 active_warp_indices_step = warp_indices[active_step_mask]
    #                 active_dt_step = dt_step[active_step_mask]

    #                 # Calculate warped times and derivatives for the active branches
    #                 active_warped_times_step = torch.zeros_like(
    #                     active_branched_times_step
    #                 )
    #                 active_warp_derivs_step = torch.zeros_like(
    #                     active_branched_times_step
    #                 )
    #                 for i in range(num_branches):
    #                     mask = active_warp_indices_step == i
    #                     if torch.any(mask):
    #                         orig_t = active_branched_times_step[mask]
    #                         # Apply warp function and scale
    #                         warped_t = warp_fns[i](orig_t)
    #                         final_warped_t = (
    #                             1 - warp_scale
    #                         ) * orig_t + warp_scale * warped_t
    #                         active_warped_times_step[mask] = torch.clamp(
    #                             final_warped_t, 0.0, 1.0
    #                         )  # Clamp time

    #                         # Apply derivative function and scale
    #                         orig_deriv = warp_deriv_fns[i](orig_t)
    #                         final_warp_deriv = (1 - warp_scale) * torch.ones_like(
    #                             orig_deriv
    #                         ) + warp_scale * orig_deriv
    #                         active_warp_derivs_step[mask] = final_warp_deriv

    #                 # Get velocity using warped time
    #                 velocity_step = self.flow_model(
    #                     active_warped_times_step,
    #                     active_branched_samples_step,
    #                     active_branched_label_step,
    #                 )

    #                 # Apply update using chain rule: v(f(t)) * f'(t) * dt
    #                 update = (
    #                     velocity_step
    #                     * active_warp_derivs_step.view(-1, 1, 1, 1)
    #                     * active_dt_step.view(-1, 1, 1, 1)
    #                 )
    #                 # Update the samples in post_branch_samples
    #                 post_branch_samples[active_step_mask] = (
    #                     active_branched_samples_step + update
    #                 )

    #             # Update times for all branches (time advances even if dt was 0 and sample didn't change)
    #             post_branch_times = post_branch_times + dt_step
    #             post_branch_times = torch.clamp(
    #                 post_branch_times, max=1.0
    #             )  # Ensure time doesn't exceed 1.0

    #             # --- 3. Simulate Each Branch to Completion (for Scoring) ---
    #             # Start simulation from the state AFTER the branching step
    #             simulated_samples = post_branch_samples.clone()
    #             simulated_times = post_branch_times.clone()
    #             # The simulation path continues using the same warp logic assigned at the branch step
    #             sim_warp_indices = warp_indices.clone()
    #             sim_label = branched_label.clone()  # Labels for simulation

    #             sim_loop_count = 0
    #             max_sim_loops = (
    #                 int(1.0 / actual_dt_step) + 50
    #             )  # Safety break for inner loop
    #             while torch.any(simulated_times < 1.0):
    #                 sim_loop_count += 1
    #                 if sim_loop_count > max_sim_loops:
    #                     print(
    #                         "Warning: Max loops reached in simulation step. Breaking."
    #                     )
    #                     # Force remaining times to 1.0 to exit loop
    #                     simulated_times = torch.clamp(simulated_times, min=1.0)
    #                     break

    #                 sim_active_mask = simulated_times < 1.0
    #                 if not torch.any(sim_active_mask):
    #                     break  # Exit if all finished

    #                 active_sim_samples = simulated_samples[sim_active_mask]
    #                 active_sim_times = simulated_times[sim_active_mask]
    #                 active_sim_label = sim_label[sim_active_mask]
    #                 active_sim_warp_indices = sim_warp_indices[sim_active_mask]

    #                 # Calculate dt for simulation step
    #                 dt_sim = torch.minimum(
    #                     actual_dt_step * torch.ones_like(active_sim_times),
    #                     1.0 - active_sim_times,
    #                 )
    #                 # Ensure dt is non-negative
    #                 dt_sim = torch.clamp(dt_sim, min=0.0)

    #                 # Calculate warped time and derivative for simulation step
    #                 warped_times_sim = torch.zeros_like(active_sim_times)
    #                 warp_derivs_sim = torch.zeros_like(active_sim_times)
    #                 for i in range(
    #                     num_branches
    #                 ):  # Apply the *same* warp func consistently
    #                     mask = active_sim_warp_indices == i
    #                     if torch.any(mask):
    #                         orig_t = active_sim_times[mask]
    #                         warped_t = warp_fns[i](orig_t)
    #                         final_warped_t = (
    #                             1 - warp_scale
    #                         ) * orig_t + warp_scale * warped_t
    #                         warped_times_sim[mask] = torch.clamp(
    #                             final_warped_t, 0.0, 1.0
    #                         )

    #                         orig_deriv = warp_deriv_fns[i](orig_t)
    #                         final_warp_deriv = (1 - warp_scale) * torch.ones_like(
    #                             orig_deriv
    #                         ) + warp_scale * orig_deriv
    #                         warp_derivs_sim[mask] = final_warp_deriv

    #                 # Get velocity and apply update
    #                 velocity_sim = self.flow_model(
    #                     warped_times_sim, active_sim_samples, active_sim_label
    #                 )
    #                 update = (
    #                     velocity_sim
    #                     * warp_derivs_sim.view(-1, 1, 1, 1)
    #                     * dt_sim.view(-1, 1, 1, 1)
    #                 )
    #                 simulated_samples[sim_active_mask] = active_sim_samples + update
    #                 simulated_times[sim_active_mask] = active_sim_times + dt_sim
    #                 simulated_times[sim_active_mask] = torch.clamp(
    #                     simulated_times[sim_active_mask], max=1.0
    #                 )  # Clamp time

    #             # --- 4. Score Simulated Samples using Batch FID ---
    #             total_simulated = len(
    #                 simulated_samples
    #             )  # Should be batch_size * num_branches
    #             batch_scores = []
    #             batch_indices_list = []  # Store the indices used for each random batch

    #             # Check if we have enough samples for at least one FID batch
    #             if total_simulated < batch_size:
    #                 print(
    #                     f"Warning: Not enough simulated samples ({total_simulated}) to form a batch of size {batch_size}. Cannot score with FID."
    #                 )
    #                 # Fallback: Arbitrarily select the first batch_size available, or handle error
    #                 # For now, let's just take the first ones and proceed. This might not be ideal.
    #                 winning_indices = torch.arange(
    #                     min(total_simulated, batch_size), device=self.device
    #                 )
    #                 print(
    #                     f"Falling back to selecting the first {len(winning_indices)} samples."
    #                 )
    #             else:
    #                 for i in range(num_scoring_batches):
    #                     # Randomly sample batch_size indices from the pool of simulated samples
    #                     # Ensure replacement=False if total_simulated is large enough, otherwise allow replacement?
    #                     # randperm is without replacement, good.
    #                     indices = torch.randperm(total_simulated, device=self.device)[
    #                         :batch_size
    #                     ]
    #                     batch_to_score = simulated_samples[indices]
    #                     # Ensure labels match the batch being scored if needed by FID function internals (though compute_batch_fid uses class_label)
    #                     # current_batch_labels = sim_label[indices] # Not directly used by compute_batch_fid

    #                     fid_score = self.compute_batch_fid(
    #                         batch_to_score, class_label, use_global
    #                     )
    #                     # Handle potential infinite scores if FID calculation failed
    #                     if fid_score == float("inf"):
    #                         print(
    #                             f"Warning: FID calculation failed for scoring batch {i}. Assigning high score."
    #                         )
    #                         # Assign a very high score instead of inf to allow argmin to work if others are also inf
    #                         fid_score = 1e10  # Or some other large number

    #                     batch_scores.append(fid_score)
    #                     batch_indices_list.append(indices)

    #                 # --- 5. Select Best Batch ---
    #                 if (
    #                     not batch_scores
    #                 ):  # Should not happen if total_simulated >= batch_size
    #                     print(
    #                         "Error: No batch scores generated. Selecting first samples."
    #                     )
    #                     winning_indices = torch.arange(batch_size, device=self.device)
    #                 else:
    #                     scores_tensor = torch.tensor(batch_scores, device=self.device)
    #                     # Find the minimum FID score (lower is better)
    #                     best_score = torch.min(scores_tensor)
    #                     best_batch_idx = torch.argmin(scores_tensor)
    #                     winning_indices = batch_indices_list[
    #                         best_batch_idx
    #                     ]  # These are indices into the simulated pool

    #             # --- 6. Update Current State ---
    #             # Select the state *after the branching step* corresponding to the winning batch
    #             current_samples = post_branch_samples[winning_indices]
    #             current_times = post_branch_times[winning_indices]
    #             # Label remains the same as we selected a full batch
    #             current_label = torch.full(
    #                 (batch_size,), class_label, device=self.device
    #             )

    #             # Sanity check shape
    #             if current_samples.shape[0] != batch_size:
    #                 print(
    #                     f"Error: current_samples shape is {current_samples.shape} after selection. Expected {batch_size}."
    #                 )
    #                 # This indicates a problem with indexing or the fallback logic
    #                 # Try to recover or raise error? For now, print and potentially break.
    #                 break

    #         # Return the final batch of samples (should be batch_size)
    #         if current_samples.shape[0] != batch_size:
    #             print(
    #                 f"Warning: Final sample count is {current_samples.shape[0]}, expected {batch_size}. Returning available samples."
    #             )
    #             # Potentially pad or truncate? Returning as is for now.
    #         return current_samples

    # def batch_sample_with_path_exploration_batch_fid(
    #     self,
    #     class_label,
    #     batch_size=16,
    #     num_branches=4,
    #     num_scoring_batches=10,  # How many random batches to score
    #     dt_std=0.1,  # Std deviation for dt sampling
    #     use_global=False,
    #     branch_start_time=0.0,
    #     branch_dt=None,
    # ):
    #     """
    #     Samples using dt variation path exploration, selecting based on BATCH FID score.

    #     Args:
    #         class_label: Target class label.
    #         batch_size: Final number of samples and size of batches for FID scoring.
    #         num_branches: Number of branches per sample at each step.
    #         num_scoring_batches: Number of random batches (size batch_size) to evaluate with FID.
    #         dt_std: Standard deviation for sampling dt variations (relative to branch_dt_base).
    #         use_global: Use global FID statistics instead of class-specific.
    #         branch_start_time: Time to start the branching exploration.
    #         branch_dt: Base timestep size during exploration.

    #     Returns:
    #         Tensor of [batch_size, C, H, W] generated samples.
    #     """
    #     if num_branches == 1:
    #         return self.regular_batch_sample(class_label, batch_size)

    #     assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"
    #     assert num_scoring_batches > 0, "num_scoring_batches must be positive"

    #     self.flow_model.eval()
    #     base_dt = 1 / self.num_timesteps  # For initial phase and simulation
    #     branch_dt_base = branch_dt if branch_dt is not None else base_dt

    #     with torch.no_grad():
    #         # Initialize with batch_size samples
    #         current_samples = torch.randn(
    #             batch_size,
    #             self.channels,
    #             self.image_size,
    #             self.image_size,
    #             device=self.device,
    #         )
    #         current_times = torch.zeros(batch_size, device=self.device)
    #         current_label = torch.full((batch_size,), class_label, device=self.device)

    #         # --- Regular flow until branch_start_time ---
    #         while torch.any(current_times < branch_start_time):
    #             max_dt = torch.clamp(branch_start_time - current_times, min=0.0)
    #             dt = torch.minimum(base_dt * torch.ones_like(current_times), max_dt)
    #             active_mask = dt > 0
    #             if not torch.any(active_mask):
    #                 break

    #             velocity = self.flow_model(
    #                 current_times[active_mask],
    #                 current_samples[active_mask],
    #                 current_label[active_mask],
    #             )
    #             current_samples[active_mask] = current_samples[
    #                 active_mask
    #             ] + velocity * dt[active_mask].view(-1, 1, 1, 1)
    #             current_times[active_mask] = (
    #                 current_times[active_mask] + dt[active_mask]
    #             )
    #             current_times = torch.clamp(current_times, max=branch_start_time)

    #         # --- Main Path Exploration Loop ---
    #         while torch.any(current_times < 1.0):

    #             active_batch_mask = current_times < 1.0
    #             if not torch.any(active_batch_mask):
    #                 print("All samples finished.")
    #                 break

    #             active_samples = current_samples
    #             active_times = current_times
    #             active_label = current_label
    #             num_active = len(active_samples)

    #             # --- 1. Create Branches ---
    #             branched_samples = active_samples.repeat_interleave(num_branches, dim=0)
    #             branched_times = active_times.repeat_interleave(num_branches)
    #             branched_label = active_label.repeat_interleave(num_branches)

    #             # --- 2. Take ONE Branching Step with DT Variation ---
    #             post_branch_samples = branched_samples.clone()  # State AFTER the step
    #             post_branch_times = branched_times.clone()  # Time AFTER the step

    #             # Only compute velocity for branches where t < 1 (mask used later)
    #             active_step_mask = branched_times < 1.0
    #             if torch.any(active_step_mask):
    #                 # Get velocity for ALL branches (simpler to compute for all, mask later)
    #                 velocity = self.flow_model(
    #                     branched_times, branched_samples, branched_label
    #                 )

    #                 # Sample dt values for ALL branches using SCALAR mean and std
    #                 # Use branch_dt_base (scalar) as the mean
    #                 mean_dt_scalar = branch_dt_base
    #                 std_dev_scalar = dt_std * mean_dt_scalar

    #                 # Use the normal(float, float, size=...) signature
    #                 dts = torch.normal(
    #                     mean=mean_dt_scalar,
    #                     std=std_dev_scalar,
    #                     size=(len(branched_samples),),  # Sample for ALL branches
    #                     device=self.device,
    #                 )

    #                 # Clamp dt AFTER sampling based on individual time remaining
    #                 # Ensure clamp min is a tensor or scalar consistent with dts
    #                 min_clamp = torch.tensor(0.0, device=self.device)
    #                 # max is time remaining for each branch
    #                 max_clamp = 1.0 - branched_times
    #                 dts = torch.clamp(dts, min=min_clamp, max=max_clamp)
    #                 # Ensure dts is non-negative after clamping (max could be < 0 if t > 1, though mask should prevent)
    #                 dts = torch.clamp(dts, min=0.0)

    #                 # Apply update only to active branches using the calculated dts
    #                 # Select the relevant dts and velocities using the mask
    #                 active_velocity = velocity[active_step_mask]
    #                 active_dts = dts[active_step_mask]

    #                 update = active_velocity * active_dts.view(-1, 1, 1, 1)
    #                 # Update samples in post_branch_samples using the mask
    #                 post_branch_samples[active_step_mask] = (
    #                     branched_samples[active_step_mask] + update
    #                 )

    #                 # Update times for ALL branches based on their respective sampled/clamped dts
    #                 post_branch_times = post_branch_times + dts
    #                 post_branch_times = torch.clamp(
    #                     post_branch_times, max=1.0
    #                 )  # Clamp time

    #             # --- 3. Simulate Each Branch to Completion (for Scoring) ---
    #             # Simulation uses standard Euler steps (base_dt) from the state AFTER the branching step
    #             simulated_samples = post_branch_samples.clone()
    #             simulated_times = post_branch_times.clone()
    #             sim_label = branched_label.clone()

    #             sim_loop_count = 0
    #             max_sim_loops = (
    #                 int(1.0 / base_dt) + 50
    #             )  # Safety break for inner loop (uses base_dt)
    #             while torch.any(simulated_times < 1.0):
    #                 sim_loop_count += 1
    #                 if sim_loop_count > max_sim_loops:
    #                     print(
    #                         "Warning: Max loops reached in simulation step. Breaking."
    #                     )
    #                     simulated_times = torch.clamp(simulated_times, min=1.0)
    #                     break

    #                 sim_active_mask = simulated_times < 1.0
    #                 if not torch.any(sim_active_mask):
    #                     break

    #                 active_sim_samples = simulated_samples[sim_active_mask]
    #                 active_sim_times = simulated_times[sim_active_mask]
    #                 active_sim_label = sim_label[sim_active_mask]

    #                 # Calculate dt for simulation step (standard Euler with base_dt)
    #                 dt_sim = torch.minimum(
    #                     base_dt * torch.ones_like(active_sim_times),
    #                     1.0 - active_sim_times,
    #                 )
    #                 dt_sim = torch.clamp(dt_sim, min=0.0)  # Ensure non-negative

    #                 # Get velocity and apply update
    #                 velocity_sim = self.flow_model(
    #                     active_sim_times, active_sim_samples, active_sim_label
    #                 )
    #                 update = velocity_sim * dt_sim.view(-1, 1, 1, 1)

    #                 simulated_samples[sim_active_mask] = active_sim_samples + update
    #                 simulated_times[sim_active_mask] = active_sim_times + dt_sim
    #                 simulated_times[sim_active_mask] = torch.clamp(
    #                     simulated_times[sim_active_mask], max=1.0
    #                 )  # Clamp time

    #             # --- 4. Score Simulated Samples using Batch FID ---
    #             total_simulated = len(simulated_samples)
    #             batch_scores = []
    #             batch_indices_list = []

    #             if total_simulated < batch_size:
    #                 print(
    #                     f"Warning: Not enough simulated samples ({total_simulated}) for batch size {batch_size}. Cannot score with FID."
    #                 )
    #                 winning_indices = torch.arange(
    #                     min(total_simulated, batch_size), device=self.device
    #                 )
    #                 print(
    #                     f"Falling back to selecting the first {len(winning_indices)} samples."
    #                 )
    #             else:
    #                 for i in range(num_scoring_batches):
    #                     indices = torch.randperm(total_simulated, device=self.device)[
    #                         :batch_size
    #                     ]
    #                     batch_to_score = simulated_samples[indices]
    #                     fid_score = self.compute_batch_fid(
    #                         batch_to_score, class_label, use_global
    #                     )
    #                     if fid_score == float("inf"):
    #                         print(
    #                             f"Warning: FID calculation failed for scoring batch {i}. Assigning high score."
    #                         )
    #                         fid_score = 1e10
    #                     batch_scores.append(fid_score)
    #                     batch_indices_list.append(indices)

    #                 # --- 5. Select Best Batch ---
    #                 if not batch_scores:
    #                     print(
    #                         "Error: No batch scores generated. Selecting first samples."
    #                     )
    #                     winning_indices = torch.arange(batch_size, device=self.device)
    #                 else:
    #                     scores_tensor = torch.tensor(batch_scores, device=self.device)
    #                     best_score = torch.min(scores_tensor)
    #                     best_batch_idx = torch.argmin(scores_tensor)
    #                     winning_indices = batch_indices_list[best_batch_idx]

    #             # --- 6. Update Current State ---
    #             current_samples = post_branch_samples[winning_indices]
    #             current_times = post_branch_times[winning_indices]
    #             current_label = torch.full(
    #                 (batch_size,), class_label, device=self.device
    #             )

    #             if current_samples.shape[0] != batch_size:
    #                 print(
    #                     f"Error: current_samples shape is {current_samples.shape} after selection. Expected {batch_size}."
    #                 )
    #                 break

    #         if current_samples.shape[0] != batch_size:
    #             print(
    #                 f"Warning: Final sample count is {current_samples.shape[0]}, expected {batch_size}. Returning available samples."
    #             )
    #         return current_samples

    # def batch_sample_with_random_search_batch_fid_direct(
    #     self,
    #     class_label,
    #     batch_size=16,
    #     num_branches=4,  # Determines the size of the candidate pool (batch_size * num_branches)
    #     num_scoring_batches=10,  # How many random batches to score
    #     use_global=True,
    # ):
    #     """
    #     Generates samples using a direct random search optimized for batch FID.
    #     Matches the signature of path exploration methods for consistency.
    #     1. Generates a pool of candidate samples (size = batch_size * num_branches).
    #     2. Randomly samples multiple batches (size batch_size) from the pool.
    #     3. Scores each batch using FID against the target distribution.
    #     4. Returns the batch with the best FID score.

    #     Args:
    #         class_label: Target class label.
    #         batch_size: Final number of samples and size of batches for FID scoring.
    #         num_branches: Multiplier for batch_size to determine the candidate pool size.
    #         num_scoring_batches: Number of random batches to sample and evaluate with FID.
    #         use_global: Use global FID statistics instead of class-specific.

    #     Returns:
    #         Tensor of [batch_size, C, H, W] generated samples corresponding to the best batch FID.
    #     """
    #     if num_branches == 1:
    #         return self.regular_batch_sample(class_label, batch_size)

    #     # Calculate the total number of candidates based on batch_size and num_branches
    #     num_candidates = batch_size * num_branches

    #     assert num_branches >= 1, f"num_branches ({num_branches}) must be >= 1"
    #     assert num_scoring_batches > 0, "num_scoring_batches must be positive"

    #     self.flow_model.eval()
    #     base_dt = 1 / self.num_timesteps

    #     with torch.no_grad():
    #         # --- 1. Generate Candidate Pool (size: num_candidates) ---
    #         all_candidate_samples = torch.zeros(
    #             (num_candidates, self.channels, self.image_size, self.image_size),
    #             device=self.device,
    #         )
    #         generated_count = 0

    #         # Generate candidates in manageable chunks
    #         # Using batch_size chunks for convenience, but could use larger chunks
    #         while generated_count < num_candidates:
    #             # Determine size of the next chunk to generate
    #             current_chunk_size = min(batch_size, num_candidates - generated_count)
    #             if current_chunk_size <= 0:
    #                 break  # Should not happen with loop condition, but safe check

    #             # Initialize chunk
    #             current_samples = torch.randn(
    #                 current_chunk_size,
    #                 self.channels,
    #                 self.image_size,
    #                 self.image_size,
    #                 device=self.device,
    #             )
    #             current_label = torch.full(
    #                 (current_chunk_size,), class_label, device=self.device
    #             )

    #             # Ensure timesteps attribute exists
    #             if not hasattr(self, "timesteps") or self.timesteps is None:
    #                 raise AttributeError(
    #                     "Class requires 'timesteps' attribute for standard sampling."
    #                 )

    #             # Regular flow matching for this chunk using class timesteps
    #             for step, t in enumerate(self.timesteps[:-1]):
    #                 dt = self.timesteps[step + 1] - t
    #                 t_batch = torch.full(
    #                     (current_chunk_size,), t.item(), device=self.device
    #                 )

    #                 velocity = self.flow_model(t_batch, current_samples, current_label)
    #                 current_samples = current_samples + velocity * dt

    #             # Store generated samples
    #             all_candidate_samples[
    #                 generated_count : generated_count + current_chunk_size
    #             ] = current_samples
    #             generated_count += current_chunk_size

    #         if generated_count != num_candidates:
    #             print(
    #                 f"Warning: Mismatch in generated candidates. Proceeding with {generated_count}."
    #             )
    #             # Adjust num_candidates if mismatch occurred, relevant for randperm range
    #             num_candidates = generated_count
    #             if num_candidates < batch_size:
    #                 print(
    #                     "Error: Fewer candidates generated than batch size. Cannot proceed."
    #                 )
    #                 return torch.empty(
    #                     0,
    #                     self.channels,
    #                     self.image_size,
    #                     self.image_size,
    #                     device=self.device,
    #                 )  # Return empty tensor

    #         # --- 2. Score Random Batches ---
    #         batch_scores = []
    #         batch_indices_list = []  # Store the indices used for each random batch

    #         # Check if enough candidates for even one scoring batch
    #         if num_candidates < batch_size:
    #             print("Error: Not enough candidates to form a batch for scoring.")
    #             # Return the first 'batch_size' if available, or all if fewer
    #             return all_candidate_samples[: min(batch_size, num_candidates)]

    #         for i in range(num_scoring_batches):
    #             # Randomly sample batch_size indices *without replacement* from the pool
    #             indices = torch.randperm(num_candidates, device=self.device)[
    #                 :batch_size
    #             ]
    #             batch_to_score = all_candidate_samples[indices]

    #             # Compute FID for the selected batch
    #             fid_score = self.compute_batch_fid(
    #                 batch_to_score, class_label, use_global
    #             )

    #             # Handle potential infinite scores if FID calculation failed
    #             if fid_score == float("inf"):
    #                 print(
    #                     f"Warning: FID calculation failed for scoring batch {i}. Assigning high score."
    #                 )
    #                 fid_score = 1e10  # Assign a very high score

    #             batch_scores.append(fid_score)
    #             batch_indices_list.append(indices)  # Store the actual indices

    #         # --- 3. Select Best Batch ---
    #         if (
    #             not batch_scores
    #         ):  # Should only happen if num_scoring_batches was 0 or FID failed every time
    #             print(
    #                 "Error: No valid batch scores generated. Returning first samples."
    #             )
    #             # Fallback: return the first batch_size candidates
    #             return all_candidate_samples[:batch_size]
    #         else:
    #             scores_tensor = torch.tensor(batch_scores, device=self.device)
    #             # Find the minimum FID score (lower is better)
    #             best_score = torch.min(scores_tensor)
    #             best_batch_overall_idx = torch.argmin(
    #                 scores_tensor
    #             )  # Index in the list of scored batches
    #             winning_indices = batch_indices_list[
    #                 best_batch_overall_idx
    #             ]  # Get the actual sample indices

    #             print(
    #                 f"Selected batch with FID: {best_score:.4f} (from {num_scoring_batches} evaluations)"
    #             )

    #             # Retrieve the best batch from the candidate pool
    #             final_samples = all_candidate_samples[winning_indices]

    #             return final_samples
