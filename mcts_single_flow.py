from collections import deque
import torch
from tqdm import tqdm
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)
import numpy as np
import os
import pickle
from scipy.linalg import sqrtm
from torchmetrics.image.fid import NoTrainInceptionV3
from utils import divfree_swirl_si, score_si_linear


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
            return self.batch_sample_ode(class_label, batch_size)

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
            return self.batch_sample_ode(class_label, batch_size)

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
        return torch.clamp(unnormalized, 0.0, 1.0).cpu()

    def decode_latents(self, latents):
        """
        Decode latents to images using VAE.
        """
        batch_size = latents.shape[0]
        max_batch_size = 64  # this is based on a 24GB RTX 6000

        if batch_size <= max_batch_size:
            # Convert latents to half precision to match VAE parameters
            latents_half = latents.half()
            images = self.vae.decode(latents_half / 0.18215).sample
            return torch.clamp((images + 1) / 2, 0.0, 1.0).float()
        else:
            # Process in batches of max_batch_size
            decoded_images = []
            for i in range(0, batch_size, max_batch_size):
                batch_latents = latents[i : i + max_batch_size]
                batch_latents_half = batch_latents.half()
                batch_images = self.vae.decode(batch_latents_half / 0.18215).sample
                batch_images = torch.clamp((batch_images + 1) / 2, 0.0, 1.0).float()
                decoded_images.append(batch_images)

            return torch.cat(decoded_images, dim=0)

    def batch_sample_ode(self, class_label, batch_size=16):
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

    def batch_sample_sde(self, class_label, batch_size=16, noise_scale=0.05):
        """
        Flow matching sampling with added noise (SDE sampling).

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            noise_scale: Scale of the noise to add during sampling
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

            # Generate samples using timesteps with noise
            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full((batch_size,), t.item(), device=self.device)

                # Flow step with SDE noise term
                velocity = self.flow_model(t_batch, current_samples, current_label)

                # Add noise scaled by dt and noise_scale
                noise = torch.randn_like(current_samples) * torch.sqrt(dt) * noise_scale

                # Euler-Maruyama update
                current_samples = current_samples + velocity * dt + noise

            return self.unnormalize_images(current_samples)

    def batch_sample_ode_divfree(
        self,
        class_label,
        batch_size=16,
        lambda_div=0.2,
    ):
        is_tensor = torch.is_tensor(class_label)
        self.flow_model.eval()

        with torch.no_grad():
            x = torch.randn(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )

            if is_tensor:
                y = class_label
            else:
                y = torch.full(
                    (batch_size,), class_label, device=self.device, dtype=torch.long
                )

            for step, t in enumerate(self.timesteps[:-1]):
                dt = self.timesteps[step + 1] - t
                t_batch = torch.full((batch_size,), t.item(), device=self.device)

                u_t = self.flow_model(t_batch, x, y)  # drift
                w = lambda_div * divfree_swirl_si(x, t_batch, y, u_t)

                x = x + (u_t + w) * dt  # Euler ODE step

            return self.unnormalize_images(x)

    def batch_sample_ode_divfree_path_exploration(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        lambda_div=0.2,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
    ):
        """
        Flow matching sampling with ODE and divergence-free path exploration.
        Explores multiple paths by adding different divergence-free vector fields
        at each branching step, then simulates deterministic paths to evaluate branches.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            lambda_div: Scale factor for the divergence-free field
            selector: Selection criteria - options include "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
        """
        if num_branches == 1 and num_keep == 1:
            return self.batch_sample_ode(class_label, batch_size)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"

        score_fn, use_global = self._get_score_function(selector, use_global)

        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps

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

            # Handle both tensor and single class label cases
            if torch.is_tensor(class_label):
                current_label = class_label
            else:
                current_label = torch.full(
                    (batch_size,), class_label, device=self.device
                )

            # Regular flow until branch_start_time
            while torch.all(current_times < branch_start_time):
                t_batch = current_times
                u_t = self.flow_model(t_batch, current_samples, current_label)
                w = lambda_div * divfree_swirl_si(
                    current_samples, t_batch, current_label, u_t
                )

                dt = min(base_dt, branch_start_time - current_times[0].item())
                current_samples = current_samples + (u_t + w) * dt
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

                # Take one branching step with different divergence-free fields
                u_t = self.flow_model(branched_times, branched_samples, branched_label)

                # Add divergence-free field for branching
                w = lambda_div * divfree_swirl_si(
                    branched_samples, branched_times, branched_label, u_t
                )

                dt = torch.clamp(
                    torch.full((len(branched_samples),), base_dt, device=self.device),
                    min=torch.tensor(0.0, device=self.device),
                    max=1.0 - branched_times,
                )

                # Apply the branching step with ODE + div-free field
                branched_samples = branched_samples + (u_t + w) * dt.view(-1, 1, 1, 1)
                branched_times = branched_times + dt

                # Simulate each branch to completion (t=1) WITHOUT div-free fields (deterministic)
                simulated_samples = branched_samples.clone()
                simulated_times = branched_times.clone()

                while torch.any(simulated_times < 1.0):
                    # Only update samples that haven't reached t=1
                    active_mask = simulated_times < 1.0
                    if not torch.any(active_mask):
                        break

                    active_times = simulated_times[active_mask]
                    active_samples = simulated_samples[active_mask]
                    active_labels = branched_label[active_mask]

                    # Only use the drift u_t for deterministic simulation, no div-free field
                    u_t = self.flow_model(active_times, active_samples, active_labels)

                    dt = torch.min(
                        base_dt * torch.ones_like(active_times),
                        1.0 - active_times,
                    )

                    simulated_samples[active_mask] = active_samples + u_t * dt.view(
                        -1, 1, 1, 1
                    )
                    simulated_times[active_mask] = active_times + dt

                # Evaluate final samples
                if use_global:
                    final_scores = score_fn(simulated_samples)
                else:
                    final_scores = score_fn(simulated_samples, branched_label)

                # Select best branches for each batch element
                selected_samples = []
                selected_times = []
                selected_labels = []

                for idx in range(len(current_samples)):
                    # Get branches for this batch element
                    batch_mask = batch_indices == idx
                    batch_samples = branched_samples[batch_mask]
                    batch_times = branched_times[batch_mask]
                    batch_labels = branched_label[batch_mask]
                    batch_scores = final_scores[batch_mask]

                    # Select top num_keep branches based on final scores
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])
                    selected_labels.append(batch_labels[top_k_indices])

                # Update current state with selected branches
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)
                current_label = torch.cat(selected_labels, dim=0)

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

            # Need to track batch indices
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
                    "labels": current_label[batch_mask],
                }

            # Select best sample for each batch element
            for i in range(batch_size):
                batch_data = samples_by_batch[i]
                best_idx = torch.argmax(batch_data["scores"])
                final_samples.append(batch_data["samples"][best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_sde_path_exploration(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        noise_scale=0.05,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
    ):
        """
        Flow matching sampling with SDE and path exploration.
        Explores multiple paths by adding different noise samples at each branching step,
        then simulates deterministic paths to evaluate branches.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            noise_scale: Scale of the noise to add during branching
            selector: Selection criteria - options include "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
        """
        if num_branches == 1 and num_keep == 1:
            return self.batch_sample_ode(class_label, batch_size)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"

        score_fn, use_global = self._get_score_function(selector, use_global)

        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps

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

            # Handle both tensor and single class label cases
            if torch.is_tensor(class_label):
                current_label = class_label
            else:
                current_label = torch.full(
                    (batch_size,), class_label, device=self.device
                )

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

                # Take one branching step with different noise samples
                velocity = self.flow_model(
                    branched_times, branched_samples, branched_label
                )
                dt = torch.clamp(
                    torch.full((len(branched_samples),), base_dt, device=self.device),
                    min=torch.tensor(0.0, device=self.device),
                    max=1.0 - branched_times,
                )

                # Generate different noise samples for each branch
                noise = (
                    torch.randn_like(branched_samples)
                    * torch.sqrt(dt.view(-1, 1, 1, 1))
                    * noise_scale
                )

                # Apply the branching step with SDE
                branched_samples = (
                    branched_samples + velocity * dt.view(-1, 1, 1, 1) + noise
                )
                branched_times = branched_times + dt

                # Simulate each branch to completion (t=1) WITHOUT noise (deterministic)
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

                    # No noise during simulation phase - deterministic
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
                selected_labels = []

                for idx in range(len(current_samples)):
                    # Get branches for this batch element
                    batch_mask = batch_indices == idx
                    batch_samples = branched_samples[batch_mask]
                    batch_times = branched_times[batch_mask]
                    batch_labels = branched_label[batch_mask]
                    batch_scores = final_scores[batch_mask]

                    # Select top num_keep branches based on final scores
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])
                    selected_labels.append(batch_labels[top_k_indices])

                # Update current state with selected branches
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)
                current_label = torch.cat(selected_labels, dim=0)

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

            # Need to track batch indices
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
                    "labels": current_label[batch_mask],
                }

            # Select best sample for each batch element
            for i in range(batch_size):
                batch_data = samples_by_batch[i]
                best_idx = torch.argmax(batch_data["scores"])
                final_samples.append(batch_data["samples"][best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_with_random_search_sde(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        selector="fid",
        use_global=False,
        noise_scale=0.05,
    ):
        """
        Simple random search sampling method using SDE (stochastic differential equation) that:
        1. Runs num_branches independent flow matching batches with noise
        2. Evaluates all samples using selected metric
        3. Returns the best sample for each class position

        Args:
            class_label: Tensor of class labels for the batch
            batch_size: Number of final samples to generate
            num_branches: Number of branches to evaluate per sample
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", or "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            noise_scale: Scale of the noise to add during sampling
        """
        assert (
            len(class_label) == batch_size
        ), "class_label tensor length must match batch_size"

        if num_branches == 1:
            return self.batch_sample_ode(class_label, batch_size)

        score_fn, use_global = self._get_score_function(selector, use_global)

        self.flow_model.eval()

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

                # SDE flow matching for this batch
                for step, t in enumerate(self.timesteps[:-1]):
                    dt = self.timesteps[step + 1] - t
                    t_batch = torch.full((batch_size,), t.item(), device=self.device)

                    # Flow step with SDE noise term
                    velocity = self.flow_model(t_batch, current_samples, class_label)

                    # Add noise scaled by dt and noise_scale
                    noise = (
                        torch.randn_like(current_samples) * torch.sqrt(dt) * noise_scale
                    )

                    # Euler-Maruyama update
                    current_samples = current_samples + velocity * dt + noise

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

    def batch_sample_score_sde_path_exploration(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        noise_scale_factor=1.0,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
    ):
        """
        Flow matching sampling with Score SDE and path exploration.
        Explores multiple paths by adding different score-based noise samples at each branching step,
        then simulates deterministic paths to evaluate branches.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches per batch element at each step
            num_keep: Number of samples to keep before next branching
            noise_scale_factor: Scale factor for the g_t = t^2 noise schedule
            selector: Selection criteria - options include "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching (0.0 to 1.0)
        """
        if num_branches == 1 and num_keep == 1:
            return self.batch_sample_ode(class_label, batch_size)

        assert (
            num_branches % num_keep == 0
        ), "num_branches must be divisible by num_keep"
        assert 0.0 <= branch_start_time < 1.0, "branch_start_time must be in [0, 1)"

        score_fn, use_global = self._get_score_function(selector, use_global)

        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps

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

            # Handle both tensor and single class label cases
            if torch.is_tensor(class_label):
                current_label = class_label
            else:
                current_label = torch.full(
                    (batch_size,), class_label, device=self.device
                )

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

                # Take one branching step with Score SDE
                dt = torch.clamp(
                    torch.full((len(branched_samples),), base_dt, device=self.device),
                    min=torch.tensor(0.0, device=self.device),
                    max=1.0 - branched_times,
                )

                # Get velocity from flow model
                velocity = self.flow_model(
                    branched_times, branched_samples, branched_label
                )

                # Compute score using score_si_linear function
                from utils import score_si_linear

                score = score_si_linear(branched_samples, branched_times, velocity)

                # Compute noise schedule g_t = t^2 scaled by factor
                g_t = (branched_times**2) * noise_scale_factor
                g_t = g_t.view(-1, *([1] * (branched_samples.ndim - 1)))

                # Compute drift coefficient: f_t(x_t) = u_t(x_t) - (g_t^2/2) * score
                drift_correction = -(g_t**2) / 2.0 * score
                drift = velocity + drift_correction

                # Generate different score-based noise samples for each branch
                noise = (
                    torch.randn_like(branched_samples)
                    * g_t
                    * torch.sqrt(dt.view(-1, 1, 1, 1))
                )

                # Apply the branching step with Score SDE
                branched_samples = (
                    branched_samples + drift * dt.view(-1, 1, 1, 1) + noise
                )
                branched_times = branched_times + dt

                # Simulate each branch to completion (t=1) WITHOUT noise (deterministic)
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

                    # No noise during simulation phase - deterministic
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
                selected_labels = []

                for idx in range(len(current_samples)):
                    # Get branches for this batch element
                    batch_mask = batch_indices == idx
                    batch_samples = branched_samples[batch_mask]
                    batch_times = branched_times[batch_mask]
                    batch_labels = branched_label[batch_mask]
                    batch_scores = final_scores[batch_mask]

                    # Select top num_keep branches based on final scores
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])
                    selected_labels.append(batch_labels[top_k_indices])

                # Update current state with selected branches
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)
                current_label = torch.cat(selected_labels, dim=0)

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

            # Need to track batch indices
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
                    "labels": current_label[batch_mask],
                }

            # Select best sample for each batch element
            for i in range(batch_size):
                batch_data = samples_by_batch[i]
                best_idx = torch.argmax(batch_data["scores"])
                final_samples.append(batch_data["samples"][best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_random_search_then_divfree_path_exploration(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        lambda_div=0.2,
        selector="fid",
        use_global=False,
        branch_start_time=0.0,
    ):
        """
        Two-stage inference scaling method that combines random search with divfree path exploration:
        1. First stage: Run random search to find high-scoring initial noise samples
        2. Second stage: Use the winning initial noises from stage 1 as starting points for divfree path exploration

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches for both the random search and path exploration stages
            num_keep: Number of samples to keep during path exploration branching
            lambda_div: Scale factor for the divergence-free field in path exploration
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            branch_start_time: Time point at which to start branching in path exploration (0.0 to 1.0)
        """
        assert (
            len(class_label) == batch_size if torch.is_tensor(class_label) else True
        ), "class_label tensor length must match batch_size"

        if num_branches == 1:
            # If no search needed, just do divfree path exploration
            return self.batch_sample_ode_divfree_path_exploration(
                class_label,
                batch_size,
                num_branches,
                num_keep,
                lambda_div,
                selector,
                use_global,
                branch_start_time,
            )

        score_fn, use_global = self._get_score_function(selector, use_global)
        self.flow_model.eval()
        base_dt = 1 / self.num_timesteps

        # Handle both tensor and single class label cases
        if torch.is_tensor(class_label):
            current_label = class_label
        else:
            current_label = torch.full((batch_size,), class_label, device=self.device)

        with torch.no_grad():
            # =========================
            # STAGE 1: RANDOM SEARCH
            # =========================

            # Store initial noises and their corresponding final samples for evaluation
            all_initial_noises = []
            all_final_samples = []

            for _ in range(num_branches):
                # Generate random initial noise for this branch
                initial_noise = torch.randn(
                    batch_size,
                    self.channels,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )

                # Store the initial noise
                all_initial_noises.append(initial_noise)

                # Run regular ODE flow matching from this noise
                current_samples = initial_noise.clone()

                for step, t in enumerate(self.timesteps[:-1]):
                    t_batch = torch.full((batch_size,), t.item(), device=self.device)
                    velocity = self.flow_model(t_batch, current_samples, current_label)
                    current_samples = current_samples + velocity * base_dt

                all_final_samples.append(current_samples)

            # Evaluate all final samples from random search
            all_scores = []
            for final_samples in all_final_samples:
                if use_global:
                    scores = score_fn(final_samples)
                else:
                    scores = score_fn(final_samples, current_label)
                all_scores.append(scores)

            # Stack scores [num_branches, batch_size]
            all_scores = torch.stack(all_scores, dim=0)

            # Find best initial noise for each position
            best_branch_indices = torch.argmax(all_scores, dim=0)  # Shape: [batch_size]

            # Extract the winning initial noises
            winning_initial_noises = torch.zeros(
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
            )

            for i in range(batch_size):
                best_branch = best_branch_indices[i]
                winning_initial_noises[i] = all_initial_noises[best_branch][i]

            # =========================
            # STAGE 2: DIVFREE PATH EXPLORATION FROM WINNING NOISES
            # =========================

            if num_branches == 1 and num_keep == 1:
                # If no path exploration needed, just run ODE from winning noises
                current_samples = winning_initial_noises
                current_times = torch.zeros(batch_size, device=self.device)

                for step, t in enumerate(self.timesteps[:-1]):
                    t_batch = torch.full((batch_size,), t.item(), device=self.device)
                    u_t = self.flow_model(t_batch, current_samples, current_label)
                    w = lambda_div * divfree_swirl_si(
                        current_samples, t_batch, current_label, u_t
                    )
                    current_samples = current_samples + (u_t + w) * base_dt

                return self.unnormalize_images(current_samples)

            # Run divfree path exploration starting from the winning initial noises
            current_samples = winning_initial_noises
            current_times = torch.zeros(batch_size, device=self.device)

            # Regular flow until branch_start_time
            while torch.all(current_times < branch_start_time):
                t_batch = current_times
                u_t = self.flow_model(t_batch, current_samples, current_label)
                w = lambda_div * divfree_swirl_si(
                    current_samples, t_batch, current_label, u_t
                )

                dt = min(base_dt, branch_start_time - current_times[0].item())
                current_samples = current_samples + (u_t + w) * dt
                current_times += dt

            # Main path exploration loop - continue until all samples reach t=1
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

                # Take one branching step with different divergence-free fields
                u_t = self.flow_model(branched_times, branched_samples, branched_label)
                w = lambda_div * divfree_swirl_si(
                    branched_samples, branched_times, branched_label, u_t
                )

                dt = torch.clamp(
                    torch.full((len(branched_samples),), base_dt, device=self.device),
                    min=torch.tensor(0.0, device=self.device),
                    max=1.0 - branched_times,
                )

                # Apply the branching step with ODE + div-free field
                branched_samples = branched_samples + (u_t + w) * dt.view(-1, 1, 1, 1)
                branched_times = branched_times + dt

                # Simulate each branch to completion (t=1) WITHOUT div-free fields (deterministic)
                simulated_samples = branched_samples.clone()
                simulated_times = branched_times.clone()

                while torch.any(simulated_times < 1.0):
                    # Only update samples that haven't reached t=1
                    active_mask = simulated_times < 1.0
                    if not torch.any(active_mask):
                        break

                    active_times = simulated_times[active_mask]
                    active_samples = simulated_samples[active_mask]
                    active_labels = branched_label[active_mask]

                    # Only use the drift u_t for deterministic simulation, no div-free field
                    u_t = self.flow_model(active_times, active_samples, active_labels)

                    dt = torch.min(
                        base_dt * torch.ones_like(active_times),
                        1.0 - active_times,
                    )

                    simulated_samples[active_mask] = active_samples + u_t * dt.view(
                        -1, 1, 1, 1
                    )
                    simulated_times[active_mask] = active_times + dt

                # Evaluate final samples
                if use_global:
                    final_scores = score_fn(simulated_samples)
                else:
                    final_scores = score_fn(simulated_samples, branched_label)

                # Select best branches for each batch element
                selected_samples = []
                selected_times = []
                selected_labels = []

                for idx in range(len(current_samples)):
                    # Get branches for this batch element
                    batch_mask = batch_indices == idx
                    batch_samples = branched_samples[batch_mask]
                    batch_times = branched_times[batch_mask]
                    batch_labels = branched_label[batch_mask]
                    batch_scores = final_scores[batch_mask]

                    # Select top num_keep branches based on final scores
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=min(num_keep, len(batch_scores)), dim=0
                    )

                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])
                    selected_labels.append(batch_labels[top_k_indices])

                # Update current state with selected branches
                current_samples = torch.cat(selected_samples, dim=0)
                current_times = torch.cat(selected_times, dim=0)
                current_label = torch.cat(selected_labels, dim=0)

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

            # Track batch indices for final selection
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
                    "labels": current_label[batch_mask],
                }

            # Select best sample for each batch element
            for i in range(batch_size):
                batch_data = samples_by_batch[i]
                best_idx = torch.argmax(batch_data["scores"])
                final_samples.append(batch_data["samples"][best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def _sample_with_divfree_noise(
        self, start_sample, label, start_time=0.0, lambda_div=0.2, save_at_time=None
    ):
        """
        Helper function to sample from start_time to t=1 with divergence-free noise.

        Args:
            start_sample: Starting sample (shape: [1, C, H, W])
            label: Class label (shape: [1])
            start_time: Time to start sampling from (0.0 to 1.0)
            lambda_div: Scale factor for divergence-free field
            save_at_time: Optional time to save intermediate sample (if None, only return final)

        Returns:
            If save_at_time is None: Final sample at t=1
            If save_at_time is given: (intermediate_sample, final_sample)
        """
        current_sample = start_sample.clone()
        base_dt = 1 / self.num_timesteps
        current_time = start_time
        intermediate_sample = None

        while current_time < 1.0:
            dt = min(base_dt, 1.0 - current_time)
            t_batch = torch.full((1,), current_time, device=self.device)

            # Get velocity and add divergence-free noise
            u_t = self.flow_model(t_batch, current_sample, label)
            w = lambda_div * divfree_swirl_si(current_sample, t_batch, label, u_t)

            # Update sample
            current_sample = current_sample + (u_t + w) * dt
            current_time += dt

            # Save intermediate sample if we've reached the save_at_time
            if (
                save_at_time is not None
                and intermediate_sample is None
                and np.isclose(current_time, save_at_time, atol=base_dt / 2)
            ):
                intermediate_sample = current_sample.clone()

        if save_at_time is not None:
            return intermediate_sample, current_sample
        else:
            return current_sample

    def _sample_with_sde_noise(
        self, start_sample, label, start_time=0.0, noise_scale=0.05, save_at_time=None
    ):
        """
        Helper function to sample from start_time to t=1 with SDE noise.

        Args:
            start_sample: Starting sample (shape: [1, C, H, W])
            label: Class label (shape: [1])
            start_time: Time to start sampling from (0.0 to 1.0)
            noise_scale: Scale of SDE noise
            save_at_time: Optional time to save intermediate sample (if None, only return final)

        Returns:
            If save_at_time is None: Final sample at t=1
            If save_at_time is given: (intermediate_sample, final_sample)
        """
        current_sample = start_sample.clone()
        base_dt = 1 / self.num_timesteps
        current_time = start_time
        intermediate_sample = None

        while current_time < 1.0:
            dt = min(base_dt, 1.0 - current_time)
            t_batch = torch.full((1,), current_time, device=self.device)

            # Get velocity and add SDE noise
            velocity = self.flow_model(t_batch, current_sample, label)
            noise = torch.randn_like(current_sample) * noise_scale * np.sqrt(dt)

            # Update sample
            current_sample = current_sample + velocity * dt + noise
            current_time += dt

            # Save intermediate sample if we've reached the save_at_time
            if (
                save_at_time is not None
                and intermediate_sample is None
                and np.isclose(current_time, save_at_time, atol=base_dt / 2)
            ):
                intermediate_sample = current_sample.clone()

        if save_at_time is not None:
            return intermediate_sample, current_sample
        else:
            return current_sample

    def _sample_with_divfree_max_noise(
        self,
        start_samples,
        labels,
        start_time=0.0,
        lambda_div=0.2,
        repulsion_strength=0.02,
        noise_schedule_end_factor=0.5,
        save_at_time=None,
    ):
        """
        Helper function to sample from start_time to t=1 with divfree_max noise.
        This handles multiple samples at once to compute repulsion forces between them.

        Args:
            start_samples: Starting samples (shape: [batch_size, C, H, W])
            labels: Class labels (shape: [batch_size])
            start_time: Time to start sampling from (0.0 to 1.0)
            lambda_div: Scale factor for divergence-free field
            repulsion_strength: Strength of repulsion forces
            noise_schedule_end_factor: End factor for time-dependent noise scaling
            save_at_time: Optional time to save intermediate sample (if None, only return final)

        Returns:
            If save_at_time is None: Final samples at t=1
            If save_at_time is given: (intermediate_samples, final_samples)
        """
        current_samples = start_samples.clone()
        batch_size = current_samples.shape[0]
        base_dt = 1 / self.num_timesteps
        current_time = start_time
        intermediate_samples = None

        while current_time < 1.0:
            dt = min(base_dt, 1.0 - current_time)
            t_batch = torch.full((batch_size,), current_time, device=self.device)

            # Get velocity
            u_t = self.flow_model(t_batch, current_samples, labels)

            # Standard divfree term - normal Gaussian noise projected to be divergence-free
            from utils import (
                divfree_swirl_si,
                make_divergence_free,
                particle_guidance_forces,
            )

            w_unscaled = divfree_swirl_si(current_samples, t_batch, labels, u_t)
            w_divfree = lambda_div * w_unscaled

            # Add repulsion term using vectorized approach
            raw_repulsion_forces = particle_guidance_forces(
                current_samples, current_time, alpha_t=1.0, kernel_type="euclidean"
            )

            # Regularize repulsion magnitude to match Gaussian magnitude
            dims = tuple(range(1, w_unscaled.ndim))
            gaussian_magnitude = torch.linalg.vector_norm(w_unscaled, dim=dims).mean()
            repulsion_magnitude = torch.linalg.vector_norm(
                raw_repulsion_forces, dim=dims
            ).mean()

            if repulsion_magnitude > 1e-8:  # Avoid division by zero
                # Scale forces so their average magnitude equals gaussian magnitude
                regularization_factor = gaussian_magnitude / repulsion_magnitude
                regularized_repulsion = raw_repulsion_forces * regularization_factor
            else:
                regularized_repulsion = raw_repulsion_forces

            # Apply user-specified repulsion strength AFTER regularization
            scaled_repulsion = regularized_repulsion * repulsion_strength

            # Make repulsion forces divergence-free to maintain mathematical properties
            repulsion_divfree = make_divergence_free(
                scaled_repulsion, current_samples, t_batch, u_t
            )

            # Combine divfree term and repulsion term
            total_perturbation = w_divfree + repulsion_divfree

            # Apply time-dependent scaling: full strength at t=0, reduced at t=1
            noise_scale_factor = 1.0 + (noise_schedule_end_factor - 1.0) * current_time
            scaled_perturbation = total_perturbation * noise_scale_factor

            # Update samples
            current_samples = current_samples + (u_t + scaled_perturbation) * dt
            current_time += dt

            # Save intermediate sample if we've reached the save_at_time
            if (
                save_at_time is not None
                and intermediate_samples is None
                and np.isclose(current_time, save_at_time, atol=base_dt / 2)
            ):
                intermediate_samples = current_samples.clone()

        if save_at_time is not None:
            return intermediate_samples, current_samples
        else:
            return current_samples

    def batch_sample_noise_search_ode_divfree(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        rounds=9,
        lambda_div=0.2,
        selector="fid",
        use_global=False,
        use_final_samples_for_restart=False,
    ):
        """
        Multi-round noise search with divergence-free ODE sampling.
        Each round starts from a different time point and uses noise injection
        to explore around the current best candidates.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches (N) to generate per round
            num_keep: Number of candidates (K) to keep per round
            rounds: Number of iterative refinement rounds (default 3)
            lambda_div: Scale factor for the divergence-free field
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            use_final_samples_for_restart: If True, use final samples from t=1.0 as restart points (legacy mode).
                                         If False, use proper intermediate samples (default, correct behavior).
        """
        assert (
            len(class_label) == batch_size if torch.is_tensor(class_label) else True
        ), "class_label tensor length must match batch_size"

        if num_branches == 1 and rounds == 1:
            # If no search needed, just do basic ODE sampling
            return self.batch_sample_ode_divfree(class_label, batch_size, lambda_div)

        score_fn, use_global = self._get_score_function(selector, use_global)
        self.flow_model.eval()

        # Handle both tensor and single class label cases
        if torch.is_tensor(class_label):
            current_label = class_label
        else:
            current_label = torch.full((batch_size,), class_label, device=self.device)

        round_start_times = [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95]

        with torch.no_grad():
            # Initialize candidates: start with random noise for round 1
            current_candidates = []
            for i in range(batch_size):
                # For round 1, start with random initial conditions
                candidates = torch.randn(
                    num_keep,
                    self.channels,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )
                current_candidates.append(candidates)

            # Track top K samples from all rounds for global selection
            all_round_top_samples = []  # List of lists for each batch element
            all_round_top_labels = []  # Corresponding labels

            for i in range(batch_size):
                all_round_top_samples.append([])
                all_round_top_labels.append([])

            # Multi-round noise search
            for round_idx in range(rounds):
                start_time = round_start_times[round_idx]
                print(
                    f"Noise search round {round_idx + 1}/{rounds}, start_time={start_time:.2f}"
                )

                all_round_samples = []
                all_round_labels = []
                all_intermediate_samples = []  # For next round

                # Determine if we need to save intermediate for next round
                next_start_time = None
                if round_idx + 1 < len(round_start_times):
                    next_start_time = round_start_times[round_idx + 1]

                # Generate samples for each batch element
                for batch_idx in range(batch_size):
                    batch_samples = []
                    batch_intermediate = []  # For next round

                    # For each candidate from previous round
                    for candidate_idx in range(num_keep):
                        # Generate num_branches samples from this candidate
                        for branch_idx in range(num_branches):
                            if round_idx == 0:
                                # Round 1: start from random noise at t=0
                                sample_start = torch.randn(
                                    1,
                                    self.channels,
                                    self.image_size,
                                    self.image_size,
                                    device=self.device,
                                )
                            else:
                                # Later rounds: start from intermediate sample at start_time
                                sample_start = current_candidates[batch_idx][
                                    candidate_idx : candidate_idx + 1
                                ]

                            # Sample from start_time to t=1, optionally saving intermediate for next round
                            if (
                                next_start_time is not None
                                and not use_final_samples_for_restart
                            ):
                                # Correct mode: save intermediate samples for next round
                                intermediate_sample, final_sample = (
                                    self._sample_with_divfree_noise(
                                        sample_start,
                                        current_label[batch_idx : batch_idx + 1],
                                        start_time=start_time,
                                        lambda_div=lambda_div,
                                        save_at_time=next_start_time,
                                    )
                                )
                                batch_intermediate.append(intermediate_sample)
                            else:
                                # Legacy mode or last round: just sample normally
                                final_sample = self._sample_with_divfree_noise(
                                    sample_start,
                                    current_label[batch_idx : batch_idx + 1],
                                    start_time=start_time,
                                    lambda_div=lambda_div,
                                )

                            batch_samples.append(final_sample)

                    # Stack samples for this batch element
                    batch_samples = torch.cat(batch_samples, dim=0)
                    all_round_samples.append(batch_samples)

                    # Store intermediate samples for next round candidate selection
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        batch_intermediate = torch.cat(batch_intermediate, dim=0)
                        all_intermediate_samples.append(batch_intermediate)

                    # Create corresponding labels
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_keep * num_branches
                    )
                    all_round_labels.append(batch_labels)

                # Evaluate all samples from this round
                all_samples = torch.cat(all_round_samples, dim=0)
                all_labels = torch.cat(all_round_labels, dim=0)

                if use_global:
                    all_scores = score_fn(all_samples)
                else:
                    all_scores = score_fn(all_samples, all_labels)

                # Select top candidates for next round AND accumulate for global selection
                new_candidates = []
                start_idx = 0

                for batch_idx in range(batch_size):
                    batch_size_this = num_keep * num_branches
                    end_idx = start_idx + batch_size_this

                    batch_scores = all_scores[start_idx:end_idx]
                    batch_samples = all_samples[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]

                    # Keep top num_keep samples for next round
                    top_indices = torch.topk(batch_scores, num_keep).indices
                    top_samples = batch_samples[top_indices]
                    top_labels = batch_labels[top_indices]

                    # For next round: use INTERMEDIATE samples (correct mode) or FINAL samples (legacy mode)
                    if round_idx + 1 < rounds:
                        # Check if we have legacy mode enabled (only applies to noise search functions)
                        use_legacy_mode = (
                            "use_final_samples_for_restart" in locals()
                            and use_final_samples_for_restart
                        )

                        if use_legacy_mode:
                            # Legacy mode: use final samples as restart points
                            new_candidates.append(top_samples)
                        elif all_intermediate_samples:
                            # Correct mode: use intermediate samples
                            batch_intermediates = all_intermediate_samples[batch_idx]
                            top_intermediates = batch_intermediates[top_indices]
                            new_candidates.append(top_intermediates)
                        else:
                            # Fallback: use final samples if no intermediates available
                            new_candidates.append(top_samples)

                    # Accumulate top K samples from this round for global selection
                    all_round_top_samples[batch_idx].append(top_samples)
                    all_round_top_labels[batch_idx].append(top_labels)

                    start_idx = end_idx

                current_candidates = new_candidates

            # Global final selection: select best from ALL rounds' top K samples
            final_samples = []
            for batch_idx in range(batch_size):
                # Concatenate top K samples from all rounds for this batch element
                batch_all_samples = torch.cat(all_round_top_samples[batch_idx], dim=0)
                batch_all_labels = torch.cat(all_round_top_labels[batch_idx], dim=0)

                # Score all accumulated samples
                if use_global:
                    all_candidate_scores = score_fn(batch_all_samples)
                else:
                    all_candidate_scores = score_fn(batch_all_samples, batch_all_labels)

                # Select globally best sample
                best_idx = torch.argmax(all_candidate_scores)
                final_samples.append(batch_all_samples[best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_noise_search_ode_divfree_max(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        rounds=9,
        lambda_div=0.2,
        repulsion_strength=0.02,
        noise_schedule_end_factor=0.5,
        selector="fid",
        use_global=False,
        use_final_samples_for_restart=False,
    ):
        """
        Multi-round noise search with divergence-free max ODE sampling.
        Each round starts from a different time point and uses divfree_max noise injection
        to explore around the current best candidates with repulsion forces.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches (N) to generate per round
            num_keep: Number of candidates (K) to keep per round
            rounds: Number of iterative refinement rounds (default 3)
            lambda_div: Scale factor for the divergence-free field
            repulsion_strength: Strength of repulsion forces between samples
            noise_schedule_end_factor: End factor for time-dependent noise scaling
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            use_final_samples_for_restart: If True, use final samples from t=1.0 as restart points (legacy mode).
                                         If False, use proper intermediate samples (default, correct behavior).
        """
        assert (
            len(class_label) == batch_size if torch.is_tensor(class_label) else True
        ), "class_label tensor length must match batch_size"

        if num_branches == 1 and rounds == 1:
            # If no search needed, just do basic ODE sampling
            return self.batch_sample_ode_divfree(class_label, batch_size, lambda_div)

        score_fn, use_global = self._get_score_function(selector, use_global)
        self.flow_model.eval()

        # Handle both tensor and single class label cases
        if torch.is_tensor(class_label):
            current_label = class_label
        else:
            current_label = torch.full((batch_size,), class_label, device=self.device)

        round_start_times = [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95]

        with torch.no_grad():
            # Initialize candidates: start with random noise for round 1
            current_candidates = []
            for i in range(batch_size):
                # For round 1, start with random initial conditions
                candidates = torch.randn(
                    num_keep,
                    self.channels,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )
                current_candidates.append(candidates)

            # Track top K samples from all rounds for global selection
            all_round_top_samples = []  # List of lists for each batch element
            all_round_top_labels = []  # Corresponding labels

            for i in range(batch_size):
                all_round_top_samples.append([])
                all_round_top_labels.append([])

            # Multi-round noise search
            for round_idx in range(rounds):
                start_time = round_start_times[round_idx]
                print(
                    f"Divfree-max noise search round {round_idx + 1}/{rounds}, start_time={start_time:.2f}"
                )

                all_round_samples = []
                all_round_labels = []
                all_intermediate_samples = []  # For next round

                # Determine if we need to save intermediate for next round
                next_start_time = None
                if round_idx + 1 < len(round_start_times):
                    next_start_time = round_start_times[round_idx + 1]

                # Generate samples for each batch element
                for batch_idx in range(batch_size):
                    batch_samples = []
                    batch_intermediate = []  # For next round

                    # Collect all candidates for this batch element to process together
                    # (this ensures repulsion forces are calculated within the same class)
                    if round_idx == 0:
                        # Round 1: start from random noise at t=0
                        samples_to_process = torch.randn(
                            num_keep * num_branches,
                            self.channels,
                            self.image_size,
                            self.image_size,
                            device=self.device,
                        )
                    else:
                        # Later rounds: expand candidates to create branches
                        candidates = current_candidates[
                            batch_idx
                        ]  # [num_keep, C, H, W]
                        # Repeat each candidate num_branches times
                        samples_to_process = candidates.repeat_interleave(
                            num_branches, dim=0
                        )  # [num_keep*num_branches, C, H, W]

                    # Create labels for all samples in this batch
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_keep * num_branches
                    )

                    # Process all samples for this batch element together to get repulsion
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        # Correct mode: save intermediate samples for next round
                        intermediate_samples, final_samples = (
                            self._sample_with_divfree_max_noise(
                                samples_to_process,
                                batch_labels,
                                start_time=start_time,
                                lambda_div=lambda_div,
                                repulsion_strength=repulsion_strength,
                                noise_schedule_end_factor=noise_schedule_end_factor,
                                save_at_time=next_start_time,
                            )
                        )
                        batch_intermediate.append(intermediate_samples)
                    else:
                        # Legacy mode or last round: just sample normally
                        final_samples = self._sample_with_divfree_max_noise(
                            samples_to_process,
                            batch_labels,
                            start_time=start_time,
                            lambda_div=lambda_div,
                            repulsion_strength=repulsion_strength,
                            noise_schedule_end_factor=noise_schedule_end_factor,
                        )

                    batch_samples.append(final_samples)

                    # Stack samples for this batch element
                    batch_samples = torch.cat(batch_samples, dim=0)
                    all_round_samples.append(batch_samples)

                    # Store intermediate samples for next round candidate selection
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        batch_intermediate = torch.cat(batch_intermediate, dim=0)
                        all_intermediate_samples.append(batch_intermediate)

                    # Create corresponding labels
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_keep * num_branches
                    )
                    all_round_labels.append(batch_labels)

                # Evaluate all samples from this round
                all_samples = torch.cat(all_round_samples, dim=0)
                all_labels = torch.cat(all_round_labels, dim=0)

                if use_global:
                    all_scores = score_fn(all_samples)
                else:
                    all_scores = score_fn(all_samples, all_labels)

                # Select top candidates for next round AND accumulate for global selection
                new_candidates = []
                start_idx = 0

                for batch_idx in range(batch_size):
                    batch_size_this = num_keep * num_branches
                    end_idx = start_idx + batch_size_this

                    batch_scores = all_scores[start_idx:end_idx]
                    batch_samples = all_samples[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]

                    # Keep top num_keep samples for next round
                    top_indices = torch.topk(batch_scores, num_keep).indices
                    top_samples = batch_samples[top_indices]
                    top_labels = batch_labels[top_indices]

                    # For next round: use INTERMEDIATE samples (correct mode) or FINAL samples (legacy mode)
                    if round_idx + 1 < rounds:
                        # Check if we have legacy mode enabled (only applies to noise search functions)
                        use_legacy_mode = (
                            "use_final_samples_for_restart" in locals()
                            and use_final_samples_for_restart
                        )

                        if use_legacy_mode:
                            # Legacy mode: use final samples as restart points
                            new_candidates.append(top_samples)
                        elif all_intermediate_samples:
                            # Correct mode: use intermediate samples
                            batch_intermediates = all_intermediate_samples[batch_idx]
                            top_intermediates = batch_intermediates[top_indices]
                            new_candidates.append(top_intermediates)
                        else:
                            # Fallback: use final samples if no intermediates available
                            new_candidates.append(top_samples)

                    # Accumulate top K samples from this round for global selection
                    all_round_top_samples[batch_idx].append(top_samples)
                    all_round_top_labels[batch_idx].append(top_labels)

                    start_idx = end_idx

                current_candidates = new_candidates

            # Global final selection: select best from ALL rounds' top K samples
            final_samples = []
            for batch_idx in range(batch_size):
                # Concatenate top K samples from all rounds for this batch element
                batch_all_samples = torch.cat(all_round_top_samples[batch_idx], dim=0)
                batch_all_labels = torch.cat(all_round_top_labels[batch_idx], dim=0)

                # Score all accumulated samples
                if use_global:
                    all_candidate_scores = score_fn(batch_all_samples)
                else:
                    all_candidate_scores = score_fn(batch_all_samples, batch_all_labels)

                # Select globally best sample
                best_idx = torch.argmax(all_candidate_scores)
                final_samples.append(batch_all_samples[best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_noise_search_sde(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        rounds=9,
        noise_scale=0.05,
        selector="fid",
        use_global=False,
        use_final_samples_for_restart=False,
    ):
        """
        Multi-round noise search with SDE sampling.
        Each round starts from a different time point and uses SDE noise injection
        to explore around the current best candidates.

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches (N) to generate per round
            num_keep: Number of candidates (K) to keep per round
            rounds: Number of iterative refinement rounds (default 3)
            noise_scale: Scale of the SDE noise to add during sampling
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            use_final_samples_for_restart: If True, use final samples from t=1.0 as restart points (legacy mode).
                                         If False, use proper intermediate samples (default, correct behavior).
        """
        assert (
            len(class_label) == batch_size if torch.is_tensor(class_label) else True
        ), "class_label tensor length must match batch_size"

        if num_branches == 1 and rounds == 1:
            # If no search needed, just do basic SDE sampling
            return self.batch_sample_sde(class_label, batch_size, noise_scale)

        score_fn, use_global = self._get_score_function(selector, use_global)
        self.flow_model.eval()

        # Handle both tensor and single class label cases
        if torch.is_tensor(class_label):
            current_label = class_label
        else:
            current_label = torch.full((batch_size,), class_label, device=self.device)

        round_start_times = [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95]

        with torch.no_grad():
            # Initialize candidates: start with random noise for round 1
            current_candidates = []
            for i in range(batch_size):
                # For round 1, start with random initial conditions
                candidates = torch.randn(
                    num_keep,
                    self.channels,
                    self.image_size,
                    self.image_size,
                    device=self.device,
                )
                current_candidates.append(candidates)

            # Track top K samples from all rounds for global selection
            all_round_top_samples = []  # List of lists for each batch element
            all_round_top_labels = []  # Corresponding labels

            for i in range(batch_size):
                all_round_top_samples.append([])
                all_round_top_labels.append([])

            # Multi-round noise search
            for round_idx in range(rounds):
                start_time = round_start_times[round_idx]
                print(
                    f"Noise search round {round_idx + 1}/{rounds}, start_time={start_time:.2f}"
                )

                all_round_samples = []
                all_round_labels = []
                all_intermediate_samples = []  # For next round

                # Determine if we need to save intermediate for next round
                next_start_time = None
                if round_idx + 1 < len(round_start_times):
                    next_start_time = round_start_times[round_idx + 1]

                # Generate samples for each batch element
                for batch_idx in range(batch_size):
                    batch_samples = []
                    batch_intermediate = []  # For next round

                    # For each candidate from previous round
                    for candidate_idx in range(num_keep):
                        # Generate num_branches samples from this candidate
                        for branch_idx in range(num_branches):
                            if round_idx == 0:
                                # Round 1: start from random noise at t=0
                                sample_start = torch.randn(
                                    1,
                                    self.channels,
                                    self.image_size,
                                    self.image_size,
                                    device=self.device,
                                )
                            else:
                                # Later rounds: start from intermediate sample at start_time
                                sample_start = current_candidates[batch_idx][
                                    candidate_idx : candidate_idx + 1
                                ]

                            # Sample from start_time to t=1, optionally saving intermediate for next round
                            if (
                                next_start_time is not None
                                and not use_final_samples_for_restart
                            ):
                                # Correct mode: save intermediate samples for next round
                                intermediate_sample, final_sample = (
                                    self._sample_with_sde_noise(
                                        sample_start,
                                        current_label[batch_idx : batch_idx + 1],
                                        start_time=start_time,
                                        noise_scale=noise_scale,
                                        save_at_time=next_start_time,
                                    )
                                )
                                batch_intermediate.append(intermediate_sample)
                            else:
                                # Legacy mode or last round: just sample normally
                                final_sample = self._sample_with_sde_noise(
                                    sample_start,
                                    current_label[batch_idx : batch_idx + 1],
                                    start_time=start_time,
                                    noise_scale=noise_scale,
                                )

                            batch_samples.append(final_sample)

                    # Store intermediate samples for next round candidate selection
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        batch_intermediate = torch.cat(batch_intermediate, dim=0)
                        all_intermediate_samples.append(batch_intermediate)

                    # Stack samples for this batch element
                    batch_samples = torch.cat(batch_samples, dim=0)
                    all_round_samples.append(batch_samples)

                    # Create corresponding labels
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_keep * num_branches
                    )
                    all_round_labels.append(batch_labels)

                # Evaluate all samples from this round
                all_samples = torch.cat(all_round_samples, dim=0)
                all_labels = torch.cat(all_round_labels, dim=0)

                if use_global:
                    all_scores = score_fn(all_samples)
                else:
                    all_scores = score_fn(all_samples, all_labels)

                # Select top candidates for next round AND accumulate for global selection
                new_candidates = []
                start_idx = 0

                for batch_idx in range(batch_size):
                    batch_size_this = num_keep * num_branches
                    end_idx = start_idx + batch_size_this

                    batch_scores = all_scores[start_idx:end_idx]
                    batch_samples = all_samples[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]

                    # Keep top num_keep samples for next round
                    top_indices = torch.topk(batch_scores, num_keep).indices
                    top_samples = batch_samples[top_indices]
                    top_labels = batch_labels[top_indices]

                    # For next round: use INTERMEDIATE samples (correct mode) or FINAL samples (legacy mode)
                    if round_idx + 1 < rounds:
                        # Check if we have legacy mode enabled (only applies to noise search functions)
                        use_legacy_mode = (
                            "use_final_samples_for_restart" in locals()
                            and use_final_samples_for_restart
                        )

                        if use_legacy_mode:
                            # Legacy mode: use final samples as restart points
                            new_candidates.append(top_samples)
                        elif all_intermediate_samples:
                            # Correct mode: use intermediate samples
                            batch_intermediates = all_intermediate_samples[batch_idx]
                            top_intermediates = batch_intermediates[top_indices]
                            new_candidates.append(top_intermediates)
                        else:
                            # Fallback: use final samples if no intermediates available
                            new_candidates.append(top_samples)

                    # Accumulate top K samples from this round for global selection
                    all_round_top_samples[batch_idx].append(top_samples)
                    all_round_top_labels[batch_idx].append(top_labels)

                    start_idx = end_idx

                current_candidates = new_candidates

            # Global final selection: select best from ALL rounds' top K samples
            final_samples = []
            for batch_idx in range(batch_size):
                # Concatenate top K samples from all rounds for this batch element
                batch_all_samples = torch.cat(all_round_top_samples[batch_idx], dim=0)
                batch_all_labels = torch.cat(all_round_top_labels[batch_idx], dim=0)

                # Score all accumulated samples
                if use_global:
                    all_candidate_scores = score_fn(batch_all_samples)
                else:
                    all_candidate_scores = score_fn(batch_all_samples, batch_all_labels)

                # Select globally best sample
                best_idx = torch.argmax(all_candidate_scores)
                final_samples.append(batch_all_samples[best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_random_search_then_noise_search_ode_divfree(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        rounds=9,
        lambda_div=0.2,
        selector="fid",
        use_global=False,
        use_final_samples_for_restart=False,
    ):
        """
        Two-stage inference scaling method that combines random search with noise search:
        1. First stage: Run random search to find high-scoring initial noise samples
        2. Second stage: Use the winning initial noises from stage 1 as starting points for multi-round noise search

        Args:
            class_label: Target class(es) to generate. Can be a single integer or a tensor of class labels.
            batch_size: Number of samples to generate
            num_branches: Number of branches for both the random search and noise search stages
            num_keep: Number of samples to keep during noise search rounds
            rounds: Number of noise search refinement rounds (default 3)
            lambda_div: Scale factor for the divergence-free field in noise search
            selector: Selection criteria - "fid", "mahalanobis", "mean", "inception_score", "dino_score"
            use_global: Whether to use global statistics instead of class-specific ones
            use_final_samples_for_restart: If True, use final samples from t=1.0 as restart points (legacy mode).
                                         If False, use proper intermediate samples (default, correct behavior).
        """
        assert (
            len(class_label) == batch_size if torch.is_tensor(class_label) else True
        ), "class_label tensor length must match batch_size"

        if num_branches == 1:
            # If no search needed, just do noise search
            return self.batch_sample_noise_search_ode_divfree(
                class_label,
                batch_size,
                num_branches,
                num_keep,
                rounds,
                lambda_div,
                selector,
                use_global,
                use_final_samples_for_restart,
            )

        score_fn, use_global = self._get_score_function(selector, use_global)
        self.flow_model.eval()

        # Handle both tensor and single class label cases
        if torch.is_tensor(class_label):
            current_label = class_label
        else:
            current_label = torch.full((batch_size,), class_label, device=self.device)

        round_start_times = [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95]

        with torch.no_grad():
            print(f"Stage 1: Random search with {num_branches} branches")

            # Stage 1: Random search to find good initial conditions
            all_samples = []
            all_labels = []

            for batch_idx in range(batch_size):
                batch_samples = []
                for branch_idx in range(num_branches):
                    # Generate sample from random initial condition
                    sample = self.batch_sample_ode_divfree(
                        current_label[batch_idx], 1, lambda_div
                    )
                    batch_samples.append(sample)

                batch_samples = torch.cat(batch_samples, dim=0)
                all_samples.append(batch_samples)

                # Create corresponding labels
                batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                    num_branches
                )
                all_labels.append(batch_labels)

            # Evaluate all samples from stage 1
            all_samples = torch.cat(all_samples, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            if use_global:
                all_scores = score_fn(all_samples)
            else:
                all_scores = score_fn(all_samples, all_labels)

            # Select top initial conditions for stage 2
            top_initial_conditions = []
            start_idx = 0

            for batch_idx in range(batch_size):
                end_idx = start_idx + num_branches

                batch_scores = all_scores[start_idx:end_idx]

                # Keep top num_keep samples as initial conditions
                top_indices = torch.topk(
                    batch_scores, min(num_keep, num_branches)
                ).indices

                # Get the corresponding initial noise (we need to regenerate or store)
                # For simplicity, we'll use the top samples as our starting points
                # In practice, we'd want to store the initial noise that led to these samples
                selected_samples = all_samples[start_idx:end_idx][top_indices]
                top_initial_conditions.append(selected_samples)

                start_idx = end_idx

            print(
                f"Stage 2: Noise search with {rounds} rounds from top initial conditions"
            )

            # Stage 2: Multi-round noise search starting from the best initial conditions
            current_candidates = []
            for batch_idx in range(batch_size):
                # Convert final samples back to noise space (approximation)
                # In practice, we'd store the actual initial noise from stage 1
                # For now, we'll use the selected samples as "candidates"
                candidates = top_initial_conditions[batch_idx]
                current_candidates.append(candidates)

            # Track top K samples from all rounds for global selection
            all_round_top_samples = []  # List of lists for each batch element
            all_round_top_labels = []  # Corresponding labels

            for i in range(batch_size):
                all_round_top_samples.append([])
                all_round_top_labels.append([])

            # Run noise search rounds (similar to batch_sample_noise_search_ode_divfree)
            for round_idx in range(rounds):
                start_time = round_start_times[round_idx]
                print(
                    f"Noise search round {round_idx + 1}/{rounds}, start_time={start_time:.2f}"
                )

                all_round_samples = []
                all_round_labels = []
                all_intermediate_samples = []  # For next round

                # Determine if we need to save intermediate for next round
                next_start_time = None
                if round_idx + 1 < len(round_start_times):
                    next_start_time = round_start_times[round_idx + 1]

                # Generate samples for each batch element
                for batch_idx in range(batch_size):
                    batch_samples = []
                    batch_intermediate = []  # For next round

                    # For each candidate from previous round
                    num_candidates = current_candidates[batch_idx].shape[0]
                    for candidate_idx in range(num_candidates):
                        # Generate num_branches samples from this candidate
                        for branch_idx in range(num_branches):
                            if round_idx == 0:
                                # Round 1: start from the winning initial conditions
                                # Generate new random noise since we want to explore around winners
                                sample_start = torch.randn(
                                    1,
                                    self.channels,
                                    self.image_size,
                                    self.image_size,
                                    device=self.device,
                                )
                            else:
                                # Later rounds: start from previous candidate at start_time
                                sample_start = current_candidates[batch_idx][
                                    candidate_idx : candidate_idx + 1
                                ]

                            # Sample from start_time to t=1, optionally saving intermediate for next round
                            if (
                                next_start_time is not None
                                and not use_final_samples_for_restart
                            ):
                                # Correct mode: save intermediate samples for next round
                                intermediate_sample, final_sample = (
                                    self._sample_with_divfree_noise(
                                        sample_start,
                                        current_label[batch_idx : batch_idx + 1],
                                        start_time=start_time,
                                        lambda_div=lambda_div,
                                        save_at_time=next_start_time,
                                    )
                                )
                                batch_intermediate.append(intermediate_sample)
                            else:
                                # Legacy mode or last round: just sample normally
                                final_sample = self._sample_with_divfree_noise(
                                    sample_start,
                                    current_label[batch_idx : batch_idx + 1],
                                    start_time=start_time,
                                    lambda_div=lambda_div,
                                )

                            batch_samples.append(final_sample)

                    # Stack samples for this batch element
                    batch_samples = torch.cat(batch_samples, dim=0)
                    all_round_samples.append(batch_samples)

                    # Store intermediate samples for next round candidate selection
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        batch_intermediate = torch.cat(batch_intermediate, dim=0)
                        all_intermediate_samples.append(batch_intermediate)

                    # Create corresponding labels
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_candidates * num_branches
                    )
                    all_round_labels.append(batch_labels)

                # Evaluate all samples from this round
                all_samples = torch.cat(all_round_samples, dim=0)
                all_labels = torch.cat(all_round_labels, dim=0)

                if use_global:
                    all_scores = score_fn(all_samples)
                else:
                    all_scores = score_fn(all_samples, all_labels)

                # Select top candidates for next round AND accumulate for global selection
                new_candidates = []
                start_idx = 0

                for batch_idx in range(batch_size):
                    num_candidates = current_candidates[batch_idx].shape[0]
                    batch_size_this = num_candidates * num_branches
                    end_idx = start_idx + batch_size_this

                    batch_scores = all_scores[start_idx:end_idx]
                    batch_samples = all_samples[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]

                    # Keep top num_keep samples for next round
                    top_indices = torch.topk(batch_scores, num_keep).indices
                    top_samples = batch_samples[top_indices]
                    top_labels = batch_labels[top_indices]

                    # For next round: use INTERMEDIATE samples (correct mode) or FINAL samples (legacy mode)
                    if round_idx + 1 < rounds:
                        # Check if we have legacy mode enabled (only applies to noise search functions)
                        use_legacy_mode = (
                            "use_final_samples_for_restart" in locals()
                            and use_final_samples_for_restart
                        )

                        if use_legacy_mode:
                            # Legacy mode: use final samples as restart points
                            new_candidates.append(top_samples)
                        elif all_intermediate_samples:
                            # Correct mode: use intermediate samples
                            batch_intermediates = all_intermediate_samples[batch_idx]
                            top_intermediates = batch_intermediates[top_indices]
                            new_candidates.append(top_intermediates)
                        else:
                            # Fallback: use final samples if no intermediates available
                            new_candidates.append(top_samples)

                    # Accumulate top K samples from this round for global selection
                    all_round_top_samples[batch_idx].append(top_samples)
                    all_round_top_labels[batch_idx].append(top_labels)

                    start_idx = end_idx

                current_candidates = new_candidates

            # Global final selection: select best from ALL rounds' top K samples
            final_samples = []
            for batch_idx in range(batch_size):
                # Concatenate top K samples from all rounds for this batch element
                batch_all_samples = torch.cat(all_round_top_samples[batch_idx], dim=0)
                batch_all_labels = torch.cat(all_round_top_labels[batch_idx], dim=0)

                # Score all accumulated samples
                if use_global:
                    all_candidate_scores = score_fn(batch_all_samples)
                else:
                    all_candidate_scores = score_fn(batch_all_samples, batch_all_labels)

                # Select globally best sample
                best_idx = torch.argmax(all_candidate_scores)
                final_samples.append(batch_all_samples[best_idx])

            return self.unnormalize_images(torch.stack(final_samples))

    def batch_sample_random_search_then_noise_search_ode_divfree_max(
        self,
        class_label,
        batch_size=16,
        num_branches=4,
        num_keep=2,
        rounds=9,
        lambda_div=0.2,
        repulsion_strength=0.02,
        noise_schedule_end_factor=0.5,
        selector="fid",
        use_global=False,
        use_final_samples_for_restart=False,
    ):
        """
        Two-stage inference scaling method that combines random search with divfree_max noise search:
        1. First stage: Run random search to find high-scoring initial noise samples
        2. Second stage: Use the winning initial noises from stage 1 as starting points for multi-round divfree_max noise search
        """
        assert (
            len(class_label) == batch_size if torch.is_tensor(class_label) else True
        ), "class_label tensor length must match batch_size"

        if num_branches == 1:
            # If no search needed, just do divfree_max noise search
            return self.batch_sample_noise_search_ode_divfree_max(
                class_label,
                batch_size,
                num_branches,
                num_keep,
                rounds,
                lambda_div,
                repulsion_strength,
                noise_schedule_end_factor,
                selector,
                use_global,
                use_final_samples_for_restart,
            )

        score_fn, use_global = self._get_score_function(selector, use_global)
        self.flow_model.eval()

        # Handle both tensor and single class label cases
        if torch.is_tensor(class_label):
            current_label = class_label
        else:
            current_label = torch.full((batch_size,), class_label, device=self.device)

        round_start_times = [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95]

        with torch.no_grad():
            print(f"Stage 1: Random search with {num_branches} branches")

            # Stage 1: Random search to find good initial conditions
            all_samples = []
            all_labels = []

            for batch_idx in range(batch_size):
                batch_samples = []
                for branch_idx in range(num_branches):
                    # Generate sample from random initial condition
                    sample = self.batch_sample_ode_divfree(
                        current_label[batch_idx], 1, lambda_div
                    )
                    batch_samples.append(sample)

                batch_samples = torch.cat(batch_samples, dim=0)
                all_samples.append(batch_samples)

                # Create corresponding labels
                batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                    num_branches
                )
                all_labels.append(batch_labels)

            # Evaluate all samples from stage 1
            all_samples = torch.cat(all_samples, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            if use_global:
                all_scores = score_fn(all_samples)
            else:
                all_scores = score_fn(all_samples, all_labels)

            # Select top initial conditions for stage 2
            top_initial_conditions = []
            start_idx = 0

            for batch_idx in range(batch_size):
                end_idx = start_idx + num_branches

                batch_scores = all_scores[start_idx:end_idx]

                # Keep top num_keep samples as initial conditions
                top_indices = torch.topk(
                    batch_scores, min(num_keep, num_branches)
                ).indices

                # Get the corresponding initial noise (we need to regenerate or store)
                selected_samples = all_samples[start_idx:end_idx][top_indices]
                top_initial_conditions.append(selected_samples)

                start_idx = end_idx

            print(
                f"Stage 2: Divfree-max noise search with {rounds} rounds from top initial conditions"
            )

            # Stage 2: Multi-round divfree_max noise search starting from the best initial conditions
            current_candidates = []
            for batch_idx in range(batch_size):
                candidates = top_initial_conditions[batch_idx]
                current_candidates.append(candidates)

            # Track top K samples from all rounds for global selection
            all_round_top_samples = []
            all_round_top_labels = []

            for i in range(batch_size):
                all_round_top_samples.append([])
                all_round_top_labels.append([])

            # Run divfree_max noise search rounds
            for round_idx in range(rounds):
                start_time = round_start_times[round_idx]
                print(
                    f"Divfree-max noise search round {round_idx + 1}/{rounds}, start_time={start_time:.2f}"
                )

                all_round_samples = []
                all_round_labels = []
                all_intermediate_samples = []  # For next round

                # Determine if we need to save intermediate for next round
                next_start_time = None
                if round_idx + 1 < len(round_start_times):
                    next_start_time = round_start_times[round_idx + 1]

                # Generate samples for each batch element
                for batch_idx in range(batch_size):
                    batch_samples = []
                    batch_intermediate = []  # For next round

                    # For each candidate from previous round
                    num_candidates = current_candidates[batch_idx].shape[0]

                    # Collect all samples for this batch element to process together
                    # (this ensures repulsion forces are calculated within the same class)
                    if round_idx == 0:
                        # Round 1: start from the winning initial conditions, but generate new samples for exploration
                        samples_to_process = torch.randn(
                            num_candidates * num_branches,
                            self.channels,
                            self.image_size,
                            self.image_size,
                            device=self.device,
                        )
                    else:
                        # Later rounds: expand candidates to create branches
                        candidates = current_candidates[batch_idx]
                        samples_to_process = candidates.repeat_interleave(
                            num_branches, dim=0
                        )

                    # Create labels for all samples in this batch
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_candidates * num_branches
                    )

                    # Sample from start_time to t=1, optionally saving intermediate for next round
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        # Correct mode: save intermediate samples for next round
                        intermediate_samples, final_samples = (
                            self._sample_with_divfree_max_noise(
                                samples_to_process,
                                batch_labels,
                                start_time=start_time,
                                lambda_div=lambda_div,
                                repulsion_strength=repulsion_strength,
                                noise_schedule_end_factor=noise_schedule_end_factor,
                                save_at_time=next_start_time,
                            )
                        )
                        batch_intermediate.append(intermediate_samples)
                    else:
                        # Legacy mode or last round: just sample normally
                        final_samples = self._sample_with_divfree_max_noise(
                            samples_to_process,
                            batch_labels,
                            start_time=start_time,
                            lambda_div=lambda_div,
                            repulsion_strength=repulsion_strength,
                            noise_schedule_end_factor=noise_schedule_end_factor,
                        )

                    batch_samples.append(final_samples)

                    # Stack samples for this batch element
                    batch_samples = torch.cat(batch_samples, dim=0)
                    all_round_samples.append(batch_samples)

                    # Store intermediate samples for next round candidate selection
                    if (
                        next_start_time is not None
                        and not use_final_samples_for_restart
                    ):
                        batch_intermediate = torch.cat(batch_intermediate, dim=0)
                        all_intermediate_samples.append(batch_intermediate)

                    # Create corresponding labels
                    batch_labels = current_label[batch_idx : batch_idx + 1].repeat(
                        num_candidates * num_branches
                    )
                    all_round_labels.append(batch_labels)

                # Evaluate all samples from this round
                all_samples = torch.cat(all_round_samples, dim=0)
                all_labels = torch.cat(all_round_labels, dim=0)

                if use_global:
                    all_scores = score_fn(all_samples)
                else:
                    all_scores = score_fn(all_samples, all_labels)

                # Select top candidates for next round AND accumulate for global selection
                new_candidates = []
                start_idx = 0

                for batch_idx in range(batch_size):
                    num_candidates = current_candidates[batch_idx].shape[0]
                    batch_size_this = num_candidates * num_branches
                    end_idx = start_idx + batch_size_this

                    batch_scores = all_scores[start_idx:end_idx]
                    batch_samples = all_samples[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]

                    # Keep top num_keep samples for next round
                    top_indices = torch.topk(batch_scores, num_keep).indices
                    top_samples = batch_samples[top_indices]
                    top_labels = batch_labels[top_indices]

                    # For next round: use INTERMEDIATE samples (correct mode) or FINAL samples (legacy mode)
                    if round_idx + 1 < rounds:
                        # Check if we have legacy mode enabled (only applies to noise search functions)
                        use_legacy_mode = (
                            "use_final_samples_for_restart" in locals()
                            and use_final_samples_for_restart
                        )

                        if use_legacy_mode:
                            # Legacy mode: use final samples as restart points
                            new_candidates.append(top_samples)
                        elif all_intermediate_samples:
                            # Correct mode: use intermediate samples
                            batch_intermediates = all_intermediate_samples[batch_idx]
                            top_intermediates = batch_intermediates[top_indices]
                            new_candidates.append(top_intermediates)
                        else:
                            # Fallback: use final samples if no intermediates available
                            new_candidates.append(top_samples)

                    # Accumulate top K samples from this round for global selection
                    all_round_top_samples[batch_idx].append(top_samples)
                    all_round_top_labels[batch_idx].append(top_labels)

                    start_idx = end_idx

                current_candidates = new_candidates

            # Global final selection: select best from ALL rounds' top K samples
            final_samples = []
            for batch_idx in range(batch_size):
                # Concatenate top K samples from all rounds for this batch element
                batch_all_samples = torch.cat(all_round_top_samples[batch_idx], dim=0)
                batch_all_labels = torch.cat(all_round_top_labels[batch_idx], dim=0)

                # Score all accumulated samples
                if use_global:
                    all_candidate_scores = score_fn(batch_all_samples)
                else:
                    all_candidate_scores = score_fn(batch_all_samples, batch_all_labels)

                # Select globally best sample
                best_idx = torch.argmax(all_candidate_scores)
                final_samples.append(batch_all_samples[best_idx])

            return self.unnormalize_images(torch.stack(final_samples))
