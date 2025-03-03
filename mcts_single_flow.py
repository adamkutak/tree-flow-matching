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

        # Initialize inception model for FID computation
        self.inception = InceptionV3([0], normalize_input=True).to(device)
        self.inception.eval()

        # Load reference statistics for CIFAR-10
        with open("cifar10_fid_stats_64dim.pkl", "rb") as f:
            cifar_stats = pickle.load(f)

        # Initialize per-class FID metrics and buffers
        self.fids = {
            i: {"mu": None, "sigma": None, "features": deque(maxlen=buffer_size)}
            for i in range(num_classes)
        }

        for class_idx in range(num_classes):
            self.fids[class_idx]["mu"] = cifar_stats[f"class_{class_idx}_mu"]
            self.fids[class_idx]["sigma"] = cifar_stats[f"class_{class_idx}_sigma"]

        print("Initializing per-class buffers...")
        self.initialize_class_buffers(buffer_size)

    def initialize_class_buffers(self, n_samples):
        """Initialize buffers for each class with generated samples."""
        self.flow_model.eval()
        with torch.no_grad():
            samples_per_class = n_samples
            for class_idx in range(self.num_classes):
                # Generate samples for this specific class
                labels = torch.full((samples_per_class,), class_idx, device=self.device)
                x = torch.randn(
                    samples_per_class,
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
                        (samples_per_class,), t.item(), device=self.device
                    )
                    velocity = self.flow_model(t_batch, x, labels)
                    x = x + velocity * dt

                # Extract and store features for this class
                features = self.extract_inception_features(x)
                self.fids[class_idx]["features"].extend(list(features))

                print(
                    f"Class {class_idx} initialized with {len(self.fids[class_idx]['features'])} samples"
                )

        for class_idx in range(self.num_classes):
            feats = np.array(list(self.fids[class_idx]["features"]))
            mu = np.mean(feats, axis=0)
            sigma = np.cov(feats, rowvar=False)
            baseline_fid = self.calculate_frechet_distance(
                mu, sigma, self.fids[class_idx]["mu"], self.fids[class_idx]["sigma"]
            )
            self.fids[class_idx]["baseline_fid"] = baseline_fid

    def extract_inception_features(self, images):
        """Extract inception features using pytorch-fid."""
        # Resize images to inception input size
        images = F.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Move images to [-1, 1] range as expected by inception
        images = images * 2 - 1

        with torch.no_grad():
            features = self.inception(images)[0]
            # Global average pooling if features have spatial dimensions
            if len(features.shape) > 2:
                features = features.mean([2, 3])
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

    def batch_sample_wdt(
        self, class_label, batch_size=16, num_branches=4, num_keep=2, dt_std=0.1
    ):
        """
        Efficient batched sampling method that maintains constant number of samples per batch element.
        Creates branches by varying dt instead of adding noise.
        Args:
            class_label: Target class to generate
            batch_size: Number of final samples to generate
            num_branches: Number of branches per batch element (constant throughout)
            num_keep: Number of samples to keep before expansion
            dt_std: Standard deviation for sampling different dt values
        Returns:
            Tensor of shape [batch_size, C, H, W]
        """
        if num_branches == 1 and num_keep == 1:
            return self.regular_batch_sample(class_label, batch_size)

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

            # Track current time for each sample
            current_times = torch.zeros(batch_size * num_branches, device=self.device)

            # Track which batch element each sample belongs to
            batch_indices = torch.arange(
                batch_size, device=self.device
            ).repeat_interleave(num_branches)

            base_dt = 1 / self.num_timesteps
            # Generate samples with branching
            while torch.any(current_times < 1.0):
                # Flow step - process all samples in one batch
                velocity = self.flow_model(
                    current_times, current_samples, current_label
                )

                # Sample different dt values for each sample
                dts = torch.normal(
                    mean=base_dt,
                    std=dt_std * base_dt,
                    size=(len(current_samples),),
                    device=self.device,
                )
                dts = torch.clamp(
                    dts,
                    min=torch.tensor(0.0, device=self.device),
                    max=1.0 - current_times,
                )

                # Apply different step sizes to create branches
                generated = current_samples + velocity * dts.view(-1, 1, 1, 1)
                new_times = current_times + dts

                # Get value predictions for all samples at once
                value_scores = self.value_model(new_times, generated, current_label)

                # Select top num_keep samples for each batch element
                selected_samples = []
                selected_times = []

                for batch_idx in range(batch_size):
                    # Get samples for this batch element
                    batch_mask = batch_indices == batch_idx
                    batch_samples = generated[batch_mask]
                    batch_scores = value_scores[batch_mask]
                    batch_times = new_times[batch_mask]

                    # Select top num_keep samples
                    top_k_values, top_k_indices = torch.topk(
                        batch_scores, k=num_keep, dim=0
                    )
                    selected_samples.append(batch_samples[top_k_indices])
                    selected_times.append(batch_times[top_k_indices])

                # Stack selected samples and times
                current_samples = torch.cat(
                    selected_samples, dim=0
                )  # shape: [batch_size * num_keep, C, H, W]
                current_times = torch.cat(
                    selected_times, dim=0
                )  # shape: [batch_size * num_keep]

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

            # Final selection - take best sample from each batch element's num_branches samples
            final_samples = []
            for batch_idx in range(batch_size):
                batch_mask = batch_indices == batch_idx
                batch_samples = current_samples[batch_mask]
                batch_scores = value_scores[batch_mask]
                best_idx = torch.argmax(batch_scores)
                final_samples.append(batch_samples[best_idx])

            return torch.stack(final_samples)  # shape: [batch_size, C, H, W]

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

    def calculate_frechet_distance_components(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate the components of the Frechet distance between two distributions.

        Returns:
            mean_distance: The squared distance between means
            cov_distance: The trace term for covariance matrices
            covmean_term: The trace of the square root of product of covariances
        """
        # Mean term: ||μ1 - μ2||^2
        mean_distance = np.sum((mu1 - mu2) ** 2)

        # Covariance terms: Tr(Σ1) + Tr(Σ2) - 2*Tr(sqrt(Σ1*Σ2))
        trace_sigma1 = np.trace(sigma1)
        trace_sigma2 = np.trace(sigma2)

        # Calculate sqrt(Σ1*Σ2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 2*Tr(sqrt(Σ1*Σ2))
        covmean_term = 2 * np.trace(covmean)

        # Tr(Σ1) + Tr(Σ2)
        cov_distance = trace_sigma1 + trace_sigma2

        return mean_distance, cov_distance, covmean_term

    def analyze_fid_components(self, generated_samples, class_idx):
        """
        Analyze the components of the FID score for a set of generated samples.

        Args:
            generated_samples: Tensor of generated images
            class_idx: Class index for comparison

        Returns:
            Dictionary with FID components
        """
        # Extract features from generated samples
        features = self.extract_inception_features(generated_samples)

        # Calculate statistics for generated samples
        mu_gen = np.mean(features, axis=0)
        sigma_gen = np.cov(features, rowvar=False)

        # Get reference statistics for this class
        mu_ref = self.fids[class_idx]["mu"]
        sigma_ref = self.fids[class_idx]["sigma"]

        # Calculate FID components
        mean_dist, cov_dist, covmean_term = self.calculate_frechet_distance_components(
            mu_gen, sigma_gen, mu_ref, sigma_ref
        )

        # Calculate total FID
        fid_score = mean_dist + cov_dist - covmean_term

        return {
            "fid_total": fid_score,
            "mean_distance": mean_dist,
            "covariance_distance": cov_dist,
            "covmean_term": covmean_term,
            "mean_distance_percent": (
                100 * mean_dist / fid_score if fid_score > 0 else 0
            ),
            "covariance_percent": (
                100 * (cov_dist - covmean_term) / fid_score if fid_score > 0 else 0
            ),
        }
