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
)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from train_mnist_classifier import MNISTClassifier


class TrajectoryBuffer:
    def __init__(self, max_size=10000):
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
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        device="cuda:0",
        num_timesteps=10,
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

        # Initialize models
        self.flow_model = UNetModel(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_classes=10,
            class_cond=True,
        ).to(self.device)

        # Use the ValueModel directly instead of separate value_head
        self.value_model = ValueModel(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_classes=10,
            class_cond=True,
        ).to(self.device)

        # Initialize MNIST classifier for rewards
        self.classifier = MNISTClassifier().to(self.device)
        self.load_classifier()
        self.classifier.eval()

        # Initialize optimizers
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters())
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters())

        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        self.trajectory_buffer = TrajectoryBuffer()

        # Try to load pre-trained models
        if self.load_models():
            print("Successfully loaded pre-trained flow and value models")
        else:
            print("No pre-trained models found, starting from scratch")

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

    def compute_sample_quality(self, samples, target_labels):
        """Compute quality score using MNIST classifier confidence."""
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(samples)
            target_probs = logits[torch.arange(len(samples)), target_labels]
            return target_probs

    def generate_training_trajectory(self, y, num_branches=5):
        """Generate a complete trajectory for training the value model."""
        trajectories = []
        ts = []

        with torch.no_grad():
            # Initialize samples
            current_samples = torch.randn(
                len(y) * num_branches, 1, 28, 28, device=self.device
            )
            current_labels = y.repeat_interleave(num_branches)

            trajectories.append(current_samples.clone())
            ts.append(0.0)

            # Generate trajectory with branching
            for step, t in enumerate(self.timesteps[:-1]):
                # Flow model step
                dt = self.timesteps[step + 1] - t

                # Euler step
                velocity = self.flow_model(
                    torch.full_like(t, t, device=self.device),
                    current_samples,
                    current_labels,
                )
                current_samples = current_samples + velocity * dt

                trajectories.append(current_samples.clone())
                ts.append(float(self.timesteps[step + 1]))

                # Add noise for branching (except at last step)
                if step < len(self.timesteps) - 2:
                    noise_scale = 0.1 * (1 - float(t))  # Decrease noise over time
                    perturbations = torch.randn_like(current_samples) * noise_scale
                    current_samples = current_samples + perturbations

            # Compute final quality scores
            final_scores = self.compute_sample_quality(current_samples, current_labels)

        return trajectories, ts, current_labels, final_scores

    def train(self, train_loader, n_epochs=3, initial_flow_epochs=10, value_epochs=4):
        """Train both flow and value models."""
        # Initial flow model training
        print("\nInitial flow model training...")
        for epoch in range(initial_flow_epochs):
            print(f"\nInitial flow epoch {epoch + 1}/{initial_flow_epochs}")
            flow_loss = self.train_flow_matching(
                train_loader,
                desc=f"Initial flow training {epoch + 1}/{initial_flow_epochs}",
            )

        # Main training loop
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Train flow model for one epoch
            flow_loss = self.train_flow_matching(train_loader)

            # Generate trajectories
            print("Generating trajectories for value training...")
            self.flow_model.eval()  # Set to eval mode for trajectory generation
            self.trajectory_buffer = (
                TrajectoryBuffer()
            )  # Reset buffer for new trajectories
            with torch.no_grad():
                for batch_idx, (_, y) in enumerate(
                    tqdm(train_loader, desc="Generating trajectories")
                ):
                    y = y.to(self.device)
                    trajectories, ts, labels, scores = (
                        self.generate_training_trajectory(y)
                    )
                    self.trajectory_buffer.add_trajectory(
                        trajectories, ts, labels, scores
                    )

            # Train value model for multiple epochs
            print(f"Training value model for {value_epochs} epochs...")
            for value_epoch in range(value_epochs):
                self.train_value_model(
                    n_epochs=1,
                    batch_size=64,
                    desc=f"Value epoch {value_epoch + 1}/{value_epochs}",
                )

            # Save after each main epoch
            self.save_models()

    def train_flow_matching(self, train_loader, desc="Training flow"):
        """Train flow model for one epoch."""
        self.flow_model.train()
        pbar = tqdm(train_loader, desc=desc)
        for batch_idx, (x1, y) in enumerate(pbar):
            x1, y = x1.to(self.device), y.to(self.device)
            x0 = torch.randn_like(x1)

            # Train flow matching
            self.flow_optimizer.zero_grad()
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
            vt = self.flow_model(t, xt, y)
            flow_loss = torch.mean((vt - ut) ** 2)
            flow_loss.backward()
            self.flow_optimizer.step()

            pbar.set_postfix({"flow_loss": f"{flow_loss.item():.4f}"})

        return flow_loss.item()

    def train_value_model(self, n_epochs=1, batch_size=64, desc="Training value"):
        """Train value model on collected trajectories."""
        self.value_model.train()

        for epoch in range(n_epochs):
            n_batches = len(self.trajectory_buffer.trajectories) // batch_size
            pbar = tqdm(range(n_batches), desc=desc)

            total_loss = 0
            for _ in pbar:
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
                pbar.set_postfix({"value_loss": f"{value_loss.item():.4f}"})

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"Average value loss: {avg_loss:.4f}")

    def sample(self, class_label, num_branches=5, num_keep=5):
        """
        Sample a single image for a given class label using MCTS-guided selection.
        Args:
            class_label: Single integer label (0-9)
            num_branches: Number of branches to create at each step
            num_keep: Number of samples to keep at each step
        Returns:
            Single best final sample
        """
        self.flow_model.eval()
        self.value_model.eval()

        with torch.no_grad():
            # Initialize samples
            current_samples = torch.randn(num_branches, 1, 28, 28, device=self.device)
            current_label = torch.full((num_branches,), class_label, device=self.device)

            # Generate samples with branching
            for step, t in enumerate(tqdm(self.timesteps[:-1], desc="Generating")):
                dt = self.timesteps[step + 1] - t

                # Flow step
                velocity = self.flow_model(
                    torch.full_like(t, t, device=self.device),
                    current_samples,
                    current_label,
                )
                generated = current_samples + velocity * dt

                # Get value predictions
                value_scores = self.value_model(
                    torch.full((len(generated),), float(t), device=self.device),
                    generated,
                    current_label,
                )

                # Select top samples
                top_k_values, top_k_indices = torch.topk(
                    value_scores, k=num_keep, dim=0
                )
                current_samples = generated[top_k_indices]
                current_label = current_label[top_k_indices]

                # Branch for next iteration (except last step)
                if step < len(self.timesteps) - 2:
                    noise_scale = 0.1 * (1 - float(t))  # Decrease noise over time
                    current_samples = current_samples.repeat_interleave(
                        num_branches, dim=0
                    )
                    current_label = current_label[:num_keep].repeat_interleave(
                        num_branches
                    )

                    perturbations = torch.randn_like(current_samples) * noise_scale
                    current_samples = current_samples + perturbations

            # Select final best sample using value model
            final_values = self.value_model(
                torch.ones(len(current_samples), device=self.device),
                current_samples,
                current_label,
            )
            best_idx = final_values.argmax()
            return current_samples[best_idx]

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

    def load_models(self, path="saved_models"):
        """Load flow and value models if they exist."""
        flow_path = f"{path}/single_flow_model.pt"
        value_path = f"{path}/single_value_model.pt"

        flow_exists = os.path.exists(flow_path)
        value_exists = os.path.exists(value_path)

        if flow_exists:
            checkpoint = torch.load(
                flow_path, weights_only=True, map_location=self.device
            )
            self.flow_model.load_state_dict(checkpoint["model"])
            print(f"Flow model loaded from {flow_path}")

        if value_exists:
            checkpoint = torch.load(
                value_path, weights_only=True, map_location=self.device
            )
            self.value_model.load_state_dict(checkpoint["model"])
            print(f"Value model loaded from {value_path}")

        return flow_exists and value_exists
