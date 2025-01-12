import torch
import torch.utils.data as data
import torchdiffeq
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchcfm.models.unet import UNetModel
from torchcfm.models.unet import SigmaConditionedUNet
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from train_mnist_classifier import MNISTClassifier


class TrajectoryBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.trajectories = []  # List of (state, step, final_score) tuples

    def add_trajectory(self, states, scores):
        """
        Add a complete trajectory and its final score
        states: list of tensors for each step
        scores: final scores for each branch
        """
        for branch_idx in range(len(scores)):
            branch_states = [step_states[branch_idx] for step_states in states]
            for step, state in enumerate(branch_states):
                self.trajectories.append((state.cpu(), step, scores[branch_idx].cpu()))

        # Trim buffer if needed
        if len(self.trajectories) > self.max_size:
            self.trajectories = self.trajectories[-self.max_size :]

    def sample_batch(self, batch_size):
        """Sample a random batch of (state, step, score) tuples"""
        if len(self.trajectories) < batch_size:
            return None

        indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        batch = [self.trajectories[i] for i in indices]

        states = torch.stack([b[0] for b in batch])
        steps = torch.tensor([b[1] for b in batch])
        scores = torch.tensor([b[2] for b in batch])

        return states, steps, scores


class MCTSFlowSampler:
    def __init__(
        self,
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        device="cuda",
        num_noise_levels=5,
    ):
        self.device = device
        self.num_noise_levels = num_noise_levels
        self.noise_levels = self.generate_noise_levels(num_noise_levels)

        # Initialize models
        self.flow_model = SigmaConditionedUNet(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_classes=10,
            class_cond=True,
        ).to(device)

        self.value_model = ValueModel(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_classes=10,
            class_cond=True,
        ).to(device)

        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters())
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters())

        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

        self.trajectory_buffer = TrajectoryBuffer()

        # MNIST classifier
        self.classifier = MNISTClassifier().to(device)
        self.load_classifier()
        self.classifier.eval()

    def load_classifier(self, path="saved_models/mnist_classifier.pt"):
        """Load pre-trained MNIST classifier."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No pre-trained classifier found at {path}. "
                f"Please run train_mnist_classifier.py first."
            )
        self.classifier.load_state_dict(torch.load(path))
        print(f"Loaded pre-trained classifier from {path}")

    def generate_noise_levels(self, num_levels=5):
        x = np.exp(np.linspace(np.log(1e-4), np.log(1.0), num_levels))
        return np.flip(x).tolist()

    def simulate_to_end(self, x, y, current_step):
        """Simulate the flow process to the end from a given point."""
        self.flow_model.eval()
        with torch.no_grad():
            current_samples = x
            for step in range(current_step, len(self.noise_levels) - 1):
                current_sigma = torch.full(
                    (len(current_samples),), self.noise_levels[step], device=self.device
                )
                traj = torchdiffeq.odeint(
                    lambda t, x: self.flow_model(t, x, y, sigma=current_sigma),
                    current_samples,
                    torch.linspace(0, 1, 2, device=self.device),
                    atol=1e-4,
                    rtol=1e-4,
                )
                current_samples = traj[-1]
            return current_samples

    def compute_sample_quality(self, samples, target_labels):
        """
        Compute quality score using MNIST classifier confidence
        samples: tensor of shape [batch_size, 1, 28, 28]
        target_labels: tensor of shape [batch_size]
        returns: tensor of shape [batch_size] with scores between 0 and 1
        """
        self.classifier.eval()
        with torch.no_grad():
            # Get classifier predictions
            logits = self.classifier(samples)
            # Get probability for target class
            target_probs = logits[torch.arange(len(samples)), target_labels]
            return target_probs

    def save_models(self, path="saved_models"):
        """Save flow and value models separately."""
        os.makedirs(path, exist_ok=True)

        # Save flow model
        flow_path = f"{path}/flow_model.pt"
        torch.save(
            {
                "model": self.flow_model.state_dict(),
                "noise_levels": self.noise_levels,
            },
            flow_path,
        )
        print(f"Flow model saved to {flow_path}")

        # Save value model
        value_path = f"{path}/value_model.pt"
        torch.save(
            {
                "model": self.value_model.state_dict(),
            },
            value_path,
        )
        print(f"Value model saved to {value_path}")

    def load_models(self, path="saved_models"):
        """Load flow and value models if they exist."""
        flow_path = f"{path}/flow_model.pt"
        value_path = f"{path}/value_model.pt"

        flow_exists = os.path.exists(flow_path)
        value_exists = os.path.exists(value_path)

        if flow_exists:
            checkpoint = torch.load(flow_path, weights_only=True)
            self.flow_model.load_state_dict(checkpoint["model"])
            self.noise_levels = checkpoint["noise_levels"]
            print(f"Flow model loaded from {flow_path}")

        if value_exists:
            checkpoint = torch.load(value_path, weights_only=True)
            self.value_model.load_state_dict(checkpoint["model"])
            print(f"Value model loaded from {value_path}")

        return flow_exists, value_exists

    def train(self, train_loader, n_epochs=3):
        """Train both models, skipping if saved versions exist."""
        flow_exists, value_exists = self.load_models()

        print("Noise levels:", [f"{x:.4f}" for x in self.noise_levels])

        # Train flow model if needed
        if not flow_exists:
            print("Training flow model...")
            self.train_flow_model(train_loader, n_epochs)
            self.save_models()  # Save after flow training
        else:
            print("Using pre-trained flow model.")

        # Train value model if needed
        if not value_exists:
            print("\nTraining value model...")
            self.train_value_model(n_epochs)
            self.save_models()  # Save after value training
        else:
            print("Using pre-trained value model.")

    def train_flow_model(self, train_loader, n_epochs):
        for epoch in range(n_epochs):
            print(f"\nFlow Model - Epoch {epoch + 1}/{n_epochs}")
            epoch_flow_losses = {i: [] for i in range(len(self.noise_levels) - 1)}

            pbar = tqdm(train_loader)
            for batch_idx, (x1, y) in enumerate(pbar):
                x1, y = x1.to(self.device), y.to(self.device)

                # Train flow model
                transition_idx = (
                    0
                    if torch.rand(1).item() < 0.75
                    else (1 + (batch_idx % (len(self.noise_levels) - 2)))
                )

                current_sigma = self.noise_levels[transition_idx]
                next_sigma = self.noise_levels[transition_idx + 1]

                x0 = (
                    torch.randn_like(x1)
                    if current_sigma == 1.0
                    else x1 + torch.randn_like(x1) * current_sigma
                )
                x_target = x1 + torch.randn_like(x1) * next_sigma

                self.flow_optimizer.zero_grad()
                t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x_target)
                sigma = torch.full((x1.shape[0],), current_sigma, device=self.device)
                vt = self.flow_model(t, xt, y, sigma=sigma)
                flow_loss = torch.mean((vt - ut) ** 2)
                flow_loss.backward()
                self.flow_optimizer.step()

                epoch_flow_losses[transition_idx].append(flow_loss.item())
                pbar.set_postfix({"flow_loss": f"{flow_loss.item():.4f}"})

                # Generate trajectories for value model training
                if batch_idx % 5 == 0:  # Generate trajectories periodically
                    with torch.no_grad():
                        trajectories, scores = self.generate_training_trajectory(
                            y[:2]
                        )  # Use smaller batch
                        self.trajectory_buffer.add_trajectory(trajectories, scores)

    def train_value_model(self, n_epochs, batch_size=64):
        for epoch in range(n_epochs):
            print(f"\nValue Model - Epoch {epoch + 1}/{n_epochs}")
            epoch_value_losses = []

            n_batches = len(self.trajectory_buffer.trajectories) // batch_size
            pbar = tqdm(range(n_batches))

            for _ in pbar:
                batch = self.trajectory_buffer.sample_batch(batch_size)
                if batch is None:
                    continue

                states, steps, labels, scores = batch
                states = states.to(self.device)
                labels = labels.to(self.device)
                scores = scores.to(self.device)

                self.value_optimizer.zero_grad()
                value_pred = self.value_model(
                    torch.ones(len(states), device=self.device),
                    states,
                    labels,  # Use actual labels now
                )

                value_loss = F.mse_loss(value_pred, scores)
                value_loss.backward()
                self.value_optimizer.step()

                epoch_value_losses.append(value_loss.item())
                pbar.set_postfix({"value_loss": f"{value_loss.item():.4f}"})

    def generate_training_trajectory(self, y, num_branches=5):
        """Generate a complete trajectory for training the value model"""
        num_steps = len(self.noise_levels) - 1
        trajectories = []

        with torch.no_grad():
            # Initialize samples and labels
            current_samples = torch.randn(
                len(y) * num_branches, 1, 28, 28, device=self.device
            )
            current_labels = y.repeat_interleave(num_branches)

            trajectories.append(current_samples.clone())

            # Generate trajectory
            for step in range(num_steps):
                current_sigma = torch.full(
                    (len(current_samples),), self.noise_levels[step], device=self.device
                )

                # Flow model step
                traj = torchdiffeq.odeint(
                    lambda t, x: self.flow_model(
                        t, x, current_labels, sigma=current_sigma
                    ),
                    current_samples,
                    torch.linspace(0, 1, 2, device=self.device),
                    atol=1e-4,
                    rtol=1e-4,
                )
                current_samples = traj[-1]
                trajectories.append(current_samples.clone())

                # Add noise for next step (except last step)
                if step < num_steps - 1:
                    noise_scale = 0.1
                    perturbations = torch.randn_like(current_samples) * noise_scale
                    current_samples = current_samples + perturbations

            # Compute final quality scores using classifier
            final_scores = self.compute_sample_quality(current_samples, current_labels)

        return trajectories, final_scores

    def sample(self, num_samples, class_labels, num_branches=5, num_keep=5):
        """Sample using flow model with MCTS-guided selection and branching."""
        num_steps = len(self.noise_levels) - 1
        self.flow_model.eval()
        self.value_model.eval()

        intermediate_samples = []

        with torch.no_grad():
            # Initialize samples
            initial_count = num_samples * num_branches
            current_samples = torch.randn(initial_count, 1, 28, 28, device=self.device)
            current_labels = class_labels.repeat_interleave(num_branches)

            intermediate_samples.append(current_samples[:8].clone())

            # Generate samples
            for step in tqdm(range(num_steps), desc="Generating samples"):
                current_sigma = torch.full(
                    (len(current_samples),), self.noise_levels[step], device=self.device
                )

                # Flow model step
                traj = torchdiffeq.odeint(
                    lambda t, x: self.flow_model(
                        t, x, current_labels, sigma=current_sigma
                    ),
                    current_samples,
                    torch.linspace(0, 1, 2, device=self.device),
                    atol=1e-4,
                    rtol=1e-4,
                )
                generated = traj[-1]

                # Get value predictions
                value_scores = self.value_model(
                    torch.ones(len(generated), device=self.device),
                    generated,
                    current_labels,
                )

                # Select top samples
                top_k_values, top_k_indices = torch.topk(
                    value_scores, k=num_keep, dim=0
                )

                # Update current samples and labels
                current_samples = generated[top_k_indices]
                intermediate_samples.append(current_samples[:8].clone())
                current_labels = current_labels[top_k_indices]

                # Branch for next iteration (except last step)
                if step < num_steps - 1:
                    noise_scale = 0.1
                    current_samples = current_samples.repeat_interleave(
                        num_branches, dim=0
                    )
                    current_labels = current_labels.repeat_interleave(num_branches)

                    perturbations = torch.randn_like(current_samples) * noise_scale
                    current_samples = current_samples + perturbations

            return current_samples, current_labels

    def save_models(self, path="saved_mcts_model"):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "flow_model": self.flow_model.state_dict(),
                "value_model": self.value_model.state_dict(),
                "noise_levels": self.noise_levels,
            },
            f"{path}/models.pt",
        )
        print(f"Models saved to {path}/models.pt")

    def load_models(self, path="saved_mcts_model"):
        model_path = f"{path}/models.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=True)
            self.flow_model.load_state_dict(checkpoint["flow_model"])
            self.value_model.load_state_dict(checkpoint["value_model"])
            self.noise_levels = checkpoint["noise_levels"]
            print(f"Models loaded from {model_path}")
            return True
        return False


class ValueModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_layer = torch.nn.Conv2d(self.out_channels, 1, 1)

    def forward(self, t, x, y):
        features = super().forward(t, x, y)
        return torch.sigmoid(self.final_layer(features).mean(dim=[1, 2, 3]))
