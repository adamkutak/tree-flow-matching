import torch
import torch.utils.data as data
import torchdiffeq
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from tree_flow_matching import SigmaConditionedUNet
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


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

        # Initialize optimizers
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters())
        self.policy_optimizer = torch.optim.Adam(self.value_model.parameters())

        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    def save_models(self, path="saved_mcts_model"):
        """Save both flow and policy models."""
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "flow_model": self.flow_model.state_dict(),
                "policy_model": self.value_model.state_dict(),
                "noise_levels": self.noise_levels,
            },
            f"{path}/models.pt",
        )
        print(f"Models saved to {path}/models.pt")

    def load_models(self, path="saved_mcts_model"):
        """Load both flow and policy models if they exist."""
        model_path = f"{path}/models.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=True)
            self.flow_model.load_state_dict(checkpoint["flow_model"])
            self.value_model.load_state_dict(checkpoint["policy_model"])
            self.noise_levels = checkpoint["noise_levels"]
            print(f"Models loaded from {model_path}")
            return True
        return False

    def generate_noise_levels(self, num_levels=5):
        """Generate noise levels with more density near 0."""
        x = np.exp(np.linspace(np.log(1e-4), np.log(1.0), num_levels))
        return np.flip(x).tolist()

    def train(self, train_loader, n_epochs=3):
        """Train both flow and policy models."""
        if self.load_models():
            print("Using pre-trained models. Skipping training.")
            return

        noise_levels = self.generate_noise_levels(self.num_noise_levels)
        print("Noise levels:", [f"{x:.4f}" for x in noise_levels])

        # Track losses for each transition
        flow_losses = {i: [] for i in range(len(noise_levels) - 1)}
        policy_losses = []

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            epoch_flow_losses = {i: [] for i in range(len(noise_levels) - 1)}
            epoch_policy_losses = []

            pbar = tqdm(train_loader)
            for batch_idx, (x1, y) in enumerate(pbar):
                x1, y = x1.to(self.device), y.to(self.device)

                # Modified transition selection: 75% chance to train on first transition
                if torch.rand(1).item() < 0.75:
                    transition_idx = 0  # First transition (noise 1.0)
                else:
                    # Randomly select from other transitions
                    transition_idx = 1 + (batch_idx % (len(noise_levels) - 2))

                current_sigma = noise_levels[transition_idx]
                next_sigma = noise_levels[transition_idx + 1]

                # Generate start and end points for this transition
                if current_sigma == 1.0:
                    x0 = torch.randn_like(x1)
                else:
                    x0 = x1 + torch.randn_like(x1) * current_sigma
                x_target = x1 + torch.randn_like(x1) * next_sigma

                # Train flow matching
                self.flow_optimizer.zero_grad()
                t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x_target)

                # Forward pass with current sigma conditioning
                sigma = torch.full((x1.shape[0],), current_sigma, device=self.device)
                vt = self.flow_model(t, xt, y, sigma=sigma)

                # Compute flow loss and update
                flow_loss = torch.mean((vt - ut) ** 2)
                flow_loss.backward()
                self.flow_optimizer.step()

                # Train policy model
                self.policy_optimizer.zero_grad()
                noise_amounts = torch.rand(
                    x1.shape[0], device=self.device
                )  # One noise per image
                noisy_x1 = x1 + torch.randn_like(x1) * noise_amounts.view(-1, 1, 1, 1)
                predicted_noise = self.value_model(
                    torch.ones(x1.shape[0], device=self.device), noisy_x1, y
                )
                policy_loss = F.mse_loss(predicted_noise, noise_amounts)
                policy_loss.backward()
                self.policy_optimizer.step()

                # Record losses
                epoch_flow_losses[transition_idx].append(flow_loss.item())
                epoch_policy_losses.append(policy_loss.item())

                # Update progress bar
                pbar.set_postfix(
                    {
                        "transition": f"{current_sigma:.5f}->{next_sigma:.5f}",
                        "flow_loss": f"{flow_loss.item():.4f}",
                        "policy_loss": f"{policy_loss.item():.4f}",
                    }
                )

            # Log average losses for this epoch
            print("\nFlow model transitions:")
            for idx in epoch_flow_losses:
                if len(epoch_flow_losses[idx]) > 0:
                    avg_loss = sum(epoch_flow_losses[idx]) / len(epoch_flow_losses[idx])
                    flow_losses[idx].append(avg_loss)
                    print(
                        f"  {noise_levels[idx]:.5f}->{noise_levels[idx+1]:.5f} "
                        f"average loss: {avg_loss:.4f}"
                    )

            avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
            policy_losses.append(avg_policy_loss)
            print(f"Policy model average loss: {avg_policy_loss:.4f}")

        self.save_models()

    def sample(
        self,
        num_samples,
        class_labels,
        num_branches=5,
        num_keep=5,
    ):
        """Sample using flow model with policy-guided selection and branching."""
        num_steps = len(self.noise_levels) - 1
        self.flow_model.eval()
        self.value_model.eval()

        # Store intermediate results
        intermediate_samples = []

        with torch.no_grad():
            # Start with initial samples (num_samples * num_branches)
            initial_count = num_samples * num_branches
            current_samples = torch.randn(initial_count, 1, 28, 28, device=self.device)
            # Store initial noisy samples
            intermediate_samples.append(
                current_samples[:8].clone()
            )  # Store first 8 samples

            # Repeat class labels for each branch
            current_labels = class_labels.repeat_interleave(num_branches)
            current_sigmas = torch.ones(initial_count, device=self.device)

            pbar = tqdm(range(num_steps), desc="Generating samples")
            for step in pbar:
                # 1. Simulate ODE with current noise level
                current_sigma = torch.full(
                    (len(current_samples),), self.noise_levels[step], device=self.device
                )
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

                # 2. Get policy scores
                noise_estimates = self.value_model(
                    torch.ones(len(generated), device=self.device),
                    generated,
                    current_labels,
                )

                # 3. Select top samples
                top_k_values, top_k_indices = torch.topk(
                    -noise_estimates,  # Negative because we want lowest noise
                    k=num_keep,
                    dim=0,
                )

                # Keep best samples and their labels
                current_samples = generated[top_k_indices]
                intermediate_samples.append(
                    current_samples[:8].clone()
                )  # Store first 8 samples
                current_labels = current_labels[top_k_indices]
                current_sigmas = current_sigma[top_k_indices]

                # 4. Generate branches for next iteration (except on last step)
                if step < num_steps - 1:
                    # Use a small fixed noise scale for perturbations (e.g., 0.1 or 0.01)
                    # noise_scale = 0.5 * self.noise_levels[step]
                    noise_scale = 0.1
                    # Expand samples
                    current_samples = current_samples.repeat_interleave(
                        num_branches, dim=0
                    )
                    current_labels = current_labels.repeat_interleave(num_branches)
                    current_sigmas = current_sigmas.repeat_interleave(num_branches)

                    perturbations = torch.randn_like(current_samples) * noise_scale
                    current_samples = current_samples + perturbations

            # Display progress grid
            # plt.figure(figsize=(12, 2 * len(intermediate_samples)))
            # for i, samples in enumerate(intermediate_samples):
            #     plt.subplot(len(intermediate_samples), 1, i + 1)
            #     grid = make_grid(samples, nrow=8, normalize=True)
            #     plt.imshow(grid.cpu().permute(1, 2, 0))
            #     plt.title(
            #         f"Step {i}, Noise Level: {self.noise_levels[min(i, len(self.noise_levels)-1)]:.4f}"
            #     )
            #     plt.axis("off")
            # plt.tight_layout()
            # plt.show()
            return current_samples, current_labels


class ValueModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_layer = torch.nn.Conv2d(self.out_channels, 1, 1)

    def forward(self, t, x, y):
        features = super().forward(t, x, y)
        return self.final_layer(features).mean(dim=[1, 2, 3])
