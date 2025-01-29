import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader, Dataset
from mcts_single_flow import MCTSFlowSampler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class SyntheticRewardNet(nn.Module):
    """Complex neural network that provides a reward signal for testing"""

    def __init__(self, input_dim, hidden_dims=[128, 256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Create a network with different activation functions per layer
        activations = [
            nn.ReLU(),
            nn.Tanh(),
            nn.SiLU(),  # Also known as Swish
            nn.GELU(),
        ]

        for i, dim in enumerate(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    activations[i % len(activations)],
                    nn.LayerNorm(dim),
                ]
            )
            prev_dim = dim

        # Final layer to scalar output
        layers.extend(
            [nn.Linear(prev_dim, 1), nn.Sigmoid()]  # Normalize output to [0,1]
        )

        self.net = nn.Sequential(*layers)

        # Initialize with random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)


class HarderSyntheticRewardNet(nn.Module):
    """
    A more complex, random network that maps R^input_dim -> [0,1].
    Designed to be harder for a typical feed-forward net to learn exactly.
    """

    def __init__(
        self,
        input_dim,
        fourier_dim=256,
        hidden_dims=[512, 256, 512, 128],
        num_branches=4,
    ):
        super().__init__()

        # 1) Random Fourier Features for high-frequency components
        #    (We freeze these after random init.)
        #    Transform x -> [cos(Wx + b); sin(Wx + b)] of dimension 2*fourier_dim.
        self.fourier_dim = fourier_dim
        self.W = nn.Parameter(
            torch.randn(input_dim, fourier_dim) * 2.0, requires_grad=False
        )
        self.b = nn.Parameter(torch.randn(fourier_dim) * 3.14, requires_grad=False)

        # 2) Multi-branch sub-networks (each random, then combined)
        #    We'll store them in a ModuleList for flexible creation.
        self.branches = nn.ModuleList()
        for _ in range(num_branches):
            branch_layers = []
            prev_dim = 2 * fourier_dim  # after we map input -> Fourier features
            activations = [nn.ReLU(), nn.Tanh(), nn.SiLU(), nn.GELU()]

            for i, dim in enumerate(hidden_dims):
                branch_layers.append(nn.Linear(prev_dim, dim))
                # pick an activation randomly or cycle through
                branch_layers.append(activations[i % len(activations)])
                # optional layer norm
                branch_layers.append(nn.LayerNorm(dim))
                prev_dim = dim

            # final linear (branch output dimension)
            branch_layers.append(nn.Linear(prev_dim, 64))  # bigger intermediate feature
            self.branches.append(nn.Sequential(*branch_layers))

        # 3) Gating / Attention-like aggregator
        #    We'll produce a gating vector from the combined branch outputs,
        #    then do a weighted sum -> single scalar -> sigmoid.
        agg_in_dim = num_branches * 64
        self.agg_gate = nn.Linear(agg_in_dim, num_branches)  # produce branch-gating
        self.agg_final = nn.Linear(64, 1)  # final linear to scalar
        self.sigmoid = nn.Sigmoid()

        # 4) Random initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0, 0.01)

    def forward(self, x):
        # x: [batch_size, input_dim]
        # 1) Fourier features
        #    phi(x) = [ cos(xW + b), sin(xW + b) ]
        #    shape = [batch_size, 2 * fourier_dim]
        with torch.no_grad():
            # (batch_size, fourier_dim)
            proj = x @ self.W + self.b
        # cos_proj, sin_proj each [batch_size, fourier_dim]
        cos_proj = torch.cos(proj)
        sin_proj = torch.sin(proj)
        fourier_feats = torch.cat([cos_proj, sin_proj], dim=-1)

        # 2) Pass through each branch
        branch_outputs = []
        for branch in self.branches:
            out_b = branch(fourier_feats)
            branch_outputs.append(out_b)

        # Concatenate all branch outputs: shape = [batch_size, num_branches * 64]
        concat = torch.cat(branch_outputs, dim=-1)

        # 3) Gating
        # gate_scores: [batch_size, num_branches]
        gate_scores = self.agg_gate(concat)
        # softmax over branches dimension
        gate_weights = F.softmax(gate_scores, dim=-1)  # [batch_size, num_branches]

        # reshape for broadcasting
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_branches, 1]

        # chunk the concat again so we can do gating
        # each chunk is [batch_size, 64]
        chunked = torch.chunk(concat, len(self.branches), dim=-1)

        # Weighted sum of branch outputs
        # stack them: shape = [batch_size, num_branches, 64]
        stacked = torch.stack(chunked, dim=1)
        # multiply by gate_weights -> sum across branch dimension
        gated_sum = (stacked * gate_weights).sum(dim=1)  # [batch_size, 64]

        # 4) Final linear -> scalar -> [0,1]
        out = self.agg_final(gated_sum)  # [batch_size, 1]
        reward = self.sigmoid(out).squeeze(-1)  # [batch_size]

        return reward


class SyntheticDataset(Dataset):
    """Dataset for flow matching with synthetic data"""

    def __init__(self, n_samples, input_dim, num_classes):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate target points by sampling from a mixture of Gaussians
        # conditioned on the class label
        self.targets = []
        self.labels = []

        for _ in range(n_samples):
            label = torch.randint(0, num_classes, (1,)).item()

            # Create class-conditional mean
            mean = torch.zeros(input_dim)
            # Use label to influence the mean vector in a structured way
            mean[
                label
                * (input_dim // num_classes) : (label + 1)
                * (input_dim // num_classes)
            ] = 1.0

            # Add some randomness to create a mixture
            noise = torch.randn(input_dim) * 0.1
            target = mean + noise

            self.targets.append(target)
            self.labels.append(label)

        self.targets = torch.stack(self.targets)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.targets[idx], self.labels[idx]


def evaluate_with_viz(sampler, num_samples=1, branch_keep_pairs=None, num_classes=10):
    """Evaluate sample quality and visualize results."""
    if branch_keep_pairs is None:
        branch_keep_pairs = [(3, 2), (8, 3), (16, 7)]

    # Choose a random class label for this evaluation
    class_label = torch.randint(0, num_classes, (1,)).item()
    print(f"\nEvaluation for digit {class_label}:")

    for num_branches, num_keep in branch_keep_pairs:
        print(f"\nTesting with branches={num_branches}, keep={num_keep}")

        all_samples = []
        all_scores = []

        # Generate multiple samples for the same class
        for _ in range(num_samples):
            # Generate single sample
            sample = sampler.simple_sample(
                class_label=class_label,
                num_branches=num_branches,
                num_keep=num_keep,
            )

            # Compute classifier confidence
            with torch.no_grad():
                confidence_score = sampler.compute_sample_quality(
                    sample.unsqueeze(0),
                    torch.tensor([class_label], device=sampler.device),
                )

            all_samples.append(sample)
            all_scores.append(confidence_score.item())

        # Convert to tensors
        samples = torch.stack(all_samples)
        scores = torch.tensor(all_scores)

        # Report statistics
        mean_confidence = scores.mean().item()
        std_confidence = scores.std().item()
        print(f"Mean confidence: {mean_confidence:.4f} ± {std_confidence:.4f}")

        # Visualize samples
        plt.figure(figsize=(15, 6))
        grid = make_grid(samples, nrow=10, normalize=True, padding=2)
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.title(
            f"Digit {class_label} Samples (branches={num_branches}, keep={num_keep})"
        )
        plt.axis("off")
        plt.show()


def visualize_samples(samples, title="Generated Samples"):
    """Visualize a grid of generated samples."""
    plt.figure(figsize=(15, 6))
    grid = make_grid(samples, nrow=10, normalize=True, padding=2)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()


def evaluate_synthetic_samples(
    sampler, num_samples=10, branch_keep_pairs=None, num_classes=10
):
    """Evaluate sample quality for synthetic data across all classes"""
    if branch_keep_pairs is None:
        branch_keep_pairs = [(3, 2), (8, 3), (16, 7)]

    print("\nEvaluating samples:")

    for num_branches, num_keep in branch_keep_pairs:
        print(f"\nTesting with branches={num_branches}, keep={num_keep}")

        # Store scores for each class
        class_scores = []

        # Sample from each class
        for class_label in range(num_classes):
            all_scores = []

            # Generate multiple samples for this class
            for _ in range(num_samples):
                sample = sampler.simple_sample(
                    class_label=class_label,
                    num_branches=num_branches,
                    num_keep=num_keep,
                )

                with torch.no_grad():
                    score = sampler.compute_sample_quality(
                        sample.unsqueeze(0),
                        torch.tensor([class_label], device=sampler.device),
                    )
                all_scores.append(score.item())

            # Add average score for this class
            class_scores.append(torch.tensor(all_scores).mean().item())

        # Compute overall statistics
        overall_mean = sum(class_scores) / num_classes
        overall_std = torch.tensor(class_scores).std().item()
        print(
            f"Average score across all classes: {overall_mean:.4f} ± {overall_std:.4f}"
        )


def analyze_reward_distribution(reward_net, input_dim, num_classes, num_samples=1000):
    """Analyze the distribution of rewards within each class using SyntheticDataset"""
    print("\nAnalyzing reward distribution across classes:")

    device = next(reward_net.parameters()).device
    reward_net.eval()

    # Create a synthetic dataset with enough samples to ensure good representation of each class
    dataset = SyntheticDataset(
        n_samples=num_samples * num_classes,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    # Organize samples by class
    class_samples = [[] for _ in range(num_classes)]
    for target, label in zip(dataset.targets, dataset.labels):
        class_samples[label].append(target)

    class_stats = []
    overall_rewards = []

    with torch.no_grad():
        for class_label in range(num_classes):
            # Get samples for this class
            samples = torch.stack(class_samples[class_label]).to(device)

            # Compute rewards
            rewards = reward_net(samples)

            # Compute statistics
            mean_reward = rewards.mean().item()
            std_reward = rewards.std().item()
            min_reward = rewards.min().item()
            max_reward = rewards.max().item()

            class_stats.append(
                {
                    "mean": mean_reward,
                    "std": std_reward,
                    "min": min_reward,
                    "max": max_reward,
                    "n_samples": len(samples),
                }
            )
            overall_rewards.extend(rewards.cpu().tolist())

            print(f"\nClass {class_label} (n={len(samples)}):")
            print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
            print(f"Range: [{min_reward:.4f}, {max_reward:.4f}]")

    # Compute overall statistics
    overall_rewards = torch.tensor(overall_rewards)
    overall_mean = overall_rewards.mean().item()
    overall_std = overall_rewards.std().item()

    print("\nOverall Statistics:")
    print(f"Mean reward across all classes: {overall_mean:.4f} ± {overall_std:.4f}")

    # Compute class separability
    means = torch.tensor([stats["mean"] for stats in class_stats])
    mean_diff = (means.unsqueeze(0) - means.unsqueeze(1)).abs()
    avg_class_separation = mean_diff[mean_diff > 0].mean().item()
    print(f"Average separation between class means: {avg_class_separation:.4f}")

    # Compute average within-class variance
    avg_within_class_std = sum(stats["std"] for stats in class_stats) / num_classes
    print(f"Average within-class standard deviation: {avg_within_class_std:.4f}")

    return class_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Simplified data dimensions for testing
    input_dim = 1024
    num_classes = 100

    # Create synthetic dataset
    train_dataset = SyntheticDataset(
        n_samples=10000, input_dim=input_dim, num_classes=num_classes
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    reward_net = SyntheticRewardNet(input_dim).to(device)

    # analyze_reward_distribution(reward_net, input_dim, num_classes)

    # Initialize sampler with simplified dimensions
    sampler = MCTSFlowSampler(
        dim=input_dim,
        device=device,
        num_timesteps=10,
        num_classes=num_classes,
        reward_net=reward_net,
    )

    # Training configuration
    n_epochs_per_cycle = 1
    n_training_cycles = 20
    branch_keep_pairs = [(1, 1), (2, 1), (3, 2), (8, 3), (16, 7)]

    # Training loop with periodic evaluation
    for cycle in range(n_training_cycles):
        print(f"\nTraining Cycle {cycle + 1}/{n_training_cycles}")

        sampler.train(
            train_loader,
            n_epochs=n_epochs_per_cycle,
            initial_flow_epochs=5,
            value_epochs=50,
        )

        # Evaluate
        evaluate_synthetic_samples(
            sampler, branch_keep_pairs=branch_keep_pairs, num_classes=num_classes
        )


if __name__ == "__main__":
    main()
