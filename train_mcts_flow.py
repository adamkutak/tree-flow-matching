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
from fid_is_rewards import FIDISRewardNet


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


def evaluate_samples(sampler, num_samples=10, branch_keep_pairs=None, num_classes=100):
    """Evaluate sample quality for CIFAR-100 data"""
    if branch_keep_pairs is None:
        branch_keep_pairs = [(3, 2), (8, 3), (16, 7)]

    print("\nEvaluating samples:")

    for num_branches, num_keep in branch_keep_pairs:
        print(f"\nTesting with branches={num_branches}, keep={num_keep}")

        # Generate samples for a few random classes
        test_classes = np.random.choice(num_classes, 5, replace=False)
        for class_label in test_classes:
            samples = []
            scores = []

            # Generate multiple samples for this class
            for _ in range(num_samples):
                sample = sampler.simple_sample(
                    class_label=class_label,
                    num_branches=num_branches,
                    num_keep=num_keep,
                )
                samples.append(sample)

                with torch.no_grad():
                    score = sampler.compute_sample_quality(
                        sample.unsqueeze(0),
                        torch.tensor([class_label], device=sampler.device),
                    )
                scores.append(score.item())

            # Visualize samples
            samples_grid = torch.stack(samples)
            visualize_samples(
                samples_grid,
                f"Class {class_label} (branches={num_branches}, keep={num_keep})",
            )

            # Print statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(
                f"Class {class_label} - Mean score: {mean_score:.4f} ± {std_score:.4f}"
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # CIFAR-100 dimensions and setup
    image_size = 32
    channels = 3
    num_classes = 100

    # Setup CIFAR-100 dataset with appropriate transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    # Take only 1000 samples for faster training/testing
    subset_indices = range(1000)  # You can adjust this number
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    # Initialize reward network
    reward_net = FIDISRewardNet().to(device)

    # Initialize sampler with CIFAR-100 dimensions
    sampler = MCTSFlowSampler(
        image_size=image_size,
        channels=channels,
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
            initial_flow_epochs=1,
            value_epochs=1,
            flow_epochs=1,
        )

        # Evaluate
        evaluate_samples(
            sampler, branch_keep_pairs=branch_keep_pairs, num_classes=num_classes
        )


if __name__ == "__main__":
    main()
