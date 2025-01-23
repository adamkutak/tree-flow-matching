import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader, Dataset
from mcts_single_flow import MCTSFlowSampler
import numpy as np
from tqdm import tqdm
from train_cifar100_classifier import CIFAR100Classifier


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
            sample = sampler.sample(
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
                sample = sampler.sample(
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

    # Initialize sampler with simplified dimensions
    sampler = MCTSFlowSampler(
        dim=input_dim,
        device=device,
        num_timesteps=10,
        num_classes=num_classes,
    )

    # Training configuration
    n_epochs_per_cycle = 1
    n_training_cycles = 20
    branch_keep_pairs = [(1, 1), (2, 1), (3, 2), (8, 3)]

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
