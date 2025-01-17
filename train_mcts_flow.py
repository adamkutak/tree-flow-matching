import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from mcts_single_flow import MCTSFlowSampler
import numpy as np
from tqdm import tqdm
from train_cifar100_classifier import CIFAR100Classifier


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    num_classes = 100

    # Load and initialize classifier
    classifier = CIFAR100Classifier().to(device)
    classifier_path = "saved_models/cifar100_classifier.pt"
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"No pre-trained classifier found at {classifier_path}. "
            f"Please run train_cifar100_classifier.py first."
        )
    classifier.load_state_dict(
        torch.load(classifier_path, weights_only=True, map_location=device)
    )
    print(f"Loaded pre-trained classifier from {classifier_path}")

    # Updated transform for CIFAR-100
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )

    subset_size = None
    if subset_size:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Initialize sampler with CIFAR-100 dimensions
    sampler = MCTSFlowSampler(
        dim=(3, 32, 32),  # CIFAR-100 dimensions
        device=device,
        num_timesteps=10,
        num_classes=100,  # CIFAR-100 has 100 classes
        num_channels=64,
        num_res_blocks=2,
        classifier=classifier,
    )
    # Training configuration
    n_epochs_per_cycle = 1
    n_training_cycles = 20
    branch_keep_pairs = [(1, 1), (8, 3), (16, 7)]

    # Training loop with periodic evaluation
    for cycle in range(n_training_cycles):
        print(f"\nTraining Cycle {cycle + 1}/{n_training_cycles}")

        # Train for n epochs
        sampler.train(
            train_loader,
            n_epochs=n_epochs_per_cycle,
            initial_flow_epochs=5,
            value_epochs=50,
        )

        # Evaluate
        evaluate_with_viz(
            sampler, branch_keep_pairs=branch_keep_pairs, num_classes=num_classes
        )


if __name__ == "__main__":
    main()
