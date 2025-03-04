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
import torchmetrics.image.fid as FID
import torchmetrics.image.inception as IS
from torchvision import datasets, transforms


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


def visualize_samples(all_samples_dict, class_label, real_images, figsize=(15, 12)):
    """
    Visualize samples from different branch/keep configurations in a single grid.
    Each row represents a different configuration, with real images in the first row.

    all_samples_dict: Dictionary with (branch, keep) tuples as keys and samples as values
    real_images: Tensor of real images to show in the first row
    """
    plt.figure(figsize=figsize)

    # Combine all samples into a single grid, with real images as first row
    all_samples_list = [real_images]  # Start with real images
    row_labels = ["Real Images"]  # First row label

    for (branches, keep), samples in all_samples_dict.items():
        all_samples_list.append(samples)
        row_labels.append(f"branches={branches}, keep={keep}")

    # Stack all samples into a single tensor
    all_samples = torch.cat(all_samples_list, dim=0)

    # Create grid with samples from each configuration in separate rows
    nrow = all_samples_list[0].size(0)  # number of samples per configuration
    grid = make_grid(all_samples, nrow=nrow, normalize=True, padding=2)

    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.title(f"Generated Samples for Class {class_label}")

    # Add row labels on the left side
    num_configs = len(row_labels)
    for idx, label in enumerate(row_labels):
        plt.text(
            -10,
            (idx + 0.5) * grid.size(1) / num_configs,
            label,
            rotation=0,
            verticalalignment="center",
        )

    plt.axis("off")
    plt.show()


# def evaluate_samples(sampler, num_samples=10, branch_keep_pairs=None, num_classes=10):
#     """Evaluate sample quality for CIFAR-10 data using the model's quality score"""
#     if branch_keep_pairs is None:
#         branch_keep_pairs = [(3, 2), (8, 3), (16, 7)]

#     # Choose a single random class for evaluation
#     class_label = np.random.randint(num_classes)

#     # Load real CIFAR-10 images for visualization
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#     cifar10 = datasets.CIFAR10(
#         root="./data", train=True, download=True, transform=transform
#     )

#     # Get images only from the selected class
#     class_indices = [i for i, (_, label) in enumerate(cifar10) if label == class_label]
#     selected_indices = np.random.choice(class_indices, num_samples, replace=True)
#     real_images = torch.stack([cifar10[i][0] for i in selected_indices]).to(
#         sampler.device
#     )

#     print(f"\nEvaluating samples for class {class_label}:")

#     # Dictionary to store samples for visualization
#     all_samples_dict = {}

#     for num_branches, num_keep in branch_keep_pairs:
#         print(f"\nTesting with branches={num_branches}, keep={num_keep}")
#         samples = sampler.batch_sample(
#             class_label=class_label,
#             batch_size=num_samples,
#             num_branches=num_branches,
#             num_keep=num_keep,
#         )

#         with torch.no_grad():
#             scores = sampler.compute_sample_quality(
#                 samples,
#                 torch.full(
#                     (num_samples,),
#                     class_label,
#                     device=sampler.device,
#                 ),
#             )

#         all_samples_dict[(num_branches, num_keep)] = samples

#         # Print statistics
#         mean_score = scores.mean().item()
#         std_score = scores.std().item()
#         print(f"Quality score: {mean_score:.4f} ± {std_score:.4f}")

#     # Visualize all samples in a single plot, including real images
#     visualize_samples(all_samples_dict, class_label, real_images)


def calculate_metrics(
    sampler, num_branches, num_keep, device, n_samples=2000, sigma=0.1, fid=None
):
    """
    Calculate FID metrics for a specific branch/keep configuration across all classes.
    """

    fid.reset()

    # Generate samples evenly across all classes
    samples_per_class = n_samples // sampler.num_classes
    generation_batch_size = 16
    metric_batch_size = 64
    generated_samples = []

    print(
        f"\nGenerating {n_samples} samples for branches={num_branches}, keep={num_keep}"
    )

    # for 1x1, we don't need noise (this is just normal flow matching integration)
    noise_scale = sigma
    if num_branches == 1 and num_keep == 1:
        noise_scale = 0

    # Generate samples for each class
    for class_label in range(sampler.num_classes):
        num_batches = samples_per_class // generation_batch_size

        # Generate full batches
        for _ in range(num_batches):
            sample = sampler.batch_sample_wdt_intermediate_fid(
                class_label=class_label,
                batch_size=generation_batch_size,
                num_branches=num_branches,
                num_keep=num_keep,
                dt_std=0.1,
            )
            generated_samples.extend(sample.cpu())

    # Process generated samples in batches for metrics
    generated_tensor = torch.stack(generated_samples)
    for i in range(0, len(generated_tensor), metric_batch_size):
        batch = generated_tensor[i : i + metric_batch_size].to(device)
        fid.update(batch, real=False)
        batch.cpu()
        torch.cuda.empty_cache()

    # Compute final scores
    fid_score = fid.compute()

    return fid_score


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

    # CIFAR-10 dimensions and setup
    image_size = 32
    channels = 3
    num_classes = 10

    # Setup CIFAR-10 dataset with appropriate transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Initialize sampler with CIFAR-10 dimensions
    sampler = MCTSFlowSampler(
        image_size=image_size,
        channels=channels,
        device=device,
        num_timesteps=10,
        num_classes=num_classes,
        buffer_size=1000,
        load_models=True,
        flow_model="large_flow_model.pt",
        value_model=None,
        num_channels=256,
    )

    # Training configuration
    n_epochs_per_cycle = 1
    n_training_cycles = 100
    branch_keep_pairs = [(1, 1), (4, 1), (8, 4), (16, 4)]

    # Initialize metrics
    fid = FID.FrechetInceptionDistance(normalize=True, reset_real_features=False).to(
        device
    )
    cifar10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Randomly sample real images
    indices = np.random.choice(len(cifar10), 5000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    # Process real images in batches
    real_batch_size = 100
    print("Processing real images...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid.update(batch, real=True)

    for cycle in range(n_training_cycles):
        print(f"\nTraining Cycle {cycle + 1}/{n_training_cycles}")

        # sampler.train(
        #     train_loader,
        #     n_epochs=n_epochs_per_cycle,
        #     initial_flow_epochs=0,
        #     value_epochs=10,
        #     flow_epochs=10,
        #     use_tqdm=True,
        # )

        # Evaluate metrics across classes after each training cycle
        for num_branches, num_keep in branch_keep_pairs:
            fid_score = calculate_metrics(
                sampler,
                num_branches,
                num_keep,
                device,
                sigma=0,
                n_samples=1000,
                fid=fid,
            )
            print(f"Cycle {cycle + 1} - (branches={num_branches}, keep={num_keep}):")
            print(f"   FID Score: {fid_score:.4f}")


if __name__ == "__main__":
    main()
