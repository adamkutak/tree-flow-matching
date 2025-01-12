import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchdiffeq
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
from evaluation import (
    visualize_samples,
    calculate_diversity_metrics,
    visualize_class_samples,
    calculate_fid,
)
from tree_flow_matching import TreeFlowMatching
from mcts_flow_model import MCTSFlowSampler


def get_device():
    """Set up CUDA device if available."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def load_mnist_data(batch_size=128, max_batches=None, train=True):
    """
    Load and prepare MNIST dataset.
    Args:
        batch_size: Size of each batch
        max_batches: Maximum number of batches per epoch (None for all data)
        train: If True, load training data, else load test data
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST("../data", train=train, download=True, transform=transform)

    if max_batches is not None and train:
        # Only apply max_batches to training data
        subset_size = max_batches * batch_size
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Only shuffle training data
        drop_last=train,  # Only drop last batch for training data
    )

    return loader


def train_standard_cfm(train_loader, n_epochs=10, sigma=0.0, device=None):
    """Train a standard conditional flow matching model."""
    if device is None:
        device = get_device()

    # Initialize model and optimizer
    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    # Training loop
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x1 = data[0].to(device)
            y = data[1].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")

    return model


def train_tree_cfm(train_loader, n_epochs=10, sigma=0.0, device=None):
    """Train tree-structured conditional flow matching model."""
    if device is None:
        device = get_device()

    # Initialize TreeFlowMatcher
    tree_fm = TreeFlowMatching(device=device, sigma=sigma)

    # Training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        tree_fm.train_epoch(train_loader)

    return tree_fm


def evaluate_standard_cfm(model, test_loader, device=None, num_samples=100):
    """Evaluate standard CFM model using neural ODE for sampling."""
    if device is None:
        device = get_device()

    model.eval()
    results = {}

    # Generate samples using neural ODE
    print("Generating samples...")
    generated_class_list = torch.arange(10, device=device).repeat(
        num_samples // 10 + 1
    )[:num_samples]

    with torch.no_grad():
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(t, x, generated_class_list),
            torch.randn(num_samples, 1, 28, 28, device=device),
            torch.linspace(0, 1, 2, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
        generated_samples = traj[-1]

    # Calculate all metrics
    print("\nCalculating metrics...")
    results["diversity_metrics"] = calculate_diversity_metrics(generated_samples)
    results["fid_score"] = calculate_fid(
        generated_samples, test_loader, num_samples, device
    )

    # Visualizations
    print("\nGenerating visualizations...")
    visualize_samples(generated_samples)
    visualize_class_samples(generated_samples, generated_class_list)

    print(f"FID score: {results['fid_score']}")
    print(f"Diversity metrics: {results['diversity_metrics']}")

    return results, generated_samples


def evaluate_tree_cfm(tree_fm, test_loader, device=None, num_samples=100):
    """Evaluate tree CFM model using tree-based sampling."""
    if device is None:
        device = get_device()

    tree_fm.flow_model.eval()
    tree_fm.value_model.eval()
    results = {}

    # Generate class labels
    generated_class_list = torch.arange(10, device=device).repeat(
        num_samples // 10 + 1
    )[:num_samples]

    # Generate samples using tree-based sampling
    print("Generating samples...")
    generated_samples = tree_fm.sample(
        num_samples=num_samples,
        class_labels=generated_class_list,
        # num_branches=8,
        # num_select=3,
    )

    # Calculate all metrics
    print("\nCalculating metrics...")
    results["diversity_metrics"] = calculate_diversity_metrics(generated_samples)
    results["fid_score"] = calculate_fid(
        generated_samples, test_loader, num_samples, device
    )

    # Visualizations
    print("\nGenerating visualizations...")
    visualize_samples(generated_samples[:100])
    visualize_class_samples(generated_samples, generated_class_list)

    print(f"FID score: {results['fid_score']}")
    print(f"Diversity metrics: {results['diversity_metrics']}")

    return results, generated_samples


def train_mcts_cfm(train_loader, n_epochs=10, device=None):
    """Train MCTS-based conditional flow matching model."""
    if device is None:
        device = get_device()

    # Initialize MCTS Flow Sampler
    mcts_fm = MCTSFlowSampler(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        device=device,
        num_noise_levels=5,
    )

    # Train the model
    mcts_fm.train(train_loader, n_epochs=n_epochs)

    return mcts_fm


def evaluate_mcts_cfm(mcts_fm, test_loader, device=None, samples_per_class=1):
    """Evaluate MCTS CFM model using MCTS-based sampling."""
    if device is None:
        device = get_device()

    mcts_fm.flow_model.eval()
    mcts_fm.policy_model.eval()
    results = {}

    # Generate samples for each class independently
    print("Generating samples...")
    all_samples = []
    all_labels = []

    for class_label in range(2):
        print(f"\nGenerating samples for class {class_label}...")
        class_labels = torch.full((samples_per_class,), class_label, device=device)

        samples, labels = mcts_fm.sample(
            num_samples=samples_per_class,
            class_labels=class_labels,
            num_branches=8,
            num_keep=4,
        )

        all_samples.append(samples)
        all_labels.append(labels)

    # Combine all generated samples
    generated_samples = torch.cat(all_samples, dim=0)
    generated_labels = torch.cat(all_labels, dim=0)

    # Calculate all metrics
    # print("\nCalculating metrics...")
    # results["diversity_metrics"] = calculate_diversity_metrics(generated_samples)
    # results["fid_score"] = calculate_fid(
    #     generated_samples, test_loader, len(generated_samples), device
    # )
    # print(f"FID score: {results['fid_score']}")
    # print(f"Diversity metrics: {results['diversity_metrics']}")

    # Visualizations
    print("\nGenerating visualizations...")
    visualize_samples(generated_samples[:100])
    visualize_class_samples(generated_samples, generated_labels)

    return results, generated_samples


def main():
    # Set up parameters
    batch_size = 64
    max_batches = 200  # Set to None to use all data
    n_epochs = 20
    sigma = 0.0
    device = get_device()

    # Create save directory
    savedir = "models/cond_mnist"
    os.makedirs(savedir, exist_ok=True)

    # Load data
    train_loader = load_mnist_data(batch_size, max_batches, train=True)
    test_loader = load_mnist_data(batch_size, max_batches=10, train=False)

    # Train standard CFM
    # print("Training standard CFM...")
    # standard_model = train_standard_cfm(train_loader, n_epochs, sigma, device)
    # print("\nEvaluating standard CFM...")
    # evaluate_standard_cfm(standard_model, test_loader, device)

    # Train MCTS CFM
    print("\nTraining MCTS CFM...")
    mcts_model = train_mcts_cfm(train_loader, n_epochs, device)
    print("\nEvaluating MCTS CFM...")
    evaluate_mcts_cfm(mcts_model, test_loader, device)

    # Train custom CFM
    # print("\nTraining Tree CFM...")
    # tree_model = train_tree_cfm(train_loader, n_epochs, sigma, device)
    # print("\nEvaluating Tree CFM...")
    # evaluate_tree_cfm(tree_model, test_loader, device)


if __name__ == "__main__":
    main()
