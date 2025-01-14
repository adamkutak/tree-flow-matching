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
from train_mnist_classifier import MNISTClassifier


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

    save_path = "saved_models/standard_cfm.pt"

    # Initialize model
    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True,
    ).to(device)

    # Check if saved model exists
    if os.path.exists(save_path):
        print(f"Loading pre-trained model from {save_path}")
        model.load_state_dict(
            torch.load(save_path, weights_only=True, map_location=device)
        )
        return model

    print("Training new model...")
    optimizer = torch.optim.Adam(model.parameters())
    FM = ConditionalFlowMatcher(sigma=sigma)

    # Training loop
    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for data in pbar:
            optimizer.zero_grad()
            x1 = data[0].to(device)
            y = data[1].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


def evaluate_standard_cfm(model, test_loader, device=None, num_samples=5):
    """Evaluate standard CFM model using neural ODE for sampling."""
    if device is None:
        device = get_device()

    # Initialize classifier for evaluation
    classifier = MNISTClassifier().to(device)
    classifier_path = "saved_models/mnist_classifier.pt"
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"No pre-trained classifier found at {classifier_path}. "
            f"Please run train_mnist_classifier.py first."
        )
    classifier.load_state_dict(
        torch.load(classifier_path, weights_only=True, map_location=device)
    )
    classifier.eval()

    def compute_sample_quality(samples, target_labels):
        """Compute quality score using MNIST classifier confidence."""
        with torch.no_grad():
            logits = classifier(samples)
            target_probs = logits[torch.arange(len(samples)), target_labels]
            return target_probs

    model.eval()
    results = {}

    # Generate samples for each class
    print("\nGenerating and evaluating samples for each class...")
    all_samples = []
    all_scores = []

    for class_label in range(10):
        print(f"\nGenerating samples for digit {class_label}")
        class_labels = torch.full((num_samples,), class_label, device=device)

        # Generate samples using neural ODE
        with torch.no_grad():
            traj = torchdiffeq.odeint(
                lambda t, x: model.forward(t, x, class_labels),
                torch.randn(num_samples, 1, 28, 28, device=device),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
            )
            samples = traj[-1]

            # Compute classifier confidence
            confidence_scores = compute_sample_quality(samples, class_labels)
            mean_confidence = confidence_scores.mean().item()
            std_confidence = confidence_scores.std().item()
            print(f"Mean confidence: {mean_confidence:.4f} ± {std_confidence:.4f}")

            # Visualize samples for this class
            plt.figure(figsize=(15, 3))
            grid = make_grid(samples, nrow=num_samples, normalize=True, padding=2)
            plt.imshow(grid.cpu().permute(1, 2, 0))
            plt.title(f"Generated Samples for Digit {class_label}")
            plt.axis("off")
            plt.show()

            all_samples.append(samples[:10])
            all_scores.extend(confidence_scores.tolist())

    # Report overall statistics
    scores = torch.tensor(all_scores)
    mean_confidence = scores.mean().item()
    std_confidence = scores.std().item()
    print(f"\nOverall confidence: {mean_confidence:.4f} ± {std_confidence:.4f}")

    return results, all_samples


def main():
    # Set up parameters
    batch_size = 128
    max_batches = None  # Set to None to use all data
    n_epochs = 10
    sigma = 0.0
    device = get_device()

    # Create save directory
    savedir = "models/cond_mnist"
    os.makedirs(savedir, exist_ok=True)

    # Load data
    train_loader = load_mnist_data(batch_size, max_batches, train=True)
    test_loader = load_mnist_data(batch_size, max_batches=10, train=False)

    # Train standard CFM
    print("Training standard CFM...")
    standard_model = train_standard_cfm(train_loader, n_epochs, sigma, device)
    print("\nEvaluating standard CFM...")
    evaluate_standard_cfm(standard_model, test_loader, device)


if __name__ == "__main__":
    main()
