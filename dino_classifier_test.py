import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from mcts_single_flow import MCTSFlowSampler
from imagenet_dataset import ImageNet32Dataset


def test_dino_classifier():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize MCTSFlowSampler (loading the DINO model)
    sampler = MCTSFlowSampler(
        image_size=32,
        channels=3,
        device=device,
        num_timesteps=10,
        num_classes=1000,
        buffer_size=10,
        load_models=True,
        flow_model="large_flow_model_imagenet32.pt",
        dataset="imagenet32",
        num_channels=256,
        inception_layer=0,
        flow_model_config={
            "num_res_blocks": 3,
            "attention_resolutions": "16,8",
        },
        load_dino=True,
        dino_classifier_path="saved_models/dino_imagenet32_best.pt",
    )

    # Set up dataset transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load ImageNet32 dataset
    dataset = ImageNet32Dataset(root_dir="./data", train=False, transform=transform)

    # Create a subset for testing (e.g., 1000 samples)
    num_test_samples = 1000
    indices = np.random.choice(len(dataset), num_test_samples, replace=False)

    # DataLoader with batch size
    batch_size = 64
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, indices), batch_size=batch_size, shuffle=False
    )

    # Enable debug mode for sampler to see accuracy information
    sampler.debug_mode = True

    # Test the model
    total_correct = 0
    total_samples = 0

    print("Testing DINO classifier on ImageNet32 dataset...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Get scores using the batch_compute_dino_score method
            scores = sampler.batch_compute_dino_score(images, labels)

            # Get model predictions
            logits = sampler.dino_model(images)
            predictions = torch.argmax(logits, dim=1)

            # Calculate accuracy
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

    # Print final accuracy
    accuracy = (total_correct / total_samples) * 100
    print(
        f"Overall DINO Classifier Accuracy: {total_correct}/{total_samples} ({accuracy:.2f}%)"
    )

    return accuracy


if __name__ == "__main__":
    test_dino_classifier()
