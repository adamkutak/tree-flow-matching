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

    # Test the model
    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    print("Testing DINO classifier on ImageNet32 dataset...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            logits = sampler.dino_model(images)

            # Calculate Top-1, Top-5, and Top-10 accuracy
            # For Top-1
            _, top1_preds = logits.topk(1, 1, True, True)
            top1_preds = top1_preds.t()
            top1_correct_batch = top1_preds.eq(labels.view(1, -1).expand_as(top1_preds))
            top1_correct += top1_correct_batch.sum().item()

            # For Top-5
            _, top5_preds = logits.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            top5_correct_batch = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
            top5_correct += (top5_correct_batch.sum(0) > 0).sum().item()

            # For Top-10
            _, top10_preds = logits.topk(10, 1, True, True)
            top10_preds = top10_preds.t()
            top10_correct_batch = top10_preds.eq(
                labels.view(1, -1).expand_as(top10_preds)
            )
            top10_correct += (top10_correct_batch.sum(0) > 0).sum().item()

            total_samples += labels.size(0)

            # Print batch accuracy information
            batch_top1 = (top1_correct_batch.sum().item() / labels.size(0)) * 100
            batch_top5 = (
                (top5_correct_batch.sum(0) > 0).sum().item() / labels.size(0)
            ) * 100
            batch_top10 = (
                (top10_correct_batch.sum(0) > 0).sum().item() / labels.size(0)
            ) * 100
            print(
                f"Batch Top-1: {batch_top1:.2f}%, Top-5: {batch_top5:.2f}%, Top-10: {batch_top10:.2f}%"
            )

    # Print final accuracy
    top1_accuracy = (top1_correct / total_samples) * 100
    top5_accuracy = (top5_correct / total_samples) * 100
    top10_accuracy = (top10_correct / total_samples) * 100

    print(f"Overall DINO Classifier Accuracy:")
    print(f"Top-1: {top1_correct}/{total_samples} ({top1_accuracy:.2f}%)")
    print(f"Top-5: {top5_correct}/{total_samples} ({top5_accuracy:.2f}%)")
    print(f"Top-10: {top10_correct}/{total_samples} ({top10_accuracy:.2f}%)")

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
    }


if __name__ == "__main__":
    test_dino_classifier()
