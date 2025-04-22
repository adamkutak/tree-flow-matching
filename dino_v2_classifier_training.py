# train_dino_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imagenet_dataset import ImageNet32Dataset


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=1000, freeze_backbone=True):
        super().__init__()
        # Load the base DINO model without the linear classifier
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        # Freeze the backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get the feature dimension from the backbone
        feature_dim = self.backbone.embed_dim  # Should be 768 for ViT-B

        # Add a custom linear layer for ImageNet32 classification
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the DINOv2 classifier.

        Args:
            x: Tensor of shape [batch_size, C, H, W], normalized with ImageNet stats
                (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        """
        # Resize images to be compatible with DINO's patch size (14x14)
        if x.size(-1) < 224:
            x = nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Extract features from the backbone
        features = self.backbone(x)

        # Apply the classifier
        logits = self.classifier(features)

        return logits


def train_dino_classifier(
    device="cuda:0",
    batch_size=64,
    num_epochs=50,
    lr=0.001,
    save_interval=1,
    save_dir="./saved_models",
):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set up transformations for training
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load ImageNet32 dataset
    train_dataset = ImageNet32Dataset(
        root_dir="./data", train=True, transform=transform
    )
    val_dataset = ImageNet32Dataset(root_dir="./data", train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    model = DINOv2Classifier(num_classes=1000, freeze_backbone=True).to(device)

    # Check for existing best model checkpoint
    best_model_path = os.path.join(save_dir, "dino_imagenet32_best.pt")
    best_accuracy = 0.0

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded existing model from {best_model_path}")

        # Evaluate the loaded model to get its accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Evaluating loaded model"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        best_accuracy = 100.0 * correct / total
        print(f"Loaded model accuracy: {best_accuracy:.2f}%")

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Keep track of training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                accuracy = 100.0 * correct / total
                pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy})

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = 100.0 * correct / total

        history["val_loss"].append(epoch_loss)
        history["val_accuracy"].append(epoch_accuracy)

        print(
            f"Epoch {epoch+1}: Train Loss = {history['train_loss'][-1]:.4f}, "
            f"Val Loss = {epoch_loss:.4f}, Val Accuracy = {epoch_accuracy:.2f}%"
        )

        # Update learning rate based on validation accuracy
        scheduler.step(epoch_accuracy)

        # Save model if it's better than the previous best
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(
                model.state_dict(), os.path.join(save_dir, "dino_imagenet32_best.pt")
            )
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        # Save model at specified intervals
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"dino_imagenet32_epoch{epoch+1}.pt"),
            )
            print(f"Model saved at epoch {epoch+1}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dino_imagenet32_training_history.png"))
    plt.close()

    print(f"Training completed! Best validation accuracy: {best_accuracy:.2f}%")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DINOv2 classifier for ImageNet32"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--save_interval", type=int, default=1, help="Save model every N epochs"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="Directory to save models",
    )

    args = parser.parse_args()

    train_dino_classifier(
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
    )
