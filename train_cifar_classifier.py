import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR100
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast


class CIFAR100Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 and remove its final layer
        self.resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add our classifier layer
        self.cifar_classifier = nn.Linear(
            self.resnet.fc.in_features, 100
        )  # ResNet18's final feature dim is 512

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.cifar_classifier(x)
        return x


def train_cifar_classifier(
    save_path="saved_models/cifar100_classifier.pt",
    num_epochs=100,
    initial_lr=0.001,
    weight_decay=1e-4,
):
    """Train and save a CIFAR100 classifier using all available data."""
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    # Get base path without extension for checkpoint saves
    base_path = save_path.rsplit(".", 1)[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Data augmentation for all data
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    # Load both training and test datasets with the same transform
    train_dataset = CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # Combine datasets
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    print(f"Training on full dataset of {len(full_dataset)} images")

    # Create dataloader for full dataset
    data_loader = DataLoader(
        full_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    # Initialize model
    model = CIFAR100Classifier().to(device)

    # Load existing model if available
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        try:
            model.load_state_dict(torch.load(save_path, weights_only=True))
            print("Successfully loaded existing model")
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Training from scratch instead")
    else:
        print("No existing model found. Training from scratch.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay, amsgrad=True
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / total,
                    "acc": 100.0 * correct / total,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{base_path}_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": running_loss / total,
                    "accuracy": 100.0 * correct / total,
                },
                checkpoint_path,
            )
            print(f"\nSaved checkpoint to {checkpoint_path}")

        # Update learning rate
        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved final model to {save_path}")
    print(f"Final Accuracy: {100.0 * correct / total:.2f}%")

    return 100.0 * correct / total


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Train with advanced methods
    final_accuracy = train_cifar_classifier(
        save_path="saved_models/cifar100_classifier.pt",
        num_epochs=100,
        initial_lr=0.001,
        weight_decay=1e-4,
    )
    print(f"Training completed with final accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
