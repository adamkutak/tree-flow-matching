import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training


class CIFAR100Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 and remove its final layer
        self.resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add our classifier layer
        self.cifar_classifier = nn.Linear(
            512, 100
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
    scheduler_type="cosine",  # or "plateau"
):
    """Train and save a CIFAR100 classifier with advanced training methods."""
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    # Get base path without extension for checkpoint saves
    base_path = save_path.rsplit(".", 1)[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    # Advanced data augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    # Data loading
    train_dataset = CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model
    model = CIFAR100Classifier().to(device)

    # Load existing model if available
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        try:
            model.load_state_dict(torch.load(save_path))
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc="Testing existing model"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            initial_acc = 100.0 * test_correct / test_total
            print(f"Loaded model accuracy: {initial_acc:.2f}%")
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
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:  # plateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
                    "train_acc": 100.0 * correct / total,
                },
                checkpoint_path,
            )
            print(f"\nSaved checkpoint to {checkpoint_path}")

        # Evaluate on test set every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc="Testing"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            test_acc = 100.0 * test_correct / test_total
            print(f"\nTest Accuracy: {test_acc:.2f}%")

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with accuracy: {best_acc:.2f}%")

        # Update learning rate
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(test_acc)

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    return best_acc


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
        scheduler_type="cosine",
    )
    print(f"Training completed with best accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
