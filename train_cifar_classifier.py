import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import os
from tqdm import tqdm


class CIFAR100Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)

        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the final layer and make it trainable
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 100)
        self.resnet.fc.requires_grad_(True)

    def forward(self, x):
        return self.resnet(x)


def train_cifar_classifier(
    save_path="saved_models/cifar100_classifier.pt", num_epochs=10
):
    """Train and save a CIFAR100 classifier."""
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Data loading and preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
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

    train_dataset = CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize model and training components
    model = CIFAR100Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": running_loss / total, "acc": 100.0 * correct / total}
            )

    # Final evaluation on test set
    print("\nEvaluating on test set...")
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
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # Save the final model
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

    return test_acc


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Train the classifier
    final_accuracy = train_cifar_classifier(num_epochs=100)
    print(f"Training completed with final accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
