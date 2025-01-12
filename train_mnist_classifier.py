import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

    accuracy = 100.0 * correct / total
    print(f"\nTest accuracy: {accuracy:.2f}%")
    return accuracy


def train_and_save_classifier(n_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "saved_models/mnist_classifier.pt"

    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize or load model
    model = MNISTClassifier().to(device)
    if os.path.exists(save_path):
        print(f"Loading existing classifier from {save_path}")
        model.load_state_dict(torch.load(save_path, weights_only=True))
        initial_accuracy = evaluate_model(model, test_loader, device)
        print(f"Initial accuracy: {initial_accuracy:.2f}%")
    else:
        print("Initializing new classifier")
        os.makedirs("saved_models", exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    best_accuracy = 0
    model.train()
    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            running_correct += pred.eq(target.view_as(pred)).sum().item()
            running_total += len(data)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": running_loss / (batch_idx + 1),
                    "Accuracy": 100.0 * running_correct / running_total,
                }
            )

        # Evaluate on test set
        current_accuracy = evaluate_model(model, test_loader, device)

        # Save if better
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    print(f"Final best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    train_and_save_classifier()
