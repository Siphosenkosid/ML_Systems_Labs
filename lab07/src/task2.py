
import copy
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Paths
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
MODELS = RESULTS / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Settings
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 3
SPARSITY_LEVELS = [0, 0.25, 0.5, 0.75, 0.9, 0.95]

# Same CNN architecture as Task 1
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load Fashion-MNIST
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root=ROOT / "data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root=ROOT / "data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def fine_tune(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        print(
            f"  Fine-tuning epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {total_loss / len(train_data):.4f}",
            flush=True
        )


def get_prunable_layers(model):
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def calculate_sparsity(parameters):
    zeros = 0
    total = 0

    for module, name in parameters:
        weights = getattr(module, name)
        zeros += (weights == 0).sum().item()
        total += weights.numel()

    return 100 * zeros / total


def main():
    print("LAB 07 - TASK 2A: GLOBAL MAGNITUDE PRUNING")
    print("-" * 55)
    print("Device:", DEVICE)

    baseline_path = MODELS / "baseline_cnn.pth"

    if not baseline_path.exists():
        raise FileNotFoundError(
            "Run Task 1 first to create baseline_cnn.pth"
        )

    baseline = BaselineCNN().to(DEVICE)
    baseline.load_state_dict(
        torch.load(baseline_path, map_location=DEVICE, weights_only=True)
    )

    baseline_accuracy = evaluate(baseline)
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")

    results = []

    for sparsity in SPARSITY_LEVELS:
        print(f"\nTarget sparsity: {sparsity * 100:.0f}%")

        # Start from the same trained baseline
        model = copy.deepcopy(baseline)
        parameters = get_prunable_layers(model)

        if sparsity > 0:
            prune.global_unstructured(
                parameters,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )

        before_accuracy = evaluate(model)

        # Fixed masks remain active during fine-tuning
        fine_tune(model)

        after_accuracy = evaluate(model)
        actual_sparsity = calculate_sparsity(parameters)

        print(f"  Accuracy before: {before_accuracy:.2f}%")
        print(f"  Accuracy after:  {after_accuracy:.2f}%")
        print(f"  Actual sparsity: {actual_sparsity:.2f}%")

        # Remove pruning wrappers after fine-tuning
        # The zeroed weights remain zero
        if sparsity > 0:
            for module, name in parameters:
                prune.remove(module, name)

        model_path = MODELS / (
            f"global_pruned_{int(sparsity * 100)}.pth"
        )
        torch.save(model.state_dict(), model_path)

        results.append({
            "target_sparsity": sparsity * 100,
            "actual_sparsity": actual_sparsity,
            "accuracy_before": before_accuracy,
            "accuracy_after": after_accuracy,
            "model_size_mb": model_path.stat().st_size / (1024 ** 2)
        })

    output_path = RESULTS / "task2_global_results.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nGLOBAL PRUNING SUMMARY")
    print("-" * 55)

    for result in results:
        print(
            f"Sparsity: {result['actual_sparsity']:.1f}% "
            f"| Before: {result['accuracy_before']:.2f}% "
            f"| After: {result['accuracy_after']:.2f}%"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
