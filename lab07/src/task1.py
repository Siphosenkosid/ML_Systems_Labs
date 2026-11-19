
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Project paths
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
MODELS = RESULTS / "models"
DATA = ROOT / "data"

MODELS.mkdir(parents=True, exist_ok=True)

# Settings
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# Fashion-MNIST dataset
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root=DATA,
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root=DATA,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False
)

# Oversized CNN
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
        x = self.features(x)
        return self.classifier(x)


model = BaselineCNN().to(DEVICE)

parameters = sum(p.numel() for p in model.parameters())

print("LAB 07 - TASK 1: BASELINE CNN")
print("-" * 50)
print("Device:", DEVICE)
print("Total parameters:", parameters)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

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

    average_loss = total_loss / len(train_data)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f}"
    )

# Testing
model.eval()
correct = 0
total = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

# Save trained model
model_path = MODELS / "baseline_cnn.pth"
torch.save(model.state_dict(), model_path)

size_mb = model_path.stat().st_size / (1024 ** 2)

# Measure inference latency
sample = torch.randn(1, 1, 28, 28).to(DEVICE)

with torch.inference_mode():
    for _ in range(20):
        model(sample)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(100):
        model(sample)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start) / 100 * 1000

# Save results
results = {
    "model": "BaselineCNN",
    "parameters": parameters,
    "accuracy_percent": accuracy,
    "size_mb": size_mb,
    "latency_ms": latency_ms,
    "device": str(DEVICE),
    "epochs": EPOCHS
}

with open(RESULTS / "task1_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nTASK 1 RESULTS")
print("-" * 50)
print(f"Test accuracy: {accuracy:.2f}%")
print(f"Parameters: {parameters:,}")
print(f"Model size: {size_mb:.2f} MB")
print(f"Latency: {latency_ms:.4f} ms")
print(f"Model saved to: {model_path}")
print("Results saved to: task1_results.json")
