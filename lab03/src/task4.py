
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from task2 import MLP, CNN, AttentionModel

# Reproducibility
torch.manual_seed(42)

# Use CPU, matching our latency measurements
device = torch.device("cpu")

# Dataset preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_data, val_data = random_split(
    full_train,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=128, shuffle=False
)

test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)


# Evaluate accuracy on a dataset
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# Train one model for five epochs
def train_model(name, model):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    print(f"\nTraining {name}")
    print("=" * 50)

    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        average_loss = running_loss / len(train_data)
        val_accuracy = evaluate(model, val_loader)

        print(
            f"Epoch {epoch}/5 | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.2f}%",
            flush=True
        )

    # Evaluate on the official test set
    test_accuracy = evaluate(model, test_loader)

    print(f"{name} Test Accuracy: {test_accuracy:.2f}%")

    return test_accuracy


models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

results = []

for name, model in models.items():
    accuracy = train_model(name, model)

    results.append({
        "Model": name,
        "Parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "Test Accuracy (%)": round(accuracy, 2)
    })

# Save results for Task 5
with open("task4_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Model", "Parameters", "Test Accuracy (%)"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nFinal Test Results")
print("=" * 50)

for result in results:
    print(
        f"{result['Model']}: "
        f"{result['Test Accuracy (%)']:.2f}%"
    )

print("\nResults saved to task4_results.csv")
