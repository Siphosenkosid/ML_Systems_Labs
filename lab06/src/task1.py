
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Load configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Reproducibility
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Output paths
results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
checkpoint_path = PROJECT_ROOT / config["paths"]["checkpoint"]

results_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

# Dataset
transform = transforms.ToTensor()

full_train = datasets.FashionMNIST(
    root=PROJECT_ROOT / "data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root=PROJECT_ROOT / "data",
    train=False,
    download=True,
    transform=transform
)

generator = torch.Generator().manual_seed(seed)

train_data, val_data = random_split(
    full_train,
    [55000, 5000],
    generator=generator
)

batch_size = config["data"]["batch_size"]
num_workers = config["data"]["num_workers"]

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# Model
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.network(x)


model = MLP(config["model"]["hidden_size"]).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

# Evaluation
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


# MLflow configuration
tracking_path = PROJECT_ROOT / "lab06/results/mlruns"
tracking_path.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(tracking_path.as_uri())
mlflow.set_experiment(
    config["mlflow"]["experiment_name"]
)

# Training
with mlflow.start_run(run_name="Task1_MLP"):

    mlflow.log_params({
        "seed": seed,
        "model": config["model"]["name"],
        "hidden_size": config["model"]["hidden_size"],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": config["training"]["epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "optimizer": config["training"]["optimizer"],
        "device": str(device)
    })

    best_accuracy = 0.0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total
        val_loss, val_accuracy = evaluate(val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }, step=epoch + 1)

        print(
            f"Epoch {epoch + 1}/"
            f"{config['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_accuracy,
                "config": config
            }, checkpoint_path)

    # Restore best checkpoint before testing
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = evaluate(test_loader)

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("best_val_accuracy", best_accuracy)

    mlflow.log_artifact(str(CONFIG_PATH))
    mlflow.log_artifact(str(checkpoint_path))

    summary_path = results_dir / "task1_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("LAB 06 - TASK 1\n")
        file.write("YAML-Driven Training Pipeline\n")
        file.write("-" * 40 + "\n")
        file.write(f"Device: {device}\n")
        file.write(f"Seed: {seed}\n")
        file.write(f"Best Validation Accuracy: {best_accuracy:.2f}%\n")
        file.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        file.write(f"Test Loss: {test_loss:.4f}\n")
        file.write(f"Checkpoint: {checkpoint_path}\n")

    mlflow.log_artifact(str(summary_path))

    print("\nTASK 1 COMPLETED")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Results saved: {summary_path}")
