
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.yaml")
RESULTS = ROOT / "lab06" / "results"
PLOTS = RESULTS / "plots"

RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# Same MLP architecture as previous tasks
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_experiment(
    mode, train_data, val_data, config, device
):
    seed = config["seed"]
    set_seed(seed)

    epochs = 5
    workers = 2

    if mode == "Normal":
        batch_size = 256
        accumulation_steps = 1
    else:
        batch_size = 64
        accumulation_steps = 4

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=256,
        shuffle=False,
        num_workers=workers,
        persistent_workers=True
    )

    model = MLP(
        config["model"]["hidden_size"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        updates = 0

        optimizer.zero_grad(set_to_none=True)

        epoch_start = time.perf_counter()

        # Number of mini-batches in this epoch
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            # Number of micro-batches in this update group
            group_start = (
                batch_idx // accumulation_steps
            ) * accumulation_steps

            group_end = min(
                group_start + accumulation_steps,
                total_batches
            )

            group_size = group_end - group_start

            # Handle the final incomplete accumulation group
            # using its actual total sample count.
            if mode == "Accumulation":
                remaining_samples = len(train_data) - (
                    group_start * batch_size
                )
                group_samples = min(
                    accumulation_steps * batch_size,
                    remaining_samples
                )
            else:
                group_samples = labels.size(0)

            scaled_loss = (
                loss * labels.size(0) / group_samples
            )
            scaled_loss.backward()

            should_step = (
                (batch_idx + 1) % accumulation_steps == 0
                or batch_idx + 1 == total_batches
            )

            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start
        train_loss = running_loss / total

        val_accuracy = evaluate(
            model, val_loader, device
        )

        history.append({
            "mode": mode,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "epoch_time": epoch_time,
            "updates": updates
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}% | "
            f"Updates: {updates} | "
            f"Time: {epoch_time:.2f}s"
        )

    return history


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = datasets.FashionMNIST(
        root=ROOT / "data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_data, val_data = random_split(
        dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    print("\nLAB 06 - TASK 6: GRADIENT ACCUMULATION")
    print("-" * 55)
    print(f"Device: {device}")
    print("Normal batch size: 256")
    print("Accumulation: 64 x 4 = 256")
    print("Epochs: 5")

    print("\n1. Normal Training")
    normal = train_experiment(
        "Normal", train_data, val_data, config, device
    )

    print("\n2. Gradient Accumulation")
    accumulated = train_experiment(
        "Accumulation", train_data, val_data,
        config, device
    )

    all_results = normal + accumulated

    # Save epoch results
    csv_path = RESULTS / "task6_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "epoch",
                "train_loss",
                "val_accuracy",
                "epoch_time",
                "updates"
            ]
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Plot training loss
    plt.figure(figsize=(8, 5))

    for name, history in [
        ("Normal Batch 256", normal),
        ("Accumulation 64 x 4", accumulated)
    ]:
        plt.plot(
            [r["epoch"] for r in history],
            [r["train_loss"] for r in history],
            marker="o",
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Task 6: Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task6_loss.png", dpi=150)
    plt.close()

    # Plot validation accuracy
    plt.figure(figsize=(8, 5))

    for name, history in [
        ("Normal Batch 256", normal),
        ("Accumulation 64 x 4", accumulated)
    ]:
        plt.plot(
            [r["epoch"] for r in history],
            [r["val_accuracy"] for r in history],
            marker="o",
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Task 6: Gradient Accumulation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task6_accuracy.png", dpi=150)
    plt.close()

    normal_acc = normal[-1]["val_accuracy"]
    accumulated_acc = accumulated[-1]["val_accuracy"]
    difference = abs(normal_acc - accumulated_acc)

    summary_path = RESULTS / "task6_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("LAB 06 - TASK 6\n")
        file.write("Gradient Accumulation Comparison\n")
        file.write("-" * 45 + "\n")
        file.write(f"Device: {device}\n")
        file.write("Effective batch size: 256\n")
        file.write("Epochs: 5\n\n")
        file.write(
            f"Normal final accuracy: {normal_acc:.2f}%\n"
        )
        file.write(
            f"Accumulation final accuracy: "
            f"{accumulated_acc:.2f}%\n"
        )
        file.write(
            f"Accuracy difference: {difference:.2f} "
            "percentage points\n"
        )
        file.write(
            f"Normal optimizer updates per epoch: "
            f"{normal[-1]['updates']}\n"
        )
        file.write(
            f"Accumulation updates per epoch: "
            f"{accumulated[-1]['updates']}\n"
        )

    print("\nTASK 6 COMPLETED")
    print(f"Normal Accuracy: {normal_acc:.2f}%")
    print(
        f"Accumulation Accuracy: "
        f"{accumulated_acc:.2f}%"
    )
    print(
        f"Accuracy Difference: {difference:.2f} "
        "percentage points"
    )
    print(f"CSV saved: {csv_path}")
    print(f"Summary saved: {summary_path}")
    print("Plots saved in lab06/results/plots")


if __name__ == "__main__":
    main()
