
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


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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


def run_experiment(
    optimizer_name,
    scheduler_name,
    train_data,
    val_data,
    config,
    device
):
    seed = config["seed"]
    set_seed(seed)

    batch_size = 128
    workers = 2
    epochs = 10

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
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=True
    )

    model = MLP(
        config["model"]["hidden_size"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Equal initial learning rates for a controlled comparison
    lr = config["training"]["learning_rate"]

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.5
        )

    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

    history = []
    experiment_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total

        val_loss, val_accuracy = evaluate(
            model, val_loader, device, criterion
        )

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history.append({
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    duration = time.perf_counter() - experiment_start

    return {
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "final_accuracy": history[-1]["val_accuracy"],
        "best_accuracy": max(
            row["val_accuracy"] for row in history
        ),
        "final_loss": history[-1]["val_loss"],
        "time_seconds": duration,
        "history": history
    }


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

    optimizers = ["SGD", "Adam"]
    schedulers = [
        "StepLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau"
    ]

    results = []
    all_history = []

    print("\nLAB 06 - TASK 5")
    print("OPTIMIZER AND SCHEDULER COMPARISON")
    print("-" * 55)
    print(f"Device: {device}")
    print("Batch size: 128")
    print("Workers: 2")
    print("Epochs per experiment: 10")
    print("Total experiments: 6")

    for optimizer_name in optimizers:
        for scheduler_name in schedulers:
            print(
                f"\nTesting {optimizer_name} + {scheduler_name}"
            )

            result = run_experiment(
                optimizer_name,
                scheduler_name,
                train_data,
                val_data,
                config,
                device
            )

            results.append(result)
            all_history.extend(result["history"])

            print(
                f"Final Accuracy: "
                f"{result['final_accuracy']:.2f}%"
            )

    # Save comparison CSV
    comparison_path = RESULTS / "task5_comparison.csv"

    with open(
        comparison_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Optimizer",
            "Scheduler",
            "Final Accuracy (%)",
            "Best Accuracy (%)",
            "Final Validation Loss",
            "Time (s)"
        ])

        for result in results:
            writer.writerow([
                result["optimizer"],
                result["scheduler"],
                round(result["final_accuracy"], 2),
                round(result["best_accuracy"], 2),
                round(result["final_loss"], 4),
                round(result["time_seconds"], 2)
            ])

    # Save epoch history
    history_path = RESULTS / "task5_history.csv"

    with open(
        history_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "optimizer",
                "scheduler",
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "learning_rate"
            ]
        )
        writer.writeheader()
        writer.writerows(all_history)

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))

    for result in results:
        plt.plot(
            [row["epoch"] for row in result["history"]],
            [row["val_accuracy"] for row in result["history"]],
            marker="o",
            label=(
                result["optimizer"] + " + " +
                result["scheduler"]
            )
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Task 5: Optimizer and Scheduler Comparison")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task5_accuracy.png", dpi=150)
    plt.close()

    # Plot learning-rate changes
    plt.figure(figsize=(10, 6))

    for result in results:
        plt.plot(
            [row["epoch"] for row in result["history"]],
            [row["learning_rate"] for row in result["history"]],
            marker="o",
            label=(
                result["optimizer"] + " + " +
                result["scheduler"]
            )
        )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Task 5: Learning-Rate Schedules")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task5_learning_rates.png", dpi=150)
    plt.close()

    best = max(
        results,
        key=lambda result: result["final_accuracy"]
    )

    summary_path = RESULTS / "task5_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("LAB 06 - TASK 5\n")
        file.write("Optimizer and Scheduler Comparison\n")
        file.write("-" * 50 + "\n")

        for result in results:
            file.write(
                f"{result['optimizer']} + "
                f"{result['scheduler']} | "
                f"Final Accuracy: "
                f"{result['final_accuracy']:.2f}% | "
                f"Best Accuracy: "
                f"{result['best_accuracy']:.2f}% | "
                f"Time: {result['time_seconds']:.2f}s\n"
            )

        file.write(
            f"\nBest combination: "
            f"{best['optimizer']} + {best['scheduler']}\n"
        )
        file.write(
            f"Final Accuracy: {best['final_accuracy']:.2f}%\n"
        )

    print("\nTASK 5 COMPLETED")
    print(
        f"Best combination: "
        f"{best['optimizer']} + {best['scheduler']}"
    )
    print(
        f"Best final accuracy: "
        f"{best['final_accuracy']:.2f}%"
    )
    print(f"Comparison saved: {comparison_path}")
    print(f"History saved: {history_path}")
    print(f"Summary saved: {summary_path}")
    print("Plots saved in lab06/results/plots")


if __name__ == "__main__":
    main()
