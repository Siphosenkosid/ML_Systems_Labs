
import csv
import random
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Project paths
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.yaml")
RESULTS = ROOT / "lab06" / "results"
PLOTS = RESULTS / "plots"

RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# Model from Task 1
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


def run_experiment(
    batch_size,
    train_data,
    val_data,
    config,
    device
):
    seed = config["seed"]
    set_seed(seed)

    workers = 2
    epochs = 5

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()

    epoch_results = []
    total_samples = 0
    total_training_time = 0.0

    # Python allocation measurement (not total process RAM)
    tracemalloc.start()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            samples += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - epoch_start
        total_training_time += elapsed
        total_samples += samples

        train_loss = running_loss / samples
        val_accuracy = evaluate(model, val_loader, device)

        epoch_results.append({
            "batch_size": batch_size,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "epoch_time": elapsed,
            "throughput": samples / elapsed
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}% | "
            f"Throughput: {samples / elapsed:.2f} samples/s"
        )

    _, peak_python_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )
        memory_type = "CUDA allocated memory"
    else:
        peak_memory_mb = peak_python_memory / (1024 ** 2)
        memory_type = "Python traced allocations"

    return {
        "batch_size": batch_size,
        "throughput": total_samples / total_training_time,
        "peak_memory_mb": peak_memory_mb,
        "memory_type": memory_type,
        "final_accuracy": epoch_results[-1]["val_accuracy"],
        "epoch_results": epoch_results
    }


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = transforms.ToTensor()

    dataset = datasets.FashionMNIST(
        root=ROOT / "data",
        train=True,
        download=True,
        transform=transform
    )

    train_data, val_data = random_split(
        dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(seed)
    )

    batch_sizes = [32, 64, 128, 256, 512]

    results = []
    all_epochs = []

    print("\nLAB 06 - TASK 4: BATCH-SIZE STUDY")
    print("-" * 55)
    print(f"Device: {device}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print("Workers: 2")
    print("Epochs: 5")

    for batch_size in batch_sizes:
        print(f"\nTesting batch size = {batch_size}")

        result = run_experiment(
            batch_size,
            train_data,
            val_data,
            config,
            device
        )

        results.append(result)
        all_epochs.extend(result["epoch_results"])

        print(
            f"Final Accuracy: {result['final_accuracy']:.2f}%"
        )
        print(
            f"Average Throughput: "
            f"{result['throughput']:.2f} samples/s"
        )
        print(
            f"Peak Memory: {result['peak_memory_mb']:.2f} MB"
        )

    # Save comparison table
    csv_path = RESULTS / "task4_batch_sizes.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Batch Size",
            "Throughput (samples/s)",
            "Peak Memory (MB)",
            "Memory Type",
            "Final Validation Accuracy (%)"
        ])

        for result in results:
            writer.writerow([
                result["batch_size"],
                round(result["throughput"], 2),
                round(result["peak_memory_mb"], 2),
                result["memory_type"],
                round(result["final_accuracy"], 2)
            ])

    # Save epoch history
    history_path = RESULTS / "task4_history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "batch_size",
                "epoch",
                "train_loss",
                "val_accuracy",
                "epoch_time",
                "throughput"
            ]
        )
        writer.writeheader()
        writer.writerows(all_epochs)

    # Plot throughput
    plt.figure(figsize=(8, 5))
    plt.plot(
        batch_sizes,
        [r["throughput"] for r in results],
        marker="o"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Task 4: Batch Size vs Throughput")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task4_throughput.png", dpi=150)
    plt.close()

    # Plot memory
    plt.figure(figsize=(8, 5))
    plt.plot(
        batch_sizes,
        [r["peak_memory_mb"] for r in results],
        marker="o"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Peak Measured Memory (MB)")
    plt.title("Task 4: Batch Size vs Memory")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task4_memory.png", dpi=150)
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        batch_sizes,
        [r["final_accuracy"] for r in results],
        marker="o"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Final Validation Accuracy (%)")
    plt.title("Task 4: Batch Size vs Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "task4_accuracy.png", dpi=150)
    plt.close()

    fastest = max(results, key=lambda r: r["throughput"])
    most_accurate = max(
        results, key=lambda r: r["final_accuracy"]
    )

    summary_path = RESULTS / "task4_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("LAB 06 - TASK 4: BATCH-SIZE STUDY\n")
        file.write("-" * 50 + "\n")
        file.write(f"Device: {device}\n")
        file.write("Workers: 2\n")
        file.write("Epochs per batch size: 5\n\n")

        for result in results:
            file.write(
                f"Batch: {result['batch_size']} | "
                f"Throughput: {result['throughput']:.2f} | "
                f"Memory: {result['peak_memory_mb']:.2f} MB | "
                f"Accuracy: {result['final_accuracy']:.2f}%\n"
            )

        file.write(
            f"\nFastest batch size: {fastest['batch_size']}\n"
        )
        file.write(
            f"Most accurate batch size: "
            f"{most_accurate['batch_size']}\n"
        )
        file.write(
            "\nCPU memory values represent Python traced "
            "allocations, not total system RAM.\n"
        )

    print("\nTASK 4 COMPLETED")
    print(f"Fastest batch size: {fastest['batch_size']}")
    print(
        f"Most accurate batch size: "
        f"{most_accurate['batch_size']}"
    )
    print(f"Results saved: {csv_path}")
    print(f"History saved: {history_path}")
    print(f"Summary saved: {summary_path}")
    print("Plots saved in lab06/results/plots")


if __name__ == "__main__":
    main()
