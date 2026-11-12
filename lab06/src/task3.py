
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


def run_experiment(workers, dataset, config, device):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        generator=generator,
        persistent_workers=workers > 0
    )

    model = MLP(
        config["model"]["hidden_size"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    start = time.perf_counter()
    data_wait = 0.0
    compute_time = 0.0
    total_samples = 0

    # Measure one full training epoch
    iterator = iter(loader)

    while True:
        wait_start = time.perf_counter()

        try:
            images, labels = next(iterator)
        except StopIteration:
            break

        data_wait += time.perf_counter() - wait_start

        images = images.to(device)
        labels = labels.to(device)

        compute_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        compute_time += time.perf_counter() - compute_start
        total_samples += labels.size(0)

    elapsed = time.perf_counter() - start
    throughput = total_samples / elapsed

    return {
        "workers": workers,
        "epoch_time": elapsed,
        "data_wait": data_wait,
        "compute_time": compute_time,
        "throughput": throughput
    }


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = datasets.FashionMNIST(
        root=ROOT / "data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_data, _ = random_split(
        dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(seed)
    )

    worker_counts = [0, 1, 2, 4, 8]
    results = []

    print("\nLAB 06 - TASK 3: DATALOADER WORKERS")
    print("-" * 55)
    print(f"Device: {device}")
    print(f"Training samples: {len(train_data)}")

    for workers in worker_counts:
        print(f"\nTesting num_workers = {workers}...")

        result = run_experiment(
            workers, train_data, config, device
        )

        results.append(result)

        print(f"Epoch time: {result['epoch_time']:.2f} s")
        print(f"Data wait: {result['data_wait']:.2f} s")
        print(f"Compute time: {result['compute_time']:.2f} s")
        print(
            f"Throughput: {result['throughput']:.2f} samples/s"
        )

    # Save CSV
    csv_path = RESULTS / "task3_workers.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Workers",
            "Epoch Time (s)",
            "Data Wait (s)",
            "Compute Time (s)",
            "Throughput (samples/s)"
        ])

        for result in results:
            writer.writerow([
                result["workers"],
                round(result["epoch_time"], 4),
                round(result["data_wait"], 4),
                round(result["compute_time"], 4),
                round(result["throughput"], 2)
            ])

    # Plot throughput
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["workers"] for r in results],
        [r["throughput"] for r in results],
        marker="o"
    )

    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Throughput (samples/second)")
    plt.title("Task 3: DataLoader Worker Performance")
    plt.xticks(worker_counts)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS / "task3_workers.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Find fastest configuration
    best = max(
        results,
        key=lambda result: result["throughput"]
    )

    summary_path = RESULTS / "task3_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("LAB 06 - TASK 3\n")
        file.write("DataLoader Worker Comparison\n")
        file.write("-" * 45 + "\n")

        for result in results:
            file.write(
                f"Workers: {result['workers']} | "
                f"Time: {result['epoch_time']:.2f}s | "
                f"Data wait: {result['data_wait']:.2f}s | "
                f"Throughput: {result['throughput']:.2f}\n"
            )

        file.write(
            f"\nBest worker count: {best['workers']}\n"
        )
        file.write(
            f"Best throughput: {best['throughput']:.2f} samples/s\n"
        )

    print("\nTASK 3 COMPLETED")
    print(f"Best worker count: {best['workers']}")
    print(
        f"Best throughput: {best['throughput']:.2f} samples/s"
    )
    print(f"CSV saved: {csv_path}")
    print(f"Plot saved: {plot_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
