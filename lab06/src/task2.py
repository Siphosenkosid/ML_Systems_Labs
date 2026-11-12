
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Paths
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.yaml")
RESULTS = ROOT / "lab06" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load YAML configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Reproducibility
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Fashion-MNIST
transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
    root=ROOT / "data",
    train=True,
    download=True,
    transform=transform
)

train_data, _ = random_split(
    dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(seed)
)

loader = DataLoader(
    train_data,
    batch_size=config["data"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"]
)

# Same MLP architecture as Task 1
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

# Timing helpers
def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


times = {
    "Data Loading": 0.0,
    "Forward Pass": 0.0,
    "Backward Pass": 0.0,
    "Optimizer Step": 0.0
}

# Profile one complete epoch
model.train()
start_epoch = time.perf_counter()
data_start = time.perf_counter()

with profile(
    activities=(
        [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if device.type == "cuda"
        else [ProfilerActivity.CPU]
    ),
    record_shapes=True,
    profile_memory=True
) as prof:

    for images, labels in loader:

        synchronize()
        times["Data Loading"] += (
            time.perf_counter() - data_start
        )

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        synchronize()
        start = time.perf_counter()

        with record_function("Forward Pass"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        synchronize()
        times["Forward Pass"] += time.perf_counter() - start

        synchronize()
        start = time.perf_counter()

        with record_function("Backward Pass"):
            loss.backward()

        synchronize()
        times["Backward Pass"] += time.perf_counter() - start

        synchronize()
        start = time.perf_counter()

        with record_function("Optimizer Step"):
            optimizer.step()

        synchronize()
        times["Optimizer Step"] += time.perf_counter() - start

        data_start = time.perf_counter()

synchronize()
epoch_time = time.perf_counter() - start_epoch

# Calculate time percentages
measured_total = sum(times.values())

percentages = {
    name: (duration / measured_total) * 100
    for name, duration in times.items()
}

# Save profiling table
csv_path = RESULTS / "task2_profile.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Operation", "Time (s)", "Share (%)"])

    for name in times:
        writer.writerow([
            name,
            round(times[name], 4),
            round(percentages[name], 2)
        ])

# Save detailed profiler results
profiler_path = RESULTS / "task2_profiler.txt"

with open(profiler_path, "w", encoding="utf-8") as file:
    file.write(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20
        )
    )

# Save readable summary
summary_path = RESULTS / "task2_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("LAB 06 - TASK 2: PYTORCH PROFILING\n")
    file.write("-" * 45 + "\n")
    file.write(f"Device: {device}\n")
    file.write(f"Total epoch time: {epoch_time:.4f} s\n\n")

    for name in times:
        file.write(
            f"{name}: {times[name]:.4f} s "
            f"({percentages[name]:.2f}%)\n"
        )

    file.write(
        "\nPercentages are based on the sum of "
        "the four measured stages.\n"
    )

# Print results
print("\nLAB 06 - TASK 2: PYTORCH PROFILING")
print("-" * 45)
print(f"Device: {device}")
print(f"Total epoch time: {epoch_time:.4f} seconds\n")

for name in times:
    print(
        f"{name}: {times[name]:.4f} seconds "
        f"({percentages[name]:.2f}%)"
    )

print("\nTASK 2 COMPLETED")
print(f"Summary saved: {summary_path}")
print(f"Profiler saved: {profiler_path}")
print(f"CSV saved: {csv_path}")
