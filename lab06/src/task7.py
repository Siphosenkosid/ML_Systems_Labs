
import argparse
import csv
import random
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
CHECKPOINTS = RESULTS / "checkpoints"

CHECKPOINT = CHECKPOINTS / "task7_checkpoint.pth"
BASELINE_FILE = RESULTS / "task7_baseline.csv"
RESUMED_FILE = RESULTS / "task7_resumed.csv"

TOTAL_EPOCHS = 5
INTERRUPT_EPOCH = 3
BATCH_SIZE = 128
NUM_WORKERS = 2


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


def create_loaders(seed):
    dataset = datasets.FashionMNIST(
        root=ROOT / "data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_data, val_data = random_split(
        dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader


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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    return total_loss / total


def save_history(path, history):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(history)


def build_training(config, device):
    model = MLP(
        config["model"]["hidden_size"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def train_range(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    start_epoch,
    end_epoch,
    history
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, device, criterion
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{TOTAL_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    return history


def save_checkpoint(model, optimizer, history):
    state = {
        "epoch": INTERRUPT_EPOCH,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state()
    }

    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    torch.save(state, CHECKPOINT)


def load_checkpoint(model, optimizer, device):
    # weights_only=False is required for the saved RNG state.
    # Only load checkpoints created by this script that you trust.
    state = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    return state["epoch"], state["history"]


def run_baseline(config, device, train_loader, val_loader):
    print("\n1. UNINTERRUPTED TRAINING")

    set_seed(config["seed"])

    model, optimizer, criterion = build_training(
        config, device
    )

    history = train_range(
        model, optimizer, criterion,
        train_loader, val_loader, device,
        1, TOTAL_EPOCHS, []
    )

    save_history(BASELINE_FILE, history)


def run_interrupt(config, device, train_loader, val_loader):
    print("\n2. TRAINING UNTIL EPOCH 3")

    set_seed(config["seed"])

    model, optimizer, criterion = build_training(
        config, device
    )

    history = train_range(
        model, optimizer, criterion,
        train_loader, val_loader, device,
        1, INTERRUPT_EPOCH, []
    )

    save_checkpoint(model, optimizer, history)

    print(f"\nCheckpoint saved: {CHECKPOINT}")
    print("Training interrupted after epoch 3.")
    print("Run the script again with --mode resume")


def run_resume(config, device, train_loader, val_loader):
    print("\n3. RESUMING TRAINING")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Checkpoint missing. Run --mode interrupt first."
        )

    model, optimizer, criterion = build_training(
        config, device
    )

    last_epoch, history = load_checkpoint(
        model, optimizer, device
    )

    print(f"Checkpoint loaded from epoch {last_epoch}")
    print(f"Resuming from epoch {last_epoch + 1}")

    history = train_range(
        model, optimizer, criterion,
        train_loader, val_loader, device,
        last_epoch + 1, TOTAL_EPOCHS, history
    )

    save_history(RESUMED_FILE, history)

    print("\nTASK 7 COMPLETED")
    print(f"Resumed history saved: {RESUMED_FILE}")


def read_history(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "val_accuracy": float(row["val_accuracy"])
        }
        for row in rows
    ]


def compare_results():
    if not BASELINE_FILE.exists() or not RESUMED_FILE.exists():
        raise FileNotFoundError(
            "Run baseline, interrupt and resume before compare."
        )

    baseline = read_history(BASELINE_FILE)
    resumed = read_history(RESUMED_FILE)

    if len(baseline) != TOTAL_EPOCHS or len(resumed) != TOTAL_EPOCHS:
        raise ValueError("Expected five epochs in each history.")

    for metric, filename, ylabel in [
        ("train_loss", "task7_loss.png", "Training Loss"),
        ("val_accuracy", "task7_accuracy.png", "Validation Accuracy (%)")
    ]:
        plt.figure(figsize=(8, 5))

        plt.plot(
            [r["epoch"] for r in baseline],
            [r[metric] for r in baseline],
            marker="o",
            label="Uninterrupted"
        )

        plt.plot(
            [r["epoch"] for r in resumed],
            [r[metric] for r in resumed],
            marker="x",
            linestyle="--",
            label="Resumed after epoch 3"
        )

        plt.axvline(
            3,
            linestyle=":",
            label="Checkpoint at epoch 3"
        )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title("Task 7: " + ylabel + " Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS / filename, dpi=150)
        plt.close()

    loss_diff = max(
        abs(a["train_loss"] - b["train_loss"])
        for a, b in zip(baseline, resumed)
    )

    accuracy_diff = max(
        abs(a["val_accuracy"] - b["val_accuracy"])
        for a, b in zip(baseline, resumed)
    )

    summary = (
        "LAB 06 - TASK 7\n"
        "Checkpoint and Resume Training\n"
        + "-" * 45 + "\n"
        + "Checkpoint epoch: 3\n"
        + "Resumed from epoch: 4\n"
        + "Total epochs: 5\n\n"
        + f"Uninterrupted final accuracy: "
        f"{baseline[-1]['val_accuracy']:.2f}%\n"
        + f"Resumed final accuracy: "
        f"{resumed[-1]['val_accuracy']:.2f}%\n"
        + f"Maximum training loss difference: "
        f"{loss_diff:.8f}\n"
        + f"Maximum validation accuracy difference: "
        f"{accuracy_diff:.4f} percentage points\n"
    )

    (RESULTS / "task7_summary.txt").write_text(
        summary, encoding="utf-8"
    )

    print("\n" + summary)
    print("Comparison plots saved in lab06/results/plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "interrupt", "resume", "compare"]
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        compare_results()
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = create_loaders(
        config["seed"]
    )

    print("\nLAB 06 - TASK 7")
    print("CHECKPOINT AND RESUME TRAINING")
    print("-" * 55)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    if args.mode == "baseline":
        run_baseline(
            config, device, train_loader, val_loader
        )

    elif args.mode == "interrupt":
        run_interrupt(
            config, device, train_loader, val_loader
        )

    elif args.mode == "resume":
        run_resume(
            config, device, train_loader, val_loader
        )


if __name__ == "__main__":
    main()

