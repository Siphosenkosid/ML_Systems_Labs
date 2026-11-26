import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.ao.quantization import quantize_dynamic


# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RESULTS = ROOT / "lab08" / "results"
MODELS = RESULTS / "models"
PLOTS = RESULTS / "plots"

BASELINE_PATH = (
    ROOT / "lab07" / "results" / "models" / "baseline_cnn.pth"
)

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# Settings
torch.manual_seed(42)
torch.set_num_threads(1)

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
BACKEND = "fbgemm"

torch.backends.quantized.engine = BACKEND


# Original CNN architecture from Lab 07
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


# Evaluate model accuracy
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# Measure average inference latency
def measure_latency(model, sample):
    model.eval()

    warmup = 50
    repetitions = 500

    with torch.inference_mode():

        for _ in range(warmup):
            model(sample)

        start = time.perf_counter()

        for _ in range(repetitions):
            model(sample)

        elapsed = time.perf_counter() - start

    return elapsed / repetitions * 1000


# Calculate saved model size
def model_size(path):
    return path.stat().st_size / (1024 ** 2)


def main():

    print("LAB 08 - TASK 2: DYNAMIC QUANTIZATION")
    print("=" * 60)
    print("PyTorch version:", torch.__version__)
    print("Device:", DEVICE)
    print("Quantization backend:", BACKEND)
    print("CPU threads:", torch.get_num_threads())

    # Load Fashion-MNIST test dataset
    test_data = datasets.FashionMNIST(
        root=DATA,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test samples:", len(test_data))

    # Load trained baseline model
    baseline = BaselineCNN().to(DEVICE)

    baseline.load_state_dict(
        torch.load(
            BASELINE_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    baseline.eval()

    print("\nBaseline model loaded successfully.")

    # Use the same input for both latency tests
    sample = test_data[0][0].unsqueeze(0)

    # Evaluate baseline
    print("\nEvaluating baseline model...")

    baseline_accuracy = evaluate(baseline, test_loader)
    baseline_latency = measure_latency(baseline, sample)
    baseline_size = model_size(BASELINE_PATH)

    # Apply dynamic INT8 quantization
    print("\nApplying dynamic quantization...")

    quantized_model = quantize_dynamic(
        baseline,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )

    quantized_model.eval()

    # Identify quantized layers
    quantized_layers = []

    for name, module in quantized_model.named_modules():
        if isinstance(
            module,
            torch.ao.nn.quantized.dynamic.Linear
        ):
            quantized_layers.append(name)

    print("\nQuantized layers:")

    for name in quantized_layers:
        print("-", name)

    # Save quantized model
    quantized_path = MODELS / "dynamic_quantized.pth"

    torch.save(
        quantized_model.state_dict(),
        quantized_path
    )

    # Evaluate quantized model
    print("\nEvaluating quantized model...")

    quantized_accuracy = evaluate(
        quantized_model, test_loader
    )

    quantized_latency = measure_latency(
        quantized_model, sample
    )

    quantized_size = model_size(quantized_path)

    # Calculate performance differences
    accuracy_drop = baseline_accuracy - quantized_accuracy

    size_reduction = (
        (baseline_size - quantized_size)
        / baseline_size * 100
    )

    compression_ratio = baseline_size / quantized_size

    speedup = baseline_latency / quantized_latency

    # Comparison table
    results = [
        {
            "Model": "Baseline FP32",
            "Accuracy (%)": baseline_accuracy,
            "Size (MB)": baseline_size,
            "Latency (ms)": baseline_latency
        },
        {
            "Model": "Dynamic INT8",
            "Accuracy (%)": quantized_accuracy,
            "Size (MB)": quantized_size,
            "Latency (ms)": quantized_latency
        }
    ]

    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv(
        RESULTS / "task2_comparison.csv",
        index=False
    )

    # Create text report
    report = f"""
LAB 08 - TASK 2: DYNAMIC QUANTIZATION
============================================================

PyTorch version: {torch.__version__}
Device: CPU
Quantization backend: {BACKEND}
CPU threads: {torch.get_num_threads()}
Test samples: {len(test_data)}
Evaluation batch size: {BATCH_SIZE}
Latency batch size: 1
Warm-up iterations: 50
Timing iterations: 500

QUANTIZED LAYERS
------------------------------------------------------------
{chr(10).join(quantized_layers)}

MODEL COMPARISON
------------------------------------------------------------
{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}

PERFORMANCE ANALYSIS
------------------------------------------------------------
Accuracy drop: {accuracy_drop:.4f} percentage points
Size reduction: {size_reduction:.2f}%
Compression ratio: {compression_ratio:.2f}x
Inference speedup: {speedup:.2f}x

OBSERVATIONS
------------------------------------------------------------
Dynamic quantization converts Linear weights to INT8.
Convolutional layers remain in FP32.
Activation quantization parameters are calculated at runtime.
No retraining or calibration is required.
Actual speedup depends on hardware and model architecture.
"""

    print(report)

    (RESULTS / "task2_results.txt").write_text(
        report,
        encoding="utf-8"
    )

    # Save JSON for Task 5
    metrics = {
        "backend": BACKEND,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "quantized_layers": quantized_layers,
        "baseline": {
            "accuracy_percent": baseline_accuracy,
            "size_mb": baseline_size,
            "latency_ms": baseline_latency
        },
        "dynamic_quantized": {
            "accuracy_percent": quantized_accuracy,
            "size_mb": quantized_size,
            "latency_ms": quantized_latency
        },
        "accuracy_drop_pp": accuracy_drop,
        "size_reduction_percent": size_reduction,
        "compression_ratio": compression_ratio,
        "speedup": speedup
    }

    with open(
        RESULTS / "task2_results.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    # Generate comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    columns = [
        "Accuracy (%)",
        "Size (MB)",
        "Latency (ms)"
    ]

    titles = [
        "Test Accuracy",
        "Model Size",
        "Inference Latency"
    ]

    for ax, column, title in zip(axes, columns, titles):

        bars = ax.bar(
            df["Model"],
            df[column],
            color=["steelblue", "seagreen"]
        )

        ax.set_title(title)
        ax.set_ylabel(column)
        ax.set_ylim(0, max(df[column]) * 1.18)

        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=3
        )

    plt.suptitle("FP32 vs Dynamic INT8 Quantization")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "task2_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nQuantized model saved to:", quantized_path)
    print("All Task 2 results saved successfully.")


if __name__ == "__main__":
    main()