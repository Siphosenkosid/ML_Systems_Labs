import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    fuse_modules,
    get_default_qconfig,
    prepare,
    convert
)

from task2 import (
    BaselineCNN,
    evaluate,
    measure_latency,
    model_size
)


# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RESULTS = ROOT / "lab08" / "results"
MODELS = RESULTS / "models"
PLOTS = RESULTS / "plots"

BASELINE_PATH = (
    ROOT / "lab07" / "results" / "models" / "baseline_cnn.pth"
)

MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# Settings
torch.manual_seed(42)
torch.set_num_threads(1)

BACKEND = "fbgemm"
BATCH_SIZE = 128
CALIBRATION_SIZE = 500

torch.backends.quantized.engine = BACKEND


# CNN prepared for static quantization
class StaticCNN(BaselineCNN):

    def __init__(self):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)

        return x

    def fuse_model(self):

        fuse_modules(
            self.features,
            [["0", "1"], ["3", "4"]],
            inplace=True
        )

        fuse_modules(
            self.classifier,
            [["1", "2"], ["3", "4"]],
            inplace=True
        )


def main():

    print("LAB 08 - TASK 3: STATIC QUANTIZATION")
    print("=" * 60)
    print("PyTorch version:", torch.__version__)
    print("Device: CPU")
    print("Quantization backend:", BACKEND)
    print("CPU threads:", torch.get_num_threads())

    # Load Fashion-MNIST datasets
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

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Randomly select 500 training images
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(train_data),
        generator=generator
    )[:CALIBRATION_SIZE].tolist()

    calibration_data = Subset(
        train_data,
        indices
    )

    calibration_loader = DataLoader(
        calibration_data,
        batch_size=50,
        shuffle=False,
        num_workers=0
    )

    print("Training samples:", len(train_data))
    print("Test samples:", len(test_data))
    print("Calibration samples:", len(calibration_data))

    # Check calibration class distribution
    class_counts = torch.bincount(
        train_data.targets[indices],
        minlength=10
    )

    print("\nCalibration class distribution:")

    for label, count in enumerate(class_counts):

        print(
            f"{train_data.classes[label]}: "
            f"{count.item()}"
        )

    # Save calibration indices
    pd.DataFrame({
        "Training index": indices,
        "Class": train_data.targets[indices].tolist()
    }).to_csv(
        RESULTS / "task3_calibration_samples.csv",
        index=False
    )

    # Load trained weights
    weights = torch.load(
        BASELINE_PATH,
        map_location="cpu",
        weights_only=True
    )

    baseline = BaselineCNN()
    baseline.load_state_dict(weights)
    baseline.eval()

    sample = test_data[0][0].unsqueeze(0)

    # Evaluate original model
    print("\nEvaluating baseline model...")

    baseline_accuracy = evaluate(
        baseline, test_loader
    )

    baseline_latency = measure_latency(
        baseline, sample
    )

    baseline_size = model_size(BASELINE_PATH)

    # Prepare static quantization model
    static_model = StaticCNN()
    static_model.load_state_dict(weights)
    static_model.eval()

    # Fuse compatible layers
    print("\nFusing model layers...")

    static_model.fuse_model()

    # Configure quantization
    static_model.qconfig = get_default_qconfig(
        BACKEND
    )

    print("Preparing model for calibration...")

    prepare(
        static_model,
        inplace=True
    )

    # Calibration
    print("\nCalibrating with 500 training images...")

    calibrated_samples = 0

    with torch.inference_mode():

        for images, _ in calibration_loader:

            static_model(images)

            calibrated_samples += images.size(0)

    print("Calibrated samples:", calibrated_samples)

    # Convert prepared model to INT8
    print("\nConverting model to INT8...")

    quantized_model = convert(
        static_model,
        inplace=False
    )

    quantized_model.eval()

    # Identify quantized layers
    quantized_layers = []

    for name, module in quantized_model.named_modules():

        module_type = type(module).__name__
        module_path = type(module).__module__

        if (
            "quantized" in module_path
            and module_type in [
                "Conv2d",
                "ConvReLU2d",
                "Linear",
                "LinearReLU"
            ]
        ):

            quantized_layers.append(
                f"{name}: {module_type}"
            )

    print("\nQuantized layers:")

    for layer in quantized_layers:
        print("-", layer)

    # Save static quantized model
    quantized_path = (
        MODELS / "static_quantized.pth"
    )

    torch.save(
        quantized_model.state_dict(),
        quantized_path
    )

    # Evaluate static quantized model
    print("\nEvaluating static quantized model...")

    static_accuracy = evaluate(
        quantized_model,
        test_loader
    )

    static_latency = measure_latency(
        quantized_model,
        sample
    )

    static_size = model_size(quantized_path)

    # Calculate performance differences
    accuracy_drop = (
        baseline_accuracy - static_accuracy
    )

    size_reduction = (
        (baseline_size - static_size)
        / baseline_size * 100
    )

    compression_ratio = (
        baseline_size / static_size
    )

    speedup = (
        baseline_latency / static_latency
    )

    # Load dynamic quantization results
    with open(
        RESULTS / "task2_results.json"
    ) as f:

        dynamic_results = json.load(f)

    dynamic = dynamic_results["dynamic_quantized"]

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
            "Accuracy (%)": dynamic["accuracy_percent"],
            "Size (MB)": dynamic["size_mb"],
            "Latency (ms)": dynamic["latency_ms"]
        },
        {
            "Model": "Static INT8",
            "Accuracy (%)": static_accuracy,
            "Size (MB)": static_size,
            "Latency (ms)": static_latency
        }
    ]

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS / "task3_comparison.csv",
        index=False
    )

    # Calibration distribution report
    distribution = "\n".join(
        f"{train_data.classes[i]}: {count.item()}"
        for i, count in enumerate(class_counts)
    )

    # Text report
    report = f"""
LAB 08 - TASK 3: STATIC QUANTIZATION
============================================================

PyTorch version: {torch.__version__}
Device: CPU
Backend: {BACKEND}
CPU threads: {torch.get_num_threads()}
Calibration samples: {calibrated_samples}
Random seed: 42
Test samples: {len(test_data)}
Evaluation batch size: {BATCH_SIZE}
Latency batch size: 1
Warm-up iterations: 50
Timing iterations: 500

CALIBRATION CLASS DISTRIBUTION
------------------------------------------------------------
{distribution}

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

WHY REPRESENTATIVE CALIBRATION DATA IS IMPORTANT
------------------------------------------------------------
Calibration estimates the activation ranges of the model.

The calibration subset should represent the real input
distribution, including different clothing classes and
image characteristics.

Unrepresentative calibration data may produce unsuitable
quantization scales and zero-points, causing clipping,
rounding errors and reduced model accuracy.

Training images are used instead of test images to avoid
using the test dataset during model calibration.

NOTE
------------------------------------------------------------
Dynamic quantization metrics were loaded from Task 2.
The baseline and static models were measured in this run.
"""

    print(report)

    (RESULTS / "task3_results.txt").write_text(
        report,
        encoding="utf-8"
    )

    # Save JSON metrics for Task 5
    metrics = {
        "backend": BACKEND,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "calibration_samples": calibrated_samples,
        "random_seed": 42,
        "calibration_distribution": {
            train_data.classes[i]: count.item()
            for i, count in enumerate(class_counts)
        },
        "quantized_layers": quantized_layers,
        "baseline": {
            "accuracy_percent": baseline_accuracy,
            "size_mb": baseline_size,
            "latency_ms": baseline_latency
        },
        "static_quantized": {
            "accuracy_percent": static_accuracy,
            "size_mb": static_size,
            "latency_ms": static_latency
        },
        "accuracy_drop_pp": accuracy_drop,
        "size_reduction_percent": size_reduction,
        "compression_ratio": compression_ratio,
        "speedup": speedup
    }

    with open(
        RESULTS / "task3_results.json",
        "w"
    ) as f:

        json.dump(metrics, f, indent=4)

    # Model comparison plots
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )

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

    for ax, column, title in zip(
        axes, columns, titles
    ):

        bars = ax.bar(
            df["Model"],
            df[column],
            color=[
                "steelblue",
                "seagreen",
                "darkorange"
            ]
        )

        ax.set_title(title)
        ax.set_ylabel(column)
        ax.set_ylim(0, max(df[column]) * 1.2)

        ax.tick_params(
            axis="x",
            labelrotation=15
        )

        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=3
        )

    plt.suptitle(
        "Baseline vs Dynamic vs Static Quantization"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS / "task3_comparison.png",
        dpi=300
    )

    plt.close()

    # Calibration class distribution plot
    plt.figure(figsize=(10, 5))

    plt.bar(
        train_data.classes,
        class_counts.tolist()
    )

    plt.title("Calibration Dataset Class Distribution")
    plt.xlabel("Clothing Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "task3_calibration_distribution.png",
        dpi=300
    )

    plt.close()

    print("\nStatic model saved to:", quantized_path)
    print("All Task 3 results saved successfully.")


if __name__ == "__main__":
    main()