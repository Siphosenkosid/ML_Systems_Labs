import copy
import json
import time
import statistics
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
    convert,
    quantize_dynamic
)

from task2 import BaselineCNN, evaluate, model_size


# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
LAB07_MODELS = ROOT / "lab07" / "results" / "models"

RESULTS = ROOT / "lab08" / "results"
MODELS = RESULTS / "models"
PLOTS = RESULTS / "plots"

MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# Settings
SEED = 42
BACKEND = "fbgemm"
BATCH_SIZE = 128
CALIBRATION_SIZE = 500

WARMUP = 50
ITERATIONS = 500
REPETITIONS = 5

torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.backends.quantized.engine = BACKEND


# Structured pruned CNN from Lab 07
class StructuredCNN(nn.Module):

    def __init__(self, channels1=16, channels2=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, channels1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channels1, channels2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels2 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Distilled student CNN from Lab 07
class StudentCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Wrapper for static quantization
class QuantizableCNN(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.quant = QuantStub()

        self.features = copy.deepcopy(
            model.features
        )

        self.classifier = copy.deepcopy(
            model.classifier
        )

        self.dequant = DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)

        return x

    def fuse_model(self):

        # Fuse convolution and ReLU layers
        fuse_modules(
            self.features,
            [["0", "1"], ["3", "4"]],
            inplace=True
        )

        # Identify Linear and ReLU pairs
        pairs = []

        for i in range(len(self.classifier) - 1):

            if (
                isinstance(
                    self.classifier[i], nn.Linear
                )
                and isinstance(
                    self.classifier[i + 1], nn.ReLU
                )
            ):

                pairs.append(
                    [str(i), str(i + 1)]
                )

        if pairs:

            fuse_modules(
                self.classifier,
                pairs,
                inplace=True
            )


# Load original FP32 models
def load_model(model, path):

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}"
        )

    weights = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    return model


# Apply static quantization
def static_quantize(model, calibration_loader):

    quant_model = QuantizableCNN(model)
    quant_model.eval()

    quant_model.fuse_model()

    quant_model.qconfig = get_default_qconfig(
        BACKEND
    )

    prepare(
        quant_model,
        inplace=True
    )

    calibrated = 0

    with torch.inference_mode():

        for images, _ in calibration_loader:

            quant_model(images)

            calibrated += images.size(0)

    print(f"  Calibrated images: {calibrated}")

    quant_model = convert(
        quant_model,
        inplace=False
    )

    quant_model.eval()

    return quant_model


# Count logical model parameters
def count_parameters(model):

    return sum(
        p.numel() for p in model.parameters()
    )


# Measure median inference latency
def benchmark_latency(model, sample):

    model.eval()

    measurements = []

    with torch.inference_mode():

        # Warm-up
        for _ in range(WARMUP):
            model(sample)

        # Repeat timing measurements
        for _ in range(REPETITIONS):

            start = time.perf_counter()

            for _ in range(ITERATIONS):
                model(sample)

            elapsed = time.perf_counter() - start

            latency = (
                elapsed * 1000 / ITERATIONS
            )

            measurements.append(latency)

    return statistics.median(measurements)


# Save quantized model
def save_model(model, filename):

    path = MODELS / filename

    torch.save(
        model.state_dict(),
        path
    )

    return path


# Generate deployment recommendations
def deployment_recommendations(df):

    baseline_accuracy = df.iloc[0]["Accuracy (%)"]

    # Cloud: highest accuracy
    cloud = df.sort_values(
        ["Accuracy (%)", "Latency (ms)"],
        ascending=[False, True]
    ).iloc[0]

    # Edge: fastest model with limited accuracy loss
    edge_candidates = df[
        df["Accuracy (%)"] >= baseline_accuracy - 1.0
    ]

    edge = edge_candidates.sort_values(
        "Latency (ms)"
    ).iloc[0]

    # Mobile: smallest model with acceptable accuracy
    mobile_candidates = df[
        df["Accuracy (%)"] >= baseline_accuracy - 2.5
    ]

    mobile = mobile_candidates.sort_values(
        ["Size (MB)", "Latency (ms)"]
    ).iloc[0]

    # TinyML: smallest model candidate
    tinyml = df.sort_values(
        "Size (MB)"
    ).iloc[0]

    report = f"""
DEPLOYMENT RECOMMENDATIONS
=================================================================

1. CLOUD
Recommended model: {cloud['Model']}
Accuracy: {cloud['Accuracy (%)']:.2f}%
Size: {cloud['Size (MB)']:.4f} MB
Latency: {cloud['Latency (ms)']:.4f} ms

Cloud environments generally have more computational
resources. The model with the highest accuracy is
selected because model size is less restrictive.


2. EDGE
Recommended model: {edge['Model']}
Accuracy: {edge['Accuracy (%)']:.2f}%
Size: {edge['Size (MB)']:.4f} MB
Latency: {edge['Latency (ms)']:.4f} ms

Edge devices require fast local inference while
maintaining reliable accuracy.

The fastest model within 1 percentage point of
baseline accuracy is selected.


3. MOBILE
Recommended model: {mobile['Model']}
Accuracy: {mobile['Accuracy (%)']:.2f}%
Size: {mobile['Size (MB)']:.4f} MB
Latency: {mobile['Latency (ms)']:.4f} ms

Mobile devices have limited storage, memory and
battery capacity.

The smallest model within 2.5 percentage points
of baseline accuracy is selected.


4. TINYML
Candidate model: {tinyml['Model']}
Accuracy: {tinyml['Accuracy (%)']:.2f}%
Size: {tinyml['Size (MB)']:.4f} MB
Latency: {tinyml['Latency (ms)']:.4f} ms

TinyML requires extremely small models suitable
for microcontrollers.

The smallest model is selected as a candidate
for further conversion and optimisation.

Actual deployment requires checking flash memory,
activation RAM, supported operators and runtime
compatibility.


LIMITATIONS
-----------------------------------------------------------------
All latency measurements were performed on a desktop
CPU using the FBGEMM backend.

Actual performance on cloud, edge, mobile and TinyML
hardware must be measured separately.

The accuracy thresholds are experimental selection
criteria rather than universal deployment requirements.

Checkpoint size does not represent total runtime
memory consumption.
"""

    return report


def main():

    print("LAB 08 - TASK 5: COMBINED OPTIMISATION")
    print("=" * 65)

    print("PyTorch version:", torch.__version__)
    print("Device: CPU")
    print("Backend:", BACKEND)
    print("CPU threads:", torch.get_num_threads())

    # Load Fashion-MNIST
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

    # Select the same calibration samples as Task 3
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(train_data),
        generator=generator
    )[:CALIBRATION_SIZE].tolist()

    calibration_loader = DataLoader(
        Subset(train_data, indices),
        batch_size=50,
        shuffle=False,
        num_workers=0
    )

    sample = test_data[0][0].unsqueeze(0)

    print("Test samples:", len(test_data))
    print("Calibration samples:", len(indices))

    # Load baseline model
    print("\nLoading baseline CNN...")

    baseline_path = (
        LAB07_MODELS / "baseline_cnn.pth"
    )

    baseline = load_model(
        BaselineCNN(),
        baseline_path
    )

    # Load structured pruned model
    print("Loading structured pruned CNN...")

    pruned_path = (
        LAB07_MODELS / "structured_pruned_50.pth"
    )

    pruned = load_model(
        StructuredCNN(16, 32),
        pruned_path
    )

    # Load distilled student model
    print("Loading distilled student CNN...")

    distilled_path = (
        LAB07_MODELS / "distilled_student.pth"
    )

    distilled = load_model(
        StudentCNN(),
        distilled_path
    )

    # Count parameters
    baseline_params = count_parameters(baseline)
    pruned_params = count_parameters(pruned)
    distilled_params = count_parameters(distilled)

    print("\nMODEL PARAMETERS")
    print("-" * 65)

    print(f"Baseline: {baseline_params:,}")
    print(f"Pruned: {pruned_params:,}")
    print(f"Distilled: {distilled_params:,}")

    # Dynamic quantization
    # Recreate directly instead of loading a quantized checkpoint
    print("\nPreparing dynamic INT8 model...")

    dynamic_model = quantize_dynamic(
        baseline,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )

    dynamic_model.eval()

    dynamic_path = save_model(
        dynamic_model,
        "task5_dynamic_quantized.pth"
    )

    print("  Dynamic quantization completed.")

    # Static quantization
    # Recreate directly instead of loading a quantized checkpoint
    print("\nPreparing static INT8 model...")

    static_model = static_quantize(
        baseline,
        calibration_loader
    )

    static_model.eval()

    static_path = save_model(
        static_model,
        "task5_static_quantized.pth"
    )

    print("  Static quantization completed.")

    # Combine pruning and static quantization
    print("\nApplying static quantization to pruned model...")

    pruned_quantized = static_quantize(
        pruned,
        calibration_loader
    )

    pruned_quantized_path = save_model(
        pruned_quantized,
        "pruned_quantized.pth"
    )

    print("  Pruned + INT8 completed.")

    # Combine distillation and static quantization
    print("\nApplying static quantization to distilled model...")

    distilled_quantized = static_quantize(
        distilled,
        calibration_loader
    )

    distilled_quantized_path = save_model(
        distilled_quantized,
        "distilled_quantized.pth"
    )

    print("  Distilled + INT8 completed.")

    # Seven required model variants
    variants = [
        (
            "Baseline FP32",
            baseline,
            baseline_path,
            baseline_params,
            "FP32"
        ),
        (
            "Pruned FP32",
            pruned,
            pruned_path,
            pruned_params,
            "FP32"
        ),
        (
            "Distilled FP32",
            distilled,
            distilled_path,
            distilled_params,
            "FP32"
        ),
        (
            "Dynamic INT8",
            dynamic_model,
            dynamic_path,
            baseline_params,
            "Mixed FP32/INT8"
        ),
        (
            "Static INT8",
            static_model,
            static_path,
            baseline_params,
            "Static INT8"
        ),
        (
            "Pruned + INT8",
            pruned_quantized,
            pruned_quantized_path,
            pruned_params,
            "Static INT8"
        ),
        (
            "Distilled + INT8",
            distilled_quantized,
            distilled_quantized_path,
            distilled_params,
            "Static INT8"
        )
    ]

    # Evaluate all seven models
    results = []

    print("\nBENCHMARKING ALL SEVEN MODELS")
    print("=" * 65)

    for name, model, path, parameters, precision in variants:

        print(f"\nEvaluating: {name}", flush=True)

        accuracy = evaluate(
            model,
            test_loader
        )

        latency = benchmark_latency(
            model,
            sample
        )

        size = model_size(path)

        results.append({
            "Model": name,
            "Parameters": parameters,
            "Precision": precision,
            "Accuracy (%)": accuracy,
            "Size (MB)": size,
            "Latency (ms)": latency
        })

        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Size: {size:.4f} MB")
        print(f"  Latency: {latency:.4f} ms")

    df = pd.DataFrame(results)

    # Compare against baseline
    baseline_accuracy = df.iloc[0]["Accuracy (%)"]
    baseline_size = df.iloc[0]["Size (MB)"]
    baseline_latency = df.iloc[0]["Latency (ms)"]

    df["Accuracy Drop (pp)"] = (
        baseline_accuracy - df["Accuracy (%)"]
    )

    df["Size Reduction (%)"] = (
        100 * (1 - df["Size (MB)"] / baseline_size)
    )

    df["Compression Ratio"] = (
        baseline_size / df["Size (MB)"]
    )

    df["Speedup"] = (
        baseline_latency / df["Latency (ms)"]
    )

    # Save comparison table
    df.to_csv(
        RESULTS / "task5_comparison.csv",
        index=False
    )

    # Generate deployment recommendations
    recommendations = deployment_recommendations(df)

    (RESULTS / "task5_deployment_recommendations.txt").write_text(
        recommendations,
        encoding="utf-8"
    )

    # Generate final report
    report = f"""
LAB 08 - TASK 5: COMBINED MODEL OPTIMISATION
=================================================================

PyTorch version: {torch.__version__}
Device: CPU
Backend: {BACKEND}
CPU threads: {torch.get_num_threads()}
Test samples: {len(test_data)}
Calibration samples: {CALIBRATION_SIZE}
Random seed: {SEED}
Evaluation batch size: {BATCH_SIZE}
Latency batch size: 1
Warm-up iterations: {WARMUP}
Timing iterations per repetition: {ITERATIONS}
Timing repetitions: {REPETITIONS}
Reported latency: Median of five average measurements

MODEL COMPARISON
-----------------------------------------------------------------
{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}

OBSERVATIONS
-----------------------------------------------------------------
Structured pruning reduces the number of convolutional
filters and connected weights.

Knowledge distillation transfers information from
a larger teacher model to a smaller student model.

Quantization reduces numerical precision and model
storage requirements.

Combining these techniques can further reduce model
size and computational requirements.

Dynamic quantization was applied to Linear layers.

Static quantization was applied to supported
convolutional and Linear layers.

The dynamic and static models were recreated from
the trained FP32 baseline using the same settings
as Tasks 2 and 3.

No additional model training was performed.

The parameter column represents the number of logical
weights and biases before quantization.

A negative accuracy drop means the model achieved
higher accuracy than the original baseline.

All sizes are serialized checkpoint sizes in MiB.

{recommendations}
"""

    print("\n" + report)

    (RESULTS / "task5_results.txt").write_text(
        report,
        encoding="utf-8"
    )

    # Save JSON results
    metrics = {
        "pytorch_version": torch.__version__,
        "backend": BACKEND,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "test_samples": len(test_data),
        "calibration_samples": CALIBRATION_SIZE,
        "random_seed": SEED,
        "warmup_iterations": WARMUP,
        "timing_iterations": ITERATIONS,
        "timing_repetitions": REPETITIONS,
        "latency_statistic": "median",
        "models": df.to_dict(orient="records")
    }

    with open(
        RESULTS / "task5_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    # Plot colours
    colors = [
        "steelblue",
        "darkorange",
        "mediumpurple",
        "seagreen",
        "teal",
        "crimson",
        "goldenrod"
    ]

    # Plot 1: Accuracy vs latency
    plt.figure(figsize=(11, 7))

    for i, row in df.iterrows():

        plt.scatter(
            row["Latency (ms)"],
            row["Accuracy (%)"],
            color=colors[i],
            s=120,
            label=row["Model"]
        )

    plt.title("Accuracy vs Inference Latency")
    plt.xlabel("Inference Latency (ms)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS / "task5_accuracy_vs_latency.png",
        dpi=300
    )

    plt.close()

    # Plot 2: Accuracy vs model size
    plt.figure(figsize=(11, 7))

    for i, row in df.iterrows():

        plt.scatter(
            row["Size (MB)"],
            row["Accuracy (%)"],
            color=colors[i],
            s=120,
            label=row["Model"]
        )

    plt.title("Accuracy vs Model Size")
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS / "task5_accuracy_vs_size.png",
        dpi=300
    )

    plt.close()

    # Plot 3: Overall comparison
    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 7)
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
            color=colors
        )

        ax.set_title(title)
        ax.set_ylabel(column)

        ax.set_ylim(
            0,
            max(df[column]) * 1.2
        )

        ax.tick_params(
            axis="x",
            labelrotation=45
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8
        )

    plt.suptitle(
        "Combined Model Optimisation Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS / "task5_comparison.png",
        dpi=300
    )

    plt.close()

    # Final confirmation
    print("\nQUANTIZED MODELS SAVED")
    print("-" * 65)

    print(dynamic_path)
    print(static_path)
    print(pruned_quantized_path)
    print(distilled_quantized_path)

    print("\nRESULT FILES SAVED")
    print("-" * 65)

    print(RESULTS / "task5_comparison.csv")
    print(RESULTS / "task5_results.txt")
    print(RESULTS / "task5_results.json")
    print(RESULTS / "task5_deployment_recommendations.txt")

    print("\nPLOTS SAVED")
    print("-" * 65)

    print(PLOTS / "task5_accuracy_vs_latency.png")
    print(PLOTS / "task5_accuracy_vs_size.png")
    print(PLOTS / "task5_comparison.png")

    print("\nAll Task 5 results saved successfully.")


if __name__ == "__main__":
    main()