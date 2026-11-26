import time
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

ONNX_PATH = MODELS / "baseline_cnn.onnx"

MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# Settings
torch.manual_seed(42)
torch.set_num_threads(1)

BATCH_SIZE = 128
TOLERANCE = 1e-4
OPSET_VERSION = 17


# Evaluate ONNX model
def evaluate_onnx(session, loader):

    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0

    for images, labels in loader:

        outputs = session.run(
            None,
            {input_name: images.numpy()}
        )[0]

        predictions = np.argmax(
            outputs, axis=1
        )

        correct += np.sum(
            predictions == labels.numpy()
        )

        total += labels.size(0)

    return float(100 * correct / total)


# Measure ONNX inference latency
def measure_onnx_latency(session, sample):

    input_name = session.get_inputs()[0].name

    input_data = {
        input_name: sample.numpy()
    }

    warmup = 50
    repetitions = 500

    for _ in range(warmup):
        session.run(None, input_data)

    start = time.perf_counter()

    for _ in range(repetitions):
        session.run(None, input_data)

    elapsed = time.perf_counter() - start

    return elapsed / repetitions * 1000


def main():

    print("LAB 08 - TASK 4: ONNX EXPORT")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("ONNX version:", onnx.__version__)
    print("ONNX Runtime version:", ort.__version__)
    print("Device: CPU")
    print("ONNX opset:", OPSET_VERSION)
    print("Tolerance:", TOLERANCE)

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

    # Load trained baseline CNN
    model = BaselineCNN()

    model.load_state_dict(
        torch.load(
            BASELINE_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    print("\nBaseline model loaded successfully.")

    # Example input
    sample = test_data[0][0].unsqueeze(0)

    # Export PyTorch model to ONNX
    print("\nExporting model to ONNX...")

    with torch.no_grad():

        torch.onnx.export(
            model,
            sample,
            str(ONNX_PATH),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )

    print("ONNX model exported successfully.")

    # Validate ONNX model structure
    print("\nChecking ONNX model structure...")

    onnx_model = onnx.load(str(ONNX_PATH))

    onnx.checker.check_model(onnx_model)

    print("ONNX structure validation: PASSED")

    # Configure ONNX Runtime
    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    print("\nRuntime providers:", session.get_providers())

    # Numerical validation using all test images
    print("\nComparing PyTorch and ONNX outputs...")

    input_name = session.get_inputs()[0].name

    max_difference = 0.0
    total_difference = 0.0
    total_values = 0

    matching_predictions = 0
    total_samples = 0

    with torch.inference_mode():

        for images, labels in test_loader:

            pytorch_output = model(images).numpy()

            onnx_output = session.run(
                None,
                {input_name: images.numpy()}
            )[0]

            difference = np.abs(
                pytorch_output - onnx_output
            )

            max_difference = max(
                max_difference,
                float(difference.max())
            )

            total_difference += float(
                difference.sum(dtype=np.float64)
            )

            total_values += difference.size

            pytorch_predictions = np.argmax(
                pytorch_output, axis=1
            )

            onnx_predictions = np.argmax(
                onnx_output, axis=1
            )

            matching_predictions += int(
                np.sum(
                    pytorch_predictions
                    == onnx_predictions
                )
            )

            total_samples += labels.size(0)

    mean_difference = (
        total_difference / total_values
    )

    prediction_agreement = (
        100 * matching_predictions / total_samples
    )

    validation_passed = (
        max_difference <= TOLERANCE
    )

    validation_status = (
        "PASSED" if validation_passed else "FAILED"
    )

    print(f"Maximum difference: {max_difference:.10f}")
    print(f"Mean difference: {mean_difference:.10f}")
    print(f"Prediction agreement: {prediction_agreement:.2f}%")
    print("Numerical validation:", validation_status)

    # Evaluate PyTorch model
    print("\nEvaluating PyTorch model...")

    pytorch_accuracy = evaluate(
        model, test_loader
    )

    pytorch_latency = measure_latency(
        model, sample
    )

    pytorch_size = model_size(BASELINE_PATH)

    # Evaluate ONNX model
    print("\nEvaluating ONNX model...")

    onnx_accuracy = evaluate_onnx(
        session, test_loader
    )

    onnx_latency = measure_onnx_latency(
        session, sample
    )

    onnx_size = model_size(ONNX_PATH)

    # Calculate speedup
    speedup = pytorch_latency / onnx_latency

    accuracy_difference = (
        pytorch_accuracy - onnx_accuracy
    )

    # Comparison table
    results = [
        {
            "Model": "PyTorch FP32",
            "Accuracy (%)": pytorch_accuracy,
            "Size (MB)": pytorch_size,
            "Latency (ms)": pytorch_latency
        },
        {
            "Model": "ONNX FP32",
            "Accuracy (%)": onnx_accuracy,
            "Size (MB)": onnx_size,
            "Latency (ms)": onnx_latency
        }
    ]

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS / "task4_comparison.csv",
        index=False
    )

    # Create report
    report = f"""
LAB 08 - TASK 4: ONNX EXPORT AND VALIDATION
============================================================

PyTorch version: {torch.__version__}
ONNX version: {onnx.__version__}
ONNX Runtime version: {ort.__version__}
ONNX opset: {OPSET_VERSION}
Device: CPU
Runtime provider: CPUExecutionProvider
CPU threads: 1
Test samples: {len(test_data)}
Evaluation batch size: {BATCH_SIZE}
Latency batch size: 1
Warm-up iterations: 50
Timing iterations: 500

ONNX VALIDATION
------------------------------------------------------------
Structure validation: PASSED
Numerical tolerance: {TOLERANCE}
Maximum absolute difference: {max_difference:.10f}
Mean absolute difference: {mean_difference:.10f}
Prediction agreement: {prediction_agreement:.2f}%
Numerical validation: {validation_status}

MODEL COMPARISON
------------------------------------------------------------
{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}

PERFORMANCE ANALYSIS
------------------------------------------------------------
Accuracy difference: {accuracy_difference:.4f} percentage points
ONNX inference speedup: {speedup:.2f}x

OBSERVATIONS
------------------------------------------------------------
ONNX provides a portable model representation.
ONNX Runtime executes the exported model independently
of the PyTorch model implementation.

Graph optimization may improve inference performance.

The exported model remains in FP32 precision.
ONNX export alone does not perform INT8 quantization.
"""

    print(report)

    (RESULTS / "task4_results.txt").write_text(
        report,
        encoding="utf-8"
    )

    # Save JSON metrics
    metrics = {
        "device": "cpu",
        "runtime_provider": "CPUExecutionProvider",
        "opset_version": OPSET_VERSION,
        "cpu_threads": 1,
        "validation": {
            "tolerance": TOLERANCE,
            "maximum_absolute_difference": max_difference,
            "mean_absolute_difference": mean_difference,
            "prediction_agreement_percent": prediction_agreement,
            "passed": validation_passed
        },
        "pytorch": {
            "accuracy_percent": pytorch_accuracy,
            "size_mb": pytorch_size,
            "latency_ms": pytorch_latency
        },
        "onnx": {
            "accuracy_percent": onnx_accuracy,
            "size_mb": onnx_size,
            "latency_ms": onnx_latency
        },
        "speedup": speedup
    }

    with open(
        RESULTS / "task4_results.json",
        "w"
    ) as f:

        json.dump(metrics, f, indent=4)

    # Generate comparison plots
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 5)
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
            color=["steelblue", "seagreen"]
        )

        ax.set_title(title)
        ax.set_ylabel(column)
        ax.set_ylim(0, max(df[column]) * 1.2)

        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=3
        )

    plt.suptitle("PyTorch vs ONNX Runtime")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "task4_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nONNX model saved to:", ONNX_PATH)
    print("All Task 4 results saved successfully.")

    if not validation_passed:
        print("WARNING: Numerical tolerance was exceeded.")


if __name__ == "__main__":
    main()