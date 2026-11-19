
import csv
import io
import statistics
import time
from pathlib import Path

import torch

from task2 import BaselineCNN, DEVICE, MODELS, RESULTS
from task2_structured import StructuredCNN

# Settings
torch.manual_seed(42)
WARMUP = 50
REPETITIONS = 5
RUNS_PER_REPETITION = 200

SPARSE_DIR = RESULTS / "sparse"
SPARSE_DIR.mkdir(parents=True, exist_ok=True)


def load_model(name):
    if name == "baseline":
        model = BaselineCNN()
        path = MODELS / "baseline_cnn.pth"

    elif name.startswith("structured_"):
        percent = int(name.split("_")[1])
        channels1 = round(32 * (1 - percent / 100))
        channels2 = round(64 * (1 - percent / 100))

        model = StructuredCNN(channels1, channels2)
        path = MODELS / f"structured_pruned_{percent}.pth"

    else:
        method, percent = name.split("_")
        model = BaselineCNN()
        path = MODELS / f"{method}_pruned_{percent}.pth"

    state = torch.load(
        path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)

    return model.to(DEVICE).eval(), path


def measure_latency(model):
    # Batch size 1, CPU inference
    sample = torch.randn(1, 1, 28, 28, device=DEVICE)
    measurements = []

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(sample)

        for _ in range(REPETITIONS):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            for _ in range(RUNS_PER_REPETITION):
                model(sample)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            measurements.append(
                elapsed * 1000 / RUNS_PER_REPETITION
            )

    return statistics.median(measurements)


def count_sparsity(model):
    zeros = 0
    total = 0

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weights = module.weight.detach()
            zeros += (weights == 0).sum().item()
            total += weights.numel()

    return 100 * zeros / total


def save_sparse_format(model, name):
    # Sparse COO representation of model tensors
    sparse_state = {}

    for key, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()

        # Convert weight matrices to sparse COO
        if key.endswith("weight") and tensor.ndim >= 2:
            sparse_state[key] = tensor.to_sparse_coo().coalesce()
        else:
            sparse_state[key] = tensor.clone()

    path = SPARSE_DIR / f"{name}_sparse.pth"
    torch.save(sparse_state, path)

    # Verify that sparse tensors recover original values
    loaded = torch.load(
        path, map_location="cpu", weights_only=True
    )

    for key, original in model.state_dict().items():
        restored = loaded[key]
        if restored.layout != torch.strided:
            restored = restored.to_dense()

        assert torch.equal(
            original.detach().cpu(), restored
        ), f"Mismatch in {key}"

    return path.stat().st_size / (1024 ** 2)


def main():
    print("LAB 07 - TASK 3: LATENCY AND STORAGE")
    print("-" * 55)
    print("Device:", DEVICE)
    print("Batch size: 1")
    print("Warmup:", WARMUP)
    print("Repetitions:", REPETITIONS)
    print("Runs per repetition:", RUNS_PER_REPETITION)

    model_names = [
        "baseline",
        "global_50",
        "global_90",
        "global_95",
        "layerwise_50",
        "layerwise_90",
        "layerwise_95",
        "structured_10",
        "structured_25",
        "structured_50"
    ]

    results = []

    for name in model_names:
        print(f"\nTesting: {name}", flush=True)

        model, dense_path = load_model(name)

        latency = measure_latency(model)
        sparsity = count_sparsity(model)

        dense_size = (
            dense_path.stat().st_size / (1024 ** 2)
        )

        sparse_size = save_sparse_format(model, name)

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        print(f"  Parameters: {parameters:,}")
        print(f"  Weight sparsity: {sparsity:.2f}%")
        print(f"  Dense size: {dense_size:.3f} MB")
        print(f"  Sparse size: {sparse_size:.3f} MB")
        print(f"  CPU latency: {latency:.4f} ms")

        results.append({
            "model": name,
            "parameters": parameters,
            "weight_sparsity_percent": sparsity,
            "dense_size_mb": dense_size,
            "sparse_size_mb": sparse_size,
            "latency_ms": latency
        })

    output = RESULTS / "task3_results.csv"

    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nTASK 3 SUMMARY")
    print("-" * 55)

    for row in results:
        print(
            f"{row['model']:16} | "
            f"Dense: {row['dense_size_mb']:.2f} MB | "
            f"Sparse: {row['sparse_size_mb']:.2f} MB | "
            f"Latency: {row['latency_ms']:.4f} ms"
        )

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
