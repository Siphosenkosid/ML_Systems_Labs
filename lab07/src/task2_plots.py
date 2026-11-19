
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
PLOTS = RESULTS / "plots"

PLOTS.mkdir(parents=True, exist_ok=True)

# Load saved results
global_df = pd.read_csv(
    RESULTS / "task2_global_results.csv"
)

layer_df = pd.read_csv(
    RESULTS / "task2_layerwise_results.csv"
)

structured_df = pd.read_csv(
    RESULTS / "task2_structured_results.csv"
)

# Graph 1: Global pruning
plt.figure(figsize=(8, 5))

plt.plot(
    global_df["actual_sparsity"],
    global_df["accuracy_before"],
    marker="o",
    label="Before fine-tuning"
)

plt.plot(
    global_df["actual_sparsity"],
    global_df["accuracy_after"],
    marker="s",
    label="After fine-tuning"
)

plt.title("Global Magnitude Pruning")
plt.xlabel("Weight sparsity (%)")
plt.ylabel("Test accuracy (%)")
plt.xticks([0, 25, 50, 75, 90, 95])
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(PLOTS / "task2_global_accuracy.png", dpi=300)
plt.close()

# Graph 2: Layer-wise pruning
plt.figure(figsize=(8, 5))

plt.plot(
    layer_df["actual_sparsity"],
    layer_df["accuracy_before"],
    marker="o",
    label="Before fine-tuning"
)

plt.plot(
    layer_df["actual_sparsity"],
    layer_df["accuracy_after"],
    marker="s",
    label="After fine-tuning"
)

plt.title("Layer-wise Magnitude Pruning")
plt.xlabel("Weight sparsity (%)")
plt.ylabel("Test accuracy (%)")
plt.xticks([0, 25, 50, 75, 90, 95])
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(PLOTS / "task2_layerwise_accuracy.png", dpi=300)
plt.close()

# Graph 3: Comparison of all methods
plt.figure(figsize=(9, 5))

plt.plot(
    global_df["actual_sparsity"],
    global_df["accuracy_after"],
    marker="o",
    label="Global pruning"
)

plt.plot(
    layer_df["actual_sparsity"],
    layer_df["accuracy_after"],
    marker="s",
    label="Layer-wise pruning"
)

plt.plot(
    [0] + structured_df[
        "target_filter_pruning_percent"
    ].tolist(),
    [90.94] + structured_df[
        "accuracy_after"
    ].tolist(),
    marker="^",
    label="Structured filter pruning"
)

plt.title("Pruning Methods: Accuracy Comparison")
plt.xlabel("Pruning target (%)")
plt.ylabel("Test accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(PLOTS / "task2_pruning_comparison.png", dpi=300)
plt.close()

print("LAB 07 - TASK 2: PLOTS")
print("-" * 50)
print("Global pruning graph saved.")
print("Layer-wise pruning graph saved.")
print("Pruning comparison graph saved.")
print("All plots saved in:", PLOTS)
