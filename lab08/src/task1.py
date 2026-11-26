import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Results folders
results_dir = Path(__file__).resolve().parent.parent / "results"
plots_dir = results_dir / "plots"

results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# Example weight tensor
W = np.array([
    [-0.8, -0.4, 0.2],
    [0.1, 0.3, 0.5]
], dtype=np.float64)


# Symmetric quantization
def quantize(weights, bits, per_channel=False):

    qmax = 2 ** (bits - 1) - 1
    zero_point = 0

    if per_channel:
        scale = np.max(
            np.abs(weights), axis=1, keepdims=True
        ) / qmax
    else:
        scale = np.max(np.abs(weights)) / qmax

    # Avoid division by zero
    scale = np.where(scale == 0, 1.0, scale)

    # Quantize
    quantized = np.clip(
        np.round(weights / scale) + zero_point,
        -qmax,
        qmax
    ).astype(np.int8)

    # Reconstruct
    reconstructed = (quantized - zero_point) * scale

    # Calculate reconstruction error
    mse = np.mean((weights - reconstructed) ** 2)

    return scale, quantized, reconstructed, mse


results = []

report = [
    "LAB 08 - TASK 1: MANUAL QUANTIZATION",
    "=" * 55,
    f"\nOriginal weight tensor:\n{W}"
]

# Compare bit widths and scaling methods
for bits in [8, 4, 2]:

    for per_channel in [False, True]:

        method = (
            "Per-channel" if per_channel else "Per-tensor"
        )

        scale, quantized, reconstructed, mse = quantize(
            W, bits, per_channel
        )

        report.extend([
            "\n" + "-" * 55,
            f"Precision: {bits}-bit",
            f"Method: {method}",
            "Zero-point: 0",
            f"Scale:\n{scale}",
            f"Quantized tensor:\n{quantized}",
            f"Reconstructed tensor:\n{reconstructed}",
            f"MSE: {mse:.10f}"
        ])

        results.append({
            "Bits": bits,
            "Method": method,
            "MSE": mse
        })


# Create comparison table
df = pd.DataFrame(results)

report.extend([
    "\nQUANTIZATION ERROR COMPARISON",
    "=" * 55,
    df.to_string(index=False),
    "\nConclusion:",
    "Lower bit widths reduce numerical precision.",
    "Per-channel scaling adapts to each channel.",
    "The MSE measures reconstruction error."
])

# Print and save report
report_text = "\n".join(report)

print(report_text)

(results_dir / "task1_results.txt").write_text(
    report_text, encoding="utf-8"
)

# Save CSV results
df.to_csv(
    results_dir / "task1_comparison.csv",
    index=False
)

# Plot quantization error
comparison = df.pivot(
    index="Bits",
    columns="Method",
    values="MSE"
).reindex([8, 4, 2])

comparison.plot(kind="bar", figsize=(8, 5))

plt.title("Quantization Reconstruction Error")
plt.xlabel("Bit Width")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=0)
plt.yscale("log")
plt.legend(title="Scaling Method")
plt.tight_layout()

plt.savefig(
    plots_dir / "task1_quantization_error.png",
    dpi=300
)

plt.close()

print("\nAll Task 1 results saved successfully.")