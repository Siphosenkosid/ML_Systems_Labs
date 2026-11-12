
# Lab 05 - Task 6
# PyTorch Framework Performance
# Eager vs Traced Execution and Tensor Contiguity

from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# -----------------------------------------
# 1. Configuration
# -----------------------------------------

torch.manual_seed(42)
torch.set_num_threads(1)

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 100
REPEATS = 5
ITERATIONS = 500


# -----------------------------------------
# 2. Define a small neural network
# -----------------------------------------

class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# -----------------------------------------
# 3. Timing function
# -----------------------------------------

def benchmark(function, warmup=WARMUP,
              repeats=REPEATS, iterations=ITERATIONS):

    # Warm-up before measuring
    for _ in range(warmup):
        function()

    times = []

    for _ in range(repeats):
        start = perf_counter()

        for _ in range(iterations):
            function()

        elapsed = perf_counter() - start

        # Average time per operation in milliseconds
        time_ms = (elapsed / iterations) * 1000
        times.append(time_ms)

    return median(times), times


# -----------------------------------------
# 4. Main experiment
# -----------------------------------------

def main():

    lines = [
        "LAB 05 - TASK 6: PYTORCH FRAMEWORK PERFORMANCE",
        "-" * 65,
        "",
        f"PyTorch version: {torch.__version__}",
        "Device: CPU",
        f"Warm-up iterations: {WARMUP}",
        f"Timing repetitions: {REPEATS}",
        f"Operations per repetition: {ITERATIONS}",
        ""
    ]

    # -------------------------------------
    # PART 1: Inspect computation graph
    # -------------------------------------

    model = SmallModel()
    model.eval()

    example_input = torch.randn(32, 2)

    with torch.no_grad():
        eager_output = model(example_input)

        # Trace model execution
        traced_model = torch.jit.trace(
            model,
            example_input
        )

        traced_output = traced_model(example_input)

    graph_text = str(traced_model.inlined_graph)

    graph_file = RESULTS_DIR / "task6_graph.txt"
    graph_file.write_text(
        graph_text,
        encoding="utf-8"
    )

    max_output_diff = torch.max(
        torch.abs(eager_output - traced_output)
    ).item()

    lines.extend([
        "1. COMPUTATION GRAPH INSPECTION",
        "Model architecture: 2 -> 8 -> 1",
        "Hidden activation: ReLU",
        "Output activation: Sigmoid",
        f"Input shape: {tuple(example_input.shape)}",
        f"Output shape: {tuple(eager_output.shape)}",
        f"Model parameters: {sum(p.numel() for p in model.parameters())}",
        f"Maximum output difference: {max_output_diff:.12e}",
        f"Graph saved to: {graph_file.name}",
        "",
        "Graph operations:"
    ])

    graph_ops = [
        "aten::linear",
        "aten::relu",
        "aten::sigmoid"
    ]

    for op in graph_ops:
        lines.append(
            f"{op}: {'Found' if op in graph_text else 'Not found'}"
        )

    lines.append("")

    # -------------------------------------
    # PART 2: Eager vs traced execution
    # -------------------------------------

    with torch.inference_mode():

        def eager_forward():
            return model(example_input)

        def traced_forward():
            return traced_model(example_input)

        eager_median, eager_times = benchmark(
            eager_forward
        )

        traced_median, traced_times = benchmark(
            traced_forward
        )

    speedup = eager_median / traced_median

    lines.extend([
        "2. EAGER VS TRACED EXECUTION",
        "",
        "Eager times (ms per forward pass):",
    ])

    for i, value in enumerate(eager_times, 1):
        lines.append(f"Run {i}: {value:.6f} ms")

    lines.append("")
    lines.append("Traced times (ms per forward pass):")

    for i, value in enumerate(traced_times, 1):
        lines.append(f"Run {i}: {value:.6f} ms")

    lines.extend([
        "",
        f"Eager median: {eager_median:.6f} ms",
        f"Traced median: {traced_median:.6f} ms",
        f"Eager / traced ratio: {speedup:.3f}x",
        (
            "Traced execution was faster."
            if speedup > 1
            else "Eager execution was faster."
            if speedup < 1
            else "Both had equal median times."
        ),
        ""
    ])

    # -------------------------------------
    # PART 3: Tensor properties
    # -------------------------------------

    original = torch.randn(512, 512)

    # Transposing creates a non-contiguous view
    non_contiguous = original.T

    # Make a contiguous copy of the same values
    contiguous = non_contiguous.contiguous()

    lines.extend([
        "3. TENSOR PROPERTIES",
        "",
        "Original tensor:",
        f"Shape: {tuple(original.shape)}",
        f"Dtype: {original.dtype}",
        f"Strides: {original.stride()}",
        f"Contiguous: {original.is_contiguous()}",
        "",
        "Transposed view:",
        f"Shape: {tuple(non_contiguous.shape)}",
        f"Dtype: {non_contiguous.dtype}",
        f"Strides: {non_contiguous.stride()}",
        f"Contiguous: {non_contiguous.is_contiguous()}",
        "",
        "Contiguous copy:",
        f"Shape: {tuple(contiguous.shape)}",
        f"Dtype: {contiguous.dtype}",
        f"Strides: {contiguous.stride()}",
        f"Contiguous: {contiguous.is_contiguous()}",
        "",
        f"Values identical: {torch.equal(non_contiguous, contiguous)}",
        ""
    ])

    # -------------------------------------
    # PART 4: Contiguity performance
    # -------------------------------------

    with torch.inference_mode():

        def contiguous_operation():
            return torch.sin(contiguous)

        def non_contiguous_operation():
            return torch.sin(non_contiguous)

        cont_median, cont_times = benchmark(
            contiguous_operation
        )

        noncont_median, noncont_times = benchmark(
            non_contiguous_operation
        )

    ratio = noncont_median / cont_median

    lines.extend([
        "4. CONTIGUOUS VS NON-CONTIGUOUS PERFORMANCE",
        "Operation: torch.sin",
        "Tensor shape: (512, 512)",
        "",
        "Contiguous times (ms):"
    ])

    for i, value in enumerate(cont_times, 1):
        lines.append(f"Run {i}: {value:.6f} ms")

    lines.append("")
    lines.append("Non-contiguous times (ms):")

    for i, value in enumerate(noncont_times, 1):
        lines.append(f"Run {i}: {value:.6f} ms")

    lines.extend([
        "",
        f"Contiguous median: {cont_median:.6f} ms",
        f"Non-contiguous median: {noncont_median:.6f} ms",
        f"Non-contiguous / contiguous ratio: {ratio:.3f}x",
        (
            "Non-contiguous operation was slower."
            if ratio > 1
            else "Non-contiguous operation was faster."
            if ratio < 1
            else "Both had equal median times."
        ),
        ""
    ])

    # -------------------------------------
    # PART 5: Performance chart
    # -------------------------------------

    labels = [
        "Eager",
        "Traced",
        "Contiguous",
        "Non-contiguous"
    ]

    values = [
        eager_median,
        traced_median,
        cont_median,
        noncont_median
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values)

    ax.set_ylabel("Median time per operation (ms)")
    ax.set_title("Lab 05 - Task 6 Performance Comparison")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()

    chart_file = RESULTS_DIR / "task6_performance.png"

    fig.savefig(
        chart_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------
    # PART 6: Conclusion
    # -------------------------------------

    lines.extend([
        "5. CONCLUSION",
        (
            "Tracing improved execution speed for this model."
            if traced_median < eager_median
            else "Tracing did not improve execution speed for this model."
        ),
        (
            "Non-contiguous layout increased operation time."
            if noncont_median > cont_median
            else "Non-contiguous layout did not increase operation time."
        ),
        "Results depend on model size, tensor layout,",
        "operation type, and execution overhead.",
        "",
        f"Performance chart: {chart_file.name}"
    ])

    # -------------------------------------
    # PART 7: Save all results
    # -------------------------------------

    output = "\n".join(lines)

    results_file = RESULTS_DIR / "task6_results.txt"

    results_file.write_text(
        output + "\n",
        encoding="utf-8"
    )

    print(output)

    print("\nAll Task 6 results saved successfully.")
    print(f"Results: {results_file}")
    print(f"Graph: {graph_file}")
    print(f"Chart: {chart_file}")


if __name__ == "__main__":
    main()
