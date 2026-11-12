
# Lab 05 - Task 1
# Designing a Value Node for Automatic Differentiation

from pathlib import Path


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


def main():
    # Create Value nodes
    a = Value(2.0)
    b = Value(3.0)
    c = Value(6.0, (a, b), "*")

    # Collect results
    results = [
        "LAB 05 - TASK 1: VALUE NODE",
        "-" * 40,
        f"Node a: {a}",
        f"Node b: {b}",
        f"Node c: {c}",
        "",
        "Node c information:",
        f"Data: {c.data}",
        f"Gradient: {c.grad}",
        f"Operation: {c._op}",
        f"Number of parents: {len(c._prev)}",
        f"Local derivative rule callable: {callable(c._backward)}"
    ]

    # Display results in terminal
    output = "\n".join(results)
    print(output)

    # Save results in lab05/results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "task1_results.txt"
    output_file.write_text(output + "\n", encoding="utf-8")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
