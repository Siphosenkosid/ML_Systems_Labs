
# Lab 05 - Task 2
# Operations and Computation Graph

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

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Power
    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be a number.")

        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # ReLU activation
    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # Sum operation for scalar Value nodes
    def sum(self):
        return self


def main():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)

    # Test each operation
    addition = a + b
    multiplication = a * b
    power = a ** 2
    relu_positive = Value(5.0).relu()
    relu_negative = Value(-3.0).relu()

    # Build the complete computation graph
    product = a * b
    total = product + c
    squared = total ** 2
    output = squared.relu().sum()

    results = [
        "LAB 05 - TASK 2: OPERATIONS AND COMPUTATION GRAPH",
        "-" * 55,
        "",
        "1. Addition",
        f"2 + 3 = {addition.data}",
        "",
        "2. Multiplication",
        f"2 * 3 = {multiplication.data}",
        "",
        "3. Power",
        f"2 ** 2 = {power.data}",
        "",
        "4. ReLU",
        f"ReLU(5) = {relu_positive.data}",
        f"ReLU(-3) = {relu_negative.data}",
        "",
        "5. Complete Computation Graph",
        "Expression: ((a * b + c) ** 2).relu().sum()",
        f"a = {a.data}",
        f"b = {b.data}",
        f"c = {c.data}",
        f"a * b = {product.data}",
        f"a * b + c = {total.data}",
        f"(a * b + c) ** 2 = {squared.data}",
        f"Final output = {output.data}",
        "",
        "6. Graph Information",
        f"Final operation: {output._op}",
        f"Final node parents: {len(output._prev)}",
        f"Gradient initialized to: {output.grad}",
    ]

    # Display results in terminal
    text = "\n".join(results)
    print(text)

    # Save results automatically
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "task2_results.txt"
    output_file.write_text(text + "\n", encoding="utf-8")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
