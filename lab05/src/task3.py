
# Lab 05 - Task 3
# Backpropagation using Topological Sorting

from pathlib import Path


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
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
            self.grad += (
                exponent * self.data ** (exponent - 1)
            ) * out.grad

        out._backward = _backward
        return out

    # ReLU activation
    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (
                1.0 if self.data > 0 else 0.0
            ) * out.grad

        out._backward = _backward
        return out

    # Scalar sum
    def sum(self):
        return self

    # Backward propagation
    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        # Step 1: Topological sorting
        build_topo(self)

        # Reset gradients for this backward pass
        for node in topo:
            node.grad = 0.0

        # Step 2: Set output gradient to 1
        self.grad = 1.0

        # Step 3: Reverse traversal
        for node in reversed(topo):
            node._backward()

        return topo


def main():
    # Input values
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)

    # Forward pass
    product = a * b
    addition = product + c
    squared = addition ** 2
    output = squared.relu().sum()

    # Backward pass
    topo = output.backward()

    results = [
        "LAB 05 - TASK 3: BACKPROPAGATION",
        "-" * 45,
        "",
        "1. Forward Pass",
        f"a = {a.data}",
        f"b = {b.data}",
        f"c = {c.data}",
        f"a * b = {product.data}",
        f"a * b + c = {addition.data}",
        f"(a * b + c) ** 2 = {squared.data}",
        f"Final output = {output.data}",
        "",
        "2. Backward Pass",
        f"Output gradient = {output.grad}",
        f"Gradient of a = {a.grad}",
        f"Gradient of b = {b.grad}",
        f"Gradient of c = {c.grad}",
        "",
        "3. Topological Sorting",
        f"Number of nodes = {len(topo)}",
        "Forward topological order:",
    ]

    for i, node in enumerate(topo, start=1):
        results.append(
            f"{i}. Operation: {node._op or 'Input'}, "
            f"Data: {node.data}, Gradient: {node.grad}"
        )

    # Verify expected gradients
    expected = {"a": 42.0, "b": 28.0, "c": 14.0}
    actual = {"a": a.grad, "b": b.grad, "c": c.grad}

    results.extend([
        "",
        "4. Gradient Verification",
    ])

    for name in expected:
        difference = abs(actual[name] - expected[name])
        results.append(
            f"{name}: Expected = {expected[name]}, "
            f"Actual = {actual[name]}, Difference = {difference}"
        )

    passed = all(
        abs(actual[name] - expected[name]) < 1e-6
        for name in expected
    )

    results.append(
        f"Gradient verification: {'PASSED' if passed else 'FAILED'}"
    )

    # Test gradient accumulation with shared input
    x = Value(3.0)
    shared_output = x * x + x
    shared_output.backward()

    results.extend([
        "",
        "5. Gradient Accumulation Test",
        "Expression: x * x + x",
        f"x = {x.data}",
        f"Output = {shared_output.data}",
        f"Expected gradient = 7.0",
        f"Actual gradient = {x.grad}",
        f"Accumulation test: {'PASSED' if abs(x.grad - 7.0) < 1e-6 else 'FAILED'}"
    ])

    # Display results
    text = "\n".join(results)
    print(text)

    # Save results automatically
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "task3_results.txt"
    output_file.write_text(text + "\n", encoding="utf-8")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
