
# Lab 05 - Task 4
# Gradient Verification: Custom Autograd vs PyTorch vs Numerical

from pathlib import Path
import torch

from task3 import Value


# Function for numerical differentiation
def function(a, b, c):
    return max(0.0, (a * b + c) ** 2)


def numerical_gradient(a, b, c, variable, h=1e-5):
    values_plus = [a, b, c]
    values_minus = [a, b, c]

    values_plus[variable] += h
    values_minus[variable] -= h

    f_plus = function(*values_plus)
    f_minus = function(*values_minus)

    return (f_plus - f_minus) / (2 * h)


def main():
    # Input values
    a_val = 2.0
    b_val = 3.0
    c_val = 1.0

    # ---------------------------------
    # 1. Custom Autograd Engine
    # ---------------------------------
    a = Value(a_val)
    b = Value(b_val)
    c = Value(c_val)

    output = ((a * b + c) ** 2).relu().sum()
    output.backward()

    custom_grads = [a.grad, b.grad, c.grad]

    # ---------------------------------
    # 2. PyTorch Autograd
    # ---------------------------------
    a_t = torch.tensor(a_val, dtype=torch.float64,
                       requires_grad=True)
    b_t = torch.tensor(b_val, dtype=torch.float64,
                       requires_grad=True)
    c_t = torch.tensor(c_val, dtype=torch.float64,
                       requires_grad=True)

    torch_output = ((a_t * b_t + c_t) ** 2).relu().sum()
    torch_output.backward()

    torch_grads = [
        a_t.grad.item(),
        b_t.grad.item(),
        c_t.grad.item()
    ]

    # ---------------------------------
    # 3. Numerical Gradients
    # ---------------------------------
    numerical_grads = [
        numerical_gradient(a_val, b_val, c_val, i)
        for i in range(3)
    ]

    # ---------------------------------
    # 4. Compare Gradients
    # ---------------------------------
    names = ["a", "b", "c"]
    tolerance = 1e-6

    results = [
        "LAB 05 - TASK 4: GRADIENT VERIFICATION",
        "-" * 65,
        "",
        "Function: ReLU((a*b+c)^2)",
        f"Input values: a={a_val}, b={b_val}, c={c_val}",
        f"Custom output: {output.data}",
        f"PyTorch output: {torch_output.item()}",
        "",
        "1. GRADIENT COMPARISON",
        f"{'Variable':<10}{'Custom':<15}"
        f"{'PyTorch':<15}{'Numerical':<15}",
        "-" * 65
    ]

    for i, name in enumerate(names):
        results.append(
            f"{name:<10}"
            f"{custom_grads[i]:<15.8f}"
            f"{torch_grads[i]:<15.8f}"
            f"{numerical_grads[i]:<15.8f}"
        )

    # Calculate absolute differences
    custom_torch_diff = [
        abs(custom_grads[i] - torch_grads[i])
        for i in range(3)
    ]

    custom_numerical_diff = [
        abs(custom_grads[i] - numerical_grads[i])
        for i in range(3)
    ]

    torch_numerical_diff = [
        abs(torch_grads[i] - numerical_grads[i])
        for i in range(3)
    ]

    max_diff = max(
        custom_torch_diff
        + custom_numerical_diff
        + torch_numerical_diff
    )

    results.extend([
        "",
        "2. MAXIMUM ABSOLUTE DIFFERENCES",
        f"Custom vs PyTorch: {max(custom_torch_diff):.12e}",
        f"Custom vs Numerical: {max(custom_numerical_diff):.12e}",
        f"PyTorch vs Numerical: {max(torch_numerical_diff):.12e}",
        "",
        f"Overall maximum difference: {max_diff:.12e}",
        f"Required tolerance: {tolerance:.12e}",
        "",
        "3. FINAL VERIFICATION",
        f"Status: {'PASSED' if max_diff < tolerance else 'FAILED'}"
    ])

    # ---------------------------------
    # 5. Save Results
    # ---------------------------------
    text = "\n".join(results)
    print(text)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "task4_results.txt"
    output_file.write_text(text + "\n", encoding="utf-8")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
