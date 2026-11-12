
# Lab 05 - Task 5
# Training a 2-8-1 Neural Network
# Custom Autograd vs PyTorch

from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from task3 import Value


# -----------------------------------------
# 1. Extend our Value engine
# -----------------------------------------

def value_neg(self):
    return self * -1.0


def value_sub(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return self + (-other)


def value_rsub(self, other):
    return other + (-self)


def value_sigmoid(self):
    # Stable sigmoid calculation
    if self.data >= 0:
        z = np.exp(-self.data)
        s = 1.0 / (1.0 + z)
    else:
        z = np.exp(self.data)
        s = z / (1.0 + z)

    out = Value(float(s), (self,), "sigmoid")

    def _backward():
        self.grad += s * (1.0 - s) * out.grad

    out._backward = _backward
    return out


def value_log(self):
    if self.data <= 0:
        raise ValueError("Log input must be positive.")

    out = Value(float(np.log(self.data)), (self,), "log")

    def _backward():
        self.grad += (1.0 / self.data) * out.grad

    out._backward = _backward
    return out


Value.__neg__ = value_neg
Value.__sub__ = value_sub
Value.__rsub__ = value_rsub
Value.sigmoid = value_sigmoid
Value.log = value_log


# -----------------------------------------
# 2. Configuration
# -----------------------------------------

SEED = 42
N_SAMPLES = 100
EPOCHS = 60
LEARNING_RATE = 0.05
RUNS = 5

torch.set_num_threads(1)


# -----------------------------------------
# 3. Prepare make_moons dataset
# -----------------------------------------

X, y = make_moons(
    n_samples=N_SAMPLES,
    noise=0.15,
    random_state=SEED
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)


# -----------------------------------------
# 4. Initialize identical parameters
# -----------------------------------------

rng = np.random.default_rng(SEED)

W1 = rng.normal(0, 0.3, (2, 8))
b1 = np.zeros(8)

W2 = rng.normal(0, 0.3, (8, 1))
b2 = np.zeros(1)


# -----------------------------------------
# 5. Custom autograd neural network
# -----------------------------------------

class CustomNetwork:
    def __init__(self):
        self.w1 = [
            [Value(W1[i, j]) for j in range(8)]
            for i in range(2)
        ]
        self.b1 = [Value(v) for v in b1]

        self.w2 = [Value(W2[j, 0]) for j in range(8)]
        self.b2 = Value(b2[0])

        self.parameters = (
            [v for row in self.w1 for v in row]
            + self.b1
            + self.w2
            + [self.b2]
        )

    def forward(self, x):
        hidden = []

        for j in range(8):
            z = self.b1[j]

            for i in range(2):
                z = z + self.w1[i][j] * float(x[i])

            hidden.append(z.relu())

        z = self.b2

        for j in range(8):
            z = z + self.w2[j] * hidden[j]

        return z.sigmoid()


def custom_bce(prediction, target):
    # Numerically safe BCE for the small training experiment
    eps = 1e-12

    # Clamp through the computation graph
    if prediction.data < eps:
        prediction = Value(eps)
    elif prediction.data > 1.0 - eps:
        prediction = Value(1.0 - eps)

    return -(
        target * prediction.log()
        + (1.0 - target) * (1.0 - prediction).log()
    )


def train_custom():
    model = CustomNetwork()

    start = perf_counter()

    for epoch in range(EPOCHS):
        for x, target in zip(X_train, y_train):
            prediction = model.forward(x)
            loss = custom_bce(prediction, float(target))

            loss.backward()

            for parameter in model.parameters:
                parameter.data -= LEARNING_RATE * parameter.grad

    elapsed = perf_counter() - start

    predictions = np.array([
        model.forward(x).data for x in X_test
    ])

    accuracy = np.mean(
        (predictions >= 0.5) == y_test
    )

    return elapsed, float(accuracy)


# -----------------------------------------
# 6. PyTorch neural network
# -----------------------------------------

class TorchNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(2, 8, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(8, 1, dtype=torch.float64)

        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(W1.T.copy())
            )
            self.fc1.bias.copy_(
                torch.tensor(b1.copy())
            )
            self.fc2.weight.copy_(
                torch.tensor(W2.T.copy())
            )
            self.fc2.bias.copy_(
                torch.tensor(b2.copy())
            )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def train_torch():
    model = TorchNetwork()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.BCELoss()

    x_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(
        y_train.reshape(-1, 1)
    )

    x_test_t = torch.tensor(X_test)

    start = perf_counter()

    for epoch in range(EPOCHS):
        # Same sample-by-sample SGD as custom engine
        for i in range(len(x_train_t)):
            optimizer.zero_grad()

            prediction = model(x_train_t[i:i+1])
            loss = criterion(
                prediction,
                y_train_t[i:i+1]
            )

            loss.backward()
            optimizer.step()

    elapsed = perf_counter() - start

    with torch.no_grad():
        predictions = model(x_test_t).numpy().ravel()

    accuracy = np.mean(
        (predictions >= 0.5) == y_test
    )

    return elapsed, float(accuracy)


# -----------------------------------------
# 7. Run experiments 5 times
# -----------------------------------------

def main():
    custom_times = []
    torch_times = []

    custom_accuracies = []
    torch_accuracies = []

    lines = [
        "LAB 05 - TASK 5: NEURAL NETWORK TRAINING",
        "-" * 60,
        "Dataset: make_moons",
        f"Total samples: {N_SAMPLES}",
        f"Training samples: {len(X_train)}",
        f"Testing samples: {len(X_test)}",
        "Architecture: 2 -> 8 -> 1",
        "Hidden activation: ReLU",
        "Output activation: Sigmoid",
        "Loss: Binary Cross-Entropy",
        f"Epochs: {EPOCHS}",
        f"Learning rate: {LEARNING_RATE}",
        f"Number of runs: {RUNS}",
        "",
        "1. TRAINING RESULTS"
    ]

    for run in range(1, RUNS + 1):
        custom_time, custom_acc = train_custom()
        torch_time, torch_acc = train_torch()

        custom_times.append(custom_time)
        torch_times.append(torch_time)

        custom_accuracies.append(custom_acc)
        torch_accuracies.append(torch_acc)

        lines.extend([
            "",
            f"Run {run}",
            f"Custom time: {custom_time:.4f} seconds",
            f"Custom accuracy: {custom_acc:.2%}",
            f"PyTorch time: {torch_time:.4f} seconds",
            f"PyTorch accuracy: {torch_acc:.2%}"
        ])

    custom_median = median(custom_times)
    torch_median = median(torch_times)

    speed_ratio = custom_median / torch_median

    lines.extend([
        "",
        "2. MEDIAN OF 5 RUNS",
        f"Custom median time: {custom_median:.4f} seconds",
        f"PyTorch median time: {torch_median:.4f} seconds",
        f"Custom median accuracy: {median(custom_accuracies):.2%}",
        f"PyTorch median accuracy: {median(torch_accuracies):.2%}",
        f"Custom / PyTorch time ratio: {speed_ratio:.2f}x",
        "",
        "3. CONCLUSION",
        (
            "PyTorch was faster."
            if torch_median < custom_median
            else "Custom engine was faster."
            if custom_median < torch_median
            else "Both had equal median times."
        ),
        "Both models used the same architecture,",
        "initial parameters, data, and SGD settings."
    ])

    output = "\n".join(lines)
    print(output)

    results_dir = (
        Path(__file__).resolve().parent.parent
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = results_dir / "task5_results.txt"

    output_file.write_text(
        output + "\n",
        encoding="utf-8"
    )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
