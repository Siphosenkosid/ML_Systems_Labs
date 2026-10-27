import time
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# DATA PREPARATION
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

# Scale inputs by dividing by 16
X = X / 16.0

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# 70/30 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_onehot,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    # Numerical stability
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True
    )


# ============================================================
# INITIALIZE NETWORK
# ============================================================

def initialize_network(width):

    rng = np.random.default_rng(42)

    # Weights from N(0, 0.01)
    W1 = rng.normal(0, 0.01, (64, width))
    W2 = rng.normal(0, 0.01, (width, width))
    W3 = rng.normal(0, 0.01, (width, 10))

    # Biases
    b1 = np.zeros(width)
    b2 = np.zeros(width)
    b3 = np.zeros(10)

    return W1, b1, W2, b2, W3, b3


# ============================================================
# FORWARD PROPAGATION
# ============================================================

def forward(X, W1, b1, W2, b2, W3, b3):

    # Hidden layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Hidden layer 2
    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    # Output layer
    z3 = a2 @ W3 + b3
    output = softmax(z3)

    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "output": output
    }

    return output, cache


# ============================================================
# BACKPROPAGATION
# ============================================================

def backward(X, y_true, cache, W2, W3):

    m = X.shape[0]

    z1 = cache["z1"]
    a1 = cache["a1"]

    z2 = cache["z2"]
    a2 = cache["a2"]

    output = cache["output"]

    # ----- Output layer -----

    dz3 = (output - y_true) / m

    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)


    # ----- Hidden layer 2 -----

    da2 = dz3 @ W3.T

    dz2 = da2 * relu_derivative(z2)

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)


    # ----- Hidden layer 1 -----

    da1 = dz2 @ W2.T

    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2, dW3, db3


# ============================================================
# TIMING FUNCTION
# ============================================================

def measure_time(width, runs=100):

    print("\n" + "=" * 50)
    print(f"Timing network with W = {width}")
    print("=" * 50)

    # Initialize network
    W1, b1, W2, b2, W3, b3 = initialize_network(width)

    # Use mini-batch size 32
    X_batch = X_train[:32]
    y_batch = y_train[:32]

    forward_times = []
    backward_times = []


    # Warm-up runs
    for _ in range(10):

        output, cache = forward(
            X_batch,
            W1, b1,
            W2, b2,
            W3, b3
        )

        backward(
            X_batch,
            y_batch,
            cache,
            W2,
            W3
        )


    # ========================================================
    # 100 TIMING RUNS
    # ========================================================

    for _ in range(runs):

        # --------------------------
        # Time forward propagation
        # --------------------------

        start = time.perf_counter()

        output, cache = forward(
            X_batch,
            W1, b1,
            W2, b2,
            W3, b3
        )

        end = time.perf_counter()

        forward_times.append(end - start)


        # --------------------------
        # Time backward propagation
        # --------------------------

        start = time.perf_counter()

        backward(
            X_batch,
            y_batch,
            cache,
            W2,
            W3
        )

        end = time.perf_counter()

        backward_times.append(end - start)


    # Median of 100 runs
    median_forward = np.median(forward_times)
    median_backward = np.median(backward_times)


    print(f"Number of runs: {runs}")

    print(
        f"Median forward time: "
        f"{median_forward * 1000:.6f} ms"
    )

    print(
        f"Median backward time: "
        f"{median_backward * 1000:.6f} ms"
    )


    if median_backward > median_forward:
        print("Backward pass took longer.")
    else:
        print("Forward pass took longer.")

    return median_forward, median_backward


# ============================================================
# TASK 7
# ============================================================

forward_16, backward_16 = measure_time(
    width=16,
    runs=100
)

forward_64, backward_64 = measure_time(
    width=64,
    runs=100
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)

print(
    f"W=16 | Forward: {forward_16 * 1000:.6f} ms | "
    f"Backward: {backward_16 * 1000:.6f} ms"
)

print(
    f"W=64 | Forward: {forward_64 * 1000:.6f} ms | "
    f"Backward: {backward_64 * 1000:.6f} ms"
)