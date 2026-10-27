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

# Scale inputs to 0-1
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
# NETWORK INITIALIZATION
# ============================================================

W = 16

np.random.seed(42)

W1 = np.random.normal(0, 0.01, (64, W))
W2 = np.random.normal(0, 0.01, (W, W))
W3 = np.random.normal(0, 0.01, (W, 10))

b1 = np.zeros(W)
b2 = np.zeros(W)
b3 = np.zeros(10)


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
# FORWARD PROPAGATION
# ============================================================

def forward(X):

    # Hidden layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Hidden layer 2
    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    # Output layer
    z3 = a2 @ W3 + b3
    output = softmax(z3)

    # Save values needed during backpropagation
    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "z3": z3,
        "output": output
    }

    return output, cache


# ============================================================
# CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(y_true, y_pred):

    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0
    )

    loss = -np.mean(
        np.sum(
            y_true * np.log(y_pred),
            axis=1
        )
    )

    return loss


# ============================================================
# BACKPROPAGATION
# ============================================================

def backward(X, y_true, cache):

    # Number of examples in the batch
    m = X.shape[0]

    a1 = cache["a1"]
    z1 = cache["z1"]

    a2 = cache["a2"]
    z2 = cache["z2"]

    output = cache["output"]


    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------

    dz3 = (output - y_true) / m

    dW3 = a2.T @ dz3

    db3 = np.sum(
        dz3,
        axis=0
    )


    # --------------------------------------------------------
    # HIDDEN LAYER 2
    # --------------------------------------------------------

    da2 = dz3 @ W3.T

    dz2 = da2 * relu_derivative(z2)

    dW2 = a1.T @ dz2

    db2 = np.sum(
        dz2,
        axis=0
    )


    # --------------------------------------------------------
    # HIDDEN LAYER 1
    # --------------------------------------------------------

    da1 = dz2 @ W2.T

    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T @ dz1

    db1 = np.sum(
        dz1,
        axis=0
    )


    gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3
    }

    return gradients


# ============================================================
# NUMERICAL GRADIENT CHECKING
# ============================================================

def gradient_check(X_batch, y_batch, num_checks=10):

    epsilon = 1e-5

    # Calculate analytical gradients
    output, cache = forward(X_batch)

    gradients = backward(
        X_batch,
        y_batch,
        cache
    )

    # We will check random weights from W3
    analytic_gradient = gradients["W3"]

    rng = np.random.default_rng(42)

    passed = 0

    print("\nGRADIENT CHECK")
    print("-" * 70)

    for check in range(num_checks):

        # Pick one random weight from W3
        i = rng.integers(0, W3.shape[0])
        j = rng.integers(0, W3.shape[1])

        original_value = W3[i, j]


        # ----------------------------------------------------
        # Loss when weight is increased by epsilon
        # ----------------------------------------------------

        W3[i, j] = original_value + epsilon

        output_plus, _ = forward(X_batch)

        loss_plus = cross_entropy_loss(
            y_batch,
            output_plus
        )


        # ----------------------------------------------------
        # Loss when weight is decreased by epsilon
        # ----------------------------------------------------

        W3[i, j] = original_value - epsilon

        output_minus, _ = forward(X_batch)

        loss_minus = cross_entropy_loss(
            y_batch,
            output_minus
        )


        # Restore original weight
        W3[i, j] = original_value


        # Numerical gradient
        numerical = (
            loss_plus - loss_minus
        ) / (2 * epsilon)


        # Analytical gradient
        analytical = analytic_gradient[i, j]


        # Relative error
        denominator = max(
            1e-12,
            abs(numerical) + abs(analytical)
        )

        relative_error = (
            abs(numerical - analytical)
            / denominator
        )


        if relative_error < 1e-6:
            passed += 1
            result = "PASS"
        else:
            result = "FAIL"


        print(
            f"Check {check + 1}: "
            f"W3[{i},{j}] | "
            f"Analytical = {analytical:.10f} | "
            f"Numerical = {numerical:.10f} | "
            f"Relative error = {relative_error:.2e} | "
            f"{result}"
        )


    print("-" * 70)

    print(
        f"Passed gradient checks: "
        f"{passed}/{num_checks}"
    )

    return passed


# ============================================================
# TEST TASK 5
# ============================================================

# Use a batch of 32 examples
X_batch = X_train[:32]
y_batch = y_train[:32]


# Forward propagation
output, cache = forward(X_batch)


# Loss before training
loss = cross_entropy_loss(
    y_batch,
    output
)

print("Loss:", loss)


# Backpropagation
gradients = backward(
    X_batch,
    y_batch,
    cache
)


# Display gradient shapes
print("\nGradient shapes:")

print("dW1:", gradients["W1"].shape)
print("db1:", gradients["b1"].shape)

print("dW2:", gradients["W2"].shape)
print("db2:", gradients["b2"].shape)

print("dW3:", gradients["W3"].shape)
print("db3:", gradients["b3"].shape)


# Perform numerical gradient checking
passed = gradient_check(
    X_batch,
    y_batch,
    num_checks=10
)


# Final requirement
if passed >= 9:
    print("\nGradient check requirement PASSED.")
else:
    print("\nGradient check requirement NOT PASSED.")