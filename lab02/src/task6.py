import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# TASK 1: DATA PREPARATION
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
# CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(y_true, y_pred):

    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0
    )

    return -np.mean(
        np.sum(
            y_true * np.log(y_pred),
            axis=1
        )
    )


# ============================================================
# INITIALIZE NETWORK
# ============================================================

def initialize_network(width, seed=42):

    rng = np.random.default_rng(seed)

    # Weight initialization N(0, 0.01)
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

def backward(
    X,
    y_true,
    cache,
    W2,
    W3
):

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
# ACCURACY
# ============================================================

def accuracy(y_true, y_pred):

    # Convert one-hot labels back to class numbers
    true_classes = np.argmax(y_true, axis=1)

    # Select class with highest predicted probability
    predicted_classes = np.argmax(y_pred, axis=1)

    return np.mean(
        true_classes == predicted_classes
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_network(
    width,
    learning_rate,
    epochs=50,
    batch_size=32
):

    print("\n" + "=" * 60)
    print(
        f"Training: width={width}, "
        f"learning rate={learning_rate}"
    )
    print("=" * 60)

    # Initialize network
    W1, b1, W2, b2, W3, b3 = initialize_network(
        width
    )

    # Store loss for plotting
    loss_history = []

    # Random generator for shuffling
    rng = np.random.default_rng(42)


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(epochs):

        # Create shuffled indices
        indices = rng.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]


        # ----------------------------------------------------
        # MINI-BATCH LOOP
        # ----------------------------------------------------

        for start in range(
            0,
            len(X_train),
            batch_size
        ):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]


            # Forward propagation
            output, cache = forward(
                X_batch,
                W1,
                b1,
                W2,
                b2,
                W3,
                b3
            )


            # Backpropagation
            (
                dW1,
                db1,
                dW2,
                db2,
                dW3,
                db3
            ) = backward(
                X_batch,
                y_batch,
                cache,
                W2,
                W3
            )


            # ------------------------------------------------
            # OPTIMIZATION / GRADIENT DESCENT
            # ------------------------------------------------

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3


        # ----------------------------------------------------
        # LOSS AFTER EACH EPOCH
        # ----------------------------------------------------

        train_output, _ = forward(
            X_train,
            W1,
            b1,
            W2,
            b2,
            W3,
            b3
        )

        epoch_loss = cross_entropy_loss(
            y_train,
            train_output
        )

        loss_history.append(epoch_loss)

        print(
            f"Epoch {epoch + 1:02d}/50 "
            f"- Loss: {epoch_loss:.6f}"
        )


    # ========================================================
    # TEST ACCURACY
    # ========================================================

    test_output, _ = forward(
        X_test,
        W1,
        b1,
        W2,
        b2,
        W3,
        b3
    )

    test_accuracy = accuracy(
        y_test,
        test_output
    )

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    return loss_history, test_accuracy


# ============================================================
# TASK 6 EXPERIMENTS
# ============================================================

learning_rates = [
    0.001,
    0.01,
    0.1
]

widths = [
    16,
    64
]


# Store results
all_losses = {}
final_accuracies = {}


for width in widths:

    for learning_rate in learning_rates:

        losses, test_accuracy = train_network(
            width=width,
            learning_rate=learning_rate,
            epochs=50,
            batch_size=32
        )

        name = (
            f"W={width}, LR={learning_rate}"
        )

        all_losses[name] = losses
        final_accuracies[name] = test_accuracy


# ============================================================
# DISPLAY FINAL TEST ACCURACIES
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST ACCURACIES")
print("=" * 60)

for name, test_accuracy in final_accuracies.items():

    print(
        f"{name}: "
        f"{test_accuracy * 100:.2f}%"
    )


# ============================================================
# PLOT LOSS CURVES
# ============================================================

plt.figure(figsize=(10, 6))

for name, losses in all_losses.items():

    plt.plot(
        range(1, 51),
        losses,
        label=name
    )


plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True)

plt.tight_layout()


# ============================================================
# SAVE RESULTS
# ============================================================

# Create results folder if it does not already exist
results_folder = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(
    results_folder,
    exist_ok=True
)

# Correct required filename
plot_path = os.path.join(
    results_folder,
    "loss_curves.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

print(
    "\nLoss curves saved to:",
    plot_path
)

plt.show()