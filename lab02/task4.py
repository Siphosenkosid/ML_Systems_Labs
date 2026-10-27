import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# TASK 1: DATA PREPARATION
# ============================================================

# Load the digits dataset
digits = load_digits()

X = digits.data
y = digits.target

# Scale pixel values from 0-16 to 0-1
X = X / 16.0

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset: 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_onehot,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# TASK 2: NETWORK PARAMETERS
# ============================================================

# Hidden layer width
W = 16

# Set random seed so results can be reproduced
np.random.seed(42)

# Initialize weights from N(0, 0.01)
W1 = np.random.normal(0, 0.01, (64, W))
W2 = np.random.normal(0, 0.01, (W, W))
W3 = np.random.normal(0, 0.01, (W, 10))

# Initialize biases to zero
b1 = np.zeros(W)
b2 = np.zeros(W)
b3 = np.zeros(10)


# ============================================================
# TASK 3: ACTIVATION FUNCTIONS
# ============================================================

def relu(z):
    """
    ReLU activation function.
    Negative values become 0.
    Positive values remain unchanged.
    """
    return np.maximum(0, z)


def softmax(z):
    """
    Softmax converts the output values into probabilities.
    Each row of probabilities should sum to 1.
    """

    # Subtract maximum value for numerical stability
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    probabilities = exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True
    )

    return probabilities


# ============================================================
# TASK 3: FORWARD PROPAGATION
# ============================================================

def forward(X):
    """
    Perform forward propagation through the network.
    """

    # Hidden layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Hidden layer 2
    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    # Output layer
    z3 = a2 @ W3 + b3
    output = softmax(z3)

    return output


# ============================================================
# TASK 4: CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(y_true, y_pred):
    """
    Calculate the average cross-entropy loss.
    """

    # Small value to prevent log(0)
    epsilon = 1e-12

    # Keep probabilities in a safe numerical range
    y_pred = np.clip(y_pred, epsilon, 1.0)

    # Cross-entropy loss
    loss = -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )

    return loss


# ============================================================
# TEST TASK 3
# ============================================================

# Take the first 32 training examples
batch = X_train[:32]

# Perform forward propagation
output = forward(batch)

# Check output shape
print("Output shape:", output.shape)

# Check that Softmax probabilities sum to 1
row_sums = np.sum(output, axis=1)

print("\nFirst 5 row sums:")
print(row_sums[:5])

print("\nAll rows sum to 1 within 1e-6:")
print(np.allclose(row_sums, 1.0, atol=1e-6))


# ============================================================
# TEST TASK 4
# ============================================================

# Get the labels that correspond to the same 32 examples
y_batch = y_train[:32]

# Calculate cross-entropy loss
loss = cross_entropy_loss(y_batch, output)

print("\nUntrained network loss:")
print(loss)

# Expected sanity-check value
expected_loss = np.log(10)

print("\nExpected loss (ln(10)):")
print(expected_loss)

print("\nDifference:")
print(abs(loss - expected_loss))