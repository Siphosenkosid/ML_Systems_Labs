import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# TASK 1 DATA PREPARATION
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

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_onehot,
    test_size=0.30,
    random_state=42,
    stratify=y
)


#============================================================
# TASK 2 NETWORK PARAMETERS
# ============================================================

# Hidden layer width
W = 16

# Make random initialization reproducible
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
# TASK 3 FORWARD PROPAGATION
# ============================================================

# ReLU activation function
def relu(z):
    return np.maximum(0, z)


# Softmax activation function
def softmax(z):
    # Subtract the maximum value for numerical stability
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Forward pass
def forward(X):

    # ----- Hidden Layer 1 -----
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # ----- Hidden Layer 2 -----
    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    # ----- Output Layer -----
    z3 = a2 @ W3 + b3
    output = softmax(z3)

    return output


# ============================================================
# TEST THE FORWARD PASS
# ============================================================

# Use 32 training examples as one batch
batch = X_train[:32]

# Perform forward propagation
output = forward(batch)

# Check 1: Output must have shape (batch, 10)
print("Output shape:", output.shape)

# Check 2: Every row should sum to 1
row_sums = np.sum(output, axis=1)

print("\nFirst 5 row sums:")
print(row_sums[:5])

print("\nAll rows sum to 1 within 1e-6:")
print(np.allclose(row_sums, 1.0, atol=1e-6))


