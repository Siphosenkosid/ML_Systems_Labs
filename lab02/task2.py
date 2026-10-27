import numpy as np

# Hidden layer width
W = 16

# Initialize weights
W1 = np.random.normal(0, 0.01, (64, W))
W2 = np.random.normal(0, 0.01, (W, W))
W3 = np.random.normal(0, 0.01, (W, 10))

# Initialize biases with zeros
b1 = np.zeros(W)
b2 = np.zeros(W)
b3 = np.zeros(10)

# Check parameter shapes
print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)

print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)

print("W3 shape:", W3.shape)
print("b3 shape:", b3.shape)