from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the digits dataset
digits = load_digits()

X = digits.data
y = digits.target

# Scale the input values from 0-16 to 0-1
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

# Check the results
print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

print("\nFirst scaled input:")
print(X_train[0])

print("\nFirst one-hot encoded label:")
print(y_train[0])