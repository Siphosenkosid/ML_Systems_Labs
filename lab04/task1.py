
from sklearn.datasets import fetch_openml

# Load the UCI Adult dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

# Display dataset dimensions
print("Dataset shape:", adult.frame.shape)

# Display the first five rows
print("\nFirst five rows:")
print(adult.frame.head())

