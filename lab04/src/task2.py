
from sklearn.datasets import fetch_openml

# Task 2: Data Ingestion

# Load the original UCI Adult dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

# Store the raw dataset without modifying it
raw_data = adult.frame.copy(deep=True)

# Record dataset dimensions
row_count, column_count = raw_data.shape

print("TASK 2: DATA INGESTION")
print("-" * 40)

print("Dataset: UCI Adult")
print("Number of rows:", row_count)
print("Number of columns:", column_count)

# Display column names
print("\nColumn names:")
print(raw_data.columns.tolist())

# Display missing values
print("\nMissing values per column:")
print(raw_data.isna().sum())

# Verify the expected dimensions
assert row_count == 48842, "Unexpected row count"
assert column_count == 15, "Unexpected column count"

print("\nData ingestion completed successfully.")
