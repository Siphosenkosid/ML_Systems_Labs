
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

# TASK 3: DATA VALIDATION

# Load the original dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy(deep=True)

print("TASK 3: DATA VALIDATION")
print("-" * 50)

# 1. Define the expected schema
numeric_columns = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "class"
]

# 2. Validate column names and types
expected_columns = numeric_columns + categorical_columns

missing_columns = sorted(set(expected_columns) - set(df.columns))
extra_columns = sorted(set(df.columns) - set(expected_columns))

type_violations = {}

for column in numeric_columns:
    if column in df.columns:
        invalid = (
            pd.to_numeric(df[column], errors="coerce").isna()
            & df[column].notna()
        )
        type_violations[column] = int(invalid.sum())

for column in categorical_columns:
    if column in df.columns:
        invalid = df[column].dropna().map(
            lambda value: not isinstance(value, str)
        )
        type_violations[column] = int(invalid.sum())

# 3. Define numerical ranges
ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356)
}

# 4. Check range violations
range_violations = {}

for column, (minimum, maximum) in ranges.items():
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = (
            (values < minimum) | (values > maximum)
        )
        range_violations[column] = int(invalid.sum())

# 5. Check missing values
# The manual defines '?' as missing
missing_counts = {}

for column in df.columns:
    missing = df[column].isna() | (
        df[column].astype("string").str.strip() == "?"
    ).fillna(False)

    missing_counts[column] = int(missing.sum())

# 6. Validate categorical values against
# the categories supplied by OpenML
category_violations = {}

for column in categorical_columns:
    if column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            allowed = set(df[column].cat.categories.astype(str))
            values = df[column].astype("string").str.strip()
            invalid = (
                values.notna()
                & (values != "?")
                & ~values.isin(allowed)
            )
            category_violations[column] = int(invalid.sum())
        else:
            category_violations[column] = (
                "No declared category list"
            )

# 7. Display validation results
print("\nMissing columns:", missing_columns)
print("Extra columns:", extra_columns)

print("\nTYPE VIOLATIONS")
print(type_violations)

print("\nRANGE VIOLATIONS")
print(range_violations)

print("\nMISSING VALUES")
print(pd.Series(missing_counts))

print("\nCATEGORY VIOLATIONS")
print(category_violations)

# 8. Save the validation report
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)

report_path = results_dir / "validation_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("LAB 4 - DATA VALIDATION REPORT\n")
    report.write("=" * 50 + "\n")
    report.write(f"Dataset shape: {df.shape}\n")
    report.write(f"Missing columns: {missing_columns}\n")
    report.write(f"Extra columns: {extra_columns}\n\n")
    report.write(f"Type violations:\n{type_violations}\n\n")
    report.write(f"Range violations:\n{range_violations}\n\n")
    report.write(f"Missing values:\n{pd.Series(missing_counts)}\n\n")
    report.write(f"Category violations:\n{category_violations}\n")

print("\nValidation report saved to:", report_path)
print("No rows were removed or modified.")
