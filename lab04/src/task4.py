
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from transformers import PercentileClipper


# TASK 4: CLEANING AND TRANSFORMATION

print("TASK 4: CLEANING AND TRANSFORMATION")
print("-" * 50)

# 1. Load the original dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy(deep=True)

# Separate input features and target
X = df.drop(columns=["class"])
y = df["class"]


# 2. Split the data before cleaning
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training rows before cleaning:", len(X_train))
print("Testing rows:", len(X_test))


# 3. Remove exact duplicates from training only
train_data = X_train.copy()
train_data["class"] = y_train

duplicate_count = int(train_data.duplicated().sum())

train_data = train_data.drop_duplicates()

X_train = train_data.drop(columns=["class"])
y_train = train_data["class"]

print("Training duplicates removed:", duplicate_count)
print("Training rows after cleaning:", len(X_train))


# 4. Define numerical and categorical features
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
    "native-country"
]


# 5. Handle missing-value markers
# OpenML already represents many '?' values as NaN
X_train = X_train.replace("?", np.nan)
X_test = X_test.replace("?", np.nan)


# 6. Create the numerical preprocessing pipeline
# Imputation and clipping are fitted on training only
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "clipper",
        PercentileClipper(lower=0.01, upper=0.99)
    )
])


# 7. Create the categorical preprocessing pipeline
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# 8. Combine numerical and categorical pipelines
preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


# 9. Fit the preprocessor on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data without fitting again
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# 10. Save the fitted preprocessor
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

preprocessor_path = results_dir / "preprocessor.joblib"

joblib.dump(preprocessor, preprocessor_path)

print("\nPreprocessor saved successfully.")
print("Location:", preprocessor_path)


# 11. Reload and verify the saved preprocessor
loaded_preprocessor = joblib.load(preprocessor_path)

X_test_reloaded = loaded_preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays for comparison
def to_array(data):
    if hasattr(data, "toarray"):
        return data.toarray()
    return np.asarray(data)


np.testing.assert_allclose(
    to_array(X_test_processed),
    to_array(X_test_reloaded)
)

print("Saved preprocessor verification: PASSED")


# 12. Check for missing values after preprocessing
train_array = to_array(X_train_processed)
test_array = to_array(X_test_processed)

assert np.isfinite(train_array).all(), (
    "Invalid values found in processed training data"
)

assert np.isfinite(test_array).all(), (
    "Invalid values found in processed testing data"
)

print("Processed data validation: PASSED")
print("\nTask 4 completed successfully.")
