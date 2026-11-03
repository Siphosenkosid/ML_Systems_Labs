
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from transformers import PercentileClipper


# TASK 7: DATASET HASHING AND REPRODUCIBILITY

print("TASK 7: HASHING AND REPRODUCIBILITY")
print("-" * 50)


# 1. Define folder paths
SRC_DIR = Path(__file__).resolve().parent
LAB_DIR = SRC_DIR.parent

config_path = SRC_DIR / "config.yaml"


# 2. Load configuration
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

results_dir = LAB_DIR / config["output"]["directory"]
results_dir.mkdir(parents=True, exist_ok=True)

print("Configuration loaded successfully.")


# 3. Load the original Adult dataset
dataset = fetch_openml(
    name=config["dataset"]["name"],
    version=config["dataset"]["version"],
    as_frame=True
)

df = dataset.frame.copy(deep=True)

target = config["features"]["target"]

X = df.drop(columns=[target])
y = df[target]

print("Original dataset shape:", df.shape)


# 4. Split data before cleaning
split_config = config["split"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=split_config["test_size"],
    random_state=split_config["random_state"],
    stratify=y if split_config["stratify"] else None
)


# 5. Remove exact duplicates from training data only
if config["preprocessing"]["remove_train_duplicates"]:

    train_data = X_train.copy()
    train_data[target] = y_train

    before = len(train_data)

    train_data = train_data.drop_duplicates()

    removed = before - len(train_data)

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]

    print("Training duplicates removed:", removed)


# 6. Handle missing-value markers
X_train = X_train.replace("?", np.nan)
X_test = X_test.replace("?", np.nan)


# 7. Define numerical and categorical columns
numeric_columns = config["features"]["numerical"]
categorical_columns = config["features"]["categorical"]


# 8. Numerical preprocessing
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy=config["preprocessing"]["numeric_imputation"]
        )
    ),
    (
        "clipper",
        PercentileClipper(
            lower=config["preprocessing"]["lower_percentile"],
            upper=config["preprocessing"]["upper_percentile"]
        )
    )
])


# 9. Categorical preprocessing
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy=config["preprocessing"]["categorical_imputation"]
        )
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# 10. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# 11. Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 12. Convert processed data into DataFrames
def make_dataframe(data, columns):

    if hasattr(data, "toarray"):
        data = data.toarray()

    return pd.DataFrame(
        data,
        columns=columns
    )


feature_names = preprocessor.get_feature_names_out()

train_processed = make_dataframe(
    X_train_processed,
    feature_names
)

test_processed = make_dataframe(
    X_test_processed,
    feature_names
)


# 13. Add target labels
train_processed[target] = y_train.to_numpy().astype(str)
test_processed[target] = y_test.to_numpy().astype(str)


# 14. Save processed datasets
train_path = (
    results_dir / config["output"]["processed_train"]
)

test_path = (
    results_dir / config["output"]["processed_test"]
)

train_processed.to_csv(
    train_path,
    index=False,
    float_format="%.12g"
)

test_processed.to_csv(
    test_path,
    index=False,
    float_format="%.12g"
)

print("Processed training shape:", train_processed.shape)
print("Processed testing shape:", test_processed.shape)


# 15. Calculate SHA256 hashes
def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


train_hash = sha256_file(train_path)
test_hash = sha256_file(test_path)


# 16. Generate combined dataset hash
manifest = {
    train_path.name: train_hash,
    test_path.name: test_hash
}

manifest_bytes = json.dumps(
    manifest,
    sort_keys=True,
    separators=(",", ":")
).encode("utf-8")

combined_hash = hashlib.sha256(
    manifest_bytes
).hexdigest()


# 17. Save hash report
hash_path = (
    results_dir / config["output"]["hash_file"]
)

with open(hash_path, "w", encoding="utf-8") as file:

    file.write("TASK 7: SHA256 DATASET HASH REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Training SHA256: {train_hash}\n")
    file.write(f"Testing SHA256: {test_hash}\n")
    file.write(f"Combined SHA256: {combined_hash}\n")


# 18. Save fitted preprocessor
preprocessor_path = results_dir / "preprocessor.joblib"

joblib.dump(
    preprocessor,
    preprocessor_path
)


# 19. Display final results
print("\nSHA256 HASHES")
print("Training:", train_hash)
print("Testing:", test_hash)
print("Combined:", combined_hash)

print("\nFILES SAVED SUCCESSFULLY")
print("Training dataset:", train_path)
print("Testing dataset:", test_path)
print("Hash report:", hash_path)
print("Preprocessor:", preprocessor_path)

print("\nTask 7 completed successfully.")
