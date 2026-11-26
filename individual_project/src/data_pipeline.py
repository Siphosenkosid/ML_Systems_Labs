import json
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Project paths
ROOT = Path(__file__).resolve().parents[2]

PROJECT = ROOT / "individual_project"
RESULTS = PROJECT / "results"
MODELS = RESULTS / "models"
PLOTS = RESULTS / "plots"
PROCESSED = PROJECT / "data" / "processed"

CONFIG_PATH = PROJECT / "src" / "config.yaml"

for folder in [RESULTS, MODELS, PLOTS, PROCESSED]:
    folder.mkdir(parents=True, exist_ok=True)


def main():

    print("INDIVIDUAL PROJECT - DATA PREPROCESSING")
    print("=" * 60)

    # Load configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_path = ROOT / config["dataset"]["path"]
    target = config["dataset"]["target"]

    # Load dataset
    df = pd.read_csv(dataset_path)

    original_rows = len(df)
    missing_values = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    original_distribution = df[target].value_counts()

    print("\n1. ORIGINAL DATASET")
    print("-" * 60)

    print("Dataset shape:", df.shape)
    print("Missing values:", missing_values)
    print("Duplicate rows:", duplicates)

    print("\nOriginal class distribution:")
    print(original_distribution)

    # Remove duplicate rows
    if config["dataset"]["remove_duplicates"]:
        df = df.drop_duplicates().reset_index(drop=True)

    cleaned_rows = len(df)
    removed_rows = original_rows - cleaned_rows

    print("\n2. DATA CLEANING")
    print("-" * 60)

    print("Duplicates removed:", removed_rows)
    print("Cleaned dataset shape:", df.shape)

    print("\nCleaned class distribution:")
    print(df[target].value_counts())

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y if config["split"]["stratify"] else None
    )

    print("\n3. TRAIN-TEST SPLIT")
    print("-" * 60)

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # Build preprocessing pipeline
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply training transformations to test data
    X_test_processed = preprocessor.transform(X_test)

    print("\n4. FEATURE PREPROCESSING")
    print("-" * 60)

    print("Scaling method: StandardScaler")
    print("Preprocessor fitted on training data only")

    print(
        "Processed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )

    # Validate processed data
    train_valid = bool(
        np.isfinite(X_train_processed).all()
    )

    test_valid = bool(
        np.isfinite(X_test_processed).all()
    )

    if not train_valid or not test_valid:
        raise ValueError(
            "Processed data contains invalid values."
        )

    print("\n5. DATA VALIDATION")
    print("-" * 60)

    print("Training data valid:", train_valid)
    print("Testing data valid:", test_valid)

    # Save processed arrays
    np.savez_compressed(
        PROCESSED / "processed_data.npz",
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy()
    )

    # Save fitted preprocessing pipeline
    joblib.dump(
        preprocessor,
        MODELS / "preprocessor.joblib"
    )

    # Save feature names
    with open(
        RESULTS / "feature_names.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(X.columns.tolist(), file, indent=4)

    # Helper for JSON class distributions
    def class_distribution(labels):

        return {
            str(k): int(v)
            for k, v in labels.value_counts().items()
        }

    # Save preprocessing metrics
    metrics = {
        "project": config["project_name"],
        "original_rows": original_rows,
        "original_columns": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicates,
        "duplicates_removed": removed_rows,
        "cleaned_rows": cleaned_rows,
        "number_of_features": int(X.shape[1]),
        "target": target,
        "original_class_distribution": {
            str(k): int(v)
            for k, v in original_distribution.items()
        },
        "cleaned_class_distribution": class_distribution(y),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "training_class_distribution": class_distribution(y_train),
        "testing_class_distribution": class_distribution(y_test),
        "fraud_percentage": float(y.mean() * 100),
        "test_size": config["split"]["test_size"],
        "random_state": config["split"]["random_state"],
        "stratified_split": config["split"]["stratify"],
        "scaling": "StandardScaler",
        "imputation": "Median",
        "training_data_valid": train_valid,
        "testing_data_valid": test_valid
    }

    with open(
        RESULTS / "data_pipeline_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    # Plot class distribution
    counts = y.value_counts().sort_index()

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        ["Legitimate", "Fraudulent"],
        counts.values,
        color=["steelblue", "crimson"]
    )

    plt.title("Credit Card Transaction Class Distribution")
    plt.ylabel("Number of Transactions")
    plt.yscale("log")

    plt.bar_label(
        bars,
        fmt="%d",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS / "class_distribution.png",
        dpi=300
    )

    plt.close()

    # Plot train-test distribution
    distributions = pd.DataFrame({
        "Training": y_train.value_counts(),
        "Testing": y_test.value_counts()
    }).sort_index()

    distributions.index = [
        "Legitimate",
        "Fraudulent"
    ]

    ax = distributions.plot(
        kind="bar",
        figsize=(9, 6),
        color=["steelblue", "darkorange"],
        logy=True
    )

    ax.set_title("Training and Testing Class Distribution")
    ax.set_ylabel("Number of Transactions")
    ax.set_xlabel("Transaction Class")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        PLOTS / "train_test_distribution.png",
        dpi=300
    )

    plt.close()

    # Save text report
    report = f"""
CREDIT CARD FRAUD DETECTION
DATA PREPROCESSING REPORT
============================================================

Original dataset rows: {original_rows}
Missing values: {missing_values}
Duplicate rows removed: {removed_rows}
Cleaned dataset rows: {cleaned_rows}
Input features: {X.shape[1]}
Target variable: {target}

TRAIN-TEST SPLIT
------------------------------------------------------------
Training samples: {len(X_train)}
Testing samples: {len(X_test)}
Random seed: {config["split"]["random_state"]}
Stratified split: {config["split"]["stratify"]}

TRAINING CLASS DISTRIBUTION
------------------------------------------------------------
{y_train.value_counts().to_string()}

TESTING CLASS DISTRIBUTION
------------------------------------------------------------
{y_test.value_counts().to_string()}

PREPROCESSING
------------------------------------------------------------
Duplicate removal
Median imputation
StandardScaler
Training-only fitting

DATA VALIDATION
------------------------------------------------------------
Training data valid: {train_valid}
Testing data valid: {test_valid}

CONCLUSION
------------------------------------------------------------
The dataset was cleaned and divided into training and
testing sets using stratified sampling to preserve the
class distribution.

Feature scaling was fitted only on the training data
to prevent preprocessing-related data leakage.

Class imbalance was preserved during preprocessing.
Class weighting will be applied during baseline training.
"""

    (RESULTS / "data_pipeline_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    print("\n6. SAVED FILES")
    print("-" * 60)

    print(PROCESSED / "processed_data.npz")
    print(MODELS / "preprocessor.joblib")
    print(RESULTS / "feature_names.json")
    print(RESULTS / "data_pipeline_metrics.json")
    print(RESULTS / "data_pipeline_report.txt")
    print(PLOTS / "class_distribution.png")
    print(PLOTS / "train_test_distribution.png")

    print("\nData preprocessing completed successfully.")


if __name__ == "__main__":
    main()