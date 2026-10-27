
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from transformers import PercentileClipper


# TASK 5: FEATURE ENGINEERING

print("TASK 5: FEATURE ENGINEERING")
print("-" * 50)

# 1. Load the original dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy(deep=True)

X = df.drop(columns=["class"])
y = (df["class"].astype(str) == ">50K").astype(int)


# 2. Split before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 3. Remove exact duplicates from training only
train_data = X_train.copy()
train_data["target"] = y_train

duplicate_count = int(train_data.duplicated().sum())
train_data = train_data.drop_duplicates()

X_train = train_data.drop(columns=["target"])
y_train = train_data["target"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training duplicates removed:", duplicate_count)


# 4. Handle missing-value markers
X_train = X_train.replace("?", np.nan)
X_test = X_test.replace("?", np.nan)


# 5. Define original feature columns
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


# 6. Create the engineered features
def add_features(data):
    data = data.copy()

    data["capital_net"] = (
        data["capital-gain"] - data["capital-loss"]
    )

    data["hours_per_age"] = (
        data["hours-per-week"] / (data["age"] - 16)
    )

    return data


X_train_engineered = add_features(X_train)
X_test_engineered = add_features(X_test)

engineered_numeric = numeric_columns + [
    "capital_net",
    "hours_per_age"
]

print("\nEngineered features created:")
print(["capital_net", "hours_per_age"])


# 7. Build a preprocessing pipeline
def build_preprocessor(numeric_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper(
            lower=0.01,
            upper=0.99
        )),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline,
         categorical_columns)
    ])


# 8. Train and evaluate a model
def evaluate_model(X_train_data, X_test_data, numeric_features):

    model = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features)),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train_data, y_train)

    predictions = model.predict(X_test_data)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, pos_label=1)

    return model, accuracy, f1


# 9. Baseline model
print("\nTraining baseline model...")

baseline_model, baseline_accuracy, baseline_f1 = evaluate_model(
    X_train,
    X_test,
    numeric_columns
)


# 10. Model with engineered features
print("Training engineered model...")

engineered_model, engineered_accuracy, engineered_f1 = evaluate_model(
    X_train_engineered,
    X_test_engineered,
    engineered_numeric
)


# 11. Compare results
results = pd.DataFrame({
    "Model": ["Baseline", "Engineered"],
    "Accuracy": [baseline_accuracy, engineered_accuracy],
    "F1 Score": [baseline_f1, engineered_f1]
})

print("\nMODEL COMPARISON")
print("-" * 50)
print(results.to_string(index=False))

print("\nAccuracy difference:",
      round(engineered_accuracy - baseline_accuracy, 4))

print("F1 difference:",
      round(engineered_f1 - baseline_f1, 4))


# 12. Save results and models
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(
    results_dir / "task5_model_comparison.csv",
    index=False
)

joblib.dump(
    baseline_model,
    results_dir / "baseline_model.joblib"
)

joblib.dump(
    engineered_model,
    results_dir / "engineered_model.joblib"
)

print("\nResults and models saved successfully.")
print("Task 5 completed successfully.")
