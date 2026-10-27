import os
import time
import statistics
import joblib
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
X, y = load_breast_cancer(return_X_y=True)

# Use the same 70/30 split as Task 2
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


def test_system_cost(model_name, model_function):
    print(f"\nTesting {model_name}...")

    # -----------------------------
    # Training time
    # -----------------------------

    # Warm-up run
    warmup_model = model_function()
    warmup_model.fit(X_train, y_train)

    training_times = []

    for _ in range(5):
        model = model_function()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        training_times.append((end - start) * 1000)

    median_training_time = statistics.median(training_times)

    # Train one final model for the remaining tests
    model = model_function()
    model.fit(X_train, y_train)

    # -----------------------------
    # Inference latency
    # -----------------------------

    sample = X_test[0].reshape(1, -1)

    # Warm-up prediction
    model.predict(sample)

    inference_times = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        inference_times.append((end - start) * 1000)

    median_inference_time = statistics.median(inference_times)

    # -----------------------------
    # Model size
    # -----------------------------

    os.makedirs("lab01/results", exist_ok=True)

    file_name = model_name.lower().replace(" ", "_")
    model_path = f"lab01/results/{file_name}.joblib"

    joblib.dump(model, model_path)

    model_size_bytes = os.path.getsize(model_path)
    model_size_kb = model_size_bytes / 1024

    # -----------------------------
    # Memory
    # -----------------------------

    process = psutil.Process()

    memory_before = process.memory_info().rss

    memory_model = model_function()
    memory_model.fit(X_train, y_train)

    memory_after_training = process.memory_info().rss

    training_memory_mb = max(
        0,
        memory_after_training - memory_before
    ) / (1024 ** 2)

    memory_before_inference = process.memory_info().rss

    memory_model.predict(sample)

    memory_after_inference = process.memory_info().rss

    inference_memory_mb = max(
        0,
        memory_after_inference - memory_before_inference
    ) / (1024 ** 2)

    # -----------------------------
    # Print results
    # -----------------------------

    print(f"Median training time: {median_training_time:.4f} ms")
    print(f"Median inference latency: {median_inference_time:.4f} ms")
    print(f"Model size: {model_size_bytes} bytes")
    print(f"Model size: {model_size_kb:.2f} KB")
    print(f"Training memory: {training_memory_mb:.2f} MB")
    print(f"Inference memory: {inference_memory_mb:.2f} MB")


# Logistic Regression
test_system_cost(
    "Logistic Regression",
    lambda: LogisticRegression(
        max_iter=1000,
        random_state=42
    )
)

# Random Forest
test_system_cost(
    "Random Forest",
    lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
)
