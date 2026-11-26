import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)


# Project paths
ROOT = Path(__file__).resolve().parents[2]

PROJECT = ROOT / "individual_project"
RESULTS = PROJECT / "results"
MODELS = RESULTS / "models"
PLOTS = RESULTS / "plots"

DATA_PATH = (
    PROJECT / "data" / "processed" / "processed_data.npz"
)

CONFIG_PATH = PROJECT / "src" / "config.yaml"

for folder in [RESULTS, MODELS, PLOTS]:
    folder.mkdir(parents=True, exist_ok=True)


def main():

    print("INDIVIDUAL PROJECT - BASELINE MODEL")
    print("=" * 60)

    # Load configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Load processed dataset
    with np.load(DATA_PATH) as data:

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

    print("\n1. DATASET")
    print("-" * 60)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of features:", X_train.shape[1])

    # Create baseline model
    model = LogisticRegression(
        class_weight=config["model"]["class_weight"],
        max_iter=config["model"]["max_iter"],
        random_state=config["split"]["random_state"],
        solver="lbfgs"
    )

    print("\n2. MODEL CONFIGURATION")
    print("-" * 60)

    print("Model: Logistic Regression")
    print("Class weight:", model.class_weight)
    print("Maximum iterations:", model.max_iter)
    print("Solver:", model.solver)

    # Train the model
    print("\n3. MODEL TRAINING")
    print("-" * 60)

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("Training completed successfully.")
    print(f"Training time: {training_time:.4f} seconds")
    print("Iterations:", model.n_iter_[0])

    # Make predictions
    start_time = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start_time

    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    average_precision = average_precision_score(
        y_test, y_prob
    )

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    majority_accuracy = float(np.mean(y_test == 0))

    print("\n4. MODEL EVALUATION")
    print("-" * 60)

    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Average precision: {average_precision:.6f}")
    print(f"Specificity: {specificity:.6f}")

    print(
        f"Majority-class accuracy: {majority_accuracy:.6f}"
    )

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0
    )

    print("\n5. CLASSIFICATION REPORT")
    print("-" * 60)

    print(report)

    # Confusion matrix
    print("\n6. CONFUSION MATRIX")
    print("-" * 60)

    print(cm)

    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    # Save trained model
    model_path = MODELS / "baseline_model.joblib"

    joblib.dump(model, model_path)

    model_size = model_path.stat().st_size / 1024

    latency = (
        prediction_time / len(X_test)
    ) * 1000

    print("\n7. MODEL PERFORMANCE")
    print("-" * 60)

    print(f"Model size: {model_size:.4f} KB")

    print(
        f"Prediction time: {prediction_time:.6f} seconds"
    )

    print(
        f"Average batch prediction time per sample: "
        f"{latency:.6f} ms"
    )

    # Save metrics
    metrics = {
        "project": config["project_name"],
        "model": "Logistic Regression",
        "class_weight": model.class_weight,
        "solver": model.solver,
        "random_state": config["split"]["random_state"],
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "number_of_features": int(X_train.shape[1]),
        "decision_threshold": 0.5,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "specificity": float(specificity),
        "majority_class_accuracy": majority_accuracy,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "training_time_seconds": float(training_time),
        "prediction_time_seconds": float(prediction_time),
        "batch_prediction_time_per_sample_ms": float(latency),
        "model_size_kb": float(model_size),
        "iterations": int(model.n_iter_[0])
    }

    with open(
        RESULTS / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    # Save comparison table
    comparison = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "Average Precision": average_precision
        },
        {
            "Model": "Always Legitimate",
            "Accuracy": majority_accuracy,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-score": 0.0,
            "ROC-AUC": 0.5,
            "Average Precision": float(np.mean(y_test))
        }
    ])

    comparison.to_csv(
        RESULTS / "baseline_comparison.csv",
        index=False
    )

    # Plot 1: Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraudulent"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title("Baseline Model - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS / "baseline_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Plot 2: ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline Model - ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS / "baseline_roc_curve.png",
        dpi=300
    )

    plt.close()

    # Plot 3: Precision-recall curve
    precisions, recalls, _ = precision_recall_curve(
        y_test, y_prob
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recalls,
        precisions,
        label=f"Average Precision = {average_precision:.4f}"
    )

    plt.axhline(
        y=np.mean(y_test),
        linestyle="--",
        color="gray",
        label="Random baseline"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Baseline Model - Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS / "baseline_precision_recall.png",
        dpi=300
    )

    plt.close()

    # Plot 4: Evaluation metrics
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "Avg Precision"
    ]

    metric_values = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        average_precision
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        metric_names,
        metric_values,
        color="steelblue"
    )

    plt.bar_label(
        bars,
        fmt="%.3f",
        padding=3
    )

    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Baseline Model - Evaluation Metrics")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        PLOTS / "baseline_metrics.png",
        dpi=300
    )

    plt.close()

    # Save text report
    text_report = f"""
CREDIT CARD FRAUD DETECTION
BASELINE MODEL REPORT
============================================================

MODEL CONFIGURATION
------------------------------------------------------------
Model: Logistic Regression
Class weight: {model.class_weight}
Solver: {model.solver}
Maximum iterations: {model.max_iter}
Decision threshold: 0.5

DATASET
------------------------------------------------------------
Training samples: {len(X_train)}
Testing samples: {len(X_test)}
Number of features: {X_train.shape[1]}

EVALUATION METRICS
------------------------------------------------------------
Accuracy: {accuracy:.6f}
Precision: {precision:.6f}
Recall: {recall:.6f}
F1-score: {f1:.6f}
ROC-AUC: {roc_auc:.6f}
Average precision: {average_precision:.6f}
Specificity: {specificity:.6f}
Majority-class accuracy: {majority_accuracy:.6f}

CLASSIFICATION REPORT
------------------------------------------------------------
{report}

CONFUSION MATRIX
------------------------------------------------------------
True negatives: {tn}
False positives: {fp}
False negatives: {fn}
True positives: {tp}

MODEL PERFORMANCE
------------------------------------------------------------
Training time: {training_time:.6f} seconds
Prediction time: {prediction_time:.6f} seconds
Batch prediction time per sample: {latency:.6f} ms
Model size: {model_size:.4f} KB

CONCLUSION
------------------------------------------------------------
Logistic Regression was trained as the initial baseline
for credit card fraud detection.

Balanced class weights were used to address the
significant imbalance between legitimate and fraudulent
transactions.

The model was evaluated on unseen testing data using
multiple classification metrics rather than relying
on accuracy alone.
"""

    (RESULTS / "baseline_report.txt").write_text(
        text_report,
        encoding="utf-8"
    )

    print("\n8. SAVED FILES")
    print("-" * 60)

    print(model_path)
    print(RESULTS / "metrics.json")
    print(RESULTS / "baseline_comparison.csv")
    print(RESULTS / "baseline_report.txt")
    print(PLOTS / "baseline_confusion_matrix.png")
    print(PLOTS / "baseline_roc_curve.png")
    print(PLOTS / "baseline_precision_recall.png")
    print(PLOTS / "baseline_metrics.png")

    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    main()