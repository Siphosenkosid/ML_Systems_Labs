
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"


def read_csv(filename):
    with open(RESULTS / filename, newline="") as file:
        return list(csv.DictReader(file))


def main():
    print("LAB 07 - TASK 5: EFFICIENCY SUMMARY")
    print("-" * 65)

    task1 = __import__("json").loads(
        (RESULTS / "task1_results.json").read_text()
    )
    global_results = read_csv("task2_global_results.csv")
    layer_results = read_csv("task2_layerwise_results.csv")
    structured = read_csv("task2_structured_results.csv")
    latency = read_csv("task3_results.csv")
    distillation = read_csv("task4_results.csv")

    latency_by_name = {r["model"]: r for r in latency}
    students = {r["model"]: r for r in distillation}

    # Summarize representative models
    summary = []

    def add(name, accuracy, parameters, size, latency_ms,
            data_note, model_note):
        summary.append({
            "model": name,
            "data": data_note,
            "model": name,
            "model_efficiency": model_note,
            "accuracy_percent": round(float(accuracy), 2),
            "parameters": int(parameters),
            "size_mb": round(float(size), 3),
            "compute_latency_ms": round(float(latency_ms), 4),
            "energy": "Not measured"
        })

    add(
        "Baseline",
        task1["accuracy_percent"],
        task1["parameters"],
        latency_by_name["baseline"]["dense_size_mb"],
        latency_by_name["baseline"]["latency_ms"],
        "Fashion-MNIST",
        "Dense CNN"
    )

    for method, source, percent in [
        ("global", global_results, 95),
        ("layerwise", layer_results, 95)
    ]:
        result = next(
            r for r in source
            if float(r["target_sparsity"]) == percent
        )
        measured = latency_by_name[f"{method}_{percent}"]

        add(
            f"{method.capitalize()} {percent}%",
            result["accuracy_after"],
            measured["parameters"],
            measured["dense_size_mb"],
            measured["latency_ms"],
            "Fashion-MNIST",
            f"{percent}% unstructured sparsity"
        )

    result = next(
        r for r in structured
        if float(r["target_filter_pruning_percent"]) == 50
    )
    measured = latency_by_name["structured_50"]

    add(
        "Structured 50%",
        result["accuracy_after"],
        result["parameters"],
        measured["dense_size_mb"],
        measured["latency_ms"],
        "Fashion-MNIST",
        "50% convolutional filter pruning"
    )

    for name in ["plain_student", "distilled_student"]:
        r = students[name]

        add(
            name,
            r["accuracy_percent"],
            r["parameters"],
            r["size_mb"],
            r["latency_ms"],
            "Fashion-MNIST; teacher outputs"
            if name == "distilled_student"
            else "Fashion-MNIST labels",
            "Compact CNN"
        )

    output = RESULTS / "task5_efficiency_summary.csv"

    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary)

    print(
        f"{'Model':20} {'Accuracy':>10} "
        f"{'Params':>12} {'Size MB':>10} {'Latency ms':>12}"
    )
    print("-" * 70)

    for row in summary:
        print(
            f"{row['model']:20} "
            f"{row['accuracy_percent']:>9.2f}% "
            f"{row['parameters']:>12,} "
            f"{row['size_mb']:>10.3f} "
            f"{row['compute_latency_ms']:>12.4f}"
        )

    print("\nCHAPTER 9 EFFICIENCY PILLARS")
    print("-" * 65)
    print("DATA: Fashion-MNIST was used for all experiments.")
    print("MODEL: Pruning and student models reduced model complexity.")
    print("COMPUTE: CPU inference latency was measured.")
    print("ENERGY: No direct energy measurements were collected.")

    print("\nCONCLUSION")
    print("-" * 65)
    print(
        "Global pruning retained accuracy better than layer-wise "
        "pruning at high sparsity."
    )
    print(
        "Structured pruning reduced model size and improved CPU latency."
    )
    print(
        "The plain student outperformed the distilled student "
        "in this experiment."
    )
    print(
        "Energy savings cannot be confirmed without direct measurements."
    )

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
