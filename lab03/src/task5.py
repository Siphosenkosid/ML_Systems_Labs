
import csv
import os
import matplotlib.pyplot as plt

# Read the actual training results from Task 4
results = []

with open("task4_results.csv", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        results.append({
            "model": row["Model"],
            "parameters": int(row["Parameters"]),
            "accuracy": float(row["Test Accuracy (%)"])
        })

# Latency measurements from Task 3 (in milliseconds)
latencies = {
    "MLP": {"batch1": 0.2803, "batch128": 1.1196},
    "CNN": {"batch1": 0.9859, "batch128": 21.6644},
    "Attention": {"batch1": 1.9515, "batch128": 14.4446}
}

# Create a folder to save the graphs
os.makedirs("results", exist_ok=True)

# Graph 1: Accuracy vs Parameters
plt.figure(figsize=(8, 5))

for result in results:
    plt.scatter(
        result["parameters"],
        result["accuracy"],
        s=100,
        label=result["model"]
    )

    plt.annotate(
        result["model"],
        (result["parameters"], result["accuracy"]),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.title("Test Accuracy vs Model Parameters")
plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("results/accuracy_vs_parameters.png", dpi=300)
plt.show()
plt.close()


# Graph 2: Accuracy vs Inference Latency
plt.figure(figsize=(8, 5))

for result in results:
    name = result["model"]

    plt.scatter(
        latencies[name]["batch1"],
        result["accuracy"],
        s=100,
        label=name
    )

    plt.annotate(
        name,
        (
            latencies[name]["batch1"],
            result["accuracy"]
        ),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.title("Test Accuracy vs Inference Latency (Batch 1)")
plt.xlabel("Median Inference Latency (ms)")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("results/accuracy_vs_latency.png", dpi=300)
plt.show()
plt.close()

print("Task 5 completed successfully!")
print("Graphs saved in the results folder.")
