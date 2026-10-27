# Resource budgets given in the lab manual

platforms = {
    "Cloud": {
        "latency_ms": 100,
        "model_size_kb": 500 * 1024
    },
    "Edge": {
        "latency_ms": 50,
        "model_size_kb": 50 * 1024
    },
    "Mobile": {
        "latency_ms": 20,
        "model_size_kb": 10 * 1024
    },
    "TinyML": {
        "latency_ms": 10,
        "model_size_kb": 100
    }
}


# Replace the first two values with your Logistic Regression
# latency and model size from Task 3.

models = {
    "Logistic Regression": {
        "latency_ms": 0.0,     # enter your Task 3 latency here
        "model_size_kb": 1.03
    },

    "Random Forest": {
        "latency_ms": 10.4484,
        "model_size_kb": 284.07
    }
}


print("\nDEPLOYMENT FIT RESULTS\n")

for model_name, measurements in models.items():

    print(model_name)

    for platform, budget in platforms.items():

        latency_ok = (
            measurements["latency_ms"]
            <= budget["latency_ms"]
        )

        size_ok = (
            measurements["model_size_kb"]
            <= budget["model_size_kb"]
        )

        if latency_ok and size_ok:
            result = "Suitable"
        else:
            result = "Not suitable"

        print(f"  {platform}: {result}")

    print()