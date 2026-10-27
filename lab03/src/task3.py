
import torch
from torchinfo import summary
from task2 import MLP, CNN, AttentionModel

# Create the three models
models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

# Display parameter counts and model summaries
for name, model in models.items():
    print(f"\n{name} Model Summary")
    print("=" * 50)

    model.eval()

    summary(
        model,
        input_size=(1, 1, 28, 28),
        col_names=["input_size", "output_size", "num_params"],
        depth=2,
        device="cpu"
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_params:,}")


from thop import profile

print("\nTHOP Profiling Results")
print("=" * 50)

# One Fashion-MNIST image
sample_input = torch.randn(1, 1, 28, 28)

for name, model in models.items():
    model.eval()

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(sample_input,),
            verbose=False
        )

    print(f"\n{name}")
    print(f"THOP MACs: {macs:,.0f}")
    print(f"THOP estimated FLOPs (2 x MACs): {2 * macs:,.0f}")
    print(f"THOP counted parameters: {params:,.0f}")



import time
import statistics

print("\nInference Latency Results")
print("=" * 50)

# Use CPU for consistent measurements
device = torch.device("cpu")

def measure_latency(model, batch_size, runs):
    model = model.to(device)
    model.eval()

    # Generate input images
    sample = torch.randn(
        batch_size, 1, 28, 28,
        device=device
    )

    # Warm-up runs
    with torch.inference_mode():
        for _ in range(10):
            model(sample)

    # Measure inference time
    times = []

    with torch.inference_mode():
        for _ in range(runs):
            start = time.perf_counter()

            model(sample)

            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            times.append(latency_ms)

    return statistics.median(times)


# Test each architecture
for name, model in models.items():

    latency_1 = measure_latency(
        model, batch_size=1, runs=100
    )

    latency_128 = measure_latency(
        model, batch_size=128, runs=20
    )

    print(f"\n{name}")
    print(f"Batch 1 median: {latency_1:.4f} ms")
    print(f"Batch 128 median: {latency_128:.4f} ms")
