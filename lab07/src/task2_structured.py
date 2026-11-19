
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from task2 import (
    BaselineCNN,
    DEVICE,
    MODELS,
    RESULTS,
    evaluate,
    fine_tune
)

torch.manual_seed(42)

PRUNING_LEVELS = [0.10, 0.25, 0.50]


class StructuredCNN(nn.Module):
    def __init__(self, channels1, channels2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, channels1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channels1, channels2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels2 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def prune_filters(baseline, amount):
    conv1 = baseline.features[0]
    conv2 = baseline.features[3]

    # Rank filters using their L2 norms
    norm1 = torch.linalg.vector_norm(
        conv1.weight.detach().flatten(1), dim=1
    )
    norm2 = torch.linalg.vector_norm(
        conv2.weight.detach().flatten(1), dim=1
    )

    keep1 = max(1, round(32 * (1 - amount)))
    keep2 = max(1, round(64 * (1 - amount)))

    idx1 = torch.topk(norm1, keep1).indices.sort().values
    idx2 = torch.topk(norm2, keep2).indices.sort().values

    model = StructuredCNN(keep1, keep2).to(DEVICE)

    with torch.no_grad():
        # First convolution
        model.features[0].weight.copy_(
            conv1.weight[idx1]
        )
        model.features[0].bias.copy_(
            conv1.bias[idx1]
        )

        # Second convolution:
        # Remove selected output filters and input channels
        model.features[3].weight.copy_(
            conv2.weight[idx2][:, idx1]
        )
        model.features[3].bias.copy_(
            conv2.bias[idx2]
        )

        # First fully connected layer:
        # Keep only inputs from surviving conv2 channels
        old_fc = baseline.classifier[1]
        reshaped = old_fc.weight.reshape(256, 64, 7, 7)

        model.classifier[1].weight.copy_(
            reshaped[:, idx2].reshape(256, -1)
        )
        model.classifier[1].bias.copy_(old_fc.bias)

        # Remaining fully connected layers
        for index in [3, 5]:
            model.classifier[index].load_state_dict(
                baseline.classifier[index].state_dict()
            )

    return model, keep1, keep2


def main():
    print("LAB 07 - TASK 2C: STRUCTURED FILTER PRUNING")
    print("-" * 55)
    print("Device:", DEVICE)

    baseline = BaselineCNN().to(DEVICE)
    baseline.load_state_dict(
        torch.load(
            MODELS / "baseline_cnn.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    baseline_params = sum(
        p.numel() for p in baseline.parameters()
    )

    print(f"Baseline accuracy: {evaluate(baseline):.2f}%")
    print(f"Baseline parameters: {baseline_params:,}")

    results = []

    for amount in PRUNING_LEVELS:
        print(f"\nStructured pruning target: {amount * 100:.0f}%")

        model, channels1, channels2 = prune_filters(
            baseline, amount
        )

        before_accuracy = evaluate(model)

        print(f"  Conv1 filters remaining: {channels1}/32")
        print(f"  Conv2 filters remaining: {channels2}/64")
        print(f"  Accuracy before: {before_accuracy:.2f}%")

        # Train reduced architecture for 3 epochs
        fine_tune(model)

        after_accuracy = evaluate(model)

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        reduction = 100 * (
            1 - parameters / baseline_params
        )

        model_path = MODELS / (
            f"structured_pruned_{int(amount * 100)}.pth"
        )

        torch.save(model.state_dict(), model_path)

        size_mb = model_path.stat().st_size / (1024 ** 2)

        print(f"  Accuracy after: {after_accuracy:.2f}%")
        print(f"  Parameters: {parameters:,}")
        print(f"  Parameter reduction: {reduction:.2f}%")
        print(f"  Model size: {size_mb:.2f} MB")

        results.append({
            "target_filter_pruning_percent": amount * 100,
            "conv1_filters": channels1,
            "conv2_filters": channels2,
            "accuracy_before": before_accuracy,
            "accuracy_after": after_accuracy,
            "parameters": parameters,
            "parameter_reduction_percent": reduction,
            "model_size_mb": size_mb
        })

    output_path = RESULTS / "task2_structured_results.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSTRUCTURED PRUNING SUMMARY")
    print("-" * 55)

    for result in results:
        print(
            f"Target: {result['target_filter_pruning_percent']:.0f}% "
            f"| Accuracy: {result['accuracy_after']:.2f}% "
            f"| Parameters: {result['parameters']:,} "
            f"| Reduction: {result['parameter_reduction_percent']:.2f}%"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
