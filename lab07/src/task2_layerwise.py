
import copy
import csv
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

from task2 import (
    BaselineCNN,
    DEVICE,
    MODELS,
    RESULTS,
    SPARSITY_LEVELS,
    evaluate,
    fine_tune,
    get_prunable_layers,
    calculate_sparsity,
)


def main():
    print("LAB 07 - TASK 2B: LAYER-WISE MAGNITUDE PRUNING")
    print("-" * 55)
    print("Device:", DEVICE)

    torch.manual_seed(42)

    # Load the original trained baseline
    baseline = BaselineCNN().to(DEVICE)

    baseline.load_state_dict(
        torch.load(
            MODELS / "baseline_cnn.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    print(f"Baseline accuracy: {evaluate(baseline):.2f}%")

    results = []

    for sparsity in SPARSITY_LEVELS:
        print(f"\nTarget sparsity: {sparsity * 100:.0f}%")

        # Independent copy of the baseline
        model = copy.deepcopy(baseline)
        parameters = get_prunable_layers(model)

        # Prune each layer independently
        if sparsity > 0:
            for module, name in parameters:
                prune.l1_unstructured(
                    module,
                    name=name,
                    amount=sparsity
                )

        before_accuracy = evaluate(model)

        # Fine-tune for 3 epochs with fixed masks
        fine_tune(model)

        after_accuracy = evaluate(model)
        actual_sparsity = calculate_sparsity(parameters)

        print(f"  Accuracy before: {before_accuracy:.2f}%")
        print(f"  Accuracy after:  {after_accuracy:.2f}%")
        print(f"  Actual sparsity: {actual_sparsity:.2f}%")

        # Keep zero weights in saved model
        if sparsity > 0:
            for module, name in parameters:
                prune.remove(module, name)

        model_path = MODELS / (
            f"layerwise_pruned_{int(sparsity * 100)}.pth"
        )

        torch.save(model.state_dict(), model_path)

        results.append({
            "target_sparsity": sparsity * 100,
            "actual_sparsity": actual_sparsity,
            "accuracy_before": before_accuracy,
            "accuracy_after": after_accuracy,
            "model_size_mb": model_path.stat().st_size / (1024 ** 2)
        })

    # Save results
    output_path = RESULTS / "task2_layerwise_results.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nLAYER-WISE PRUNING SUMMARY")
    print("-" * 55)

    for result in results:
        print(
            f"Sparsity: {result['actual_sparsity']:.1f}% "
            f"| Before: {result['accuracy_before']:.2f}% "
            f"| After: {result['accuracy_after']:.2f}%"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
