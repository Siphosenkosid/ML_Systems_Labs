
# Lab 07: Model Compression — Pruning and Distillation

## Objective
The aim of this lab was to explore model compression using pruning
and knowledge distillation. The experiments focused on reducing model
size while maintaining accuracy and improving computational efficiency.

The lab relates to Reddi's Chapter 9 on Efficient AI and Chapter 10
on Model Optimizations, particularly pruning and knowledge distillation.

## Task 1: Baseline CNN

An oversized CNN was trained on Fashion-MNIST for five epochs.
The model was used as the baseline for all subsequent experiments.

Results:
- Parameters: 856,074
- Test accuracy: 90.94%
- Model size: 3.27 MB
- Inference latency: 1.2483 ms

The model achieved good classification accuracy and provided a
reference for evaluating compression methods.

## Task 2: Magnitude and Structured Pruning

Three pruning methods were implemented: global magnitude pruning,
layer-wise magnitude pruning, and structured filter pruning.

Global and layer-wise pruning were tested at sparsity levels of
0%, 25%, 50%, 75%, 90%, and 95%. The models were fine-tuned
for three epochs with fixed pruning masks.

At 95% sparsity:
- Global pruning accuracy: 90.39%
- Layer-wise pruning accuracy: 74.51%

Global pruning maintained better accuracy at high sparsity because
it removed the smallest weights across all eligible layers rather
than enforcing equal sparsity in each layer.

Structured pruning removed entire convolutional filters using
their L2 norms.

At 50% filter pruning:
- Accuracy: 91.23%
- Parameters: 440,650
- Parameter reduction: 48.53%
- Model size: 1.68 MB

The results showed that structured pruning reduced the actual
model dimensions while maintaining good accuracy.

Plots:
- results/plots/task2_global_accuracy.png
- results/plots/task2_layerwise_accuracy.png
- results/plots/task2_pruning_comparison.png

## Task 3: Latency and Storage Comparison

Dense and sparse model formats were compared using CPU inference.

| Model | Dense Size (MB) | Sparse Size (MB) | Latency (ms) |
|---|---:|---:|---:|
| Baseline | 3.27 | 16.61 | 0.8545 |
| Global 95% | 3.27 | 0.91 | 0.9353 |
| Layer-wise 95% | 3.27 | 0.84 | 1.0531 |
| Structured 50% | 1.68 | 8.48 | 0.6534 |

Unstructured pruning reduced sparse storage requirements but did
not improve CPU latency because dense operations still processed
zero-valued weights.

Structured pruning improved inference performance by physically
reducing the model's dimensions.

## Task 4: Knowledge Distillation

Knowledge distillation was used to train a smaller CNN with
approximately 100,000 parameters.

The original CNN acted as the teacher. Two students were trained:
one using standard cross-entropy loss and another using
knowledge distillation with temperature T=4 and alpha=0.5.

| Model | Accuracy | Parameters | Size (MB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Teacher | 90.94% | 856,074 | 3.270 | 1.0587 |
| Plain student | 89.31% | 105,866 | 0.407 | 0.6012 |
| Distilled student | 88.63% | 105,866 | 0.407 | 0.6049 |

Both student models were significantly smaller and faster than
the teacher. However, the plain student performed slightly
better than the distilled student in this experiment.

## Task 5: Efficiency Summary

The experiments were evaluated using the four efficiency pillars
discussed in Chapter 9.

| Pillar | Findings |
|---|---|
| Data | Fashion-MNIST was used consistently across experiments. |
| Model | Pruning and smaller student architectures reduced model complexity. |
| Compute | Structured pruning and student models reduced CPU inference latency. |
| Energy | Energy consumption was not directly measured, so savings cannot be confirmed. |

## Conclusion

This lab demonstrated that model compression involves trade-offs
between accuracy, storage, and computational performance.

Global pruning preserved accuracy better than layer-wise pruning
at high sparsity levels. Structured pruning reduced model size
and improved CPU latency.

Knowledge distillation produced a compact student model, although
the plain student achieved higher accuracy in this experiment.

Overall, structured pruning provided the strongest balance
between accuracy and efficiency among the tested pruned models.

## References

Reddi, Chapter 9: Efficient AI.

Reddi, Chapter 10: Model Optimizations,
Sections 10.3 and 10.4.
