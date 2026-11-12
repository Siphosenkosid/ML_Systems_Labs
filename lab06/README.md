# Lab 06: Training Pipelines and Project Proposal


## Task 1: Configuration-Driven Training Pipeline

In this task, I developed a training pipeline for the Fashion-MNIST dataset using PyTorch. I used a YAML configuration file to manage training settings, including the learning rate, batch size, and random seed.

I also implemented checkpoint saving and MLflow tracking to record training parameters and results.

**Results:**
- Training epochs: 5
- Final validation accuracy: 87.90%
- Test accuracy: 87.47%

The pipeline successfully trained the model and saved the best checkpoint for future use.

## Task 2: Training Performance Profiling

In this task, I used PyTorch Profiler to examine the performance of one training epoch. I focused on data loading, the forward pass, backpropagation, and optimizer updates.

The profiling process helped identify where training time was spent and how different operations contributed to the overall execution time.

## Task 3: DataLoader Worker Comparison

I compared five DataLoader worker settings (0, 1, 2, 4, and 8) to determine their effect on training speed.

**Results:**

| Workers | Epoch Time (s) | Throughput (samples/s) |
|---|---:|---:|
| 0 | 16.15 | 3405.99 |
| 1 | 17.03 | 3229.89 |
| 2 | 15.09 | 3644.68 |
| 4 | 24.16 | 2276.87 |
| 8 | 37.86 | 1452.76 |

Two workers achieved the best performance. Increasing the number of workers beyond two made training slower, likely because of additional multiprocessing overhead.

## Task 4: Batch Size Comparison

I tested batch sizes of 32, 64, 128, 256, and 512. Each configuration was trained for five epochs to compare throughput, memory measurements, and validation accuracy.

| Batch Size | Final Accuracy | Throughput (samples/s) |
|---|---:|---:|
| 32 | 87.60% | 3749.61 |
| 64 | 87.44% | 6270.51 |
| 128 | 88.08% | 6354.51 |
| 256 | 86.78% | 6395.67 |
| 512 | 85.46% | 6603.78 |

Batch size 512 achieved the highest throughput, while batch size 128 produced the highest accuracy of 88.08%.

I selected batch size 128 as a good balance between training speed and accuracy. Memory measurements were based on Python allocations rather than total PyTorch memory usage.

## Task 5: Optimizer and Learning-Rate Scheduler Comparison

I compared SGD with momentum and Adam using StepLR, CosineAnnealingLR, and ReduceLROnPlateau. Each combination was trained for 10 epochs.

| Optimizer | Scheduler | Final Accuracy |
|---|---|---:|
| SGD | StepLR | 80.16% |
| SGD | CosineAnnealingLR | 80.16% |
| SGD | ReduceLROnPlateau | 82.78% |
| Adam | StepLR | 88.36% |
| Adam | CosineAnnealingLR | 88.52% |
| Adam | ReduceLROnPlateau | 88.16% |

Adam with CosineAnnealingLR achieved the highest final validation accuracy of 88.52%.

The experiment showed that optimizer and scheduler selection can influence model performance. Under the tested learning rate, Adam performed better than SGD.

## Task 6: Gradient Accumulation

In this task, I compared normal training with a batch size of 256 against gradient accumulation using a batch size of 64 and four accumulation steps.

Both approaches were trained for five epochs.

**Results:**
- Normal training accuracy: 86.78%
- Gradient accumulation accuracy: 86.68%
- Accuracy difference: 0.10 percentage points
- Optimizer updates: 215 per epoch for both approaches

Both methods produced very similar results. This demonstrated that gradient accumulation can simulate a larger batch size while processing smaller batches at a time.

## Task 7: Checkpoint and Resume Training

I tested whether model training could continue after an interruption without losing progress.

First, I trained the model for five epochs without interruption. I then repeated the experiment, saved a checkpoint after epoch 3, and resumed training from epoch 4.

**Results:**
- Uninterrupted accuracy: 87.10%
- Resumed accuracy: 87.10%
- Training loss difference: 0.00000000
- Validation accuracy difference: 0.00 percentage points

Both experiments produced identical results, confirming that the model and optimizer states were restored correctly.

## Task 8: Individual AI Project Proposal

For my individual project, I proposed an **AI-Based Fraud Detection System**.

The project aims to develop a lightweight neural network using PyTorch to identify potentially fraudulent credit card transactions.

I plan to use an anonymized credit card transaction dataset and evaluate the model using precision, recall, F1-score, and precision-recall AUC.

**Project targets:**
- Fraud-class F1-score: at least 80%
- Inference latency: 50 ms or less
- Model size: 5 MB or less
- Deployment: CPU-based application

I also plan to explore pruning, quantization, and knowledge distillation to optimize the model for efficient deployment.

The complete proposal is available in `task8_project_proposal.md`.

## Conclusion

In Lab 06, I explored different techniques for improving machine learning training pipelines. I learned how to manage experiments using configuration files, profile training performance, compare DataLoader settings, optimize batch sizes, and evaluate optimizers and learning-rate schedulers.

I also implemented gradient accumulation and checkpoint recovery to improve training flexibility and reliability.

Overall, these experiments helped me understand how to build, evaluate, and optimize machine learning systems for practical applications.