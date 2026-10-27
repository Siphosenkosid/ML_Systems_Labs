
# Lab 03: Deep Neural Network Architecture Comparison

**Name:**Siphosenkosi Dube  
**Student ID:** 70634  


## Introduction

In this lab, I compared three neural network architectures: MLP, CNN, and Attention. I wanted to see how their designs affect accuracy, model size, and inference speed.

## Task 1: Dataset Preparation

I used Fashion-MNIST, which contains 70,000 grayscale images of clothing in 10 categories. Each image is 28 × 28 pixels.I divided the dataset into:
- Training: 55,000 images
- Validation: 5,000 images
- Testing: 10,000 images

I converted the images into tensors and normalized them to help the models train more consistently.

**Output:**

```text
Images: torch.Size([128, 1, 28, 28])
Labels: torch.Size([128])
```
The dataset was successfully prepared for training.

## Task 2: Model Implementation


### Task 2: Model Implementation

In this task, three neural network architectures were implemented using PyTorch: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Attention-based Network.

Each model was tested using a sample input to verify its output shape. All three models produced the expected output of `[1, 10]`, representing the 10 Fashion-MNIST classes.

The trainable parameters were also calculated to compare model complexity. The CNN had the fewest parameters (20,490), followed by the Attention model (70,794), while the MLP had the most (235,146).
This task demonstrated how different architectural designs influence model complexity.

### Results

| Model | Parameters | Output Shape |
|---|---:|---|
| MLP | 235,146 | [1, 10] |
| CNN | 20,490 | [1, 10] |
| Attention | 70,794 | [1, 10] |

**Observation:** All three models produced the expected output. The CNN had the fewest parameters, while the MLP had the most.

## Task 3: Computational Cost and Inference Latency

In this task, I compared the models using FLOPs and inference latency.In order to understand why computational cost and hardware efficiency matter when designing machine learning systems.

### FLOPs Results

| Model | Approximate FLOPs |
|---|---:|
| MLP | 469,504 |
| CNN | 2,063,488 |
| Attention | 101,632* |

### Inference Latency

| Model | Batch 1 (ms) | Batch 128 (ms) |
|---|---:|---:|
| MLP | 0.2803 | 1.1196 |
| CNN | 0.9859 | 21.6644 |
| Attention | 1.9515 | 14.4446 |

**Observation:** The MLP was the fastest model, even though it had the most parameters. The CNN had fewer parameters but required more computation.This showed me that a smaller model is not always faster.

## Task 4: Model Training and Evaluation

I trained all three models for five epochs using the same dataset.

### Final Results

| Model | Test Accuracy |
|---|---:|
| MLP | 87.88% |
| CNN | 89.74% |
| Attention | 84.34% |

The CNN achieved the highest accuracy, followed by the MLP and Attention model.

**Observation:** The CNN performed best because its convolutional layers could learn useful spatial patterns from the clothing images.The Attention model performed less well with the architecture and training settings used in this experiment.

## Task 5: Results Comparison

I compared the models based on accuracy, parameter count, and inference speed.

| Model | Parameters | Accuracy | Batch 1 Latency |
|---|---:|---:|---:|
| MLP | 235,146 | 87.88% | 0.2803 ms |
| CNN | 20,490 | 89.74% | 0.9859 ms |
| Attention | 70,794 | 84.34% | 1.9515 ms |

The CNN gave the best accuracy while using the fewest parameters.

However, the MLP was faster during inference.Choosing a model involves balancing accuracy with computational requirements.

**Observation** I learned that the most accurate model is not necessarily the fastest.

## Conclusion
This lab helped me understand how different neural network architectures work in practice.
The CNN achieved the best accuracy at 89.74% and had the fewest parameters. The MLP was the fastest during inference, while the Attention model had the lowest accuracy.
I got to understand how fully connected layers, convolutional filters, attention mechanisms, and system-level requirements affect model performance.

Overall, I learned that choosing a neural network is not just about accuracy. Model size, speed, and computational cost also matter.
