# Lab 08 – Quantization and Model Export

## Overview

The aim of this lab was to explore model quantization, compression and deployment techniques. I investigated how reducing numerical precision affects model accuracy, storage size and inference speed.

The experiments were conducted using the Fashion-MNIST dataset and the models developed in Lab 07.

## Environment

- Python 3.11.8
- PyTorch 2.3.1+cpu
- ONNX 1.16.1
- ONNX Runtime 1.18.0
- Quantization backend: FBGEMM
- Device: CPU
- Dataset: Fashion-MNIST

## Task 1 – Manual Quantization

In this task, I manually quantized a weight tensor using 8-bit, 4-bit and 2-bit symmetric quantization. I compared per-tensor and per-channel quantization by calculating the mean squared error between the original and reconstructed weights.

| Precision | Per-tensor MSE | Per-channel MSE |
|---|---:|---:|
| 8-bit | 0.0000040300 | 0.0000025833 |
| 4-bit | 0.0013265306 | 0.0008503401 |
| 2-bit | 0.0650000000 | 0.0416666667 |

The results showed that reducing numerical precision increased the reconstruction error. Per-channel quantization produced approximately 35.9% lower error than per-tensor quantization in all three experiments.

The 4-bit and 2-bit experiments simulated reduced numerical precision using integer values rather than actual packed low-bit storage.

## Task 2 – Dynamic Post-Training Quantization

Dynamic quantization was applied to the linear layers of the baseline CNN using the FBGEMM backend.

| Model | Accuracy | Size (MB) | Latency (ms) |
|---|---:|---:|---:|
| Baseline FP32 | 90.94% | 3.2695 | 1.1210 |
| Dynamic INT8 | 90.92% | 0.8773 | 1.0967 |

### Results

- Accuracy drop: 0.02 percentage points
- Model size reduction: 73.17%
- Compression ratio: 3.73x
- Speedup: 1.02x

Dynamic quantization significantly reduced the model size while maintaining almost the same accuracy. However, the improvement in inference speed was small because the convolutional layers remained in floating-point precision.

## Task 3 – Static Post-Training Quantization

Static quantization was performed using 500 randomly selected training images for calibration. The convolutional and linear layers were quantized after appropriate layer fusion.

Calibration helps the model estimate representative activation ranges before converting them to integer precision.

| Model | Accuracy | Size (MB) | Latency (ms) |
|---|---:|---:|---:|
| Baseline FP32 | 90.94% | 3.2695 | 1.5043 |
| Static INT8 | 90.79% | 0.8353 | 0.8881 |

### Results

- Accuracy drop: 0.15 percentage points
- Model size reduction: 74.45%
- Compression ratio: 3.91x
- Speedup: 1.69x

Static quantization achieved a smaller model and faster inference than the baseline, with only a small reduction in accuracy.

## Task 4 – ONNX Export

The baseline CNN was exported to ONNX format and evaluated using ONNX Runtime.

The exported model was validated by comparing its outputs against the original PyTorch model.

| Validation metric | Result |
|---|---:|
| ONNX structure validation | Passed |
| Maximum absolute difference | 0.0000061989 |
| Mean absolute difference | 0.0000005130 |
| Prediction agreement | 100% |
| Numerical tolerance | 0.0001 |
| Numerical validation | Passed |

### Performance Comparison

| Model | Accuracy | Size (MB) | Latency (ms) |
|---|---:|---:|---:|
| PyTorch FP32 | 90.94% | 3.2695 | 1.0246 |
| ONNX FP32 | 90.94% | 3.2681 | 0.2101 |

ONNX Runtime achieved a speedup of approximately 4.88x and reduced inference latency by 79.49% without changing classification accuracy.

## Task 5 – Combined Model Compression

In this task, I combined quantization with structured pruning and knowledge distillation to investigate the trade-off between accuracy, model size and inference latency.

All seven models were benchmarked under the same conditions.

| Model | Parameters | Accuracy | Size (MB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline FP32 | 856,074 | 90.94% | 3.2695 | 0.7812 |
| Pruned FP32 | 440,650 | 91.23% | 1.6849 | 0.4543 |
| Distilled FP32 | 105,866 | 88.63% | 0.4072 | 0.3963 |
| Dynamic INT8 | 856,074 | 90.92% | 0.8775 | 0.8428 |
| Static INT8 | 856,074 | 90.79% | 0.8356 | 0.6385 |
| Pruned + INT8 | 440,650 | 91.23% | 0.4383 | 0.5409 |
| Distilled + INT8 | 105,866 | 88.44% | 0.1114 | 0.5314 |

The parameter counts represent the underlying model architectures before quantization.

### Key Findings

- Structured pruning achieved the highest accuracy of 91.23%, shared by the pruned FP32 and pruned INT8 models.
- The distilled FP32 model achieved the lowest inference latency among the seven models.
- Combining distillation and quantization reduced model size by 96.59%.
- The distilled INT8 model achieved a compression ratio of 29.34x.
- Quantization did not always improve inference speed because performance also depends on hardware, model architecture and quantization overhead.

### Deployment Recommendations

| Deployment environment | Recommended model | Reason |
|---|---|---|
| Cloud | Pruned FP32 | High accuracy and fast inference |
| Edge | Pruned FP32 | Good balance between accuracy and latency |
| Mobile | Distilled + INT8 | Small storage requirement |
| TinyML | Distilled + INT8 | Smallest model among the tested variants |

The pruned INT8 model is also suitable for storage-constrained edge environments.

The TinyML recommendation is a potential candidate based on model size. Actual compatibility, memory requirements and performance on microcontrollers have not been tested.

## Task 6 – Individual Project Milestone

### Credit Card Fraud Detection Using Machine Learning

For my individual project, I implemented a data preprocessing pipeline and trained a Logistic Regression baseline model using a credit card fraud detection dataset from Kaggle.

### Dataset Summary

| Description | Result |
|---|---:|
| Original transactions | 284,807 |
| Duplicates removed | 1,081 |
| Cleaned transactions | 283,726 |
| Input features | 30 |
| Training samples | 226,980 |
| Testing samples | 56,746 |

The dataset was cleaned, standardised and divided using an 80/20 stratified split. Balanced class weights were used to address class imbalance.

### Baseline Results

| Metric | Result |
|---|---:|
| Accuracy | 97.53% |
| Precision | 5.64% |
| Recall | 87.37% |
| F1-score | 10.59% |
| ROC-AUC | 96.57% |
| Average precision | 67.19% |
| Training time | 0.9541 seconds |

The model successfully detected 83 out of 95 fraudulent transactions. However, the low precision and high number of false positives showed that further optimisation is required.

The individual project contains the preprocessing pipeline, trained model, configuration file, evaluation plots and metrics.json.

Project directory: [individual_project](../individual_project)

## Folder Structure

```text
lab08/
├── src/
│   ├── task1.py
│   ├── task2.py
│   ├── task3.py
│   ├── task4.py
│   └── task5.py
├── results/
│   ├── models/
│   ├── plots/
│   ├── CSV comparison files
│   ├── JSON metrics files
│   └── Text reports
└── README.md
```

Task 6 is maintained separately in the individual_project directory.

## How to Run

Run the following commands from the repository root:

```bash
python lab08/src/task1.py
python lab08/src/task2.py
python lab08/src/task3.py
python lab08/src/task4.py
python lab08/src/task5.py
```

Tasks 2–5 require the relevant trained model checkpoints from Lab 07.

For the individual project:

```bash
python individual_project/src/data_pipeline.py
python individual_project/src/train_baseline.py
```

The credit card dataset must be downloaded separately from Kaggle and placed in the individual project data directory.

## Limitations

Inference latency varied between separate experiments due to runtime conditions. Therefore, the Task 5 comparison provides a more consistent comparison of the seven compression variants.

Model sizes represent saved file sizes in MiB, displayed as MB in the experimental results.

The deployment recommendations are based on CPU experiments rather than measurements on actual mobile, edge or TinyML hardware.

## Conclusion

This lab helped me understand how quantization, pruning and knowledge distillation can reduce model storage requirements while maintaining reasonable accuracy.

Dynamic and static quantization significantly reduced model size, while ONNX Runtime improved inference speed without reducing accuracy.

Combining multiple compression techniques achieved the largest reduction in model size. However, the experiments also showed that smaller models are not always faster, making it important to evaluate accuracy, latency and storage together before deployment.