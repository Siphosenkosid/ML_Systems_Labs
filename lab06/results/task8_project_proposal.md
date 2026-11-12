# Task 8: AI-Based Fraud Detection System

**Student Name:** Siphosenkosi Dube  
**Student ID:** 70634  
**Course:** Machine Learning Systems

### 1. Problem Statement

This project aims to develop an AI system that detects potentially fraudulent credit card transactions. The goal is to improve transaction security by identifying suspicious activities quickly and accurately.

### 2. Dataset

I plan to use the Credit Card Fraud Detection dataset, originally made available through a collaboration involving Worldline and Université Libre de Bruxelles (ULB).

The dataset contains anonymized transaction information labelled as fraudulent or legitimate. I will preprocess the data and divide it into training, validation, and testing sets.

### 3. Proposed Model

I will develop a lightweight neural network using PyTorch to classify transactions as fraudulent or legitimate.

### 4. Success Metrics

The main success metric will be the fraud-class F1-score, with a target of at least 80%. I will also evaluate precision, recall, and precision-recall AUC.

### 5. Deployment Requirements

- Inference latency: ≤50 milliseconds per transaction.
- Model size: ≤5 MB.
- Deployment environment: CPU-based application.

### 6. Model Optimization

I plan to use pruning, quantization, and knowledge distillation to reduce model size and improve inference efficiency while maintaining good prediction performance.

### 7. Expected Outcome

The expected outcome is a lightweight AI model that can identify suspicious financial transactions efficiently while meeting the required deployment constraints.