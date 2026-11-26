# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project aims to develop a machine learning system that detects fraudulent credit card transactions. The main challenge is the highly imbalanced dataset, where fraudulent transactions represent a very small percentage of all transactions.

## Dataset

The dataset was obtained from Kaggle.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Original transactions: 284,807
- Input features: 30
- Missing values: 0
- Duplicates removed: 1,081
- Cleaned transactions: 283,726
- Legitimate transactions: 283,253
- Fraudulent transactions: 473

## Data Preprocessing

The dataset was cleaned by removing duplicate records and divided into training and testing sets using an 80/20 stratified split.

StandardScaler was used to standardise the numerical features. The preprocessing pipeline was fitted only on the training data to prevent data leakage.

## Baseline Model

Logistic Regression was selected as the initial baseline because it is simple, fast and interpretable. Balanced class weights were used to address class imbalance.

### Model Configuration

- Model: Logistic Regression
- Solver: LBFGS
- Class weight: balanced
- Maximum iterations: 1000
- Random seed: 42
- Classification threshold: 0.5
- Training samples: 226,980
- Testing samples: 56,746

## Baseline Results

| Metric | Result |
|---|---:|
| Accuracy | 97.53% |
| Precision | 5.64% |
| Recall | 87.37% |
| F1-score | 10.59% |
| ROC-AUC | 96.57% |
| Average precision | 67.19% |
| Specificity | 97.55% |
| Training time | 0.9541 seconds |
| Model size | 1.0928 KB |

## Confusion Matrix

- True negatives: 55,262
- False positives: 1,389
- False negatives: 12
- True positives: 83

## Results Analysis

The baseline model successfully detected 83 out of 95 fraudulent transactions. Although the model achieved high recall and ROC-AUC, its precision was low due to the large number of false positives.

The results demonstrate why accuracy alone is not sufficient when evaluating an imbalanced dataset.

## How to Run

Run the following commands from the repository root:

`python individual_project/src/data_pipeline.py`

`python individual_project/src/train_baseline.py`

The dataset must be downloaded separately and placed in `individual_project/data/creditcard.csv`.

## Future Improvements

- Compare additional machine learning models.
- Optimise the classification threshold using validation data.
- Investigate alternative class imbalance techniques.
- Reduce false positives while maintaining high fraud detection recall.
- Investigate chronological data splitting for more realistic evaluation.

## Conclusion

The initial data pipeline and baseline model were successfully implemented. The trained model, preprocessing pipeline, evaluation plots and metrics were saved for future experiments and improvements.