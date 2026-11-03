# Lab 4: Data Pipeline — UCI Adult Dataset

**Course:** Machine Learning Systems  
**Student:** Siphosenkosi Dube  
**Dataset:** UCI Adult (Census Income)

## Task 1: Problem Definition

The objective of this lab was to develop a reliable and reproducible data pipeline for predicting whether an individual's annual income exceeds $50,000.

The UCI Adult dataset contains 48,842 records, 14 input features, and one target variable (`class`). The target has two categories: `<=50K` and `>50K`.

Accuracy and F1-score were selected to evaluate model performance. Accuracy measures the overall proportion of correct predictions, while F1-score balances precision and recall.

To prevent data leakage, the dataset was divided into training and testing sets before preprocessing. All preprocessing parameters were learned from the training data only.

## Task 2: Data Ingestion

The dataset was loaded using Scikit-learn's `fetch_openml()` function.

The original dataset contained:

- **Rows:** 48,842
- **Columns:** 15
- **Input features:** 14
- **Target variable:** `class`

Missing values were identified in the following columns:

| Column | Missing Values |
|---|---:|
| workclass | 2,799 |
| occupation | 2,809 |
| native-country | 857 |

The original dataset was preserved without modification during ingestion.

## Task 3: Data Validation

Data validation was performed to identify structural errors, invalid data types, missing values, and values outside the specified ranges.

The following numerical constraints were checked:

| Feature | Allowed Range |
|---|---|
| age | 17–90 |
| hours-per-week | 1–99 |
| capital-gain | 0–99,999 |
| capital-loss | 0–4,356 |

**Validation results:**

- Missing columns: 0
- Unexpected columns: 0
- Type violations: 0
- Numerical range violations: 0
- Categorical violations: 0

Missing values were detected in three categorical columns, as identified during data ingestion.

The validation results were saved in `results/validation_report.txt`. No records were modified or removed during this task.

## Task 4: Data Cleaning and Transformation

The dataset was divided into 70% training and 30% testing data using stratified sampling with a random seed of 42.

The cleaning process included removing exact duplicates from the training data, imputing missing values, clipping numerical outliers, and encoding categorical features.

Numerical missing values were handled using median imputation, while categorical missing values were replaced using the most frequent category.

Numerical values were clipped using the 1st and 99th percentiles calculated from the training data.

**Results:**

| Description | Result |
|---|---:|
| Training rows before cleaning | 34,189 |
| Testing rows | 14,653 |
| Training duplicates removed | 24 |
| Training rows after cleaning | 34,165 |
| Processed training shape | 34,165 × 105 |
| Processed testing shape | 14,653 × 105 |

The fitted preprocessor was saved as `results/preprocessor.joblib`.

The saved preprocessor verification and processed data validation both passed.

## Task 5: Feature Engineering

Two additional features were created to investigate whether combining existing information could improve model performance.

**1. Net Capital**

`capital_net = capital-gain - capital-loss`

This feature represents the difference between an individual's capital gains and capital losses.

**2. Working Hours Relative to Age**

`hours_per_age = hours-per-week / (age - 16)`

This feature represents weekly working hours relative to age.

Logistic Regression models were trained using the baseline features and the engineered features.

**Model comparison:**

| Model | Accuracy | F1-score |
|---|---:|---:|
| Baseline | 85.40% | 0.6601 |
| Engineered | 85.78% | 0.6696 |

The engineered model improved accuracy by approximately 0.38 percentage points and F1-score by 0.0095.

The results show that the additional features provided a small improvement in predictive performance.

The comparison results were saved in `results/task5_model_comparison.csv`.

## Task 6: Label Quality Checks

A label-quality audit was conducted to identify potential inconsistencies and data quality issues.

First, records with identical input features but different income labels were identified.

**Duplicate-conflict audit results:**

- Conflicting feature groups: 5
- Records involved in conflicts: 10

Next, a random sample of 50 records was selected using a fixed random seed of 42.

The records were examined for invalid numerical values, missing categorical information, and inconsistencies.

**Audit results:**

| Status | Records |
|---|---:|
| Passed automated checks | 47 |
| Required review | 3 |
| Total | 50 |

The three records requiring review were:

- Row 21063: Missing `workclass` and `occupation`.
- Row 20139: Missing `native-country`.
- Row 18957: Missing `workclass` and `occupation`.

No impossible numerical values or apparent education-number inconsistencies were identified in the reviewed sample.

These findings indicate that the dataset contains missing information and potential label conflicts. The missing values do not necessarily represent incorrect labels.

The results were saved in `results/label_quality_audit.csv`, `results/label_conflicts.csv`, and `results/task6_label_quality_report.txt`.

## Task 7: Dataset Hashing and Reproducibility

A reproducible preprocessing pipeline was developed using a YAML configuration file.

The configuration stores the dataset version, random seed, training and testing split, preprocessing methods, and output locations.

After preprocessing, the training dataset contained 34,165 records and the testing dataset contained 14,653 records. Both saved datasets contained 106 columns, including the target variable.

SHA256 hashing was used to generate fingerprints for the processed datasets.

**SHA256 results:**

| Dataset | SHA256 Hash |
|---|---|
| Training | `974a0ee693a2ef3fc6c43a5c9fd17bb64e1c7618f367bcb62e25be632a3f5c7d` |
| Testing | `8aa57a21d018b782a9be871f0e2986b630d0bf1941e1637ca566d2feeff8a2b7` |
| Combined | `323b5f3b613e23da9de7e96586d9e84a6ac305fdd37ed19410e85b42dde5d696` |

The processed datasets were saved as CSV files, and the fitted preprocessor was saved using Joblib.

A data card was also created to document the dataset's source, structure, preprocessing steps, limitations, potential biases, and privacy considerations.

The pipeline can be executed using:

`python lab04/src/task7.py`

The first execution successfully generated the processed datasets and SHA256 hashes. Reproducibility can be confirmed by running the pipeline again and comparing the resulting hashes.

## Conclusion

This lab demonstrated the main stages of developing a reliable machine learning data pipeline, including data ingestion, validation, cleaning, transformation, feature engineering, label-quality checks, and reproducibility.

The dataset contained missing values and a small number of potential label conflicts, which were identified and documented.

The engineered features slightly improved the Logistic Regression model's performance, achieving 85.78% accuracy and an F1-score of 0.6696.

Overall, the lab highlighted the importance of data quality, preventing data leakage, and maintaining reproducible preprocessing procedures in machine learning systems.