# Data Card — UCI Adult Dataset

## 1. Dataset Overview

The UCI Adult dataset is used to predict whether an individual's annual income exceeds $50,000 based on demographic, educational, and employment-related information.

The dataset contains 48,842 records, 14 input features, and one target variable (`class`).

## 2. Data Source

- **Dataset:** UCI Adult (Census Income)
- **Source:** UCI Machine Learning Repository, accessed through OpenML
- **OpenML version:** 2
- **Loading method:** `fetch_openml(name="adult", version=2, as_frame=True)`
- **License:** To be verified from the official dataset documentation before submission

## 3. Dataset Structure

The dataset contains six numerical features: age, final weight (`fnlwgt`), education number, capital gain, capital loss, and hours worked per week.

The eight categorical features describe employment, education, marital status, occupation, relationship, race, sex, and native country.

The target variable has two categories: `<=50K` and `>50K`.

## 4. Data Quality

Initial validation identified missing values in three columns:

| Column | Missing values |
|---|---:|
| workclass | 2,799 |
| occupation | 2,809 |
| native-country | 857 |

No violations of the specified numerical ranges were detected.

A duplicate-conflict audit identified 5 feature groups with conflicting income labels, involving 10 records.

A sample of 50 records was reviewed. Three records contained missing categorical values, and no impossible numerical values were identified under the supplied validation rules.

## 5. Data Preprocessing

The dataset was divided into 70% training and 30% testing data using stratified sampling with random seed 42.

The initial training set contained 34,189 records. After removing 24 exact duplicate records, 34,165 training records remained. The test set contained 14,653 records and was not reduced by duplicate removal.

Missing numerical values were handled using median imputation, while missing categorical values were handled using the most frequent category.

Numerical outliers were clipped using the 1st and 99th percentiles calculated from the training data. Categorical features were converted using one-hot encoding.

All fitted preprocessing parameters were learned using training data only.

## 6. Feature Engineering

Two features were developed:

- `capital_net`: capital gain minus capital loss.
- `hours_per_age`: weekly working hours divided by age minus 16.

The engineered features were evaluated using Logistic Regression.

## 7. Model Evaluation

| Model | Accuracy | F1-score |
|---|---:|---:|
| Baseline | 85.40% | 0.6601 |
| Engineered | 85.78% | 0.6696 |

The engineered model achieved a small improvement in both accuracy and F1-score on the test dataset.

## 8. Known Biases and Limitations

The dataset contains demographic attributes such as race and sex, which may reflect historical social and economic inequalities.

Income is influenced by factors not fully captured in the dataset, and historical relationships may not generalize to current populations.

The dataset should not be treated as a neutral measure of an individual's economic potential.

## 9. Privacy Considerations

The dataset contains personal demographic and employment-related attributes.

Although the dataset does not provide direct personal identifiers in the features used here, combinations of demographic information may raise privacy concerns.

The dataset should be used responsibly and should not be used to make high-stakes decisions about individuals without additional safeguards.

## 10. Reproducibility

The pipeline uses a fixed random seed, training-only preprocessing, a YAML configuration file, and SHA256 hashing.

Processed datasets and the fitted preprocessor are saved in the `results` directory.

The pipeline can be executed using:

`python lab04/task7.py`

The SHA256 hash values are recorded in `results/dataset_hash.txt`.