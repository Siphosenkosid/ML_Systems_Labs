from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os


# Load the Breast Cancer Wisconsin dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create the two baseline models
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train both models
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)
forest_predictions = random_forest_model.predict(X_test)

# Calculate accuracy
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
forest_accuracy = accuracy_score(y_test, forest_predictions)

# Store the results
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Test Accuracy": [
        round(logistic_accuracy, 4),
        round(forest_accuracy, 4)
    ]
})

# Create the results folder if it does not exist
os.makedirs("lab01/results", exist_ok=True)

# Save results to CSV
results.to_csv(
    "lab01/results/baseline_accuracy.csv",
    index=False
)

print(results)
print("\nResults saved to lab01/results/baseline_accuracy.csv")

