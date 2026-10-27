
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

# TASK 6: LABEL QUALITY CHECKS

print("TASK 6: LABEL QUALITY CHECKS")
print("-" * 50)

# 1. Load the original dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy(deep=True)

# Define the target and input features
target_column = "class"

feature_columns = [
    col for col in df.columns
    if col != target_column
]

# Create results folder
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# 2. Duplicate-conflict audit
# Find feature combinations with different income labels
label_counts = df.groupby(
    feature_columns,
    dropna=False,
    observed=True
)[target_column].nunique()

conflicting_groups = label_counts[
    label_counts > 1
]

print("\nDUPLICATE-CONFLICT AUDIT")
print("Conflicting feature groups:", len(conflicting_groups))

# Identify all rows belonging to conflicting groups
if len(conflicting_groups) > 0:
    group_keys = conflicting_groups.index.to_frame(
        index=False
    )

    conflict_rows = df.merge(
        group_keys,
        on=feature_columns,
        how="inner"
    )
else:
    conflict_rows = df.iloc[0:0].copy()

print("Rows involved in conflicts:", len(conflict_rows))

conflict_rows.to_csv(
    results_dir / "label_conflicts.csv",
    index=False
)


# 3. Select 50 random rows for quality auditing
sample = df.sample(
    n=50,
    random_state=42
).copy()

print("\nMANUAL QUALITY AUDIT")
print("Random rows selected:", len(sample))


# 4. Define validity rules
# These are the numerical limits from the lab manual
ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356)
}

# Check each sampled row
audit_results = []

for index, row in sample.iterrows():

    issues = []

    # Check numerical ranges
    for column, (minimum, maximum) in ranges.items():
        value = row[column]

        if pd.isna(value):
            issues.append(f"{column}: missing")
        elif not (minimum <= value <= maximum):
            issues.append(
                f"{column}: outside allowed range"
            )

    # Check missing categorical values
    for column in [
        "workclass",
        "occupation",
        "native-country"
    ]:
        value = row[column]

        if pd.isna(value) or str(value).strip() == "?":
            issues.append(f"{column}: missing")

    # Check income label
    if str(row[target_column]) not in ["<=50K", ">50K"]:
        issues.append("Invalid income label")

    audit_results.append({
        "row_index": index,
        "issues": "; ".join(issues),
        "status": "Review" if issues else "Pass"
    })


# 5. Save the audit results
audit_df = pd.DataFrame(audit_results)

sample.to_csv(
    results_dir / "audit_sample_50.csv",
    index=True
)

audit_df.to_csv(
    results_dir / "label_quality_audit.csv",
    index=False
)

print("\nAUDIT SUMMARY")
print(audit_df["status"].value_counts())

print("\nRows requiring review:")
print(audit_df[audit_df["status"] == "Review"])


# 6. Save the summary report
report_path = results_dir / "task6_label_quality_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("TASK 6: LABEL QUALITY REPORT\n")
    report.write("=" * 50 + "\n\n")

    report.write(
        f"Conflicting feature groups: {len(conflicting_groups)}\n"
    )
    report.write(
        f"Rows involved in conflicts: {len(conflict_rows)}\n"
    )
    report.write(
        f"Random rows audited: {len(sample)}\n"
    )
    report.write(
        f"Rows requiring review: "
        f"{(audit_df['status'] == 'Review').sum()}\n"
    )

print("\nReport saved to:", report_path)
print("Task 6 automated checks completed.")
