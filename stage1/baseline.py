
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load only the columns needed for this experiment
df = pd.read_csv(
    "data/Base.csv",
    usecols=["month", "fraud_bool"]
)

# Use month 5 for validation
validation = df[df["month"] == 5]

y_true = validation["fraud_bool"]

# Always predict legitimate (class 0)
y_pred = np.zeros(len(y_true), dtype=int)

# Calculate evaluation metrics
print("Validation applications:", len(y_true))
print("Actual fraud cases:", y_true.sum())
print("Fraud rate:", round(y_true.mean() * 100, 4), "%")

print("\nNever-Fraud Baseline")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Average Precision:", average_precision_score(y_true, y_pred))
