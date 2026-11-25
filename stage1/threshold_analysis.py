
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

# Load our dataset
df = pd.read_csv("data/Base.csv")

# Temporal split
train_df = df[df["month"].between(0, 4)].copy()
val_df = df[df["month"] == 5].copy()

# Separate features and target
drop_columns = [
    "fraud_bool",
    "month",
    "device_fraud_count"
]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["fraud_bool"]

X_val = val_df.drop(columns=drop_columns)
y_val = val_df["fraud_bool"]

# Identify feature types
numerical_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Logistic Regression model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("Training Logistic Regression...", flush=True)
model.fit(X_train, y_train)
print("Training completed!", flush=True)

# Get fraud probabilities
y_proba = model.predict_proba(X_val)[:, 1]

# Thresholds to investigate
thresholds = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01]

print("\nDecision Threshold Analysis")
print("-" * 85)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'TP':<8}"
    f"{'FP':<8}"
    f"{'FN':<8}"
    f"{'FPR':<10}"
)

for threshold in thresholds:

    # Apply the threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(y_val, y_pred, zero_division=0)

    # False Positive Rate
    fpr = fp / (fp + tn)

    print(
        f"{threshold:<12.3f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{tp:<8}"
        f"{fp:<8}"
        f"{fn:<8}"
        f"{fpr:<10.4f}"
    )

# Average Precision uses probabilities, not fixed labels
ap = average_precision_score(y_val, y_proba)
print("\nAverage Precision:", round(ap, 6))

# Precision-Recall Curve
precision_values, recall_values, _ = precision_recall_curve(
    y_val, y_proba
)

os.makedirs("reports/stage1", exist_ok=True)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values,
    label=f"Logistic Regression (AP={ap:.4f})"
)

plt.axhline(
    y=y_val.mean(),
    linestyle="--",
    label="Random ranking baseline"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/stage1/precision_recall_curve.png",
    dpi=200
)

plt.close()

print("\nPR Curve saved successfully!")

# Explore a threshold at a maximum 5% False Positive Rate
fpr_values, tpr_values, roc_thresholds = roc_curve(
    y_val,
    y_proba,
    drop_intermediate=False
)

valid_indices = np.where(fpr_values <= 0.05)[0]
best_idx = valid_indices[np.argmax(tpr_values[valid_indices])]

selected_threshold = roc_thresholds[best_idx]

selected_predictions = (
    y_proba >= selected_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, selected_predictions, labels=[0, 1]
).ravel()

print("\nExploratory result at FPR <= 5%")
print("Selected threshold:", selected_threshold)
print("Fraud detected (TP):", tp)
print("False alarms (FP):", fp)
print("Fraud missed (FN):", fn)
print("Precision:", precision_score(y_val, selected_predictions))
print("Recall:", recall_score(y_val, selected_predictions))
print("Actual FPR:", fp / (fp + tn))



# Flagged applications per 1,000

print("\nFlagged Applications per 1,000")
print("-" * 45)

print(
    f"{'Threshold':<12}"
    f"{'Flagged':<12}"
    f"{'Per 1,000':<12}"
)

for threshold in thresholds:

    y_pred = (y_proba >= threshold).astype(int)

    flagged = y_pred.sum()

    flagged_per_1000 = (
        flagged / len(y_val)
    ) * 1000

    print(
        f"{threshold:<12.3f}"
        f"{flagged:<12}"
        f"{flagged_per_1000:<12.2f}"
    )

print("\nAt selected FPR <= 5%:")

flagged = selected_predictions.sum()

print("Total flagged:", flagged)
print("Flagged per 1,000:", round(
    flagged / len(y_val) * 1000, 2
))
