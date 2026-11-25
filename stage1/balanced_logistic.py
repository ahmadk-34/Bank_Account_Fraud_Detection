
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    roc_curve
)

# Load the dataset
df = pd.read_csv("data/Base.csv")

# Temporal split
train_df = df[df["month"].between(0, 4)].copy()
val_df = df[df["month"] == 5].copy()

drop_columns = [
    "fraud_bool",
    "month",
    "device_fraud_count"
]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["fraud_bool"]

X_val = val_df.drop(columns=drop_columns)
y_val = val_df["fraud_bool"]

# Identify features
numerical_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()


def create_model(class_weight):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=class_weight
        ))
    ])


# Two models for comparison
models = {
    "Standard": create_model(None),
    "Balanced": create_model("balanced")
}

results = {}

print("Training applications:", len(y_train))
print("Validation applications:", len(y_val))
print("Training fraud rate:", round(y_train.mean() * 100, 4), "%")

for name, model in models.items():

    print(f"\nTraining {name} Logistic Regression...", flush=True)

    model.fit(X_train, y_train)

    # Predict probabilities and labels
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    ap = average_precision_score(y_val, y_proba)

    print(f"\n{name} Logistic Regression")
    print("-" * 45)
    print("Accuracy:", round(accuracy_score(y_val, y_pred), 6))
    print("Precision:", round(precision_score(y_val, y_pred, zero_division=0), 6))
    print("Recall:", round(recall_score(y_val, y_pred, zero_division=0), 6))
    print("F1:", round(f1_score(y_val, y_pred, zero_division=0), 6))
    print("Average Precision:", round(ap, 6))

    print("True Positives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Negatives:", tn)

    results[name] = {
        "probabilities": y_proba,
        "ap": ap
    }


# Compare both models at the same FPR constraint
print("\nComparison at FPR <= 5%")
print("-" * 75)

print(
    f"{'Model':<12}"
    f"{'Threshold':<14}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'TP':<8}"
    f"{'FP':<8}"
)

for name, result in results.items():

    y_proba = result["probabilities"]

    fpr_values, tpr_values, thresholds = roc_curve(
        y_val,
        y_proba,
        drop_intermediate=False
    )

    valid = np.where(fpr_values <= 0.05)[0]
    best_idx = valid[np.argmax(tpr_values[valid])]

    threshold = thresholds[best_idx]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)

    print(
        f"{name:<12}"
        f"{threshold:<14.6f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{tp:<8}"
        f"{fp:<8}"
    )


# Plot both Precision-Recall curves
os.makedirs("reports/stage1", exist_ok=True)

plt.figure(figsize=(9, 6))

for name, result in results.items():

    precision_values, recall_values, _ = precision_recall_curve(
        y_val,
        result["probabilities"]
    )

    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (AP={result['ap']:.4f})"
    )

plt.axhline(
    y_val.mean(),
    linestyle="--",
    color="gray",
    label="Random ranking baseline"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Standard vs Balanced Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/stage1/standard_vs_balanced_pr.png",
    dpi=200
)

plt.close()

print("\nComparison graph saved successfully!")



# Experiment: Score Distribution Comparison

standard_scores = results["Standard"]["probabilities"]
balanced_scores = results["Balanced"]["probabilities"]

labels = y_val.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

classes = [
    (0, "Legitimate Applications"),
    (1, "Fraudulent Applications")
]

for ax, (class_value, class_name) in zip(axes, classes):

    mask = labels == class_value

    ax.hist(
        standard_scores[mask],
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="Standard LR"
    )

    ax.hist(
        balanced_scores[mask],
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="Balanced LR"
    )

    ax.set_title(class_name)
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Density (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Fraud Score Distribution Comparison")
plt.tight_layout()

plt.savefig(
    "reports/stage1/score_distribution_comparison.png",
    dpi=200
)

plt.close()

print("\nScore Distribution Summary")
print("-" * 55)

for name, scores in [
    ("Standard", standard_scores),
    ("Balanced", balanced_scores)
]:
    for class_value, class_name in classes:
        class_scores = scores[labels == class_value]

        print(
            f"{name} - {class_name}: "
            f"Mean={np.mean(class_scores):.4f}, "
            f"Median={np.median(class_scores):.4f}"
        )

# Compare rankings using Spearman correlation
standard_ranks = pd.Series(standard_scores).rank()
balanced_ranks = pd.Series(balanced_scores).rank()

rank_correlation = standard_ranks.corr(
    balanced_ranks,
    method="pearson"
)

print("\nSpearman Rank Correlation:", round(rank_correlation, 6))
print("Score distribution graph saved successfully!")
