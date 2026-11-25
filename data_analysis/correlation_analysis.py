import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/Base.csv")

# Use training months only (0–4)
train_df = df[df["month"].between(0, 4)]

# ==========================================
# 2. Select Numerical Features
# ==========================================

numerical_df = train_df.select_dtypes(
    include="number"
).copy()

# Remove month and constant feature
numerical_df = numerical_df.drop(
    columns=["month", "device_fraud_count"],
    errors="ignore"
)

print("Number of columns:", numerical_df.shape[1])

# ==========================================
# 3. Calculate Correlation Matrix
# ==========================================

correlation_matrix = numerical_df.corr(
    method="pearson"
)

# ==========================================
# 4. Feature-to-Feature Correlation Heatmap
# ==========================================

# Exclude target variable
feature_correlations = correlation_matrix.drop(
    index="fraud_bool",
    columns="fraud_bool"
)

plt.figure(figsize=(16, 12))

sns.heatmap(
    feature_correlations,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.3
)

plt.title("Numerical Feature Correlations — Training Months 0–4")

plt.tight_layout()
plt.show()

# ==========================================
# 5. Feature-to-Fraud Correlations
# ==========================================

fraud_correlations = correlation_matrix[
    "fraud_bool"
].drop("fraud_bool")

# Sort correlations
fraud_correlations = fraud_correlations.sort_values()

print("\nCorrelations with Fraud:")
print(fraud_correlations)

# ==========================================
# 6. Feature-to-Fraud Correlation Heatmap
# ==========================================

plt.figure(figsize=(8, 10))

sns.heatmap(
    fraud_correlations.to_frame(),
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title("Numerical Feature Correlations with Fraud")

plt.tight_layout()
plt.show()

# ==========================================
# 7. Find Strongly Correlated Feature Pairs
# ==========================================

print("\nStrongly Correlated Feature Pairs (|r| >= 0.7):")

features = feature_correlations.columns

found = False

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        feature_1 = features[i]
        feature_2 = features[j]

        correlation = feature_correlations.loc[
            feature_1, feature_2
        ]

        if abs(correlation) >= 0.7:

            print(
                f"{feature_1} <-> {feature_2}: "
                f"{correlation:.3f}"
            )

            found = True

if not found:
    print("No strongly correlated feature pairs found.")