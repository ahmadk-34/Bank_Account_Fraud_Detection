import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Base.csv")

# Use months 0–5 only
development_df = df[df["month"] <= 5]

# Select categorical features
categorical_features = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os"
]

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, feature in zip(axes, categorical_features):

    # Count applications in each category
    counts = development_df[feature].value_counts(dropna=False)

    # Calculate percentages
    percentages = counts / len(development_df) * 100

    # Print results
    print(f"\n{feature} — Months 0–5:")

    for category in counts.index:
        print(
            f"{category}: {counts[category]:,} "
            f"({percentages[category]:.2f}%)"
        )

    # Plot category distribution
    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Applications")
    ax.tick_params(axis="x", labelrotation=35)

# Hide unused subplot
for ax in axes[len(categorical_features):]:
    ax.axis("off")

fig.suptitle("Categorical Feature Distributions — Months 0–5")

plt.tight_layout()
plt.show()