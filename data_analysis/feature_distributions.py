import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/Base.csv")

# Use months 0–5 only
development_df = df[df["month"] <= 5]

# Identify numerical columns
numerical_features = development_df.select_dtypes(
    include="number"
).columns.tolist()

# Exclude target, month, and constant feature
excluded_columns = {
    "fraud_bool",
    "month",
    "device_fraud_count"
}

numerical_features = [
    feature for feature in numerical_features
    if feature not in excluded_columns
]

# Print numerical features
print("Numerical features:")

for feature in numerical_features:
    print(feature)

print(f"\nTotal numerical features: {len(numerical_features)}")

# Plot all numerical features
columns_per_figure = 8

for start in range(0, len(numerical_features), columns_per_figure):

    group = numerical_features[start:start + columns_per_figure]

    # Create 8 plots per figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, feature in zip(axes, group):

        values = development_df[feature]
        unique_values = set(values.dropna().unique())

        # Binary features: use bar charts
        if unique_values.issubset({0, 1}):

            counts = values.value_counts().reindex(
                [0, 1], fill_value=0
            )

            ax.bar(["0", "1"], counts.values)

        # Other numerical features: use histograms
        else:

            sns.histplot(
                x=values,
                bins=30,
                ax=ax
            )

        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of Applications")

    # Hide unused plots
    for ax in axes[len(group):]:
        ax.axis("off")

    fig.suptitle(
        f"Numerical Feature Distributions — Months 0–5 "
        f"(Group {start // columns_per_figure + 1})"
    )

    plt.tight_layout()

plt.show()