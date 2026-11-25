import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/Base.csv")

# Use training months only (0–4)
train_df = df[df["month"].between(0, 4)]

# Identify numerical features
numerical_features = train_df.select_dtypes(
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

print(f"Total numerical features: {len(numerical_features)}")

# Create figures with 8 plots each
features_per_figure = 8

for start in range(0, len(numerical_features), features_per_figure):

    group = numerical_features[start:start + features_per_figure]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, feature in zip(axes, group):

        values = train_df[feature]
        unique_values = set(values.dropna().unique())

        # Binary features: compare fraud rates
        if unique_values.issubset({0, 1}):

            fraud_rates = (
                train_df.groupby(feature)["fraud_bool"].mean() * 100
            )

            fraud_rates = fraud_rates.reindex([0, 1], fill_value=0)

            ax.bar(
                ["0", "1"],
                fraud_rates.values
            )

            ax.set_title(f"Fraud Rate by {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Fraud Rate (%)")

        # Other numerical features: compare distributions
        else:

            sns.boxplot(
                data=train_df,
                x="fraud_bool",
                y=feature,
                ax=ax,
                showfliers=False
            )

            ax.set_title(f"{feature} vs Fraud")
            ax.set_xlabel("Application Type")
            ax.set_ylabel(feature)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Legitimate", "Fraud"])

    # Hide unused plots
    for ax in axes[len(group):]:
        ax.axis("off")

    fig.suptitle(
        f"Numerical Features vs Fraud — Training Months 0–4 "
        f"(Group {start // features_per_figure + 1})"
    )

    plt.tight_layout()

plt.show()