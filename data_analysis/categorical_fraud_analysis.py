import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Base.csv")

# Use months 0–5 only
development_df = df[df["month"] <= 5]

# Categorical features
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

    # Calculate fraud rate and number of applications per category
    statistics = development_df.groupby(feature)["fraud_bool"].agg(
        fraud_rate="mean",
        applications="count",
        fraud_cases="sum"
    )

    # Convert fraud rate to percentage
    statistics["fraud_rate"] *= 100

    # Sort by fraud rate
    statistics = statistics.sort_values(
        "fraud_rate",
        ascending=False
    )

    print(f"\n{feature}:")
    print(statistics)

    # Plot fraud rate for each category
    ax.bar(
        statistics.index.astype(str),
        statistics["fraud_rate"]
    )

    ax.set_title(f"Fraud Rate by {feature}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Fraud Rate (%)")
    ax.tick_params(axis="x", labelrotation=35)

# Hide unused subplot
for ax in axes[len(categorical_features):]:
    ax.axis("off")

fig.suptitle("Fraud Rates by Categorical Features — Months 0–5")

plt.tight_layout()
plt.show()
