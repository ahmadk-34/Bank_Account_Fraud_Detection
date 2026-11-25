import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/Base.csv")

# Select months 0–5
development_df = df[df["month"] <= 5]

# Count legitimate and fraudulent applications
fraud_counts = development_df["fraud_bool"].value_counts().sort_index()

# Calculate percentages
fraud_percentages = fraud_counts / fraud_counts.sum() * 100

print("Class Distribution:")
print(f"Legitimate: {fraud_counts[0]:,} ({fraud_percentages[0]:.2f}%)")
print(f"Fraud: {fraud_counts[1]:,} ({fraud_percentages[1]:.2f}%)")

# Create graph
plt.figure(figsize=(8, 5))

plt.bar(
    ["Legitimate", "Fraud"],
    fraud_counts.values,
    color=["steelblue", "tomato"]
)

plt.title("Class Distribution — Months 0–5")
plt.xlabel("Application Type")
plt.ylabel("Number of Applications")

plt.tight_layout()
plt.show()