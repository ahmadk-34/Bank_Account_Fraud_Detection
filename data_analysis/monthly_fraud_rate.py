import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Base.csv")

# Use months 0–5 only
development_df = df[df["month"] <= 5]

# Calculate fraud rate for each month
monthly_fraud_rate = (
    development_df.groupby("month")["fraud_bool"].mean() * 100
)

print("Fraud Rate Per Month:")
print(monthly_fraud_rate)

# Plot fraud rate
plt.figure(figsize=(9, 5))

plt.plot(
    monthly_fraud_rate.index,
    monthly_fraud_rate.values,
    marker="o"
)

plt.title("Fraud Rate Across Months (0–5)")
plt.xlabel("Month")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(6))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()