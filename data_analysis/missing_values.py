import pandas as pd

# Load dataset
df = pd.read_csv("data/Base.csv")

# Use months 0–5 only
development_df = df[df["month"] <= 5]

# Check standard missing values
missing_counts = development_df.isna().sum()

print("Standard missing values:")
print(missing_counts[missing_counts > 0])

# Check special missing-value indicators
special_columns = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w"
]

print("\nSpecial values (-1):")

for column in special_columns:
    count = (development_df[column] == -1).sum()
    percentage = count / len(development_df) * 100

    print(f"{column}: {count:,} ({percentage:.2f}%)")

# Check negative sentinel values in intended_balcon_amount
negative_balance_count = (development_df["intended_balcon_amount"] < 0).sum()
negative_balance_percentage = negative_balance_count / len(development_df) * 100

print("\nNegative values in intended_balcon_amount:")
print(f"{negative_balance_count:,} ({negative_balance_percentage:.2f}%)")