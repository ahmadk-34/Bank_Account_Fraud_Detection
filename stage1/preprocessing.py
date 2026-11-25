
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
df = pd.read_csv("data/Base.csv")

# Temporal split
train_df = df[df["month"].between(0, 4)].copy()
val_df = df[df["month"] == 5].copy()

# Separate features from the target
columns_to_drop = [
    "fraud_bool",
    "month",
    "device_fraud_count"
]

X_train = train_df.drop(columns=columns_to_drop)
y_train = train_df["fraud_bool"]

X_val = val_df.drop(columns=columns_to_drop)
y_val = val_df["fraud_bool"]

# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\nCategorical columns:")
print(categorical_features)

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Learn preprocessing parameters ONLY from training data
X_train_prepared = preprocessor.fit_transform(X_train)

# Apply the SAME transformations to validation data
X_val_prepared = preprocessor.transform(X_val)

# Show results
print("\nBefore preprocessing:")
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("\nAfter preprocessing:")
print("Training shape:", X_train_prepared.shape)
print("Validation shape:", X_val_prepared.shape)

print("\nTarget distribution:")
print("Training fraud rate:", y_train.mean() * 100, "%")
print("Validation fraud rate:", y_val.mean() * 100, "%")

print("\nPreprocessing completed successfully!")
