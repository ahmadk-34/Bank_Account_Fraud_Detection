
import pandas as pd

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
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/Base.csv")

# Split by time
train_df = df[df["month"].between(0, 4)].copy()
val_df = df[df["month"] == 5].copy()

# Separate features and labels
drop_columns = [
    "fraud_bool",
    "month",
    "device_fraud_count"
]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["fraud_bool"]

X_val = val_df.drop(columns=drop_columns)
y_val = val_df["fraud_bool"]

# Identify feature types
numerical_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

# Preprocessing
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

# Create full ML pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# Train the model
print("Training Logistic Regression...", flush=True)
model.fit(X_train, y_train)

print("Training completed!", flush=True)

# Predict on validation data
y_pred = model.predict(X_val)

# Probability of fraud
y_proba = model.predict_proba(X_val)[:, 1]

# Evaluate
print("\nLogistic Regression Results")
print("---------------------------")

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, zero_division=0))
print("Recall:", recall_score(y_val, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_val, y_pred, zero_division=0))
print("Average Precision:", average_precision_score(y_val, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nActual fraud cases:", int(y_val.sum()))
print("Predicted fraud cases:", int(y_pred.sum()))
print("Correctly detected fraud:", int(((y_val == 1) & (y_pred == 1)).sum()))
