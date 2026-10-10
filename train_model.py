
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
DATA_FILE = "ai4i2020.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Put the CSV in your project folder."
    )

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Select input features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

# 3. Prepare the input and target
X = df[features].copy()
y = df[target].astype(int)

# Convert machine type L, M, H into numeric columns
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# 4. Split the dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save the model and feature information
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features,
        "encoded_columns": X.columns.tolist()
    },
    "models/random_forest_model.joblib"
)

print("\nModel saved successfully!")
print("Location: models/random_forest_model.joblib")
