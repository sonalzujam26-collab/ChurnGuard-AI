import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/customer_churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

df = df.drop(columns=["customerID"])


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=[
        "New",
        "Early",
        "Established",
        "Long-term"
    ]
)


service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


df["TotalServices"] = 0

for column in service_columns:

    df["TotalServices"] += (
        df[column]
        .isin(["Yes", "Yes,"])
        .astype(int)
    )


df["AvgMonthlySpend"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, 1)
)


# --------------------------------------------------
# Prepare X and y
# --------------------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# Same Train-Test Split Used During Training
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Load Saved Model
# --------------------------------------------------

with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)


# --------------------------------------------------
# Generate Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)


print("Model evaluation completed.")

print("\nConfusion Matrix:")
print(cm)

print("\nROC-AUC:")
print(f"{roc_auc:.4f}")