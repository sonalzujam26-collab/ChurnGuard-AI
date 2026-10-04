import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("data/customer_churn.csv")


# -----------------------------------------
# 2. Basic cleaning
# -----------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

df = df.drop(columns=["customerID"])


# -----------------------------------------
# 3. Feature Engineering
# -----------------------------------------

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


# -----------------------------------------
# 4. Separate X and y
# -----------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# -----------------------------------------
# 5. Train-Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 6. Identify feature types
# -----------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# -----------------------------------------
# 7. Create preprocessing
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# -----------------------------------------
# 8. Preprocess data
# -----------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# -----------------------------------------
# 9. Create Logistic Regression model
# -----------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# -----------------------------------------
# 10. Train the model
# -----------------------------------------

model.fit(
    X_train_processed,
    y_train
)


print("Logistic Regression training completed!")


# -----------------------------------------
# 11. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test_processed)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# -----------------------------------------
# 12. Calculate evaluation metrics
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# -----------------------------------------
# 13. Display results
# -----------------------------------------

print("\n--- Logistic Regression Results ---")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# -----------------------------------------
# 14. Confusion Matrix
# -----------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------------------
# 15. Classification Report
# -----------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)