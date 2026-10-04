import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/customer_churn.csv")


# --------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

df = df.drop(columns=["customerID"])


# --------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["New", "Early", "Established", "Long-term"]
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
# 4. Separate Features and Target
# --------------------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Identify Feature Types
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 7. Preprocessing
# --------------------------------------------------

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


X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 8. Create XGBoost Model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# --------------------------------------------------
# 9. Train Model
# --------------------------------------------------

model.fit(
    X_train_processed,
    y_train
)

print("XGBoost training completed!")


# --------------------------------------------------
# 10. Predictions
# --------------------------------------------------

y_pred = model.predict(
    X_test_processed
)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# --------------------------------------------------
# 11. Evaluation Metrics
# --------------------------------------------------

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


# --------------------------------------------------
# 12. Display Results
# --------------------------------------------------

print("\n--- XGBoost Results ---")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# 13. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 14. Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)