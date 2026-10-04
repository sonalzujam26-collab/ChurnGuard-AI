import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


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
# 4. Features and Target
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
# 7. Create Preprocessor
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


# --------------------------------------------------
# 8. Create Final Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# --------------------------------------------------
# 9. Create Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 10. Train Pipeline
# --------------------------------------------------

pipeline.fit(
    X_train,
    y_train
)

print("Final model training completed!")


# --------------------------------------------------
# 11. Save Complete Pipeline
# --------------------------------------------------

model_path = "models/churn_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(pipeline, file)

print(f"Model saved successfully to: {model_path}")


# --------------------------------------------------
# 12. Test Saved Pipeline
# --------------------------------------------------

test_predictions = pipeline.predict(X_test)

print(
    f"Test predictions generated successfully: "
    f"{len(test_predictions)} customers"
)