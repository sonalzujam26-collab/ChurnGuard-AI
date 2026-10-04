import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

# Tenure groups
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


# Count active services
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


# Average monthly spending
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


print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


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


print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# -----------------------------------------
# 6. Identify numerical and categorical columns
# -----------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# -----------------------------------------
# 7. Create preprocessing pipeline
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
# 8. Fit preprocessing only on training data
# -----------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)