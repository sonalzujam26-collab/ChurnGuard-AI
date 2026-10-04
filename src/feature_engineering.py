import pandas as pd


# -----------------------------------------
# 1. Load the original dataset
# -----------------------------------------

df = pd.read_csv("data/customer_churn.csv")


# -----------------------------------------
# 2. Basic cleaning
# -----------------------------------------

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove customerID because it is only an identifier
df = df.drop(columns=["customerID"])


# -----------------------------------------
# 3. Create TenureGroup
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


# -----------------------------------------
# 4. Create TotalServices
# -----------------------------------------

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

# Count services that are actually active
df["TotalServices"] = 0

for column in service_columns:
    df["TotalServices"] += (
        df[column]
        .isin(["Yes", "Yes,"])
        .astype(int)
    )


# -----------------------------------------
# 5. Create AvgMonthlySpend
# -----------------------------------------

df["AvgMonthlySpend"] = (
    df["TotalCharges"] / df["tenure"].replace(0, 1)
)


# -----------------------------------------
# 6. Display the new features
# -----------------------------------------

print("\nNew Features Created:")
print(
    df[
        [
            "tenure",
            "TenureGroup",
            "TotalServices",
            "TotalCharges",
            "AvgMonthlySpend"
        ]
    ].head(10)
)


# -----------------------------------------
# 7. Display feature information
# -----------------------------------------

print("\nDataset Shape:", df.shape)

print("\nNew Feature Data Types:")
print(
    df[
        [
            "TenureGroup",
            "TotalServices",
            "AvgMonthlySpend"
        ]
    ].dtypes
)