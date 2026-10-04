import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

df = pd.read_csv("data/customer_churn.csv")


# -----------------------------------------
# 2. Clean the dataset
# -----------------------------------------

# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])

# Remove customerID because it is only an identifier
df = df.drop(columns=["customerID"])


# -----------------------------------------
# 3. Display basic information
# -----------------------------------------

print("Dataset Shape:", df.shape)

print("\nChurn Counts:")
print(df["Churn"].value_counts())


# -----------------------------------------
# 4. Overall Churn Distribution
# -----------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Overall Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 5. Churn by Contract Type
# -----------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=10)

plt.tight_layout()
plt.show()

# -----------------------------------------
# 6. Churn by Tenure
# -----------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    multiple="stack"
)

plt.title("Customer Churn Distribution by Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 7. Monthly Charges and Churn
# -----------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 8. Payment Method and Churn
# -----------------------------------------

plt.figure(figsize=(11, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=25)

plt.tight_layout()
plt.show()

# -----------------------------------------
# 9. Internet Service and Churn
# -----------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 10. Senior Citizen Status and Churn
# -----------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn"
)

plt.title("Customer Churn by Senior Citizen Status")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 11. Correlation Analysis
# -----------------------------------------

# Create a numeric version of Churn
df["ChurnNumeric"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Select numerical columns
numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ChurnNumeric"
]

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Analysis of Numerical Features")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 12. Churn Rate by Contract
# -----------------------------------------

contract_churn_rate = (
    df.groupby("Contract")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nChurn Rate by Contract:")
print(contract_churn_rate)

# -----------------------------------------
# 13. Churn Rate by Internet Service
# -----------------------------------------

internet_churn_rate = (
    df.groupby("InternetService")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nChurn Rate by Internet Service:")
print(internet_churn_rate)


# -----------------------------------------
# 14. Churn Rate by Payment Method
# -----------------------------------------

payment_churn_rate = (
    df.groupby("PaymentMethod")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)

)

print("\nChurn Rate by Payment Method:")
print(payment_churn_rate)


# -----------------------------------------
# 15. Churn Rate by Senior Citizen Status
# -----------------------------------------

senior_churn_rate = (
    df.groupby("SeniorCitizen")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nChurn Rate by Senior Citizen Status:")
print(senior_churn_rate)