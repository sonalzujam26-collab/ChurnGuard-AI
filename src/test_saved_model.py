import pickle
import pandas as pd


# --------------------------------------------------
# 1. Load Saved Model
# --------------------------------------------------

model_path = "models/churn_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

print("Saved model loaded successfully!")


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/customer_churn.csv")


# --------------------------------------------------
# 3. Prepare One Existing Customer
# --------------------------------------------------

customer = df.iloc[[0]].copy()

customer_id = customer["customerID"].iloc[0]

customer = customer.drop(columns=["customerID", "Churn"])


# --------------------------------------------------
# 4. Apply Same Feature Engineering
# --------------------------------------------------

customer["TotalCharges"] = pd.to_numeric(
    customer["TotalCharges"],
    errors="coerce"
)

customer["TenureGroup"] = pd.cut(
    customer["tenure"],
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

customer["TotalServices"] = 0

for column in service_columns:
    customer["TotalServices"] += (
        customer[column]
        .isin(["Yes", "Yes,"])
        .astype(int)
    )

customer["AvgMonthlySpend"] = (
    customer["TotalCharges"] /
    customer["tenure"].replace(0, 1)
)


# --------------------------------------------------
# 5. Make Prediction
# --------------------------------------------------

prediction = model.predict(customer)[0]

probability = model.predict_proba(customer)[0][1]


# --------------------------------------------------
# 6. Display Result
# --------------------------------------------------

prediction_label = (
    "Churn" if prediction == 1
    else "No Churn"
)

print("\n--- Customer Prediction ---")

print(f"Customer ID: {customer_id}")
print(f"Prediction : {prediction_label}")
print(f"Churn Probability: {probability:.2%}")