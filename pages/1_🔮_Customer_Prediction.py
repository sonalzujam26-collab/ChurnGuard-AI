import streamlit as st
import pandas as pd
import pickle

from src.ui import load_css


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Prediction",
    page_icon="🔮",
    layout="wide"
)

load_css()


# --------------------------------------------------
# Load Saved Model
# --------------------------------------------------

with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("🔮 Customer Churn Prediction")

st.write(
    "Enter customer information below to estimate "
    "the probability of customer churn."
)

st.divider()


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


with col3:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )


# --------------------------------------------------
# Additional Services
# --------------------------------------------------

st.divider()

st.header("🛠️ Customer Services")

col1, col2, col3, col4 = st.columns(4)

with col1:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


with col4:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    # --------------------------------------------------
    # Create Tenure Group
    # --------------------------------------------------

    if tenure <= 12:
        tenure_group = "New"

    elif tenure <= 24:
        tenure_group = "Early"

    elif tenure <= 48:
        tenure_group = "Established"

    else:
        tenure_group = "Long-term"


    # --------------------------------------------------
    # Calculate Total Services
    # --------------------------------------------------

    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    total_services = sum(
        value in ["Yes", "Yes,"]
        for value in service_values
    )


    # --------------------------------------------------
    # Calculate Average Monthly Spend
    # --------------------------------------------------

    avg_monthly_spend = (
        total_charges / max(tenure, 1)
    )


    # --------------------------------------------------
    # Create Customer DataFrame
    # --------------------------------------------------

    customer_data = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges,

        "TenureGroup": tenure_group,

        "TotalServices": total_services,

        "AvgMonthlySpend": avg_monthly_spend

    }])


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(
        customer_data
    )[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]


    # --------------------------------------------------
    # Determine Risk Level
    # --------------------------------------------------

    if probability < 0.30:

        risk_level = "LOW"

    elif probability < 0.60:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # --------------------------------------------------
    # Display Prediction
    # --------------------------------------------------

    st.divider()

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Customer may churn"
            )

        else:

            st.success(
                "✅ Prediction: Customer may not churn"
            )


    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    # --------------------------------------------------
    # Risk Classification
    # --------------------------------------------------

    st.subheader("Risk Classification")


    if risk_level == "HIGH":

        st.error(
            "🔴 HIGH RISK — The model estimates a relatively "
            "high probability of churn."
        )


    elif risk_level == "MEDIUM":

        st.warning(
            "🟠 MEDIUM RISK — The model estimates a moderate "
            "probability of churn."
        )


    else:

        st.success(
            "🟢 LOW RISK — The model estimates a relatively "
            "low probability of churn."
        )


    st.caption(
        "Risk bands used in this project: "
        "LOW < 30%, MEDIUM 30%–<60%, HIGH ≥ 60%."
    )