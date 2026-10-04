import streamlit as st
import pandas as pd

from src.ui import load_css

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Churn Insights",
    page_icon="💡",
    layout="wide"
)

load_css()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/customer_churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])


# --------------------------------------------------
# Feature for Churn Analysis
# --------------------------------------------------

df["ChurnNumeric"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("💡 Churn Insights")

st.write(
    "Explore important customer characteristics and "
    "their relationship with customer churn."
)

st.divider()


# --------------------------------------------------
# Churn Rate by Contract
# --------------------------------------------------

st.header("📄 Churn Rate by Contract Type")

contract_churn = (
    df.groupby("Contract")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)
)

st.bar_chart(contract_churn)

st.write(
    "This analysis shows how the observed churn rate "
    "varies across different contract types."
)


# --------------------------------------------------
# Churn Rate by Internet Service
# --------------------------------------------------

st.divider()

st.header("🌐 Churn Rate by Internet Service")

internet_churn = (
    df.groupby("InternetService")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)
)

st.bar_chart(internet_churn)

st.write(
    "The chart compares the observed churn rate "
    "for different internet service categories."
)


# --------------------------------------------------
# Churn Rate by Payment Method
# --------------------------------------------------

st.divider()

st.header("💳 Churn Rate by Payment Method")

payment_churn = (
    df.groupby("PaymentMethod")["ChurnNumeric"]
    .mean()
    .mul(100)
    .round(2)
)

st.bar_chart(payment_churn)

st.write(
    "This shows how the observed churn rate differs "
    "across payment methods."
)


# --------------------------------------------------
# Tenure Analysis
# --------------------------------------------------

st.divider()

st.header("⏳ Customer Tenure and Churn")

tenure_churn = (
    df.groupby("Churn")["tenure"]
    .mean()
    .round(2)
)

st.bar_chart(tenure_churn)

st.write(
    "The chart compares average customer tenure "
    "between churned and non-churned customers."
)


# --------------------------------------------------
# Monthly Charges
# --------------------------------------------------

st.divider()

st.header("💰 Monthly Charges and Churn")

monthly_charges = (
    df.groupby("Churn")["MonthlyCharges"]
    .mean()
    .round(2)
)

st.bar_chart(monthly_charges)

st.write(
    "This compares the average monthly charges "
    "for churned and non-churned customers."
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

st.divider()

st.header("📌 Key Analysis Areas")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "📄 Contract Type\n\n"
        "Compare churn rates across contract categories."
    )

with col2:
    st.info(
        "🌐 Internet Service\n\n"
        "Compare churn rates across internet service types."
    )

with col3:
    st.info(
        "💳 Payment Method\n\n"
        "Compare churn rates across payment methods."
    )