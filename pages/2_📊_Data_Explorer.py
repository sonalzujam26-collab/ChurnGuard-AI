import streamlit as st
import pandas as pd

from src.ui import load_css

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
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
# Page Title
# --------------------------------------------------

st.title("📊 Data Explorer")

st.write(
    "Explore the customer churn dataset using "
    "summary statistics, filters and visualizations."
)

st.divider()


# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------

total_customers = len(df)

churned_customers = (
    df["Churn"] == "Yes"
).sum()

non_churned_customers = (
    df["Churn"] == "No"
).sum()

churn_rate = (
    churned_customers / total_customers
) * 100


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Not Churned",
        f"{non_churned_customers:,}"
    )

with col4:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


# --------------------------------------------------
# Churn Distribution
# --------------------------------------------------

st.divider()

st.header("📈 Churn Distribution")

churn_counts = df["Churn"].value_counts()

st.bar_chart(churn_counts)


# --------------------------------------------------
# Contract Filter
# --------------------------------------------------

st.divider()

st.header("🔎 Explore Customers")

selected_contract = st.selectbox(
    "Select Contract Type",
    [
        "All",
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

if selected_contract != "All":

    filtered_df = df[
        df["Contract"] == selected_contract
    ]

else:

    filtered_df = df


st.write(
    f"Showing {len(filtered_df):,} customers"
)


# --------------------------------------------------
# Customer Data Table
# --------------------------------------------------

display_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    height=400
)


# --------------------------------------------------
# Contract vs Churn
# --------------------------------------------------

st.divider()

st.header("📊 Churn by Contract Type")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

st.bar_chart(contract_churn)


# --------------------------------------------------
# Internet Service vs Churn
# --------------------------------------------------

st.header("🌐 Churn by Internet Service")

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"]
)

st.bar_chart(internet_churn)


# --------------------------------------------------
# Payment Method vs Churn
# --------------------------------------------------

st.header("💳 Churn by Payment Method")

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"]
)

st.bar_chart(payment_churn)