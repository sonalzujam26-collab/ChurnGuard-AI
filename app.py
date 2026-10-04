import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide"
)

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide"
)


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
# Calculate Overview Values
# --------------------------------------------------

total_customers = len(df)

churned_customers = (
    df["Churn"] == "Yes"
).sum()

churn_rate = (
    churned_customers / total_customers
) * 100


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ ChurnGuard AI")

st.subheader(
    "Customer Churn Prediction & "
    "Retention Intelligence Dashboard"
)

st.write(
    "ChurnGuard AI is a machine learning application "
    "that predicts customer churn and helps understand "
    "customer characteristics associated with churn."
)

st.divider()


# --------------------------------------------------
# Overview Metrics
# --------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.divider()

st.header("🔎 About This Project")

st.write(
    "This project uses customer information such as "
    "tenure, contract type, services, payment method, "
    "monthly charges and total charges to build a "
    "customer churn prediction system."
)

st.write(
    "Three classification models were explored: "
    "Logistic Regression, Random Forest and XGBoost."
)

st.info(
    "Use the navigation options in the sidebar to "
    "explore customer predictions, data analysis and "
    "model performance."
)