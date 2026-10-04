import streamlit as st

from src.ui import load_css

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="About ChurnGuard AI",
    page_icon="ℹ️",
    layout="wide"
)

load_css()

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("ℹ️ About ChurnGuard AI")

st.subheader(
    "Customer Churn Prediction & Retention Intelligence Dashboard"
)

st.write(
    "ChurnGuard AI is an end-to-end machine learning project "
    "developed to predict whether a customer is likely to churn "
    "and to explore customer characteristics related to churn."
)

st.divider()


# --------------------------------------------------
# Project Objective
# --------------------------------------------------

st.header("🎯 Project Objective")

st.write(
    "The main objective of this project is to use customer "
    "data and machine learning classification techniques "
    "to identify customers who may be at risk of churn."
)

st.write(
    "The project also provides an interactive dashboard "
    "for customer prediction, data exploration, model "
    "comparison and churn analysis."
)


# --------------------------------------------------
# Machine Learning Workflow
# --------------------------------------------------

st.header("🔄 Machine Learning Workflow")

workflow = [
    "1. Dataset Collection",
    "2. Data Cleaning",
    "3. Exploratory Data Analysis",
    "4. Feature Engineering",
    "5. Data Preprocessing",
    "6. Model Training",
    "7. Model Evaluation",
    "8. Customer Churn Prediction",
    "9. Streamlit Dashboard"
]

for step in workflow:
    st.write(step)


# --------------------------------------------------
# Models
# --------------------------------------------------

st.divider()

st.header("🤖 Machine Learning Models")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "### Logistic Regression\n"
        "A classification algorithm used to "
        "predict the probability of customer churn."
    )

with col2:
    st.info(
        "### Random Forest\n"
        "An ensemble learning algorithm that "
        "combines multiple decision trees."
    )

with col3:
    st.info(
        "### XGBoost\n"
        "A gradient boosting algorithm used "
        "for classification tasks."
    )


# --------------------------------------------------
# Technology Stack
# --------------------------------------------------

st.divider()

st.header("🛠️ Technology Stack")

technologies = {
    "Programming Language": "Python",
    "Data Processing": "Pandas, NumPy",
    "Data Visualization": "Matplotlib, Seaborn",
    "Machine Learning": "Scikit-learn, XGBoost",
    "Dashboard": "Streamlit",
    "Version Control": "Git",
    "Code Repository": "GitHub"
}

for technology, tools in technologies.items():
    st.write(f"**{technology}:** {tools}")


# --------------------------------------------------
# Project Features
# --------------------------------------------------

st.divider()

st.header("✨ Dashboard Features")

features = [
    "Customer churn prediction",
    "Churn probability and risk level",
    "Interactive customer data exploration",
    "Churn analysis by customer characteristics",
    "Comparison of machine learning models",
    "Model performance metrics",
    "Interactive Streamlit interface"
]

for feature in features:
    st.write(f"• {feature}")


# --------------------------------------------------
# Project Structure
# --------------------------------------------------

st.divider()

st.header("📁 Project Structure")

st.code(
"""ChurnGuard-AI/
│
├── app.py
├── pages/
│   ├── 1_🔮_Customer_Prediction.py
│   ├── 2_📊_Data_Explorer.py
│   ├── 3_📈_Model_Comparison.py
│   ├── 4_💡_Churn_Insights.py
│   └── 5_ℹ️_About.py
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_preparation.py
│   ├── train_model.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_final_model.py
│   └── test_saved_model.py
│
├── assets/
└── requirements.txt
""",
    language="text"
)


# --------------------------------------------------
# Final Note
# --------------------------------------------------

st.divider()

st.success(
    "ChurnGuard AI combines data analysis, machine learning "
    "and an interactive dashboard into one end-to-end project."
)

st.caption(
    "Built as a beginner-friendly end-to-end machine learning project."
)