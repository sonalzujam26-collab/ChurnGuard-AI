import streamlit as st
import pandas as pd
import pickle

from src.ui import load_css

from src.plot_evaluation import (
    plot_confusion_matrix,
    plot_roc_curve
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📈",
    layout="wide"
)

load_css()

# --------------------------------------------------
# Model Results
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        0.7925,
        0.7889,
        0.7818
    ],
    "Precision": [
        0.6323,
        0.6279,
        0.6070
    ],
    "Recall": [
        0.5241,
        0.5053,
        0.5080
    ],
    "F1 Score": [
        0.5731,
        0.5600,
        0.5531
    ],
    "ROC-AUC": [
        0.8347,
        0.8164,
        0.8306
    ]
})


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("📈 Model Comparison")

st.write(
    "This page compares the performance of the three "
    "classification models evaluated on the same test set."
)

st.divider()


# --------------------------------------------------
# Results Table
# --------------------------------------------------

st.header("📋 Model Performance")

display_results = results.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for column in metric_columns:
    display_results[column] = (
        display_results[column] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Metric Selection
# --------------------------------------------------

st.divider()

st.header("📊 Metric Comparison")

selected_metric = st.selectbox(
    "Select a metric",
    metric_columns
)


chart_data = results.set_index("Model")[
    selected_metric
] * 100

st.bar_chart(chart_data)


# --------------------------------------------------
# Model Observations
# --------------------------------------------------

st.divider()

st.header("🔎 Observations")

st.write(
    "Logistic Regression achieved an accuracy of "
    "79.25% and a ROC-AUC of 83.47% on the test set."
)

st.write(
    "Random Forest achieved an accuracy of 78.89% "
    "and a ROC-AUC of 81.64%."
)

st.write(
    "XGBoost achieved an accuracy of 78.18% and a "
    "ROC-AUC of 83.06%."
)

st.info(
    "These results are specific to the dataset, "
    "feature engineering, model settings and test "
    "split used in this project. They should not be "
    "interpreted as universal model performance."
)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.divider()

st.header("🔍 Feature Importance")

st.write(
    "This section shows the features that have the strongest "
    "positive and negative coefficients in the final "
    "Logistic Regression model."
)


# Load final trained pipeline
with open("models/churn_model.pkl", "rb") as file:
    final_model = pickle.load(file)


# Get preprocessing and model steps
preprocessor = final_model.named_steps["preprocessor"]
logistic_model = final_model.named_steps["model"]


# Get processed feature names
feature_names = preprocessor.get_feature_names_out()


# Get Logistic Regression coefficients
coefficients = logistic_model.coef_[0]


# Create feature importance dataframe
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})


# Calculate absolute importance
feature_importance["Importance"] = (
    feature_importance["Coefficient"].abs()
)


# Sort by importance
feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)


# Display top 15 features
top_features = feature_importance.head(15)


st.subheader("Top 15 Important Features")

st.bar_chart(
    top_features.set_index("Feature")["Importance"]
)


st.dataframe(
    top_features[
        ["Feature", "Coefficient", "Importance"]
    ].round(4),
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Higher absolute coefficient values indicate stronger "
    "influence on the Logistic Regression model. A positive "
    "coefficient is associated with higher predicted churn "
    "probability, while a negative coefficient is associated "
    "with lower predicted churn probability."
)

# --------------------------------------------------
# Model Evaluation Visualizations
# --------------------------------------------------

st.divider()

st.header("📊 Model Evaluation")

st.write(
    "The following visualizations show how the final "
    "Logistic Regression model performed on the test data."
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.subheader("Confusion Matrix")

confusion_matrix_fig = plot_confusion_matrix()

st.pyplot(
    confusion_matrix_fig,
    use_container_width=True
)


st.write(
    "The confusion matrix shows the number of customers "
    "that were correctly and incorrectly classified "
    "as churned or not churned."
)


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

st.subheader("ROC Curve")

roc_curve_fig = plot_roc_curve()

st.pyplot(
    roc_curve_fig,
    use_container_width=True
)


st.write(
    "The ROC curve shows the model's ability to distinguish "
    "between customers who churned and customers who did not churn."
)