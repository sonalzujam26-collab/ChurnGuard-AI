# ChurnGuard AI – Customer Churn Prediction & Retention Intelligence Dashboard

ChurnGuard AI is a machine learning project that predicts whether a telecom customer is likely to churn. The project combines data preprocessing, exploratory data analysis, feature engineering, machine learning classification, model evaluation, and an interactive Streamlit dashboard.

## Project Objective

The main objective of ChurnGuard AI is to build a machine learning system that can:

- Predict customer churn probability
- Identify customers at higher risk of churn
- Analyze important customer and service-related patterns
- Compare different machine learning models
- Provide an interactive dashboard for customer-level predictions and data exploration

## Problem Statement

Customer churn is an important business problem for subscription-based services. Identifying customers who may leave can help organizations understand customer behavior and plan appropriate retention strategies.

This project uses historical telecom customer data to build a binary classification model where:

- `0` = Customer does not churn
- `1` = Customer churns

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information related to:

- Demographics
- Account information
- Services used
- Contract type
- Payment method
- Tenure
- Monthly charges
- Total charges
- Churn status

Dataset source:

https://github.com/IBM/telco-customer-churn-on-icp4d

## Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Data Preprocessing
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Final Model Selection
        ↓
Streamlit Dashboard
