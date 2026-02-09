import streamlit as st
import pandas as pd
import joblib

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Customer Churn & LTV Predictor",
    layout="centered"
)

st.title(" Customer Churn & LTV Predictor")
st.write("Enter customer details below:")

# ================== LOAD MODELS ==================
churn_model = joblib.load("app/churn_model.pkl")
ltv_model = joblib.load("app/ltv_model.pkl")
scaler = joblib.load("app/scaler.pkl")

# ================== USER INPUTS ==================
monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)

tenure = st.number_input(
    "Tenure (months)", min_value=0, value=12
)

contract_risk = st.selectbox(
    "Contract Risk", [0, 1, 2]
)

avg_monthly_spend = st.number_input(
    "Avg Monthly Spend", min_value=0.0, value=65.0
)

# ================== PREDICTION ==================
if st.button("Predict"):

    # 1️ Raw input (single source of truth)
    raw_input = {
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
        "ContractRisk": contract_risk,
        "AvgMonthlySpend": avg_monthly_spend
    }

    # ================== CHURN PIPELINE ==================
    churn_features = list(scaler.feature_names_in_)
    churn_df = pd.DataFrame([raw_input]).reindex(columns=churn_features)

    churn_scaled = scaler.transform(churn_df)
    churn_prob = churn_model.predict_proba(churn_scaled)[0][1]

    # ================== LTV PIPELINE ==================
    ltv_features = list(ltv_model.feature_names_in_)
    ltv_df = pd.DataFrame([raw_input]).reindex(columns=ltv_features)

    predicted_ltv = ltv_model.predict(ltv_df)[0]

    # ================== RESULTS ==================
    st.subheader("Results")

    st.write(f"**Churn Probability:** {churn_prob:.2f}")
    st.write(f"**Predicted LTV:** ₹{predicted_ltv:,.0f}")

    if churn_prob > 0.5 and predicted_ltv > 50000:
        st.error(" High Value • High Risk → Retain Immediately")
    elif churn_prob <= 0.5 and predicted_ltv > 50000:
        st.success(" High Value • Low Risk → VIP Customer")
    elif churn_prob > 0.5:
        st.warning(" High Risk • Low Value → Low-cost Retention")
    else:
        st.info(" Low Priority Customer")
