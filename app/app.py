import streamlit as st
import pandas as pd
import joblib
FEATURES = [
    "MonthlyCharges",
    "tenure",
    "ContractRisk",
    "AvgMonthlySpend"
]


# Load models
churn_model = joblib.load("app/churn_model.pkl")
ltv_model = joblib.load("app/ltv_model.pkl")
scaler = joblib.load("app/scaler.pkl")
st.write("Expected features:", scaler.feature_names_in_)


st.set_page_config(page_title="Customer Churn & LTV", layout="centered")
st.title(" Customer Churn & LTV Predictor")

st.write("Enter customer details below:")
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
contract_risk = st.selectbox("Contract Risk", [0, 1, 2])
avg_monthly_spend = st.number_input("Avg Monthly Spend", min_value=0.0, value=65.0)
if st.button("Predict"):
    FEATURES = list(scaler.feature_names_in_)

    input_df = pd.DataFrame([{
    "MonthlyCharges": monthly_charges,
    "tenure": tenure,
    "ContractRisk": contract_risk,
    "AvgMonthlySpend": avg_monthly_spend
}])
    input_df = input_df.reindex(columns=FEATURES)

    
    

    input_scaled = scaler.transform(input_df)

    churn_prob = churn_model.predict_proba(input_scaled)[0][1]
    predicted_ltv = ltv_model.predict(input_df)[0]

    st.subheader("Results")
    st.write(f" Churn Probability: **{churn_prob:.2f}**")
    st.write(f" Predicted LTV: **₹{predicted_ltv:,.0f}**")

    if churn_prob > 0.5 and predicted_ltv > 50000:
        st.error(" High Value – High Risk: Retain Immediately")
    elif churn_prob <= 0.5 and predicted_ltv > 50000:
        st.success(" High Value – Low Risk: VIP Customer")
    elif churn_prob > 0.5:
        st.warning(" High Risk – Low Value: Low-cost retention")
    else:
        st.info(" Low Priority Customer")


