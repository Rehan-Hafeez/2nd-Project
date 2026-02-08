import streamlit as st
import pandas as pd
import joblib

st.title("🏦 Loan Approval Prediction App")

model = joblib.load("loan_model.pkl")

gender = st.selectbox("Gender", ["M", "F"])
city = st.selectbox("City", ["Multan", "Faisalabad", "Peshawar", "Hyderabad", "Islamabad"])
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
bank = st.selectbox("Bank", ["Faysal Bank", "Standard Chartered"])

age = st.number_input("Age", min_value=18)
monthly_income = st.number_input("Monthly Income (PKR)", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
loan_amount = st.number_input("Loan Amount (PKR)", min_value=0)
loan_tenure = st.number_input("Loan Tenure (Months)", min_value=1)
existing_loans = st.number_input("Existing Loans", min_value=0)
default_history = st.selectbox("Default History", [0, 1])
has_credit_card = st.selectbox("Has Credit Card", [0, 1])

input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "city": [city],
    "employment_type": [employment_type],
    "bank": [bank],
    "monthly_income_pkr": [monthly_income],
    "credit_score": [credit_score],
    "loan_amount_pkr": [loan_amount],
    "loan_tenure_months": [loan_tenure],
    "existing_loans": [existing_loans],
    "default_history": [default_history],
    "has_credit_card": [has_credit_card]
})

if st.button("Predict Loan Approval"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
