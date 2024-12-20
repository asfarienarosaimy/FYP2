import streamlit as st
import pandas as pd
import pickle  # Use to load the trained ML model

# Load the trained machine learning model
# model = pickle.load(open("model.pkl", "rb"))
# For demonstration, we'll use a placeholder for the prediction function
def predict_loan_eligibility(input_data):
    # Placeholder prediction logic
    return "Eligible" if input_data['Credit History'] == 1 else "Not Eligible"

# Streamlit UI
st.title("Loan Eligibility Prediction")
st.write("Enter your information below to check your loan eligibility.")

# User Input Form
with st.form("loan_form"):
    gender = st.selectbox("Gender:", ["Male", "Female"], index=0)
    marital_status = st.selectbox("Marital Status:", ["Married", "Single"], index=1)
    dependents = st.selectbox("Number of Dependents:", ["0", "1", "2", "3+"], index=0)
    education = st.selectbox("Education Level:", ["Graduate", "Not Graduate"], index=0)
    self_employed = st.selectbox("Self Employed:", ["No", "Yes"], index=0)
    applicant_income = st.number_input("Applicant Income (USD):", min_value=0, value=0, step=100)
    coapplicant_income = st.number_input("Coapplicant Income (USD):", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (USD):", min_value=0, value=0, step=100)
    loan_amount_term = st.number_input("Loan Amount Term (months):", min_value=0, value=360, step=1)
    credit_history = st.selectbox("Credit History:", ["No", "Yes"], index=1)
    property_area = st.selectbox("Property Area:", ["Urban", "Semiurban", "Rural"], index=0)

    submit_button = st.form_submit_button(label="Predict")

# Process input and show prediction
if submit_button:
    # Convert inputs into the format required by the model
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Marital Status": 1 if marital_status == "Married" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self Employed": 1 if self_employed == "Yes" else 0,
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount": loan_amount,
        "Loan Amount Term": loan_amount_term,
        "Credit History": 1 if credit_history == "Yes" else 0,
        "Property Area": ["Urban", "Semiurban", "Rural"].index(property_area),
    }

    # Predict using the loaded model
    prediction = predict_loan_eligibility(input_data)

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"The applicant is **{prediction}** for the loan.")
