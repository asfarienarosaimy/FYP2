import streamlit as st
import pandas as pd
import pickle  # Use to load the trained ML model

# Load the trained machine learning model
# model = pickle.load(open("model.pkl", "rb"))
# For demonstration, we'll use a placeholder for the prediction function
def predict_loan_eligibility(input_data):
    # Placeholder prediction logic
    score = input_data['Applicant Income'] / (input_data['Loan Amount'] + 1)  # Avoid division by zero
    eligibility = "Eligible" if score > 0.5 and input_data['Credit History'] == 1 else "Not Eligible"
    return eligibility, score

# Streamlit UI
st.title("Loan Eligibility Prediction Dashboard")
st.write("This tool helps predict loan eligibility based on user-provided information. Enter your details below to get a detailed analysis and prediction.")

# User Input Form
with st.form("loan_form"):
    st.subheader("Applicant Information")
    gender = st.selectbox("Gender:", ["Male", "Female"], index=0)
    marital_status = st.selectbox("Marital Status:", ["Married", "Single"], index=1)
    dependents = st.selectbox("Number of Dependents:", ["0", "1", "2", "3+"], index=0)
    education = st.selectbox("Education Level:", ["Graduate", "Not Graduate"], index=0)
    self_employed = st.selectbox("Self Employed:", ["No", "Yes"], index=0)
    
    st.subheader("Financial Information")
    applicant_income = st.number_input("Applicant Income (USD):", min_value=0, value=0, step=100)
    coapplicant_income = st.number_input("Coapplicant Income (USD):", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (USD):", min_value=0, value=0, step=100)
    loan_amount_term = st.number_input("Loan Amount Term (months):", min_value=0, value=360, step=1)
    
    st.subheader("Credit and Property Details")
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
    prediction, score = predict_loan_eligibility(input_data)

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"**Eligibility:** {prediction}")
    st.write(f"**Score:** {score:.2f}")
    
    # Detailed Interpretation
    st.subheader("Detailed Analysis")
    st.write("The prediction is based on the following considerations:")
    st.write(f"- Applicant Income: {applicant_income} USD")
    st.write(f"- Loan Amount: {loan_amount} USD")
    st.write(f"- Credit History: {'Good' if credit_history == 'Yes' else 'Poor'}")
    st.write(f"- Loan Term: {loan_amount_term} months")
    st.write(f"- Property Area: {property_area}")
    st.write("These factors are evaluated to determine the loan eligibility score and final decision.")
