import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load('lr_model.joblib')

# Set the title of the application
st.title('Loan Eligibility Prediction :bank:')

# Input fields for user data
Gender = st.selectbox('Gender', ('Male', 'Female'))
Married = st.selectbox('Married', ('No', 'Yes'))
Dependents = st.selectbox('Number Of Dependents', ('0', '1', '2', '3 or More Dependents'))
Education = st.selectbox('Education Status', ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
ApplicantIncome = st.number_input('Applicant Income (in dollars)', 0)
CoapplicantIncome = st.number_input('Coapplicant Income (in dollars)', 0)
LoanAmount = st.number_input('Loan Amount (in dollars)', 0)
Loan_Amount_Term = st.select_slider(
    'Loan Amount Term',
    ['1 YEAR', '3 YEARS', '5 YEARS', '7 YEARS', '10 YEARS', '15 YEARS', '20 YEARS', '25 YEARS', '30 YEARS', '40 YEARS']
)
Credit_History = st.select_slider('Credit History (1 for Good, 0 for Bad)', [0, 1])
Property_Area = st.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))

# Define column names for the input data
columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
           'Credit_History', 'Property_Area']

# Helper function to preprocess input data
def preprocess_data():
    loan_term_mapping = {
        '1 YEAR': 12, '3 YEARS': 36, '5 YEARS': 60, '7 YEARS': 84,
        '10 YEARS': 120, '15 YEARS': 180, '20 YEARS': 240, '25 YEARS': 300,
        '30 YEARS': 360, '40 YEARS': 480
    }
    dependents_mapping = {'0': 0, '1': 1, '2': 2, '3 or More Dependents': 3}
    property_area_mapping = {'Urban': 0, 'Rural': 1, 'Semiurban': 2}

    # Map categorical values to numerical values
    data = {
        'Gender': 1 if Gender == 'Male' else 0,
        'Married': 1 if Married == 'Yes' else 0,
        'Dependents': dependents_mapping[Dependents],
        'Education': 1 if Education == 'Graduate' else 0,
        'Self_Employed': 1 if Self_Employed == 'Yes' else 0,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': loan_term_mapping[Loan_Amount_Term],
        'Credit_History': Credit_History,
        'Property_Area': property_area_mapping[Property_Area]
    }
    return pd.DataFrame([data])

# Function to predict loan eligibility
def predict():
    # Preprocess input data
    data = preprocess_data()

    # Make prediction
    prediction = model.predict(data)[0]
    prediction_probability = model.predict_proba(data)[0]

    # Display result
    if prediction == 1:
        st.success(f'🎉 Congratulations! You are eligible for the loan.')
        st.info(f'Probability of approval: {prediction_probability[1]:.2f}')
    else:
        st.error(f'Sorry, you are not eligible for the loan.')
        st.info(f'Probability of approval: {prediction_probability[1]:.2f}')
        
        # Provide insights into reasons for ineligibility
        st.subheader('Possible reasons:')
        if Credit_History == 0:
            st.warning('❌ Poor credit history significantly reduces eligibility.')
        if ApplicantIncome < 3000 and CoapplicantIncome < 2000:
            st.warning('❌ Low applicant and coapplicant income.')
        if LoanAmount > (ApplicantIncome + CoapplicantIncome) * 0.5:
            st.warning('❌ Loan amount requested is high compared to income.')
        if Loan_Amount_Term < 60:
            st.warning('❌ Short loan term increases monthly repayment burden.')

# Add button with custom styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
}
</style>
""", unsafe_allow_html=True)

# Create Predict button
if st.button('Predict', on_click=predict):
    st.write("Processing your prediction...")
