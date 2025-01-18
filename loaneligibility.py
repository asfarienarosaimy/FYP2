import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained Logistic Regression model
model = joblib.load('lr_model.joblib')

# Define Streamlit app title and header
st.title('Loan Eligibility Prediction :bank:')
st.header('Predict your loan eligibility with explanations!')

# Input fields for user data
Gender = st.selectbox('Gender', ('Male', 'Female'))
Married = st.selectbox('Married', ('No', 'Yes'))
Dependents = st.selectbox('Number Of Dependents', ('0', '1', '2', '3 or More Dependents'))
Education = st.selectbox('Education Status', ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
ApplicantIncome = st.number_input('Applicant Income (in USD)', 0)
CoapplicantIncome = st.number_input('Coapplicant Income (in USD)', 0)
LoanAmount = st.number_input('Loan Amount (in USD)', 0)
Loan_Amount_Term = st.select_slider(
    'Loan Amount Term (in years)',
    ['1 YEAR', '3 YEARS', '5 YEARS', '7 YEARS',
     '10 YEARS', '15 YEARS', '20 YEARS', '25 YEARS', '30 YEARS', '40 YEARS']
)
Credit_History = st.select_slider('Credit History (1 for Good, 0 for Bad)', [0, 1])
Property_Area = st.selectbox('Area of Property', ('Urban', 'Rural', 'Semiurban'))

# Define column names for the input data
columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Helper function for mapping input data to numerical values (if necessary for your model)
def preprocess_input(data):
    """Preprocess the input to match the model's requirements."""
    mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3 or More Dependents': 3},
        'Property_Area': {'Urban': 0, 'Semiurban': 1, 'Rural': 2}
    }
    # Convert Loan_Amount_Term to numeric (years)
    term_mapping = {
        '1 YEAR': 12, '3 YEARS': 36, '5 YEARS': 60, '7 YEARS': 84,
        '10 YEARS': 120, '15 YEARS': 180, '20 YEARS': 240,
        '25 YEARS': 300, '30 YEARS': 360, '40 YEARS': 480
    }
    data['Loan_Amount_Term'] = term_mapping[data['Loan_Amount_Term']]
    for key in mapping.keys():
        data[key] = mapping[key][data[key]]
    return data

# Prediction function with explanation
def predict():
    # Prepare the input data as a dictionary
    input_data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }
    # Preprocess the input
    processed_data = preprocess_input(input_data)
    data_df = pd.DataFrame([processed_data], columns=columns)

    # Predict eligibility
    prediction = model.predict(data_df)[0]
    probability = model.predict_proba(data_df)[0][1]  # Probability of being eligible

    # Display the prediction and probability
    if prediction == 1:
        st.success(f'You are eligible for the loan! :thumbsup:')
        st.write(f'**Confidence Level:** {probability * 100:.2f}%')
    else:
        st.error(f'Sorry, you are not eligible for the loan. :thumbsdown:')
        st.write(f'**Confidence Level:** {(1 - probability) * 100:.2f}%')

    # Explain why (basic analysis based on inputs)
    st.subheader('Factors affecting your loan eligibility:')
    if Credit_History == 0:
        st.warning('Low credit history is a significant factor.')
    if LoanAmount > ApplicantIncome * 0.4:
        st.warning('Loan amount is high relative to your income.')
    if CoapplicantIncome == 0 and ApplicantIncome < 2500:
        st.warning('Low income without a co-applicant may affect eligibility.')
    if processed_data['Loan_Amount_Term'] > 300:
        st.warning('Long loan terms might reduce approval chances.')

# Styling the button
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color: #ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color: #ff0000;
}
</style>""", unsafe_allow_html=True)

# Add the prediction button
st.button('Predict', on_click=predict)
