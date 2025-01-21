import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import base64

# Function to encode the image in Base64
def add_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    background_style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Add your background image here
add_background_image("loan background.jpg") 

# Load the trained model
model = joblib.load('lr_model.joblib')

st.title('Loan Eligibility Prediction :bank:')

# User input fields
Gender = st.selectbox('Gender', ('Male', 'Female'))
Married = st.selectbox('Married', ('No', 'Yes'))
Dependents = st.selectbox('Number Of Dependents', ('0', '1', '2', '3 or More Dependents'))
Education = st.selectbox('Education status', ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
ApplicantIncome = st.number_input('Applicant Income (RM) (1 year = 12 months)', 0)
CoapplicantIncome = st.number_input('Coapplicant Income (RM) (1 year = 12 months)', 0)
LoanAmount = st.number_input('Loan Amount (RM) (5 year = 60 months, 10 year = 120 months)', 0)
Loan_Amount_Term = st.select_slider(
    'Loan Amount Term',
    ['1 YEAR', '3 YEARS', '5 YEARS', '7 YEARS', '10 YEARS', '15 YEARS', '20 YEARS', '25 YEARS', '30 YEARS', '40 YEARS']
)
Credit_History = st.select_slider('Credit History 1 for Good 0 for Bad', [0, 1])
Property_Area = st.selectbox('Area of Property', ('Urban', 'Rural', 'Semiurban'))

# Define column names
columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Function to generate suggestions for improvement
def suggest_improvements(data):
    st.subheader("Suggestions for Improvement")
    suggestions = []
    if data['Credit_History'][0] == 0:
        suggestions.append("Maintain a good credit history to improve your chances.")
    if float(data['ApplicantIncome'][0]) < 5000:
        suggestions.append("Consider increasing your income to meet eligibility requirements.")
    if float(data['LoanAmount'][0]) > 500000:
        suggestions.append("Reduce the requested loan amount to increase approval likelihood.")
    if data['Property_Area'][0] == 'Rural':
        suggestions.append("Consider applying for a loan in urban or semiurban areas for better options.")
    if data['Education'][0] == 'Not Graduate':
        suggestions.append("Further education may enhance your eligibility for loans.")
    
    if suggestions:
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    else:
        st.write("Your application looks strong, no specific suggestions for improvement.")

# Function for prediction
def predict():
    col = np.array([
        Gender, Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area
    ])
    data = pd.DataFrame([col], columns=columns)
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success('🎉 Congratulations! You can get the loan. :thumbsup:')
    else:
        st.error('❌ Sorry, you cannot get the loan. :thumbsdown:')
        suggest_improvements(data)

         # Provide suggestions for improvement
        st.subheader("💡 Suggestions for Improvement")
        
        # Analyze inputs and give suggestions
        suggestions = []
        
        if Credit_History == 0:
            suggestions.append("Improve your credit history by paying off debts or building a good repayment record.")
        if LoanAmount > ApplicantIncome * 0.4:
            suggestions.append("Consider reducing the loan amount or increasing your income.")
        if ApplicantIncome + CoapplicantIncome < 5000:
            suggestions.append("Increase your total income (e.g., by including a co-applicant with a stable income).")
        if Education == 'Not Graduate':
            suggestions.append("Education status affects approval rates. If possible, consider further education.")
        if Dependents in ['2', '3 or More Dependents']:
            suggestions.append("Having fewer dependents reduces financial risk and increases approval chances.")

        if suggestions:
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
        else:
            st.write("No specific suggestions. Review your inputs or consult a loan officer for personalized advice.")

# Feature Importance Visualization
def plot_feature_importance():
    st.subheader("Feature Importance Visualization")
    st.write("The chart below shows the overall importance of each feature in determining loan eligibility:")

    # Replace this with your model's method to extract feature importances
    try:
        # Simulating feature importances (replace this with model.feature_importances_ if available)
        feature_importance = [0.25, 0.18, 0.20, 0.10, 0.08, 0.12, 0.10, 0.15, 0.05, 0.12, 0.05]
        columns = ['Credit_History', 'LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Education',
                   'Married', 'Dependents', 'Gender', 'Loan_Amount_Term', 'Property_Area', 'Self_Employed']

        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        # Text-based visualization 
        st.text("Feature Importance")
        for index, row in importance_df.iterrows():
            bar_length = int(row['Importance'] * 40)  # Scale the bar length for visualization
            percentage = f"{row['Importance'] * 100:.0f}%" # Calculate percentage
            st.text(f"{row['Feature']:20} {'█' * bar_length} {percentage}")

    # Now provide a detailed explanation for each feature's importance
    st.subheader("Explanation of Feature Importance")
    
    explanations = {
        'Credit_History': "Credit history is the most important factor in loan eligibility because it directly reflects the applicant's past financial behavior, such as timely repayments or defaults.",
        'LoanAmount': "The loan amount is important because it directly influences the financial risk to the lender. Larger loan amounts may require stricter eligibility checks.",
        'ApplicantIncome': "Applicant income indicates the applicant's ability to repay the loan. Higher incomes suggest better repayment capacity.",
        'CoapplicantIncome': "The income of a coapplicant boosts the overall income, making the loan application stronger, especially in cases where the applicant’s income alone may not be sufficient.",
        'Education': "Education status can affect the applicant's earning potential. Graduates are more likely to have stable and higher-paying jobs.",
        'Married': "Marital status can be an indicator of financial stability. Married applicants may have a more stable financial environment.",
        'Dependents': "The number of dependents can influence the loan eligibility, as more dependents might suggest more financial responsibilities.",
        'Gender': "Gender may have an impact, though typically less influential in modern models, it can correlate with societal patterns in loan approval.",
        'Loan_Amount_Term': "The length of the loan term impacts the repayment schedule. Longer terms might reduce monthly payments but increase the overall risk for the lender.",
        'Property_Area': "The area of the property can indicate the risk of the loan. Urban areas typically have higher property values, making them less risky for lenders.",
        'Self_Employed': "Being self-employed can influence eligibility due to income variability, with self-employed individuals often seen as riskier borrowers due to unstable income."
    }

    for feature in importance_df['Feature']:
        st.write(f"**{feature}:** {explanations.get(feature, 'No explanation available.')}")

# Style the Predict button
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
}
</style>""", unsafe_allow_html=True)

# Prediction button
st.button('Predict', on_click=predict)

# Display feature importance visualization
plot_feature_importance()
