import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

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
    
    # Extract feature importances (ensure your model supports this)
    try:
        feature_importance = model.feature_importances_  # Replace with appropriate attribute if different
        importance_df = pd.DataFrame({
            'Feature': columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
    except AttributeError:
        st.error("Feature importance is not available for this model.")

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
