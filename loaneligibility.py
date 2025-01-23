import streamlit as st
import numpy as np
import pandas as pd
import joblib
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

# Sidebar for navigation
page = st.sidebar.radio("Navigate to", ["Home", "Prediction", "Suggestions"])

# Columns for the data
columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Home Page
if page == "Home":
    
    st.title("Loan Eligibility System")
    st.markdown("Welcome to the Loan Eligibility Prediction App. Use the sidebar to navigate to different sections.")

   # Display the centered image with a caption
    col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns, center column is wider
    with col2:  # Place the image and caption in the center column
        st.image("images 2.webp", caption="Loan Application System", width=400)

# Sidebar for navigation
page = st.sidebar.radio("Navigate to", ["Home", "Prediction", "Suggestions"])

# Store prediction result globally
if "loan_result" not in st.session_state:
    st.session_state.loan_result = None  # Initialize the loan result
if "input_data" not in st.session_state:
    st.session_state.input_data = None  # Store user input for suggestions

# Prediction Page
elif page == "Prediction":
    st.title("Loan Eligibility Prediction :bank:")

    # User input fields
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Married = st.selectbox('Married', ('No', 'Yes'))
    Dependents = st.selectbox('Number Of Dependents', ('0', '1', '2', '3 or More Dependents'))
    Education = st.selectbox('Education status', ('Graduate', 'Not Graduate'))
    Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
    ApplicantIncome = st.number_input('Applicant Income (RM) (per month)', 0)
    CoapplicantIncome = st.number_input('Coapplicant Income (RM) (per month)', 0)
    LoanAmount = st.number_input('Loan Amount (RM) (to applied)', 0)
    Loan_Amount_Term = st.select_slider(
        'Loan Amount Term',
        ['1 YEAR', '3 YEARS', '5 YEARS', '7 YEARS', '9 YEARS', '11 YEARS', '13 YEARS', '15 YEARS', '17 YEARS', '19 YEARS', '21 YEARS', '23 YEARS', '25 YEARS', '27 YEARS', '29 YEARS', '31 YEARS', '33 YEARS', '35 YEARS', '37 YEARS', '40 YEARS']
    )
    Credit_History = st.selectbox('Credit_History (1 for Good 0 for Bad)', ('0', '1'))
    Property_Area = st.selectbox('Area of Property', ('Urban', 'Rural', 'Semiurban'))

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

    # Button for prediction
    if st.button('Predict'):
        predict()

# Suggestions Page
elif page == "Suggestions":
    st.title("Suggestions and Approval Feedback")

    # Function to generate suggestions or reasons for approval
    def suggest_improvements(data):
        if st.session_state.loan_result == 1:
            # Loan approved: provide reasons for approval
            st.subheader("🎉 Reasons for Loan Approval:")
            if data['Credit_History'][0] == 1:
                st.write("• Good credit history maintained.")
            if float(data['ApplicantIncome'][0]) >= 5000:
                st.write("• Applicant's income meets the required threshold.")
            if float(data['LoanAmount'][0]) <= 500000:
                st.write("• Loan amount is within the acceptable range.")
            if data['Property_Area'][0] in ['Urban', 'Semiurban']:
                st.write(f"• Property area ({data['Property_Area'][0]}) qualifies for better loan options.")
            if data['Education'][0] == 'Graduate':
                st.write("• Applicant's educational background enhances eligibility.")
        elif st.session_state.loan_result == 0:
            # Loan not approved: provide suggestions for improvement
            st.subheader("❌ Suggestions for Improvement:")
            if data['Credit_History'][0] == 0:
                st.write("• Maintain a good credit history to improve your chances.")
            if float(data['ApplicantIncome'][0]) < 5000:
                st.write("• Consider increasing your income to meet eligibility requirements.")
            if float(data['LoanAmount'][0]) > 500000:
                st.write("• Reduce the requested loan amount to increase approval likelihood.")
            if data['Property_Area'][0] == 'Rural':
                st.write("• Consider applying for a loan in urban or semiurban areas for better options.")
            if data['Education'][0] == 'Not Graduate':
                st.write("• Further education may enhance your eligibility for loans.")
        else:
            st.write("⚠️ Please make a prediction on the Prediction Page first.")

    # Display suggestions based on stored data
    if st.session_state.input_data is not None:
        suggest_improvements(st.session_state.input_data)
    else:
        st.write("⚠️ No data available. Please predict your loan eligibility on the Prediction Page.")
