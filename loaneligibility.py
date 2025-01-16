import streamlit as st
import pandas as pd
import pickle  # Use to load the trained ML model
import base64

# Function to encode the image in Base64 and add custom styles
def add_custom_styles(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    custom_styles = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #333333;  /* Darker color for headings */
        font-weight: bold;  /* Make headings bold */
    }}
    p {{
        color: #444444;  /* Slightly lighter color for body text */
        font-size: 16px;  /* Larger font size for better readability */
    }}
    .stRadio label {{
        font-size: 14px;  /* Increase font size for radio labels */
    }}
    .stButton>button {{
        font-size: 16px;  /* Increase font size for buttons */
        background-color: #4CAF50;  /* Green button background */
        color: white;  /* White text on buttons */
        border-radius: 8px;  /* Rounded corners for buttons */
    }}
    </style>
    """
    st.markdown(custom_styles, unsafe_allow_html=True)

# Add your background image and custom styles here
add_custom_styles("loan image.jpg") 

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

# User Input Sections
st.subheader("1. Applicant Information")
gender = st.radio("Gender:", ["Male", "Female"], index=0)
marital_status = st.radio("Marital Status:", ["Married", "Single"], index=1)
dependents = st.radio("Number of Dependents:", ["0", "1", "2", "3+"], index=0)
education = st.radio("Education Level:", ["Graduate", "Not Graduate"], index=0)
self_employed = st.radio("Self Employed:", ["No", "Yes"], index=0)

st.subheader("2. Financial Information")
applicant_income = st.number_input("Applicant Income (RM):", min_value=0, value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income (RM):", min_value=0, value=0, step=100)
loan_amount = st.number_input("Loan Amount (RM):", min_value=0, value=0, step=100)
loan_amount_term = st.radio("Loan Amount Term:", ["12 months (1 year)", "60 months (5 years)", "120 months (10 years)", "360 months (30 years)"], index=3)

st.subheader("3. Credit and Property Details")
credit_history = st.radio("Credit History:", ["No", "Yes"], index=1)
property_area = st.radio("Property Area:", ["Urban", "Semiurban", "Rural"], index=0)

# Process input and show prediction
if st.button("Predict"):
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
        "Loan Amount Term": int(loan_amount_term.split()[0]),
        "Credit History": 1 if credit_history == "Yes" else 0,
        "Property Area": ["Urban", "Semiurban", "Rural"].index(property_area),
    }

    # Predict using the loaded model
    prediction, score = predict_loan_eligibility(input_data)

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"*Eligibility:* {prediction}")
    st.write(f"*Score:* {score:.2f}")

    # Detailed Interpretation
    st.subheader("Detailed Analysis")
    if prediction == "Not Eligible":
        st.write("The loan application is **Not Eligible** due to the following reasons:")
        if score <= 0.5:
            st.write("- Low income-to-loan ratio.")
        if credit_history == "No":
            st.write("- Poor credit history.")
        st.write("- Other factors might include high loan amount or insufficient income.")
    else:
        st.write("The loan application is **Eligible** due to the following reasons:")
        st.write("- Sufficient income-to-loan ratio.")
        st.write("- Good credit history.")
        st.write("- Favorable property area and financial details.")

# Explain the score
st.subheader("Score Explanation")
st.write("The score is calculated as a ratio of the applicant's income to the loan amount, adjusted by other factors.")
st.write(f"- **Applicant Income:** RM{applicant_income}")
st.write(f"- **Loan Amount:** RM{loan_amount}")
st.write(f"- **Credit History:** {'Good' if credit_history == 'Yes' else 'Poor'}")
st.write(f"- **Income-to-Loan Ratio:** {applicant_income / (loan_amount + 1):.2f}")
if coapplicant_income > 0:
    st.write(f"- **Coapplicant Income Contribution:** RM{coapplicant_income}")
if score > 0.5:
    st.write("This indicates that the applicant has a sufficient income relative to the loan amount and/or a favorable credit history.")
else:
    st.write("This indicates that the applicant's income may not be sufficient relative to the loan amount and/or has an unfavorable credit history.")

# Calculate and show estimated monthly payment
loan_term_years = int(loan_amount_term.split()[0]) / 12
monthly_payment = loan_amount / loan_term_years
st.write(f"Estimated Monthly Payment: RM{monthly_payment:.2f}")

# Provide advice based on income-to-loan ratio
income_to_loan_ratio = applicant_income / (loan_amount + 1)
st.write(f"Your income-to-loan ratio is {income_to_loan_ratio:.2f}.")
if income_to_loan_ratio < 0.1:
    st.write("Warning: Your income-to-loan ratio is very low, which may affect eligibility.")
elif income_to_loan_ratio > 0.5:
    st.write("Your income-to-loan ratio is excellent, which increases eligibility.")
