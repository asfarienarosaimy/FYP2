import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model= joblib.load('lr_model.joblib')

st.title('Loan Eligibility Prediction:bank:')

Gender= st.selectbox('Gender',('Male','Female'))
Married= st.selectbox('Married',('No','Yes'))
Dependents= st.selectbox('Number Of Dependents',('0','1','2','3 or More Dependents'))
Education= st.selectbox('Education status',('Graduate','Not Graduate'))
Self_Employed= st.selectbox('Self Employed',('No','Yes'))
ApplicantIncome= st.number_input('Applicant Income',0)
CoapplicantIncome= st.number_input('Coapplicant Income',0)
LoanAmount= st.number_input('Loan Amount',0)
Loan_Amount_Term= st.select_slider('Loan Amount Term',['1 YEAR','3 YEARS','5 YEARS','7 YEARS',
                                   '10 YEARS','15 YEARS','20 YEARS','25 YEARS','30 YEARS','40 YEARS'])
Credit_History= st.select_slider('Credit History 1 for Good 0 for Bad',[0,1])
Property_Area= st.selectbox('Area of Property',('Urban','Rural','Semiurban'))


columns= ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',
          'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
  
def predict():
    col= np.array([Gender,Married,Dependents,Education,Self_Employed,
                   ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
    data= pd.DataFrame([col],columns=columns)
    prediction= model.predict(data)[0]

    if prediction == 1:
        st.success('🎉 Congratulations! You can get the loan. :thumbsup:')
    else:
        st.error('❌ Sorry, you cannot get the loan. :thumbsdown:')

    # SHAP Explanation
    explainer = shap.Explainer(model, data)  # Create the SHAP explainer
    shap_values = explainer(data)  # Calculate SHAP values for the input

    st.subheader("Key Factors Contributing to the Decision")
    st.write("Below is a SHAP explanation of the factors influencing your loan eligibility decision:")

    # Visualize SHAP values using a waterfall plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)  # For single prediction
    plt.tight_layout()
    st.pyplot(fig)

st.button('Predict',on_click=predict)
