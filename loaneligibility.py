import streamlit as st
import numpy as np
import pandas as pd
import joblib
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

def plot_feature_importance():
  st.subheader("Feature Importance Visualization")
  st.write("The chart below shows the overall importance of each fetaure in determining loan eligibility:")

  try:
      feature_importance = model.feature_importances_
      importance_df = pd.DataFrame({
        'Feature': columns,
        'Importance': feature_importance
      }).sort_values(by='Importance', ascending=False)

      fig, ax = plt.subplots()
      ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
      ax.set_xlabel('Importance Score')
      ax.set_ylabel('Feature')
      ax.set_title('Feature Importance')
      plt.tight_layout()
      st.pyplot(fig)
  except AttributeError:
      st.error("Feature importance is not available for this model.")

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


st.button('Predict',on_click=predict)

plot_feature_importance()
