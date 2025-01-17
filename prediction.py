import streamlit as st

st.title('Prediction Loan')
st.caption('prediction using supervised learning rf model')

with st.sidebar:
    st.markdown('Menu Navigation')
    eda = st.Page('loaneligibility.py', title='EDA', icon='📊')
    predict = st.Page('prediction.py', title='Prediction Loan', icon='')






