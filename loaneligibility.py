import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

test = pd.read_csv('https://raw.githubusercontent.com/asfarienarosaimy/FYP2/refs/heads/main/testing.csv')
train = pd.read_csv('https://raw.githubusercontent.com/asfarienarosaimy/FYP2/refs/heads/main/training.csv')

train_original = train.copy()
test_original = test.copy()

st.write(train.head(3))

st.write(test.head(3))

# Display the count of Loan_Status values
st.write("### Loan Status Value Counts")
st.write(train['Loan_Status'].value_counts())

# Plot Loan_Status value counts as a bar chart
st.write("### Loan Status Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
train['Loan_Status'].value_counts().plot.bar(ax=ax)
ax.set_title("Loan Status")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")
st.pyplot(fig)  # Embed the plot in the Streamlit app

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')
