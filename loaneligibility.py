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

train.info()

st.write(train['Loan_Status'].value_counts())

plt.figure(figsize=(10, 6))  # Set the figure size
train['Loan_Status'].value_counts().plot.bar(title='Loan Status')
plt.xlabel("Loan Status")  # Add x-axis label
plt.ylabel("Count")        # Add y-axis label
plt.show()                 # Display the plot
