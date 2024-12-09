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

# Plot multiple subplots
st.write("### Additional Distributions")
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

# Gender distribution
train['Gender'].value_counts(normalize=True).plot.bar(ax=axes[0, 0], title="Gender")

# Married distribution
train['Married'].value_counts(normalize=True).plot.bar(ax=axes[0, 1], title="Married")

# Self_Employed distribution
train['Self_Employed'].value_counts(normalize=True).plot.bar(ax=axes[1, 0], title="Self Employed")

# Credit_History distribution
train['Credit_History'].value_counts(normalize=True).plot.bar(ax=axes[1, 1], title="Credit History")

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the figure in Streamlit
st.pyplot(fig)

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title='Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property Area')
