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

# Plot Dependents, Education, Property Area subplots
st.write("### Other Distributions")
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Dependents distribution
train['Dependents'].value_counts(normalize=True).plot.bar(ax=axes[0], title="Dependents")

# Education distribution
train['Education'].value_counts(normalize=True).plot.bar(ax=axes[1], title="Education")

# Property_Area distribution
train['Property_Area'].value_counts(normalize=True).plot.bar(ax=axes[2], title="Property Area")

# Adjust layout
plt.tight_layout()

# Display the figure in Streamlit
st.pyplot(fig)

# Plot ApplicantIncome distribution and boxplot
st.write("### Applicant Income Distribution and Boxplot")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Distribution plot (left subplot)
sns.histplot(train['ApplicantIncome'], kde=True, ax=axes[0])  # Updated sns.distplot to sns.histplot
axes[0].set_title("Applicant Income Distribution")
axes[0].set_xlabel("Applicant Income")
axes[0].set_ylabel("Frequency")

# Box plot (right subplot)
train['ApplicantIncome'].plot.box(ax=axes[1])
axes[1].set_title("Applicant Income Box Plot")

# Display the figure in Streamlit
st.pyplot(fig)

# Plot ApplicantIncome boxplot grouped by Education
st.write("### Applicant Income by Education")
fig, ax = plt.subplots(figsize=(10, 6))

# Create the boxplot
train.boxplot(column='ApplicantIncome', by='Education', ax=ax)
ax.set_title("Applicant Income by Education")
ax.set_ylabel("Applicant Income")
plt.suptitle("")  # Remove the automatic "Boxplot grouped by Education" title

# Display the figure in Streamlit
st.pyplot(fig)

# Plot Married vs Loan_Status as a stacked bar plot
st.write("### Married vs Loan Status")
fig, ax = plt.subplots(figsize=(6, 4))

# Create a crosstab and normalize it
Married = pd.crosstab(train['Married'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, ax=ax)

# Add labels and legend
ax.set_title("Married vs Loan Status")
ax.set_xlabel("Marital Status")
ax.set_ylabel("Proportion")
ax.legend(loc='best')

# Display the figure in Streamlit
st.pyplot(fig)

# Plot Credit_History vs Loan_Status as a stacked bar plot
st.write("### Credit History vs Loan Status")
fig, ax = plt.subplots(figsize=(6, 4))

# Create a crosstab and normalize it
Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, ax=ax)

# Add labels and legend
ax.set_title("Credit History vs Loan Status")
ax.set_xlabel("Credit History")
ax.set_ylabel("Proportion")
ax.legend(title="Loan Status", loc='best')

# Display the plot in Streamlit
st.pyplot(fig)

# Plot Property_Area vs Loan_Status as a stacked bar plot
st.write("### Property Area vs Loan Status")
fig, ax = plt.subplots(figsize=(6, 4))

# Create a crosstab and normalize it
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, ax=ax)

# Add labels and legend
ax.set_title("Property Area vs Loan Status")
ax.set_xlabel("Property Area")
ax.set_ylabel("Proportion")
ax.legend(title="Loan Status", bbox_to_anchor=(1.05, 1.0), loc='upper left')  # Adjust legend position

# Display the plot in Streamlit
st.pyplot(fig)
