import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the application
st.title("Loan Dataset Viewer and Preprocessing")

# File uploader widget
uploaded_file = st.file_uploader("Upload your loan_data_set.csv file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())  # Display the first few rows of the dataset

        # Filter Applicant Income <= 40000
        df_filtered_applicant = df[df['ApplicantIncome'] <= 40000]
        st.write("### Filtered Dataset (Applicant Income ≤ 40000)")
        st.write(df_filtered_applicant.head())

        # Plot histogram for Applicant Income
        st.write("### Distribution of Applicant Income (Up to 40000)")
        fig, ax = plt.subplots()
        df_filtered_applicant['ApplicantIncome'].hist(bins=50, ax=ax)
        ax.set_xlabel('Applicant Income')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Applicant Income (Up to 40000)')
        st.pyplot(fig)

        # Coapplicant Income Analysis
        st.write("## Coapplicant Income Analysis (Up to 20000)")
        df_filtered_coapplicant = df[df['CoapplicantIncome'] <= 20000]
        st.write("Filtered Dataset (Coapplicant Income ≤ 20000):")
        st.write(df_filtered_coapplicant.head())

        # Plot Coapplicant Income Distribution
        fig2, ax2 = plt.subplots()
        df_filtered_coapplicant['CoapplicantIncome'].hist(bins=50, ax=ax2)
        ax2.set_xlabel('Coapplicant Income')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Coapplicant Income (Up to 20000)')
        st.pyplot(fig2)

        # Loan Amount Analysis
        st.write("## Loan Amount Analysis")
        fig3, ax3 = plt.subplots()
        df['LoanAmount'].hist(bins=50, ax=ax3)
        ax3.set_xlabel('Loan Amount')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Loan Amount')
        st.pyplot(fig3)

        # Gender Distribution
        st.write("## Gender Distribution")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Gender', data=df, ax=ax4)
        ax4.set_title('Gender Distribution')
        ax4.set_xlabel('Gender')
        ax4.set_ylabel('Count')
        st.pyplot(fig4)

        # Married Distribution
        st.write("## Married Distribution")
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Married', data=df, ax=ax5)
        ax5.set_title('Married Distribution')
        ax5.set_xlabel('Married')
        ax5.set_ylabel('Count')
        st.pyplot(fig5)

        # Dependents Distribution
        st.write("## Dependents Distribution")
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Dependents', data=df, ax=ax6)
        ax6.set_title('Dependents Distribution')
        ax6.set_xlabel('Number of Dependents')
        ax6.set_ylabel('Count')
        st.pyplot(fig6)

        # Education Distribution
        st.write("## Education Distribution")
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Education', data=df, ax=ax7)
        ax7.set_title('Education Distribution')
        ax7.set_xlabel('Education Level')
        ax7.set_ylabel('Count')
        st.pyplot(fig7)

        # Drop the 'Loan_ID' column
        if 'Loan_ID' in columns:
            df = drop('Loan_ID', axis=1)
            st.write("### Dataset After Dropping 'Loan_ID' Column")
            st.write(df.head()) # Display the updated dataset
        else:
            st.warning("'Loan_ID' column not found in the dataset.")

        # Check for empty values in each column
        empty_values_per_column = df.isnull().sum()
        st.write("### Missing Values per Column")
        st.write(empty_values_per_column)  # Display missing values for each column

        # Check for missing values before filling
        missing_values_before = df.isnull().sum()
        st.write("### Missing Values Before Filling")
        st.write(missing_values_before) # Display the count of missing values for each column

        # Fill missing values: for numeric columns, use median; for non-numeric, use mode
        st.write("### Filling Missing Values")
        for column in df.columns:
           if df[column].isnull().any(): # Check if the column has any missing values
               if pd.api.types.is_numeric_dtype(df[column]):
                   df[column] = df[column].fillna(df[column].median())
               else: # For non-numeric columns, use the most frequent value (mode) 
                   df[column] = df[column].fillna(df[column].mode()[0])

        st.write("Dataset after filling missing values:")
        st.write(df.head())  # Display the updated dataset

        # Step 3: Transform 'Married' column to numerical values
        st.write("### Step 3: Transform 'Married' to Numerical")
        married_mapping = {'Yes': 1, 'No': 0}  # Example mapping
        df['Married'] = df['Married'].map(married_mapping)
        df['Married'] = df['Married'].fillna(df['Married'].mode()[0]) # Fill NaN with the mode
        st.write("After transformation of 'Married' column:")
        st.write(df.head()) # Display the updated dataset

        # Step 4: Transform 'Education' to numerical values
        st.write("### Step 4: Transform 'Education' to Numerical")
        education_mapping = {'Graduate': 1, 'Not Graduate': 0}  # Example mapping
        df['Education'] = df['Education'].map(education_mapping)
        df['Education'] = df['Education'].fillna(df['Education'].mode()[0])  # Fill NaN with the mode
        st.write("After transformation of 'Education' column:")
        st.write(df.head())  # Display the updated dataset

        # Step 5: Transform 'Property_Area' to numerical values
        st.write("### Step 5: Transform 'Property_Area' to Numerical")
        property_area_mapping = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}  # Example mapping
        df['Property_Area'] = df['Property_Area'].map(property_area_mapping)
        df['Property_Area'] = df['Property_Area'].fillna(df['Property_Area'].mode()[0])  # Fill NaN with the mode
        st.write("After transformation of 'Property_Area' column:")
        st.write(df.head())  # Display the updated dataset

        # Step 6: Show the cleaned and transformed dataset
        st.write("### Final Cleaned and Transformed Dataset")
        st.write(df.head())  # Display the final cleaned dataset

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
