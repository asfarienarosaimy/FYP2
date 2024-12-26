import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Title of the application
st.title("Loan Dataset Viewer and Preprocessing")

# Title of the Streamlit app
st.title("Loan Eligibility Prediction using Random Forest")

#Streamlit title
st.title("Loan Status Prediction using Logistic Regression")

#Streamlit title
st.title("Loan Status Prediction using Decision Tree")

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
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
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

        # Random Forest Classifier
        # Handling Missing Values
        st.write("### Handling Missing Values")
        imputer = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        st.write("Missing values handled.")

        # Convert categorical variables to numeric using Label Encoding
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        st.write("Categorical variables converted to numeric.")

        # Define features (X) and target variable (y)
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted N', 'Predicted Y'], yticklabels=['Actual N', 'Actual Y'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

        # Classification Report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

        # F1 Score
        f1 = f1_score(y_test, y_pred)
        st.write(f"### F1 Score: {f1:.2f}")

        # Feature Importances
        feature_importances = model.feature_importances_
        feature_names = X.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.write("### Feature Importances")
        st.write(importance_df)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='green')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        st.pyplot(plt)

        # Predict loan status for a new applicant
        st.write("### Predict Loan Status for a New Applicant")
        new_applicant = {
            'Gender': 1,
            'Married': 1,
            'Dependents': 2,
            'Education': 1,
            'Self_Employed': 0,
            'ApplicantIncome': 5000,
            'CoapplicantIncome': 2000,
            'LoanAmount': 100,
            'Loan_Amount_Term': 360,
            'Credit_History': 1,
            'Property_Area': 1
        }

        new_applicant_df = pd.DataFrame(new_applicant, index=[0])

        predicted_loan_status = model.predict(new_applicant_df)
        loan_status = 'Approved' if predicted_loan_status[0] == 1 else 'Rejected'
        st.write(f"Predicted Loan Status: {loan_status}")

        # Logistic Regression
        st.write("# Logistic Regression")
        imputer = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logreg_model = LogisticRegression(random_state=42, max_iter=1000)
        logreg_model.fit(X_train, y_train)

        y_pred_logreg = logreg_model.predict(X_test)

        accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
        st.write(f"### Logistic Regression Accuracy: {accuracy_logreg:.4f}")

        cm_logreg = confusion_matrix(y_test, y_pred_logreg)
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted N', 'Predicted Y'],
                    yticklabels=['Actual N', 'Actual Y'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Logistic Regression')
        st.pyplot(fig)

        recall = recall_score(y_test, y_pred_logreg)
        st.write(f"### Recall for Logistic Regression: {recall:.4f}")

        f1_logreg = f1_score(y_test, y_pred_logreg)
        st.write(f"### F1 Score for Logistic Regression: {f1_logreg:.4f}")

        precision = precision_score(y_test, y_pred_logreg)
        st.write(f"### Precision for Logistic Regression: {precision:.4f}")

        feature_importances = np.abs(logreg_model.coef_[0])
        feature_names = X.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.write("### Feature Importances:")
        st.write(importance_df)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
        ax2.set_xlabel('Importance')
        ax2.set_ylabel('Feature')
        ax2.set_title('Feature Importance (Logistic Regression)')
        ax2.invert_yaxis()
        st.pyplot(fig2)

        # Decision Tree Classifier
        st.write("# Decision Tree Classifier")
        imputer = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)

        y_pred_dt = dt_model.predict(X_test)

        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        recall_dt = recall_score(y_test, y_pred_dt)
        f1_dt = f1_score(y_test, y_pred_dt)
        precision_dt = precision_score(y_test, y_pred_dt)

        st.write("### Decision Tree Model Evaluation")
        st.write(f"Accuracy: {accuracy_dt:.4f}")
        st.write(f"Recall: {recall_dt:.4f}")
        st.write(f"F1 Score: {f1_dt:.4f}")
        st.write(f"Precision: {precision_dt:.4f}")

        cm_dt = confusion_matrix(y_test, y_pred_dt)
        st.write("### Confusion Matrix for Decision Tree")
        fig_cm_dt, ax_cm_dt = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted No', 'Predicted Yes'],
                    yticklabels=['Actual No', 'Actual Yes'])
        ax_cm_dt.set_xlabel('Predicted')
        ax_cm_dt.set_ylabel('Actual')
        ax_cm_dt.set_title('Confusion Matrix for Decision Tree')
        st.pyplot(fig_cm_dt)

        feature_importances = dt_model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.write("### Feature Importance (Decision Tree)")
        st.write(importance_df)

        fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
        ax_importance.barh(importance_df['Feature'], importance_df['Importance'], color='orange')


        

    'Importance': feature
