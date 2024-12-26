import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Helper function to display dataset
def display_dataset(df, title="Dataset Preview"):
    st.write(f"### {title}")
    st.write(df.head())

# Helper function for distribution plots
def plot_distribution(df, column, title, bins=50):
    fig, ax = plt.subplots()
    df[column].hist(bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Helper function to encode categorical variables
def encode_categorical_columns(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# Helper function for missing value handling
def handle_missing_values(df):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Helper function for evaluation metrics
def evaluate_model(y_test, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    st.write(f"### {model_name} Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"Precision: {precision:.4f}")
    return accuracy, recall, f1, precision

# File uploader widget
uploaded_file = st.file_uploader("Upload your loan_data_set.csv file", type=["csv"])

if uploaded_file:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        display_dataset(df)

        # Filter and visualize data
        st.write("## Applicant Income Analysis")
        plot_distribution(df[df['ApplicantIncome'] <= 40000], 'ApplicantIncome', "Applicant Income Distribution (Up to 40000)")

        st.write("## Loan Amount Analysis")
        plot_distribution(df, 'LoanAmount', "Loan Amount Distribution")

        # Handle missing values
        df = handle_missing_values(df)
        display_dataset(df, title="Dataset After Handling Missing Values")

        # Encode categorical features
        df, encoders = encode_categorical_columns(df)
        display_dataset(df, title="Encoded Dataset")

        # Define features and target
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

        # Logistic Regression Model
        logreg_model = LogisticRegression(max_iter=1000, random_state=42)
        logreg_model.fit(X_train, y_train)
        y_pred_logreg = logreg_model.predict(X_test)
        evaluate_model(y_test, y_pred_logreg, model_name="Logistic Regression")

        # Decision Tree Model
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        evaluate_model(y_test, y_pred_dt, model_name="Decision Tree")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
