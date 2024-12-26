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

# Helper function for count plots
def plot_countplot(df, column, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=column, data=df, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Helper function for handling missing values
def handle_missing_values(df):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Helper function to train and evaluate models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    st.write(f"### {model_name} Evaluation")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"Precision: {precision:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    ax.set_title(f"Confusion Matrix ({model_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    return model

# Streamlit app
st.title("Loan Eligibility Prediction System")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        display_dataset(df, "Original Dataset Preview")

        # Data Preprocessing
        if 'Loan_ID' in df.columns:
            df.drop('Loan_ID', axis=1, inplace=True)

        # Handle missing values
        df = handle_missing_values(df)

        # Encode categorical columns
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Split data
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model = train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

        # Logistic Regression
        logreg_model = LogisticRegression(max_iter=1000, random_state=42)
        logreg_model = train_and_evaluate_model(logreg_model, X_train, X_test, y_train, y_test, "Logistic Regression")

        # Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model = train_and_evaluate_model(dt_model, X_train, X_test, y_train, y_test, "Decision Tree")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to proceed.")
