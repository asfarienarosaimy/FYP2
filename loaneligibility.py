# prompt: generate code for loan eligibility prediction calculator using random forest, logistic regression and decision tree

pip install scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Sample Loan Dataset (Replace with your actual data)
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Married': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Dependents': [0, 1, 2, 0, 2, 1, 0],
    'Education': ['Graduate', 'Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
    'Self_Employed': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'No'],
    'ApplicantIncome': [5849, 4583, 3000, 2583, 6000, 5417, 2333],
    'CoapplicantIncome': [0.0, 1508.0, 0.0, 2358.0, 0.0, 4196.0, 1516.0],
    'LoanAmount': [128.0, 128.0, 66.0, 120.0, 141.0, 267.0, 95.0],
    'Loan_Amount_Term': [360.0, 360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
    'Credit_History': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    'Property_Area': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban', 'Urban', 'Rural'],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'N']
}
df = pd.DataFrame(data)

# Preprocessing
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)
df.replace({'Loan_Status': {'Y': 1, 'N': 0}}, inplace=True) # Convert loan status to numerical values


X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train and Evaluate Models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))  # Precision, Recall, F1-score
    print(confusion_matrix(y_test, y_pred))
    print("-"*30)


# Example Prediction (replace with your values)
new_applicant = pd.DataFrame({
    'Gender_Male': [1],
    'Married_Yes': [1],
    'Dependents': [0],
    'Education_Not Graduate': [0],
    'Self_Employed_Yes': [0],
    'ApplicantIncome': [6000],
    'CoapplicantIncome': [0],
    'LoanAmount': [140],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    # Instead of manually creating Property_Area dummies,
    # use the same 'Property_Area' column as in original data
    'Property_Area': ['Urban']  
})

# Apply get_dummies to new_applicant to ensure consistent columns
new_applicant = pd.get_dummies(new_applicant, 
                               columns=['Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes', 'Property_Area'], 
                               drop_first=True)

# Align columns with training data - add missing columns and fill with 0
for col in X.columns:
    if col not in new_applicant.columns:
        new_applicant[col] = 0

# Reorder columns to match the order in training data
new_applicant = new_applicant[X.columns]

# Scale new applicant data
new_applicant_scaled = scaler.transform(new_applicant)

#... (rest of your code)

for name, model in models.items():
    prediction = model.predict(new_applicant_scaled)[0]
    print(f"{name} Prediction: {'Loan Approved' if prediction == 1 else 'Loan Rejected'}")
