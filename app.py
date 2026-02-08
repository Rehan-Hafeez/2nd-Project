import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# App Title
# -------------------------------
st.title("Loan Approval Prediction App")

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("loan_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Data Preprocessing
# -------------------------------
data = pd.read_csv("loan_data.csv")
data.columns = data.columns.str.strip()

st.write("Columns:", data.columns)

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"].map({"Y": 1, "N": 0})



categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -------------------------------
# ML Pipeline
# -------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Plot
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("Loan Approval Prediction")

def user_input():
    gender = st.selectbox("Gender", data["Gender"].unique())
    married = st.selectbox("Married", data["Married"].unique())
    education = st.selectbox("Education", data["Education"].unique())
    income = st.number_input("Applicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    credit_history = st.selectbox("Credit History", data["Credit_History"].unique())

    user_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Education": [education],
        "ApplicantIncome": [income],
        "LoanAmount": [loan_amount],
        "Credit_History": [credit_history]
    })

    return user_data

input_df = user_input()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
