import streamlit as st
import numpy as np
import joblib

st.title("❤️ Heart Disease Prediction App")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Input fields
st.header("Enter Patient Details")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
restecg = st.number_input("Resting ECG (0-2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)
thal = st.number_input("Thal (0-3)", 0, 3, 2)

# Predict button
if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ Patient is likely to have Heart Disease")
    else:
        st.success("✅ Patient is unlikely to have Heart Disease")
