# ML_Heart_Health_Predictor
❤️ Heart Disease Prediction using Machine Learning and Streamlit
<p align="center">
  <img src="assets/Screenshot 2025-09-11 015935.png" alt="App Screenshot" width="500"/>
</p>
# About Project
This project develops a machine learning pipeline to predict the likelihood of heart disease based on clinical features. It extends classroom learning into a real-world healthcare problem by building, evaluating, and deploying models using Python and Streamlit. The final solution enables both clinicians and the general public to perform quick preliminary screenings using a simple web interface.

# Background Overview

Heart disease is one of the leading causes of mortality worldwide, responsible for millions of deaths annually. According to the World Health Organization (WHO), cardiovascular diseases account for over 17 million deaths per year.
Early detection is essential for preventive healthcare, but many individuals do not undergo regular check-ups due to cost, awareness, or accessibility barriers.

By applying machine learning techniques, we can create data-driven systems that learn from past patient records to predict risks of heart disease. Such systems assist healthcare professionals in decision-making and raise awareness among the general population about potential risks.

# Problem Statement

Heart disease often remains undetected until advanced stages, leading to severe health consequences. Current challenges include:

Lack of accessible, affordable, and fast risk assessment tools.

Dependence on medical specialists for interpretation.

Limited awareness among non-medical individuals.

Thus, the need arises for a machine learning-based predictive tool that can take basic health attributes (like age, cholesterol, blood pressure) and output a risk prediction.
# Objective

The main objectives of this project are:

To preprocess and prepare the UCI/Kaggle heart disease dataset for machine learning.

To develop and compare at least two machine learning models for prediction.

To evaluate models using relevant metrics (accuracy, precision, recall, F1-score).

To deploy the best-performing model using Streamlit for user-friendly interaction.

To demonstrate how machine learning can contribute to early detection and healthcare awareness.

# Built With

🐍 Python

📓 Jupyter Notebook

📊 Pandas, NumPy, Matplotlib, Seaborn

🤖 Scikit-learn (Logistic Regression, Random Forest)

🌐 Streamlit

🖥️ VS Code / Google Colab

# Data Source

Dataset: UCI / Kaggle Heart Disease Dataset

Records: ~303 patient entries

Features include:

# age 
— Age of patient

# sex
— Gender (1 = Male, 0 = Female)

# cp 
— Chest pain type

# chol
— Serum cholesterol level

# trestbps
— Resting blood pressure

# thalach
— Maximum heart rate achieved

…and others (total ~13 features)

# Target:
0 = No Heart Disease, 1 = Heart Disease

# Methodology
# Data Collection & Preprocessing

1. Dataset loaded from Kaggle.

2. Cleaned missing values and checked for duplicates.

3. Categorical variables encoded (sex, chest pain type, etc.).

4. Features normalized using StandardScaler.

5. Train-test split (80:20).

# Model Development

Two models were implemented:

Logistic Regression (baseline, interpretable model)

Random Forest Classifier (handles non-linearities)

# Model Evaluation

Metrics used: Accuracy, Precision, Recall, F1-score.
Confusion matrices were generated to analyze misclassifications.

# Results and Impact

Best Model: Random Forest achieved the highest balanced accuracy.

Logistic Regression: Good baseline, interpretable, but slightly lower recall.

Impact: Demonstrates the feasibility of quick, affordable screening tools in healthcare using machine learning.

# Example table (replace with your exact numbers):

# Model	               Accuracy	Precision	Recall	F1-score
  Logistic Regression	 0.84	    0.82	    0.80	  0.81
  Random Forest        0.89	    0.87	    0.88	  0.87
# Application Deployment

A Streamlit web app was developed where users can input values such as age, cholesterol, and blood pressure. The app returns a prediction of whether the person is at High Risk or Low Risk for heart disease.

# 👉 Demo link here
(https://zaidmlheartproject1.streamlit.app/#enter-patient-details)

# Challenges and Solutions

# 1. Small dataset size (~300 records):
  → Used cross-validation to avoid overfitting.

# 2. Feature scaling & categorical encoding:
  → StandardScaler + OneHotEncoding applied consistently.

# 3. Model selection:
  → Compared baseline (Logistic Regression) with advanced (Random Forest).

# 4. Streamlit deployment issues (libraries, kernel mismatches):
  → Resolved by creating a clean requirements.txt and isolating environment.

# How to Use
# Deployed Version 
Click here 👉 https://zaidmlheartproject1.streamlit.app/#enter-patient-details
# Screenshots
<p align="center"> <img src="assets/Screenshot 2025-09-11 015945.png" width="700" alt="Streamlit app input form"/> </p> <p align="center"> <img src="assets/result.png" width="700" alt="Prediction output"/> </p>
<p align="center"> <img src="assets/Screenshot 2025-09-11 015513.png" width="700" alt="Streamlit app input form"/> </p> <p align="center"> <img src="assets/result.png" width="700" alt="Prediction output"/> </p>

# Acknowledgement

BIT4333 Introduction to Machine Learning — Final Project, City University Malaysia.

Supervisor: Sir Nazmirul Izzad Bin Nassir

Dataset source: UCI / Kaggle Heart Disease Dataset.

Tools: Python, Scikit-learn, Streamlit, Colab, VS Code.




