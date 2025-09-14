# 📌 Heart Disease Prediction - Train Models (Logistic Regression & Random Forest)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Upload heart.csv file
from google.colab import files
uploaded = files.upload()

# Step 2: Load dataset
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, log_reg.predict(X_test))

# Step 6: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))

print(f"Logistic Regression Accuracy: {acc_lr:.2f}")
print(f"Random Forest Accuracy: {acc_rf:.2f}")

# Step 7: Select & Save Best Model
best_model = rf if acc_rf >= acc_lr else log_reg
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Best model saved as model.pkl and scaler.pkl")

