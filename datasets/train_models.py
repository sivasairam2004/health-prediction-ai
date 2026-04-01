import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

os.makedirs("../models", exist_ok=True)

# =====================
# TRAIN DIABETES MODEL
# =====================
print("Training Diabetes model...")
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler_diabetes = StandardScaler()
X_scaled = scaler_diabetes.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
model_diabetes.fit(X_train, y_train)
acc = accuracy_score(y_test, model_diabetes.predict(X_test))
print(f"✅ Diabetes Model Accuracy: {acc*100:.2f}%")

with open("../models/diabetes_model.pkl", "wb") as f:
    pickle.dump(model_diabetes, f)
with open("../models/diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)

# =====================
# TRAIN HEART MODEL
# =====================
print("Training Heart Disease model...")
df = pd.read_csv("heart_disease.csv")
X = df.drop("target", axis=1)
y = df["target"]

scaler_heart = StandardScaler()
X_scaled = scaler_heart.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_train, y_train)
acc = accuracy_score(y_test, model_heart.predict(X_test))
print(f"✅ Heart Disease Model Accuracy: {acc*100:.2f}%")

with open("../models/heart_model.pkl", "wb") as f:
    pickle.dump(model_heart, f)
with open("../models/heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)

# =====================
# TRAIN PARKINSONS MODEL
# =====================
print("Training Parkinson's model...")
df = pd.read_csv("parkinsons.csv")
X = df.drop("status", axis=1)
y = df["status"]

scaler_parkinsons = StandardScaler()
X_scaled = scaler_parkinsons.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_parkinsons = RandomForestClassifier(n_estimators=100, random_state=42)
model_parkinsons.fit(X_train, y_train)
acc = accuracy_score(y_test, model_parkinsons.predict(X_test))
print(f"✅ Parkinson's Model Accuracy: {acc*100:.2f}%")

with open("../models/parkinsons_model.pkl", "wb") as f:
    pickle.dump(model_parkinsons, f)
with open("../models/parkinsons_scaler.pkl", "wb") as f:
    pickle.dump(scaler_parkinsons, f)

print("\n🎉 All 3 models trained and saved!")
