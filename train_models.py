import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

os.makedirs("models", exist_ok=True)

# ─── Train Diabetes Model ────────────────────────────────────────────
print("Training Diabetes model...")
df = pd.read_csv("datasets/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_diabetes = StandardScaler()
X_train_scaled = scaler_diabetes.fit_transform(X_train)
X_test_scaled = scaler_diabetes.transform(X_test)
model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
model_diabetes.fit(X_train_scaled, y_train)
acc = accuracy_score(y_test, model_diabetes.predict(X_test_scaled))
print(f"✅ Diabetes Model Accuracy: {acc*100:.2f}%")
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(model_diabetes, f)
with open("models/diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)

# ─── Train Heart Disease Model ───────────────────────────────────────
print("Training Heart Disease model...")
df2 = pd.read_csv("datasets/heart_disease.csv")
X2 = df2.drop("target", axis=1)
y2 = df2["target"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
scaler_heart = StandardScaler()
X_train2_scaled = scaler_heart.fit_transform(X_train2)
X_test2_scaled = scaler_heart.transform(X_test2)
model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_train2_scaled, y_train2)
acc2 = accuracy_score(y_test2, model_heart.predict(X_test2_scaled))
print(f"✅ Heart Disease Model Accuracy: {acc2*100:.2f}%")
with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(model_heart, f)
with open("models/heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)

print("\n🎉 All models trained and saved!")
