import pandas as pd
import numpy as np
import os

os.makedirs(".", exist_ok=True)

# =====================
# DIABETES DATASET
# =====================
print("Creating diabetes dataset...")
np.random.seed(42)
n = 768

diabetes_data = {
    'Pregnancies': np.random.randint(0, 17, n),
    'Glucose': np.random.randint(70, 200, n),
    'BloodPressure': np.random.randint(40, 130, n),
    'SkinThickness': np.random.randint(0, 60, n),
    'Insulin': np.random.randint(0, 850, n),
    'BMI': np.round(np.random.uniform(15, 55, n), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.05, 2.5, n), 3),
    'Age': np.random.randint(21, 81, n),
    'Outcome': np.random.randint(0, 2, n)
}

df_diabetes = pd.DataFrame(diabetes_data)
df_diabetes.to_csv("diabetes.csv", index=False)
print(f"✅ diabetes.csv created! Shape: {df_diabetes.shape}")

# =====================
# HEART DISEASE DATASET
# =====================
print("Creating heart disease dataset...")
n = 303

heart_data = {
    'age': np.random.randint(29, 77, n),
    'sex': np.random.randint(0, 2, n),
    'cp': np.random.randint(0, 4, n),
    'trestbps': np.random.randint(90, 200, n),
    'chol': np.random.randint(120, 570, n),
    'fbs': np.random.randint(0, 2, n),
    'restecg': np.random.randint(0, 3, n),
    'thalach': np.random.randint(70, 210, n),
    'exang': np.random.randint(0, 2, n),
    'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
    'slope': np.random.randint(0, 3, n),
    'ca': np.random.randint(0, 4, n),
    'thal': np.random.randint(0, 4, n),
    'target': np.random.randint(0, 2, n)
}

df_heart = pd.DataFrame(heart_data)
df_heart.to_csv("heart_disease.csv", index=False)
print(f"✅ heart_disease.csv created! Shape: {df_heart.shape}")

# =====================
# PARKINSON'S DATASET
# =====================
print("Creating Parkinson's dataset...")
n = 195

parkinsons_data = {
    'MDVP:Fo(Hz)': np.round(np.random.uniform(80, 270, n), 3),
    'MDVP:Fhi(Hz)': np.round(np.random.uniform(100, 600, n), 3),
    'MDVP:Flo(Hz)': np.round(np.random.uniform(60, 240, n), 3),
    'MDVP:Jitter(%)': np.round(np.random.uniform(0.001, 0.03, n), 5),
    'MDVP:Shimmer': np.round(np.random.uniform(0.01, 0.12, n), 5),
    'NHR': np.round(np.random.uniform(0.001, 0.31, n), 5),
    'HNR': np.round(np.random.uniform(8, 34, n), 3),
    'RPDE': np.round(np.random.uniform(0.25, 0.69, n), 6),
    'DFA': np.round(np.random.uniform(0.57, 0.83, n), 6),
    'spread1': np.round(np.random.uniform(-7.96, -2.43, n), 6),
    'spread2': np.round(np.random.uniform(0.006, 0.45, n), 6),
    'D2': np.round(np.random.uniform(1.4, 3.7, n), 6),
    'PPE': np.round(np.random.uniform(0.04, 0.53, n), 6),
    'status': np.random.randint(0, 2, n)
}

df_parkinsons = pd.DataFrame(parkinsons_data)
df_parkinsons.to_csv("parkinsons.csv", index=False)
print(f"✅ parkinsons.csv created! Shape: {df_parkinsons.shape}")

print("\n🎉 All 3 datasets ready!")
