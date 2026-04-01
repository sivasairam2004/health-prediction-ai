import pandas as pd
import numpy as np
import os

os.makedirs("datasets", exist_ok=True)

# ─── Diabetes Dataset ───────────────────────────────────────────────
print("Creating diabetes dataset...")
np.random.seed(42)
n = 768
diabetes_data = {
    'Pregnancies': np.random.randint(0, 17, n),
    'Glucose': np.random.randint(70, 200, n),
    'BloodPressure': np.random.randint(40, 120, n),
    'SkinThickness': np.random.randint(0, 60, n),
    'Insulin': np.random.randint(0, 400, n),
    'BMI': np.round(np.random.uniform(18, 50, n), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.08, 2.5, n), 3),
    'Age': np.random.randint(21, 81, n),
}
df_diabetes = pd.DataFrame(diabetes_data)
df_diabetes['Outcome'] = ((df_diabetes['Glucose'] > 140) | 
                          (df_diabetes['BMI'] > 35) | 
                          (df_diabetes['Age'] > 50)).astype(int)
df_diabetes.to_csv("datasets/diabetes.csv", index=False)
print(f"✅ diabetes.csv created! Shape: {df_diabetes.shape}")

# ─── Heart Disease Dataset ───────────────────────────────────────────
print("Creating heart disease dataset...")
n = 303
heart_data = {
    'age': np.random.randint(29, 77, n),
    'sex': np.random.randint(0, 2, n),
    'cp': np.random.randint(0, 4, n),
    'trestbps': np.random.randint(90, 200, n),
    'chol': np.random.randint(150, 400, n),
    'fbs': np.random.randint(0, 2, n),
    'restecg': np.random.randint(0, 3, n),
    'thalach': np.random.randint(70, 210, n),
    'exang': np.random.randint(0, 2, n),
    'oldpeak': np.round(np.random.uniform(0, 6, n), 1),
    'slope': np.random.randint(0, 3, n),
    'ca': np.random.randint(0, 4, n),
    'thal': np.random.randint(1, 4, n),
}
df_heart = pd.DataFrame(heart_data)
df_heart['target'] = ((df_heart['age'] > 55) | 
                      (df_heart['chol'] > 250) | 
                      (df_heart['trestbps'] > 140)).astype(int)
df_heart.to_csv("datasets/heart_disease.csv", index=False)
print(f"✅ heart_disease.csv created! Shape: {df_heart.shape}")

print("\n🎉 All datasets ready!")
