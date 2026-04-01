# 🏥 Health Prediction AI

AI-powered Multiple Disease Prediction Web App using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Flask](https://img.shields.io/badge/Flask-3.1-green) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange) ![HTML](https://img.shields.io/badge/HTML%2FCSS%2FJS-Frontend-yellow)

## 🚀 Features

- 🩸 **Diabetes Prediction** — Predicts diabetes risk from medical records
- ❤️ **Heart Disease Prediction** — Assesses cardiovascular disease risk
- 🧠 **Parkinson's Prediction** — Detects Parkinson's from voice measurements
- 📈 **Interactive Charts** — Compares your values vs normal range
- 🖨️ **Download PDF Report** — Save your prediction as a professional PDF
- 📊 **Prediction History** — Tracks your last 5 predictions
- 🌙 **Dark Mode UI** — Beautiful modern dark interface

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Chart.js, jsPDF
- **Backend:** Python, Flask, Flask-CORS
- **ML:** Scikit-learn (Random Forest Classifier)
- **Data:** Pandas, NumPy
- **Models:** Pre-trained Random Forest models (~75-80% accuracy)

## 📁 Project Structure

```
health-prediction-ai/
├── datasets/
│   ├── create_datasets.py   # Generate datasets
│   ├── train_models.py      # Train ML models
│   ├── diabetes.csv
│   ├── heart_disease.csv
│   └── parkinsons.csv
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── parkinsons_model.pkl
├── backend/
│   └── app.py               # Flask REST API
├── frontend/
│   └── index.html           # Main UI
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

### Step 1 — Install Python packages

```bash
pip install -r requirements.txt
```

### Step 2 — Create datasets

```bash
cd datasets
python create_datasets.py
```

### Step 3 — Train ML models

```bash
python train_models.py
cd ..
```

### Step 4 — Start Flask backend

```bash
cd backend
python app.py
```

### Step 5 — Open frontend

Open `frontend/index.html` in your browser ✅

## 🔗 API Endpoints

| Method | Endpoint              | Description                |
| ------ | --------------------- | -------------------------- |
| POST   | `/predict/diabetes`   | Predict diabetes risk      |
| POST   | `/predict/heart`      | Predict heart disease risk |
| POST   | `/predict/parkinsons` | Predict Parkinson's risk   |

## 📊 ML Models

| Disease          | Algorithm     | Features                   | Accuracy |
| ---------------- | ------------- | -------------------------- | -------- |
| 🩸 Diabetes      | Random Forest | 8 medical features         | ~75%     |
| ❤️ Heart Disease | Random Forest | 13 cardiovascular features | ~78%     |
| 🧠 Parkinson's   | Random Forest | 13 voice measurements      | ~80%     |

## ⚠️ Disclaimer

This app is for **educational purposes only**. Always consult a qualified medical professional for proper diagnosis and treatment.

## 👨‍💻 Author

**Siva Sairam** — [GitHub](https://github.com/sivasairam2004)
