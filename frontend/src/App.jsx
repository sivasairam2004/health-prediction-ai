import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:5000";

function DiabetesForm({ onResult }) {
  const [form, setForm] = useState({
    pregnancies: "", glucose: "", bloodpressure: "",
    skinthickness: "", insulin: "", bmi: "", dpf: "", age: ""
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict/diabetes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      const data = await res.json();
      onResult(data);
    } catch {
      onResult({ error: "Server not connected. Please run the Flask backend." });
    }
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="form">
      {[
        { key: "pregnancies", label: "Pregnancies", placeholder: "e.g. 2" },
        { key: "glucose", label: "Glucose Level", placeholder: "e.g. 120" },
        { key: "bloodpressure", label: "Blood Pressure", placeholder: "e.g. 80" },
        { key: "skinthickness", label: "Skin Thickness", placeholder: "e.g. 20" },
        { key: "insulin", label: "Insulin", placeholder: "e.g. 80" },
        { key: "bmi", label: "BMI", placeholder: "e.g. 25.5" },
        { key: "dpf", label: "Diabetes Pedigree Function", placeholder: "e.g. 0.5" },
        { key: "age", label: "Age", placeholder: "e.g. 35" },
      ].map(({ key, label, placeholder }) => (
        <div className="form-group" key={key}>
          <label>{label}</label>
          <input
            type="number" step="any" required placeholder={placeholder}
            value={form[key]}
            onChange={(e) => setForm({ ...form, [key]: e.target.value })}
          />
        </div>
      ))}
      <button type="submit" className="btn" disabled={loading}>
        {loading ? "Predicting..." : "🔍 Predict Diabetes"}
      </button>
    </form>
  );
}

function HeartForm({ onResult }) {
  const [form, setForm] = useState({
    age: "", sex: "", cp: "", trestbps: "", chol: "",
    fbs: "", restecg: "", thalach: "", exang: "",
    oldpeak: "", slope: "", ca: "", thal: ""
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict/heart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      const data = await res.json();
      onResult(data);
    } catch {
      onResult({ error: "Server not connected. Please run the Flask backend." });
    }
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="form">
      {[
        { key: "age", label: "Age", placeholder: "e.g. 52" },
        { key: "sex", label: "Sex (1=Male, 0=Female)", placeholder: "0 or 1" },
        { key: "cp", label: "Chest Pain Type (0-3)", placeholder: "0-3" },
        { key: "trestbps", label: "Resting Blood Pressure", placeholder: "e.g. 130" },
        { key: "chol", label: "Cholesterol", placeholder: "e.g. 250" },
        { key: "fbs", label: "Fasting Blood Sugar > 120 (1=Yes)", placeholder: "0 or 1" },
        { key: "restecg", label: "Resting ECG (0-2)", placeholder: "0-2" },
        { key: "thalach", label: "Max Heart Rate", placeholder: "e.g. 150" },
        { key: "exang", label: "Exercise Induced Angina (1=Yes)", placeholder: "0 or 1" },
        { key: "oldpeak", label: "ST Depression", placeholder: "e.g. 1.5" },
        { key: "slope", label: "Slope (0-2)", placeholder: "0-2" },
        { key: "ca", label: "Major Vessels (0-3)", placeholder: "0-3" },
        { key: "thal", label: "Thal (1=Normal, 2=Fixed, 3=Reversable)", placeholder: "1-3" },
      ].map(({ key, label, placeholder }) => (
        <div className="form-group" key={key}>
          <label>{label}</label>
          <input
            type="number" step="any" required placeholder={placeholder}
            value={form[key]}
            onChange={(e) => setForm({ ...form, [key]: e.target.value })}
          />
        </div>
      ))}
      <button type="submit" className="btn" disabled={loading}>
        {loading ? "Predicting..." : "❤️ Predict Heart Disease"}
      </button>
    </form>
  );
}

function ResultCard({ result }) {
  if (!result) return null;
  if (result.error) return <div className="result error">⚠️ {result.error}</div>;
  const isPositive = result.prediction === 1;
  return (
    <div className={`result ${isPositive ? "positive" : "negative"}`}>
      <h3>{isPositive ? "⚠️" : "✅"} {result.result}</h3>
      <p>Probability: <strong>{result.probability}%</strong></p>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${result.probability}%`,
          background: isPositive ? "#ef4444" : "#22c55e" }} />
      </div>
      <p className="advice">
        {isPositive
          ? "Please consult a doctor immediately for proper diagnosis."
          : "You appear healthy! Keep maintaining a healthy lifestyle."}
      </p>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("diabetes");
  const [result, setResult] = useState(null);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResult(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-icon">🏥</div>
        <h1>Health Prediction AI</h1>
        <p>AI-powered disease prediction using Machine Learning</p>
      </header>

      <div className="tabs">
        <button
          className={`tab ${activeTab === "diabetes" ? "active" : ""}`}
          onClick={() => handleTabChange("diabetes")}>
          🩸 Diabetes
        </button>
        <button
          className={`tab ${activeTab === "heart" ? "active" : ""}`}
          onClick={() => handleTabChange("heart")}>
          ❤️ Heart Disease
        </button>
      </div>

      <div className="container">
        <div className="card">
          <h2>{activeTab === "diabetes" ? "🩸 Diabetes Prediction" : "❤️ Heart Disease Prediction"}</h2>
          <p className="subtitle">Enter your medical details below</p>
          {activeTab === "diabetes"
            ? <DiabetesForm onResult={setResult} />
            : <HeartForm onResult={setResult} />}
        </div>
        <div className="result-section">
          <h2>📊 Prediction Result</h2>
          {result ? <ResultCard result={result} /> : (
            <div className="empty-result">
              <p>Fill in the form and click predict to see results</p>
            </div>
          )}
        </div>
      </div>

      <footer className="footer">
        <p>⚠️ This tool is for educational purposes only. Always consult a medical professional.</p>
      </footer>
    </div>
  );
}
