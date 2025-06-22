# 🧬 MedGPT – Diabetes Risk Prediction App

**MedGPT** is a machine learning-powered application that predicts the risk of diabetes using patient health indicators. Built for clinical data analysis and healthcare innovation, it combines a trained Random Forest model, SHAP explainability, and a user-friendly Streamlit interface. Docker support makes it easy to deploy anywhere.

---

## 🚀 Features

- ✅ Predict diabetes risk from 8 clinical features
- ✅ Real-time web interface via Streamlit
- ✅ SHAP-based explainability for transparency
- ✅ Docker containerization for easy deployment
- ✅ Clean, modular folder structure

---

## 🧠 Tech Stack

- Python 3.10
- scikit-learn
- pandas, numpy
- joblib
- shap
- matplotlib
- Streamlit
- Docker

---

## 📁 Folder Structure

MedGPT/
├── app/
│ ├── main.py # Streamlit frontend
│ ├── train_model.py # ML model training script
│ ├── diabetes_model.pkl # Trained ML model
│ └── diabetes_scaler.pkl # StandardScaler used during training
├── data/
│ └── diabetes.csv # Dataset file (Pima Indians Diabetes dataset)
├── Dockerfile # Docker configuration file
├── requirements.txt # List of Python dependencies
└── README.md # This documentation


---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**, a widely-used dataset in medical ML research, containing records of female patients over age 21.

**Features used for prediction:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target:** `1` (Diabetic) or `0` (Non-diabetic)

---

## ⚙️ Setup Instructions (Local)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kumail-here/MedGPT.git
cd MedGPT
