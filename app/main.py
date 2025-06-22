import streamlit as st
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="MedGPT – Diabetes Prediction", layout="centered")
st.title("🧬 MedGPT: Diabetes Risk Predictor")

# Load model and scaler
model_path = "diabetes_model.pkl"
scaler_path = "diabetes_scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Model or Scaler file not found! Please train the model first.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.markdown("Fill in the form below to check your risk of diabetes.")

# Input form
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose Level", 0, 200)
        blood_pressure = st.number_input("Blood Pressure", 0, 150)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 70.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        age = st.number_input("Age", 1, 100)

    submitted = st.form_submit_button("Predict")

# Prediction + SHAP
if submitted:
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0][1]

    # Prediction Result
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
    st.info(f"🧪 Prediction Probability: **{probability:.2f}**")

    # SHAP Explanation
    st.subheader("🔍 Feature Contribution (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_data_scaled)

        # For binary classification, class index = 1 (diabetic)
        class_idx = 1 if len(shap_values) > 1 else 0
        shap_vals = shap_values[class_idx][0]

        # Feature names
        feature_names = model.feature_names_in_
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = [shap_vals[i] for i in sorted_idx]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(sorted_features[::-1], sorted_values[::-1])
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top Feature Contributions")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be generated: {e}")