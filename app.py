import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🔥 Forest Fire FRP Predictor", layout="centered")
st.title("🌲🔥 Forest Fire Radiation (FRP) Predictor")

# Load Models
reg_model = joblib.load("frp_linear_regression.pkl")
clf_model = joblib.load("frp_classifier.pkl")

# --- SIDEBAR: User Input ---
st.sidebar.header("Input Parameters")

def user_input():
    brightness = st.sidebar.slider("Brightness", 280, 500, 330)
    bright_t31 = st.sidebar.slider("Bright_T31", 270, 350, 300)
    confidence = st.sidebar.slider("Confidence (%)", 0, 100, 80)
    scan = st.sidebar.number_input("Scan", 0.1, 2.0, step=0.1, value=1.0)
    track = st.sidebar.number_input("Track", 0.1, 2.0, step=0.1, value=1.0)
    month = st.sidebar.slider("Month", 1, 12, 7)
    year = st.sidebar.slider("Year", 2000, 2025, 2021)
    daynight = st.sidebar.selectbox("Day or Night", ("Day", "Night"))

    daynight_value = 1 if daynight == "Day" else 0

    data = {
        'brightness': brightness,
        'bright_t31': bright_t31,
        'confidence': confidence,
        'scan': scan,
        'track': track,
        'month': month,
        'year': year,
        'daynight': daynight_value
    }

    return pd.DataFrame([data])

input_df = user_input()

# --- Show Input ---
st.subheader("🔍 Input Features")
st.write(input_df)

# --- Predict ---
if st.button("🔥 Predict Fire Radiative Power"):
    frp_pred = reg_model.predict(input_df)[0]
    frp_class_pred = clf_model.predict(input_df)[0]
    class_probs = clf_model.predict_proba(input_df)[0]
    class_labels = clf_model.classes_

    st.markdown("---")
    st.subheader("📈 Regression Result (FRP Estimate)")
    st.success(f"Estimated FRP: **{frp_pred:.2f} MW**")

    st.subheader("🔒 Classification Result (FRP Intensity)")
    st.info(f"Predicted Intensity Class: **{frp_class_pred}**")

    # --- Plot: Class Probabilities ---
    st.subheader("📊 FRP Class Probabilities")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=class_labels, y=class_probs, ax=ax1, palette="coolwarm")
    ax1.set_ylabel("Probability")
    ax1.set_title("Predicted Class Probabilities")
    st.pyplot(fig1)

    # --- Plot: Feature Importances (regression coefficients) ---
    st.subheader("📌 Feature Importance (Regression Coefficients)")
    coeffs = reg_model.coef_
    features = input_df.columns.tolist()
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coeffs}).sort_values(by='Coefficient', key=abs, ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax2, palette='viridis')
    ax2.set_title("Linear Regression Coefficients")
    st.pyplot(fig2)

    # Optional: Correlation Heatmap of Input
    st.subheader("🧠 Input Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    sns.heatmap(input_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
