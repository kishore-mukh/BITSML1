import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

QUALITY_MIN = 3
st.set_page_config(
    page_title="Wine Quality Prediction",
    layout="centered"
)

st.title("Wine Quality Prediction App")
st.write("Upload wine chemical properties to predict wine quality.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "Logistic_Regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "Decision_Tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR, "KNN.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "Naive_Bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR, "XGBoost.pkl"))
}

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

model_name = st.selectbox("Select a model", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())
    X = df.copy()
    X_scaled = scaler.transform(X)
    y_pred_shifted = model.predict(X_scaled)
    y_pred = y_pred_shifted + QUALITY_MIN
    y_prob = model.predict_proba(X_scaled)
    confidence = np.max(y_prob, axis=1)
    results = df.copy()
    results["Predicted Quality"] = y_pred
    results["Confidence"] = confidence

    st.subheader("Prediction Results")
    st.dataframe(results)

    st.success("Prediction completed successfully!")
