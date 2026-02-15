import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(
    page_title="ML Assignment 2 – Wine Quality",
    layout="wide"
)

st.title("Wine Quality Classification – ML Assignment 2")

metrics_df = pd.read_csv("model/model_metrics.csv")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a classification model",
    metrics_df["Model"].tolist()
)

model_path = f"model/{model_name.replace(' ', '_')}.pkl"
model = joblib.load(model_path)
st.header("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload winequality-white.csv or test split CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    TARGET = "quality"
    X = df.drop(columns=[TARGET])
    y_true_raw = df[TARGET]
    y_true = label_encoder.transform(y_true_raw)
    if model_name in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    st.subheader("Model Performance Metrics")
    st.dataframe(metrics_df[metrics_df["Model"] == model_name])
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.subheader("Classification Report")
    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
    st.text(report)
else:
    st.info("Please upload a CSV file to begin.")
    