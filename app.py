import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Restaurant Revenue Prediction", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Restaurant Revenue Prediction")

val1 = st.number_input("Number of Items", min_value=0, value=50)
val2 = st.number_input("Orders Placed", min_value=0.0, value=100.0)
val3 = st.number_input("Restaurant ID", min_value=0, value=110)

if st.button("Predict"):
    features = np.array([[val1, val2, val3]])
    prediction = model.predict(features)
    st.success(f"Predicted Revenue: {prediction[0]:,.2f}")
