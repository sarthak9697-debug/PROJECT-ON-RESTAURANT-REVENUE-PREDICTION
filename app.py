import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="ML Model Inference", layout="centered")

# 1. Load artifacts properly
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    # Uncomment if you actually exported a scaler:
    # scaler = joblib.load("scaler.pkl")
    # return model, scaler
    return model

model = load_artifacts()

st.title("Machine Learning Prediction App")

# 2. Add input fields (adjust according to your dataset features)
val1 = st.number_input("Number of Items", value=0.0)
val2 = st.number_input("Orders Placed", value=0.0)
val3 = st.number_input("Restaurant ID", value=0.0)

# 3. Predict button
if st.button("Predict"):
    features = np.array([[val1, val2, val3]])
    
    # If using scaler:
    # features = scaler.transform(features)
    
    prediction = model.predict(features)
    st.success(f"Output / Cluster: {prediction[0]}")