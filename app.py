import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Restaurant Revenue Prediction", layout="centered")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Restaurant Revenue Prediction")
st.write("Enter the restaurant details below to estimate revenue.")

st.subheader("Restaurant Information")

# Input fields with clear guidance
number_of_items = st.number_input(
    "Number of Items",
    min_value=0,
    value=50,
    step=1,
    help="Enter the number of food items associated with the restaurant/order. Example: 50."
)

orders_placed = st.number_input(
    "Orders Placed",
    min_value=0.0,
    value=100.0,
    step=1.0,
    help="Enter the number of orders placed. Example: 100."
)

restaurant_id = st.number_input(
    "Restaurant ID",
    min_value=0,
    value=150,
    step=1,
    help="Enter the unique ID of the restaurant. Example: 150."
)

st.caption("Example input: Number of Items = 50, Orders Placed = 100, Restaurant ID = 150")

if st.button("Predict Revenue", type="primary"):
    features = np.array([[number_of_items, orders_placed, restaurant_id]])

    prediction = model.predict(features)

    st.success(f"Predicted Revenue: {prediction[0]:,.2f}")
