import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #4A4A4A;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 1.2rem;
        border: none;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title with emoji
st.markdown("<h1 class='main-title'>💻 PC Market Price Prediction</h1>", unsafe_allow_html=True)

# Sidebar for user inputs with custom background
st.sidebar.markdown("<h2 style='color:#4A4A4A'>Customize Your PC</h2>", unsafe_allow_html=True)

device_type = st.sidebar.selectbox('Device Type', ['Laptop', 'Desktop'])

laptop = st.sidebar.selectbox('Laptop Model', ['Surface', 'ThinkPad', 'Aspire', 'VivoBook', 'MacBook', 'Pavilion', 'Inspiron'])
status = st.sidebar.selectbox('Status', ['New', 'Used', 'Refurbished'])
brand = st.sidebar.selectbox('Brand', ['Asus', 'Apple', 'Lenovo', 'HP', 'Microsoft', 'Dell', 'Acer'])
model = st.sidebar.text_input('Model', 'Pavilion')
cpu = st.sidebar.selectbox('CPU', ['i7-10510U', 'i3-10110U', 'Ryzen 5 3500U', 'M1', 'i5-10210U', 'Ryzen 7 3700U'])
ram = st.sidebar.number_input('RAM (GB)', 4, 64, step=4)
storage = st.sidebar.number_input('Storage (GB)', 128, 2048, step=128)
storage_type = st.sidebar.selectbox('Storage Type', ['SSD', 'HDD'])
gpu = st.sidebar.selectbox('GPU', ['None', 'Radeon RX 560', 'NVIDIA GTX 1650', 'Intel HD'])
screen_size = st.sidebar.number_input('Screen Size (inches)', 13.0, 17.3, step=0.1)
touch = st.sidebar.selectbox('Touch Screen', ['Yes', 'No'])
retailer_type = st.sidebar.selectbox('Retailer Type', ['Online platforms (Amazon)', 'Retailers (Croma)', 'Direct web sales (Dell, HP)'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
region = st.sidebar.selectbox('Region', ['North', 'South', 'East', 'West', 'Central'])
state = st.sidebar.selectbox('State', ['Gujarat', 'Maharashtra', 'Tamil Nadu', 'Rajasthan', 'Karnataka', 'Uttar Pradesh', 'West Bengal'])
city = st.sidebar.text_input('City', 'Mumbai')

# Add year selector
year = st.sidebar.selectbox('Year of Prediction', list(range(2024, 2031)))

# Load the trained model
pipeline = joblib.load('model_pipeline.pkl')

# When the 'Predict' button is pressed
if st.button('Predict'):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Laptop': [laptop],
        'Status': [status],
        'Brand': [brand],
        'Model': [model],
        'CPU': [cpu],
        'RAM': [ram],
        'Storage': [storage],
        'Storage Type': [storage_type],
        'GPU': [gpu],
        'Screen Size': [screen_size],
        'Touch': [touch],
        'Retailer Type': [retailer_type],
        'Gender': [gender],
        'Region': [region],
        'State': [state],
        'City': [city],
        'Year': [year]  # Updated year value
    })

    # Make a prediction
    prediction = pipeline.predict(input_data)[0]

    # Convert prediction to INR (assuming model prediction is in INR)
    prediction_inr = prediction * 1

    # Display the prediction
    st.success(f'💰 Predicted Price for {year}: ₹{prediction_inr:,.2f}')

# Add a footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Built by Shiv</p>
    """, unsafe_allow_html=True)
