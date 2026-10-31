import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

# App title and description
st.title("🚚 Delivery Delay Predictor")
st.write("""
This interactive app uses a Logistic Regression model to predict if a delivery will be delayed 
based on trip details, road conditions, and warehouse handling times.
""")

# Load the saved model
try:
    # Strip any potential accidental spaces in filename
    model = joblib.load("logistic.sav ")
except FileNotFoundError:
    try:
        model = joblib.load("logistic.sav")
    except FileNotFoundError:
        st.error("⚠️ Model file 'logistic.sav' not found. Please ensure the model is saved in the working directory.")
        st.stop()

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
    traffic_congestion = st.slider("Traffic Congestion (1-5)", min_value=1, max_value=5, value=3)
    weather_condition = st.slider("Weather Condition Score (1-5)", min_value=1, max_value=5, value=2)
    delivery_slot = st.slider("Delivery Slot Hour/Code", min_value=1, max_value=24, value=2)
    driver_experience = st.number_input("Driver Experience (years)", min_value=0, max_value=50, value=5)
    num_stops = st.slider("Number of Stops", min_value=0, max_value=20, value=3)

with col2:
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=3)
    road_condition = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
    package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, max_value=500.0, value=15.0, step=0.1)
    fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=1.0, max_value=100.0, value=12.0, step=0.1)
    warehouse_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, max_value=500, value=45)

# Predict button
if st.button("🔮 Predict Delivery Status", use_container_width=True):
    # Prepare the input data frame with original column names to match model training
    input_data = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_time
    }])
    
    # Generate Prediction and Probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("--- ")
    
    # Display results
    if prediction == 1:
        st.error(f"🚨 **Prediction: DELAY EXPECTED** (Probability: {probability:.1%})")
    else:
        st.success(f"✅ **Prediction: ON TIME** (Probability of delay: {probability:.1%})")
