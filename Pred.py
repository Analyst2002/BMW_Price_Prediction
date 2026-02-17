import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="BMW Car Price Prediction", layout="wide")

@st.cache_resource
def load_bundle():
    bundle = joblib.load("bmw_gb_price_model.joblib")
    return bundle["model"], bundle["preprocessor"]

model, preprocessor = load_bundle()

current_year = datetime.now().year

st.title("BMW Car Price Prediction")
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year", 1995, current_year, 2019)
    mileage = st.number_input("Mileage (km)", 0, 300000, 25000)
    mpg = st.number_input("Mileage Per Gallon (MPG)", 5.0, 500.0, 45.0)
    engine_size = st.number_input("Engine Size (L)", 0.6, 6.0, 2.0)
    tax = st.number_input("Road Tax (£)", 0, 600, 150)

with col2:
    model_name = st.selectbox(
        "Car Model",
        [
            "1 Series", "2 Series", "3 Series", "4 Series", "5 Series",
            "6 Series", "7 Series", "8 Series",
            "i3", "i8",
            "X1", "X2", "X3", "X4", "X5", "X6", "X7",
            "M2", "M3", "M4", "M5", "M6", "Z3", "Z4"
        ]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic", "Semi-Auto"]
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
    )



if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    input_df["car_age"] = current_year - input_df["year"]
    input_df["car_age"] = input_df["car_age"].clip(lower=1)

    input_df["mileage_per_year"] = input_df["mileage"] / input_df["car_age"]

    input_df["engineSize"] = input_df["engineSize"].replace(0, 0.1)
    input_df["power_efficiency"] = input_df["mpg"] / input_df["engineSize"]

    X_processed = preprocessor.transform(input_df)
    predicted_price = model.predict(X_processed)[0]

    st.success(f"Estimated Car Price: £{predicted_price:,.0f}")
