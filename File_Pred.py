import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="BMW Batch Price Prediction", layout="wide")

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("bmw_gb_price_model.joblib")
    return bundle["model"], bundle["preprocessor"]

model, preprocessor = load_bundle()
current_year = datetime.now().year

# -------------------------
# UI
# -------------------------
st.title("BMW Car Price Prediction (Batch Mode)")
st.subheader("Upload CSV file (without price column)")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# -------------------------
# Main Logic
# -------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("Uploaded File Preview")
    st.dataframe(df, use_container_width=True)

    required_cols = [
        "model", "year", "transmission", "mileage",
        "fuelType", "tax", "mpg", "engineSize"
    ]

    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # -------------------------
    # Feature Engineering
    # -------------------------
    df_fe = df.copy()

    df_fe["car_age"] = current_year - df_fe["year"]
    df_fe["car_age"] = df_fe["car_age"].replace(0, 1)

    df_fe["mileage_per_year"] = df_fe["mileage"] / df_fe["car_age"]
    df_fe["power_efficiency"] = df_fe["mpg"] / df_fe["engineSize"].replace(0, np.nan)

    # -------------------------
    # Prediction
    # -------------------------
    X_processed = preprocessor.transform(df_fe)
    predictions = model.predict(X_processed)

    df_result = df.copy()
    df_result["Predicted Price (£)"] = predictions.round(0).astype(int)

    st.markdown("### 💰 Prediction Results")
    st.dataframe(df_result, use_container_width=True)



    # -------------------------
    # Download Excel
    # -------------------------
    buffer = BytesIO()
    df_result.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="📥 Download Excel",
        data=buffer,
        file_name="bmw_price_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
