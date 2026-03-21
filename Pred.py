import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMW Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0a0a;
    --surface:   #131313;
    --card:      #1a1a1a;
    --border:    #2a2a2a;
    --accent:    #c8a84b;       /* BMW gold */
    --accent2:   #e8c96b;
    --text:      #f0f0f0;
    --muted:     #888;
    --danger:    #e05c5c;
    --radius:    12px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

/* hide hamburger & footer */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1208 50%, #0a0a0a 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "BMW";
    font-family: 'Bebas Neue', sans-serif;
    font-size: 12rem;
    color: rgba(200,168,75,0.04);
    position: absolute;
    right: -1rem;
    top: -2rem;
    line-height: 1;
    pointer-events: none;
    letter-spacing: 0.05em;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    line-height: 1;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.03em;
}
.hero-badge {
    display: inline-block;
    background: rgba(200,168,75,0.12);
    border: 1px solid rgba(200,168,75,0.3);
    color: var(--accent2);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.stat-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--accent);
    letter-spacing: 0.05em;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── Section headers ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding-left: 0.7rem;
    margin-bottom: 1.2rem;
}

/* ── Column labels ── */
.col-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.8rem;
}

/* ── Streamlit inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(200,168,75,0.2) !important;
}

label[data-testid="stWidgetLabel"] p {
    color: #ccc !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0a0a0a !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.8rem 2.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #1a1208 0%, #0f0f0f 100%);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 2.2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 0 40px rgba(200,168,75,0.08);
}
.result-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: var(--accent2);
    letter-spacing: 0.05em;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-note {
    font-size: 0.78rem;
    color: var(--muted);
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    bundle = joblib.load("bmw_gb_price_model.joblib")
    return bundle["model"], bundle["preprocessor"]

model, preprocessor = load_bundle()
current_year = datetime.now().year


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Machine Learning · Gradient Boosting</div>
    <div class="hero-title">BMW Price Predictor</div>
    <p class="hero-sub">Instant used-car valuations powered by a trained Gradient Boosting model · UK Market</p>
</div>
""", unsafe_allow_html=True)


# ── Stats row ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">0.94</div>
        <div class="stat-label">R² Score</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">24</div>
        <div class="stat-label">BMW Models</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">10K+</div>
        <div class="stat-label">Training Records</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">GBM</div>
        <div class="stat-label">Algorithm</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Vehicle Specifications</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="col-label">Identity</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Car Model", [
        "1 Series", "2 Series", "3 Series", "4 Series", "5 Series",
        "6 Series", "7 Series", "8 Series",
        "i3", "i8",
        "X1", "X2", "X3", "X4", "X5", "X6", "X7",
        "M2", "M3", "M4", "M5", "M6", "Z3", "Z4"
    ])
    year = st.number_input("Manufacturing Year", 1995, current_year, 2019)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])

with col2:
    st.markdown('<div class="col-label">Engine & Fuel</div>', unsafe_allow_html=True)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])
    engine_size = st.number_input("Engine Size (L)", 0.6, 6.0, 2.0, step=0.1)
    mpg = st.number_input("Fuel Efficiency (MPG)", 5.0, 500.0, 45.0, step=1.0)

with col3:
    st.markdown('<div class="col-label">Usage & Tax</div>', unsafe_allow_html=True)
    mileage = st.number_input("Mileage (km)", 0, 300000, 25000, step=1000)
    tax = st.number_input("Road Tax (£)", 0, 600, 150, step=10)
    car_age = current_year - year
    mileage_per_year = round(mileage / max(car_age, 1))
    st.metric("Estimated Mileage / Year", f"{mileage_per_year:,} km")


# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict = st.button("Predict Price")


# ── Prediction logic ──────────────────────────────────────────────────────────
if predict:
    input_df = pd.DataFrame({
        "model":        [model_name],
        "year":         [year],
        "transmission": [transmission],
        "mileage":      [mileage],
        "fuelType":     [fuel_type],
        "tax":          [tax],
        "mpg":          [mpg],
        "engineSize":   [engine_size],
    })

    input_df["car_age"] = (current_year - input_df["year"]).clip(lower=1)
    input_df["mileage_per_year"] = input_df["mileage"] / input_df["car_age"]
    input_df["engineSize"] = input_df["engineSize"].replace(0, 0.1)
    input_df["power_efficiency"] = input_df["mpg"] / input_df["engineSize"]

    X_processed = preprocessor.transform(input_df)
    predicted_price = model.predict(X_processed)[0]

    low  = predicted_price * 0.93
    high = predicted_price * 1.07

    # Summary metrics above the price
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", model_name)
    m2.metric("Year", str(year))
    m3.metric("Mileage", f"{mileage:,} km")
    m4.metric("Transmission", transmission)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Market Value</div>
        <div class="result-price">£{predicted_price:,.0f}</div>
        <div class="result-note">Confidence range &nbsp;·&nbsp; £{low:,.0f} — £{high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Gradient Boosting Regressor · UK BMW Dataset<br>
    <span style="color:#555">Predictions are estimates for informational purposes only</span>
</div>
""", unsafe_allow_html=True)