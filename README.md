# BMW Used Car Price Predictor

A machine learning web application that estimates the resale price of used BMW vehicles using a trained Gradient Boosting model on UK market data.

**Live App:** https://bmw-price-predictions.streamlit.app/

---

## Overview

This is an end-to-end machine learning project covering data analysis, feature engineering, model training, evaluation, and live deployment. The model takes vehicle specifications as input and returns an instant price estimate with a confidence range.

---

## Results

| Metric | Value |
|---|---|
| R² Score | 0.94 |
| Mean Absolute Error | ~£1,400 |
| Test Set Size | 1,078 records |
| Algorithm | Gradient Boosting Regressor |

---

## Tech Stack

- **Language:** Python
- **ML Library:** Scikit-learn
- **Web App:** Streamlit
- **Data:** UK BMW used car dataset (~10,000 listings)
- **Deployment:** Streamlit Cloud

---

## Project Structure

```
BMW_Price_Prediction/
│
├── Pred.py                  # Streamlit web application
├── BMW_EDA (1).ipynb        # Exploratory Data Analysis notebook
├── bmw_gb_price_model.joblib  # Trained model + preprocessor pipeline
├── bmw.csv                  # Dataset
├── Research Paper.pdf       # Full project research paper
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Features Used

| Feature | Description |
|---|---|
| Model | BMW model series (1 Series, X5, M3, etc.) |
| Year | Manufacturing year |
| Mileage | Total distance driven (km) |
| Transmission | Manual / Automatic / Semi-Auto |
| Fuel Type | Petrol / Diesel / Hybrid / Electric |
| Engine Size | Engine displacement in litres |
| MPG | Fuel efficiency |
| Road Tax | Annual UK road tax (£) |

**Engineered Features:**
- `car_age` — depreciation proxy derived from manufacturing year
- `mileage_per_year` — usage intensity normalised by age
- `power_efficiency` — MPG to engine size ratio capturing performance vs economy

---

## How to Run Locally

```bash
git clone https://github.com/Analyst2002/BMW_Price_Prediction.git
cd BMW_Price_Prediction
pip install -r requirements.txt
streamlit run Pred.py
```

---

## Research Paper

A detailed research paper documenting the full methodology — including EDA findings, feature engineering rationale, model selection, and evaluation — is available in the repository as `Research Paper.pdf`.

---

*Predictions are estimates based on UK market data and are intended for informational purposes only.*