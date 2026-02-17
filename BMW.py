import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import joblib

@st.cache_data
def get_data():
    return pd.read_csv("E:/Data Science/ML/BMW/bmw.csv")

df = get_data()

X = df.drop(columns=["price"])
y = df["price"]

test_size = st.sidebar.slider("Test Size", 0.05, 0.5, 0.2, 0.05)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

current_year = datetime.now().year

def feature_engineering(df):
    df = df.copy()
    df["car_age"] = current_year - df["year"]
    df["car_age"] = df["car_age"].replace(0, 1)
    df["mileage_per_year"] = df["mileage"] / df["car_age"]
    df["power_efficiency"] = df["mpg"] / df["engineSize"].replace(0, np.nan)
    return df

X_train_fe = feature_engineering(X_train)
X_test_fe = feature_engineering(X_test)

st.dataframe(df.head())
st.dataframe(X_train_fe.head())

categorical_cols = ["model", "transmission", "fuelType"]
numerical_cols = X_train_fe.drop(columns=categorical_cols).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train_fe)
X_test_processed = preprocessor.transform(X_test_fe)

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    min_samples_leaf=20,
    random_state=42
)



model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluation Metrics (Raw Price)")
st.write("MAE (£):", round(mae, 2))
st.write("RMSE (£):", round(rmse, 2))
st.write("R2:", round(r2, 4))

if st.button("Save Model"):
    bundle = {
        "model": model,
        "preprocessor": preprocessor
    }
    joblib.dump(bundle, "bmw_gb_price_model.joblib")
    st.success("Gradient Boosting model saved successfully")
