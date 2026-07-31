import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor


# 1. Load dataset and train model (Cached to avoid retraining on every click)
@st.cache_resource
def load_and_train():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


rf_model = load_and_train()

# 2. Web UI Elements
st.title("🏠 California House Price Predictor")
st.write(
    "Enter the characteristics of the property below to estimate its market price:"
)

# Input controls
med_inc = st.number_input(
    "Median Income (in $10,000s)", min_value=0.5, max_value=15.0, value=3.5
)
house_age = st.slider(
    "House Age (years)", min_value=1, max_value=52, value=25
)
ave_rooms = st.number_input("Average Rooms", min_value=1.0, value=5.0)
ave_bedrms = st.number_input("Average Bedrooms", min_value=1.0, value=1.0)
population = st.number_input("Population in block", min_value=1, value=1200)
ave_occup = st.number_input(
    "Average Household Members", min_value=1.0, value=3.0
)
latitude = st.number_input("Latitude", value=34.05)
longitude = st.number_input("Longitude", value=-118.25)

# 3. Prediction Action
if st.button("Predict Price 🚀"):
    input_data = pd.DataFrame(
        [{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }]
    )

    predicted_val = rf_model.predict(input_data)[0]
    actual_dollar_price = predicted_val * 100000

    st.success("Prediction Complete!")
    st.metric(
        label="Estimated House Value", value=f"${actual_dollar_price:,.2f}"
    )