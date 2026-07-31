import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Exploratory Data Analysis (EDA)
print("--- First 5 Rows ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# 3. Feature and Target Separation
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 4. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 5. Baseline Model: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_y_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)

print("\n=== Linear Regression Evaluation ===")
print(f"Mean Squared Error (MSE): {lr_mse:.4f}")
print(f"R2 Score: {lr_r2:.4f}")

# 6. Advanced Model: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print("\n=== Random Forest Evaluation ===")
print(f"Mean Squared Error (MSE): {rf_mse:.4f}")
print(f"R2 Score: {rf_r2:.4f}")

# 7. Predict Price for a Custom House
print("\n=== Interactive House Price Prediction ===")

# Sample input for 1 house
sample_house = pd.DataFrame([{
    'MedInc': float(input("MedInc: ")),       # Median Income ($35,000)
    'HouseAge': int(input("HouseAge: ")),    # House Age (years)
    'AveRooms': int(input("AveRooms: ")),     # Average Rooms
    'AveBedrms': int(input("AveBedrms: ")),    # Average Bedrooms
    'Population': int(input("Population: ")),  # Population
    'AveOccup': int(input("AveOccup: ")),     # Average Household Members
    'Latitude': float(input("Latitude: ")),   # Latitude
    'Longitude': float(input("Longitude: ")) # Longitude
}])

# Predict using Random Forest
predicted_val = rf_model.predict(sample_house)[0]
actual_dollar_price = predicted_val * 100000

print(f"Estimated Price for Sample House: ${actual_dollar_price:,.2f}")