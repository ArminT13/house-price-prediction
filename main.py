import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Downloading Datas
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Showing 5 Rows
print("5 Satr aval:")
print(df.head())
print("===========================")
print(df.shape)
print("===========================")
print(df.info())
print("===========================")
print(df.describe())
print("===========================")
# تفکیک ویژگی‌ها و هدف
X = df.drop(columns=['MedHouseVal'])  # همه ستون‌ها به جز قیمت
y = df['MedHouseVal']                 # فقط ستون قیمت

# بررسی ابعاد
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("(X_train):", X_train.shape[0])
print("(X_test):", X_test.shape[0])

# ۱. تعریف مدل
model = LinearRegression()

# ۲. آموزش مدل با داده‌های Train
model.fit(X_train, y_train)

# ۳. پیش‌بینی قیمت برای داده‌های Test
y_pred = model.predict(X_test)

# نمایش ۳ پیش‌بینی اول در مقایسه با قیمت‌های واقعی
print("Predict: ", y_pred[:3])
print("Real prices: ", y_test.iloc[:3].values)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# ۱. تعریف و آموزش مدل Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ۲. پیش‌بینی و ارزیابی
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print("\n--- Random Forest Results ---")
print("Random Forest MSE:", rf_mse)
print("Random Forest R2 Score:", rf_r2)