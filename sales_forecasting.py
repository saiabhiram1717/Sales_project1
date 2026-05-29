# ==========================================
# SALES FORECASTING PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================
# DATA CLEANING
# ==========================================

# Remove Missing Values
df = df.dropna()

# Remove Duplicates
df = df.drop_duplicates()

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# DATE CONVERSION
# ==========================================

df['Order Date'] = pd.to_datetime(df['Order Date'])

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['DayOfWeek'] = df['Order Date'].dt.dayofweek

print("\nFeature Engineering Completed")
print(df[['Order Date', 'Year', 'Month', 'Day']].head())

# ==========================================
# SELECT FEATURES
# ==========================================

X = df[['Year', 'Month', 'Day', 'DayOfWeek']]
y = df['Sales']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.2f}")

# ==========================================
# VISUALIZATION 1
# MONTHLY SALES TREND
# ==========================================

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10,5))

monthly_sales.plot(kind='line', marker='o')

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.grid(True)

plt.savefig("sales_trend.png")

plt.show()

# ==========================================
# VISUALIZATION 2
# ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(y_test.values[:100], label='Actual Sales')

plt.plot(y_pred[:100], label='Predicted Sales')

plt.title("Actual vs Predicted Sales")

plt.xlabel("Samples")
plt.ylabel("Sales")

plt.legend()

plt.savefig("actual_vs_predicted.png")

plt.show()

# ==========================================
# FUTURE FORECAST
# ==========================================

future_data = pd.DataFrame({
    'Year': [2026, 2026, 2026],
    'Month': [6, 7, 8],
    'Day': [1, 1, 1],
    'DayOfWeek': [0, 2, 4]
})

future_predictions = model.predict(future_data)

print("\n===== FUTURE SALES FORECAST =====")

for i, prediction in enumerate(future_predictions):
    print(f"Forecast {i+1}: {prediction:.2f}")

# ==========================================
# FORECAST GRAPH
# ==========================================

months = ['June', 'July', 'August']

plt.figure(figsize=(8,5))

plt.plot(months, future_predictions, marker='o')

plt.title("Future Sales Forecast")

plt.xlabel("Month")
plt.ylabel("Predicted Sales")

plt.grid(True)

plt.savefig("forecast.png")

plt.show()

print("\nProject Completed Successfully")
