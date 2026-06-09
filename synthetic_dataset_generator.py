import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of records
n = 10000

# --------------------------
# Generate Features
# --------------------------

fuel_price = np.random.uniform(90, 120, n)

temperature = np.random.uniform(20, 40, n)

available_bikes = np.random.randint(5, 101, n)

holiday = np.random.choice([0, 1], n, p=[0.85, 0.15])

day_of_week = np.random.randint(0, 7, n)

weather = np.random.choice(
    ["Sunny", "Cloudy", "Rainy"],
    n,
    p=[0.5, 0.3, 0.2]
)

# --------------------------
# Demand Calculation
# --------------------------

weather_effect = np.select(
    [
        weather == "Sunny",
        weather == "Cloudy",
        weather == "Rainy"
    ],
    [
        15,
        5,
        -20
    ]
)

weekend_effect = np.where(day_of_week >= 5, 15, 0)

demand_score = (
    (fuel_price - 90) * 1.5
    + weather_effect
    + weekend_effect
    + holiday * 20
    + (100 - available_bikes) * 0.3
    + np.random.normal(0, 5, n)
)

demand_score = np.clip(demand_score, 0, 100)

# --------------------------
# Dynamic Pricing Formula
# --------------------------

price = (
    100
    + demand_score * 1.8
    + (50 - available_bikes) * 0.5
    + holiday * 10
    + np.random.normal(0, 8, n)
)

price = np.clip(price, 80, None)

# --------------------------
# Create DataFrame
# --------------------------

df = pd.DataFrame({
    "fuel_price": np.round(fuel_price, 2),
    "temperature": np.round(temperature, 1),
    "available_bikes": available_bikes,
    "holiday": holiday,
    "day_of_week": day_of_week,
    "weather": weather,
    "demand_score": np.round(demand_score, 2),
    "rental_price": np.round(price, 2)
})

# --------------------------
# Save Dataset
# --------------------------

df.to_csv("ebike_dynamic_pricing_dataset.csv", index=False)

print("Dataset Generated Successfully!")
print(df.head())
print("\nShape:", df.shape)