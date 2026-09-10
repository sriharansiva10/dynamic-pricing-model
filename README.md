# Dynamic Pricing Model 🚲📈

An **XGBoost-powered dynamic pricing model** for predicting rental prices based on demand, supply, environmental, and temporal factors.

The project demonstrates how machine learning can be used to build a data-driven pricing system that adjusts predicted rental prices according to changing market conditions.

## 🚀 Overview

Dynamic pricing is a pricing strategy where the price of a product or service changes according to factors such as:

* Demand
* Availability
* Fuel prices
* Weather conditions
* Holidays
* Day of the week
* Demand score

This project uses **XGBoost Regression** to learn relationships between these factors and the target `rental_price`.

The trained model can then be used to predict rental prices for new data.

## 🧠 How It Works

The pipeline is straightforward:

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Preprocessing
   │
   ▼
Train / Test Split
   │
   ▼
XGBoost Regression
   │
   ▼
Model Evaluation
   │
   ├── RMSE
   ├── MAE
   └── R² Score
   │
   ▼
Visualizations
   │
   ├── Feature Importance
   └── Predicted vs Actual
   │
   ▼
Trained Model (.joblib)
```

If the dataset contains a `weather` column, categorical weather values are automatically converted into numerical features using one-hot encoding.

## ✨ Features

* 📊 Dynamic rental-price prediction
* 🤖 XGBoost regression model
* 🌦️ Weather feature preprocessing
* 📈 Model evaluation using RMSE, MAE, and R²
* 🔍 Feature-importance visualization
* 🎯 Predicted-vs-actual visualization
* 💾 Export of the trained model using Joblib
* ⚙️ Command-line arguments for dataset, target column, output model, and test size

## 🛠️ Tech Stack

* **Python**
* **Pandas** for data processing
* **NumPy** for numerical operations
* **Scikit-learn** for data splitting and evaluation
* **XGBoost** for machine learning
* **Matplotlib** for visualization
* **Joblib** for model serialization

The repository includes these dependencies in `requirements.txt`.

## 📁 Project Structure

```text
dynamic-pricing-model/
│
├── xgboost_pricing_model.py
├── xgb_pricing_model.joblib
├── feature_importance.png
├── predicted_vs_actual.png
├── requirements.txt
└── README.md
```

### `xgboost_pricing_model.py`

The main training script.

It:

1. Loads the dataset.
2. Preprocesses categorical weather data.
3. Splits the dataset into training and testing sets.
4. Trains an XGBoost regression model.
5. Calculates evaluation metrics.
6. Generates visualization plots.
7. Saves the trained model.

The script uses an 80/20 train-test split by default and an XGBoost configuration with 400 estimators, a maximum depth of 4, and a learning rate of 0.05.

### `xgb_pricing_model.joblib`

A serialized version of the trained XGBoost model along with the feature names required by the model.

### `feature_importance.png`

Visualizes which input features contributed most strongly to the model's predictions.

### `predicted_vs_actual.png`

Compares the model's predicted rental prices against the actual values from the test dataset.

### `requirements.txt`

Contains the Python packages required to run the project.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sriharansiva10/dynamic-pricing-model.git
cd dynamic-pricing-model
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

The training script expects a CSV dataset.

Run it with the default configuration:

```bash
python xgboost_pricing_model.py
```

The default configuration expects:

```text
ebike_dynamic_pricing_dataset.csv
```

with:

```text
rental_price
```

as the target column.

### Using a custom dataset

```bash
python xgboost_pricing_model.py \
    --data ebike_dynamic_pricing_dataset_without_weather.csv \
    --target rental_price
```

You can also specify the test-set size:

```bash
python xgboost_pricing_model.py \
    --data your_dataset.csv \
    --target rental_price \
    --test-size 0.2
```

## 📊 Model Evaluation

The model reports three evaluation metrics.

### RMSE

**Root Mean Squared Error** measures the average magnitude of prediction errors while giving larger errors more weight.

Lower is better.

### MAE

**Mean Absolute Error** measures the average absolute difference between predicted and actual prices.

Lower is better.

### R² Score

**R² (coefficient of determination)** indicates how much of the variation in rental prices is explained by the model.

Higher is generally better, with `1.0` representing a perfect fit.

## 📈 Generated Visualizations

### Feature Importance

The model generates:

```text
feature_importance.png
```

This chart helps identify which variables have the greatest influence on predicted rental prices.

### Predicted vs Actual

The model also generates:

```text
predicted_vs_actual.png
```

A prediction closer to the diagonal reference line indicates better agreement between predicted and actual rental prices.

## 💾 Model Output

After training, the model is saved as:

```text
xgb_pricing_model.joblib
```

The saved artifact contains:

```python
{
    "model": trained_xgboost_model,
    "feature_names": [...]
}
```

This allows the trained model to be loaded later without retraining it.

Example:

```python
import joblib

data = joblib.load("xgb_pricing_model.joblib")

model = data["model"]
feature_names = data["feature_names"]

prediction = model.predict(new_data)
```

## 🔬 Model Configuration

The default XGBoost configuration includes:

| Parameter     |      Value |
| ------------- | ---------: |
| Estimators    |        400 |
| Max Depth     |          4 |
| Learning Rate |       0.05 |
| Subsample     |        0.8 |
| Column Sample |        0.8 |
| Objective     | Regression |

These settings are defined in the training script and can be modified for experimentation.

## 🎯 Project Goal

The goal of this project is to demonstrate a practical machine-learning approach to **dynamic pricing**.

Instead of relying entirely on fixed pricing rules, the model learns pricing relationships from historical data and uses those patterns to estimate rental prices under different conditions.

This type of system could potentially be extended to applications such as:

* 🚲 Bike and e-bike rentals
* 🚗 Ride-sharing
* 🏨 Hotels
* ✈️ Travel services
* 🛒 E-commerce
* 📦 Logistics
* 🎫 Event and ticket pricing

## 🔮 Future Improvements

Possible extensions include:

* Real-time pricing predictions
* REST API deployment
* Web dashboard for predictions
* Hyperparameter optimization
* Cross-validation
* Automated model retraining
* Demand forecasting
* Price elasticity analysis
* Integration with live weather data
* Integration with inventory/availability data
* Model monitoring and drift detection
* Containerization with Docker
* Cloud deployment

## ⚠️ Disclaimer

This project is intended for **educational and experimental purposes**.

The quality of predictions depends heavily on the dataset, feature quality, data distribution, and model configuration. A production pricing system would require additional validation, monitoring, safeguards, and business constraints.

## 👨‍💻 Author

**Sriharan Siva**

GitHub: [@sriharansiva10](https://github.com/sriharansiva10)

---

## 🤖 Credits

Developed by **Sriharan Siva** with assistance from **ChatGPT (GPT-5.6 Luna)** for documentation, explanation, and development guidance.

> This README was prepared with the assistance of ChatGPT after reviewing the repository structure and source code.
