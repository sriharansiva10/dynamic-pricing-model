"""
Dynamic pricing model using XGBoost.

Trains a gradient-boosted regression model to predict `rental_price`
from demand/supply features (fuel price, weather, availability, holiday,
day of week, demand score), evaluates it, and saves the trained model
plus a feature-importance plot.

Usage:
    python xgboost_pricing_model.py
    python xgboost_pricing_model.py --data ebike_dynamic_pricing_dataset_without_weather.csv --target rental_price
"""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "weather" in df.columns:
        df = pd.get_dummies(df, columns=["weather"], drop_first=True)
    return df


def train_model(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    **xgb_params,
):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    default_params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        objective="reg:squarederror",
    )
    default_params.update(xgb_params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model, X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_feature_importance(model, feature_names, out_path: str):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[order], color="#1f77b4")
    plt.xticks(range(len(importances)), np.array(feature_names)[order], rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("XGBoost Feature Importance — Rental Price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_test, preds, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.4, s=12, color="#1f77b4")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, color="#e83c13", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual rental price")
    plt.ylabel("Predicted rental price")
    plt.title("Predicted vs Actual Rental Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train an XGBoost dynamic pricing model.")
    parser.add_argument(
        "--data",
        default="ebike_dynamic_pricing_dataset.csv",
        help="Path to the training CSV (default: %(default)s)",
    )
    parser.add_argument("--target", default="rental_price", help="Target column to predict")
    parser.add_argument("--model-out", default="xgb_pricing_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.data)
    model, X_train, X_test, y_train, y_test = train_model(
        df, target=args.target, test_size=args.test_size
    )

    metrics = evaluate(model, X_test, y_test)
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    preds = model.predict(X_test)
    plot_feature_importance(model, X_train.columns, "feature_importance.png")
    plot_predicted_vs_actual(y_test, preds, "predicted_vs_actual.png")

    joblib.dump({"model": model, "feature_names": list(X_train.columns)}, args.model_out)
    print(f"\nModel saved to {args.model_out}")
    print("Plots saved: feature_importance.png, predicted_vs_actual.png")


if __name__ == "__main__":
    main()
