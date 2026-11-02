"""Train a reproducible XGBoost rental-pricing model.

The saved artifact is a scikit-learn pipeline, so it includes the fitted
preprocessing and can safely handle the same raw columns at prediction time.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str, target: str) -> pd.DataFrame:
    """Load and validate training data before any transformations occur."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. "
            "Create a reproducible demo dataset with `python generate_sample_data.py` "
            "or pass --data with your CSV path."
        )

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Training data is empty.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the CSV.")
    if df[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values.")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"Target column '{target}' must be numeric.")
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a feature transformer that is retained in the trained artifact."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    if not numeric_columns and not categorical_columns:
        raise ValueError("No feature columns were provided.")

    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    time_column: str | None = None,
):
    """Use a chronological holdout when a time column is supplied."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    if time_column:
        if time_column not in features.columns:
            raise ValueError(f"Time column '{time_column}' is missing from the CSV.")
        timestamps = pd.to_datetime(features[time_column], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"Time column '{time_column}' contains invalid timestamps.")
        ordered_index = timestamps.sort_values(kind="stable").index
        split_at = int(len(ordered_index) * (1 - test_size))
        if split_at == 0 or split_at == len(ordered_index):
            raise ValueError("The dataset is too small for the requested chronological split.")
        train_index, test_index = ordered_index[:split_at], ordered_index[split_at:]
        return features.loc[train_index], features.loc[test_index], target.loc[train_index], target.loc[test_index]

    return train_test_split(features, target, test_size=test_size, random_state=random_state)


def train_model(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    time_column: str | None = None,
    early_stopping_rounds: int = 25,
    **xgb_params,
):
    features = df.drop(columns=[target])
    labels = df[target]
    X_train_full, X_test, y_train_full, y_test = split_data(
        features, labels, test_size, random_state, time_column
    )

    # Reserve the final test set for evaluation only. When early stopping is
    # enabled, split the training partition again so model selection never
    # sees the holdout.
    if early_stopping_rounds > 0:
        val_size = 0.2
        X_train, X_val, y_train, y_val = split_data(
            X_train_full, y_train_full, val_size, random_state, time_column
        )
    else:
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = None, None

    # A raw timestamp is an identifier-like value, not a stable model feature.
    # It is used for the chronological split and excluded unless callers first
    # derive durable calendar features (hour, weekday, season, and so on).
    if time_column:
        X_train = X_train.drop(columns=[time_column])
        X_test = X_test.drop(columns=[time_column])
        if X_val is not None:
            X_val = X_val.drop(columns=[time_column])

    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    if X_val is not None:
        X_val_transformed = preprocessor.transform(X_val)
    else:
        X_val_transformed = None

    default_params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "objective": "reg:squarederror",
        "n_jobs": 1,
    }
    if early_stopping_rounds > 0:
        default_params["early_stopping_rounds"] = early_stopping_rounds
    default_params.update(xgb_params)

    estimator = xgb.XGBRegressor(**default_params)
    fit_kwargs = {"verbose": False}
    if X_val_transformed is not None:
        fit_kwargs["eval_set"] = [(X_val_transformed, y_val)]
    estimator.fit(X_train_transformed, y_train, **fit_kwargs)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    return pipeline, X_train, X_test, y_train, y_test


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    predictions = model.predict(X_test)
    baseline_predictions = np.full(len(y_test), y_test.mean())
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_predictions)),
    }
    return metrics, predictions


def plot_feature_importance(model: Pipeline, out_path: str) -> None:
    estimator = model.named_steps["model"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = estimator.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[order], color="#1f77b4")
    plt.xticks(range(len(importances)), feature_names[order], rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("XGBoost Feature Importance — Rental Price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_test: pd.Series, predictions: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.4, s=12, color="#1f77b4")
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, color="#e83c13", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual rental price")
    plt.ylabel("Predicted rental price")
    plt.title("Predicted vs Actual Rental Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost dynamic-pricing model.")
    parser.add_argument("--data", default="data/ebike_dynamic_pricing_dataset.csv")
    parser.add_argument("--target", default="rental_price")
    parser.add_argument("--model-out", default="artifacts/xgb_pricing_model.joblib")
    parser.add_argument("--feature-importance-out", default="artifacts/feature_importance.png")
    parser.add_argument("--prediction-plot-out", default="artifacts/predicted_vs_actual.png")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--time-column", help="Optional timestamp column for chronological evaluation")
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    args = parser.parse_args()

    df = load_data(args.data, args.target)
    model, _, X_test, _, y_test = train_model(
        df, target=args.target, test_size=args.test_size, time_column=args.time_column,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    metrics, predictions = evaluate(model, X_test, y_test)
    for output in (args.model_out, args.feature_importance_out, args.prediction_plot_out):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(model, args.feature_importance_out)
    plot_predicted_vs_actual(y_test, predictions, args.prediction_plot_out)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "source_data": str(Path(args.data)),
        "target": args.target,
        "metrics": metrics,
        "feature_names": model.named_steps["preprocessor"].get_feature_names_out().tolist(),
        "library_versions": {"xgboost": xgb.__version__, "scikit_learn": sklearn.__version__},
    }
    joblib.dump({"pipeline": model, "metadata": metadata}, args.model_out)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
