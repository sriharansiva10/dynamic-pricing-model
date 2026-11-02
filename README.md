# Dynamic Pricing Model

An XGBoost-based rental-price forecasting project. It includes a deterministic
synthetic-data generator so that every clone can train and verify the pipeline
without depending on an untracked dataset.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_sample_data.py
python xgboost_pricing_model.py --time-column timestamp
```

The final command writes these ignored, reproducible outputs to `artifacts/`:

* `xgb_pricing_model.joblib` — fitted preprocessing pipeline, model, metrics, and metadata
* `feature_importance.png`
* `predicted_vs_actual.png`

## Training with production data

```bash
python xgboost_pricing_model.py \
  --data path/to/your_data.csv \
  --target rental_price \
  --time-column timestamp
```

The target must be numeric and non-null. Numeric features are median-imputed;
other columns are mode-imputed and one-hot encoded. The persisted pipeline uses
`handle_unknown="ignore"`, so new categorical values will not break inference.

Passing `--time-column` creates a chronological train/test split, which is the
recommended evaluation strategy for pricing data. Without it, the script uses a
deterministic random split. The metrics include both the model and a mean-price
baseline; do not promote a model unless it beats that baseline on a future holdout.
The raw time column is used only for ordering and is excluded from model inputs;
derive calendar features explicitly if they are useful to the pricing policy.

## Controls and limitations

This project predicts historical rental prices; it is not, by itself, a complete
pricing policy. Apply operational guardrails (price floors/ceilings, margin,
inventory limits, and fairness review) in the serving layer before showing a
price to customers. Validate demand elasticity and revenue impact before using
the model in production.

Useful options:

```text
--test-size 0.2
--early-stopping-rounds 25
--model-out artifacts/model.joblib
--feature-importance-out artifacts/importance.png
--prediction-plot-out artifacts/predictions.png
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The test trains the complete raw-data pipeline on synthetic data, including a
categorical weather field and chronological evaluation.
