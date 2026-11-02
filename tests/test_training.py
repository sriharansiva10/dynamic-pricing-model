import unittest

from generate_sample_data import generate_sample_data
from xgboost_pricing_model import evaluate, train_model


class TrainingTests(unittest.TestCase):
    def test_pipeline_trains_on_raw_categorical_features(self):
        data = generate_sample_data(rows=80)
        model, _, X_test, _, y_test = train_model(
            data, target="rental_price", time_column="timestamp", early_stopping_rounds=5
        )
        metrics, predictions = evaluate(model, X_test, y_test)
        self.assertEqual(len(predictions), len(y_test))
        self.assertLess(metrics["rmse"], metrics["baseline_rmse"])
        self.assertIn("weather", " ".join(model.named_steps["preprocessor"].get_feature_names_out()))


if __name__ == "__main__":
    unittest.main()
