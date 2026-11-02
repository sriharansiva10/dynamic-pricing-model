"""Generate a deterministic, synthetic dataset for a reproducible demo."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h")
    weather = rng.choice(["clear", "cloudy", "rain"], size=rows, p=[0.55, 0.3, 0.15])
    availability = rng.integers(5, 101, size=rows)
    demand_score = rng.uniform(0, 100, size=rows)
    is_holiday = ((timestamps.dayofweek >= 5) | rng.choice([False, True], size=rows, p=[0.96, 0.04])).astype(int)
    weather_adjustment = np.select([weather == "rain", weather == "cloudy"], [-1.5, -0.5], default=0.0)
    rental_price = 8 + 0.11 * demand_score - 0.045 * availability + 1.2 * is_holiday + weather_adjustment
    rental_price += rng.normal(0, 0.8, size=rows)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "weather": weather,
            "availability": availability,
            "demand_score": demand_score.round(2),
            "is_holiday": is_holiday,
            "fuel_price": rng.uniform(3.0, 4.5, size=rows).round(2),
            "rental_price": rental_price.clip(3, None).round(2),
        }
    )


def main() -> None:
    output = Path("data/ebike_dynamic_pricing_dataset.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_data().to_csv(output, index=False)
    print(f"Synthetic demo data written to {output}")


if __name__ == "__main__":
    main()
