"""Reproducible regression-model benchmark on synthetic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def run_benchmark(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Fit baseline regressors and save test metrics and a comparison plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_regression(
        n_samples=600,
        n_features=10,
        n_informative=7,
        noise=18.0,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random forest": RandomForestRegressor(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, max_iter=150
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": mse ** 0.5,
                "R2": r2_score(y_test, predictions),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metrics.to_csv(output_dir / "regression_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(metrics["model"], metrics["RMSE"])
    ax.set_title("Regression baseline comparison")
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_benchmark.png", dpi=160)

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print("\nTest metrics (sorted by RMSE):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model by test RMSE: {metrics.iloc[0]['model']}")
    print(f"Saved: {output_dir / 'regression_metrics.csv'}")
    print(f"Saved: {output_dir / 'regression_benchmark.png'}")
    return metrics


if __name__ == "__main__":
    run_benchmark()
