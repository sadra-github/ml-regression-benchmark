# ML Regression Benchmark

A compact, reproducible baseline for comparing regression models on a clearly labelled synthetic dataset.

## What it does
- Creates a synthetic regression dataset with a fixed random seed.
- Uses a train/test split to keep final evaluation data separate from model fitting.
- Compares Dummy Regressor, Linear Regression, Random Forest, and HistGradientBoosting.
- Reports MAE, RMSE, and R² in a CSV file.
- Saves a model-comparison plot and prints the best model by test RMSE.

This is a demonstration project, not a claim of performance on a real-world or NASA battery dataset.

## Requirements
Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
```

## Run
From the project root:

```bash
python -m src.benchmark
```

Outputs are written to `outputs/`:
- `regression_metrics.csv`
- `regression_benchmark.png`

## Tests

```bash
python -m unittest discover -s tests -v
```

## Reproducibility and limitations
The dataset is synthetic and generated with a fixed seed. The test split is held out from model fitting. The benchmark is intended as a small baseline template; real projects should use domain-appropriate validation, leakage checks, and uncertainty analysis.

## License
MIT. See `LICENSE`.
