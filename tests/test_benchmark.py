import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.benchmark import run_benchmark


class RegressionBenchmarkTests(unittest.TestCase):
    def test_metrics_and_plot_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics = run_benchmark(output_dir)

            self.assertEqual(len(metrics), 4)
            self.assertEqual(set(metrics["model"]), {
                "Dummy mean", "Linear regression", "Random forest",
                "HistGradientBoosting"
            })
            self.assertTrue((output_dir / "regression_metrics.csv").is_file())
            self.assertTrue((output_dir / "regression_benchmark.png").is_file())

            saved = pd.read_csv(output_dir / "regression_metrics.csv")
            self.assertEqual(len(saved), 4)
            self.assertTrue(saved["RMSE"].notna().all())
            self.assertTrue(saved["MAE"].notna().all())
            self.assertTrue(saved["R2"].notna().all())

    def test_metrics_are_sorted_by_rmse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = run_benchmark(Path(temp_dir))
            self.assertEqual(metrics["RMSE"].tolist(), sorted(metrics["RMSE"].tolist()))


if __name__ == "__main__":
    unittest.main()
