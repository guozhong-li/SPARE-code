"""No dependency on parent experiment folders or cluster installations."""
from pathlib import Path
from contextlib import redirect_stderr
import io
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from spare.data import load_datasets
from spare.resident_data import ResidentWindows, verify_windows
from spare.training import parse_args

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_help_and_invalid_backend(self):
        result = subprocess.run([sys.executable, str(ROOT / "train.py"), "--help"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--retention", result.stdout)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--data-root", ".", "--output", "unused", "--backend", "fused"])

    def test_chronological_scaling_and_exact_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            n = 300
            source = np.arange(n, dtype=float)
            frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                                  "x": source, "OT": source * 2})
            frame.to_csv(root / "custom.csv", index=False)
            preset = dict(data="custom", data_path="custom.csv", freq="h", enc_in=2, dataset="fixture")
            splits = load_datasets(preset, root, length=32, horizon=8, label=16)
            np.testing.assert_allclose(splits["train"].scaler.mean_, [104.5, 209.])
            self.assertEqual(len(splits["train"]), 171)
            self.assertEqual(len(splits["val"]), 23)
            self.assertEqual(len(splits["test"]), 53)
            for split, data in splits.items():
                loader = ResidentWindows(data, 8, 32, 16, 8, split == "train", 2021, "cpu")
                self.assertEqual(verify_windows(loader)["status"], "passed")
                covered = sum(len(batch[0]) for batch in loader)
                self.assertEqual(covered, len(data))

    def test_portable_offline_and_online_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(2021)
            values = rng.normal(size=(160, 21))
            columns = [f"v{i}" for i in range(20)] + ["OT"]
            frame = pd.DataFrame(values, columns=columns)
            frame.insert(0, "date", pd.date_range("2020-01-01", periods=160, freq="10min"))
            frame.to_csv(root / "weather.csv", index=False)
            for mode in ("offline", "online"):
                output = root / mode
                command = [sys.executable, str(ROOT / "train.py"), "--dataset", "Weather",
                           "--data-root", str(root), "--output", str(output), "--mode", mode,
                           "--length", "32", "--horizon", "8", "--retention", ".5",
                           "--d-model", "16", "--d-ff", "32", "--heads", "2", "--layers", "1",
                           "--batch-size", "4", "--preprocess-batch-size", "8", "--smoke"]
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "smoke_complete")
                self.assertEqual(len(result["history"]), 2)
                self.assertTrue(np.isfinite(result["test"]["mse"]))
                self.assertTrue(np.isfinite(result["test"]["mae"]))
                self.assertGreater(result["training_seconds"], 0)
                self.assertTrue((output / "best.pth").is_file())
                if mode == "offline":
                    self.assertEqual(result["cache_gate"]["status"], "passed")
                    self.assertGreater(result["cache_bytes"], 0)
                    self.assertGreater(result["preprocessing_seconds"], 0)
                else:
                    self.assertEqual(result["preprocessing_seconds"], 0)
                repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(repeated.returncode, 0)
                self.assertEqual(json.loads((output / "result.json").read_text())["status"], "smoke_complete")


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
