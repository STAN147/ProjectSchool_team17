"""Check the compact parameter archive without training."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_benchmark import USER_DATASETS
from saved_nca_params import BEST_PARAMS, load_modernnca_config


class SavedNCAParamsTest(unittest.TestCase):
    def test_all_thirty_datasets_and_read_only_loading(self):
        before = BEST_PARAMS.read_bytes()
        configs = json.loads(before)
        expected = {name for group in USER_DATASETS.values() for name in group} | {"walking-activity"}
        self.assertEqual(set(configs), expected)
        self.assertEqual(len(configs), 30)
        for name in configs:
            self.assertEqual(load_modernnca_config(name), configs[name])
        self.assertEqual(BEST_PARAMS.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "not found"):
            load_modernnca_config("missing")

    def test_missing_file_does_not_create_one(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                load_modernnca_config("stock", path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
