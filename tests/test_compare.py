import csv
import json
import tempfile
import unittest
from pathlib import Path

from libian_metrics.compare import (
    METRIC_KEYS,
    compare_result_files,
    save_comparison_csv,
)


def _character(value, image_count):
    return {
        **{metric: value for metric in METRIC_KEYS},
        "image_count": image_count,
    }


def _write_result(path, dataset_name, characters):
    path.write_text(
        json.dumps({"dataset_name": dataset_name, "characters": characters}),
        encoding="utf-8",
    )


class CompareTests(unittest.TestCase):
    def test_compare_uses_common_characters_and_after_minus_before(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            before_path = root / "before.json"
            after_path = root / "after.json"
            _write_result(
                before_path,
                "Chu",
                {"同": _character(0.2, 3), "少": _character(0.8, 1)},
            )
            _write_result(
                after_path,
                "Han",
                {"同": _character(0.5, 4), "少": _character(0.9, 3), "新": _character(0.7, 5)},
            )

            report = compare_result_files(str(before_path), str(after_path), min_samples=2)

            self.assertEqual(report["summary"]["common_characters"], 2)
            self.assertEqual(report["summary"]["compared_characters"], 1)
            self.assertEqual(report["summary"]["excluded_by_min_samples"], 1)
            self.assertAlmostEqual(report["characters"]["同"]["delta"]["LQI"], 0.3)
            self.assertEqual(report["summary"]["metrics"]["LQI"]["after_higher_rate"], 1.0)

    def test_comparison_csv_has_one_row_per_paired_character(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            before_path = root / "before.json"
            after_path = root / "after.json"
            csv_path = root / "comparison.csv"
            _write_result(before_path, "Chu", {"甲": _character(0.2, 2)})
            _write_result(after_path, "Han", {"甲": _character(0.4, 2)})

            report = compare_result_files(str(before_path), str(after_path))
            save_comparison_csv(report, str(csv_path))

            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["char"], "甲")
            self.assertAlmostEqual(float(rows[0]["delta_LQI"]), 0.2)

    def test_min_samples_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            compare_result_files("unused.json", "unused.json", min_samples=0)


if __name__ == "__main__":
    unittest.main()
