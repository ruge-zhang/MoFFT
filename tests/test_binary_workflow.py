"""Compatibility checks for frozen producer recipes, without running timings."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "binary_compile_wisdom", ROOT / "tools/compile_wisdom.py")
WISDOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WISDOM)


class FrozenRecipeTests(unittest.TestCase):
    def load(self, bank=None):
        entry = {"length": 256, "precision": "fp32", "radices": [16, 16]}
        if bank is not None:
            entry["kernel_bank"] = bank
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan-wisdom.json"
            path.write_text(json.dumps({"schema_version": 1, "entries": [entry]}))
            return WISDOM.load_entries(path)[0][0]

    def test_legacy_recipes_use_default_bank(self):
        self.assertEqual(self.load()["kernel_bank"], 0)

    def test_measured_bank_is_preserved(self):
        self.assertEqual(self.load(3)["kernel_bank"], 3)

    def test_invalid_bank_is_not_silently_accepted(self):
        for bank in (-1, 4, True, "1", 1.5):
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                self.load(bank)

    def test_source_label_does_not_expose_input_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-project-owner-case.json"
            path.write_text(json.dumps({"schema_version": 1, "entries": []}))
            self.assertEqual(WISDOM.load_entries(path)[2], "embedded-plan-wisdom")


if __name__ == "__main__":
    unittest.main()
