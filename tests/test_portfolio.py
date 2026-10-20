import json
import tempfile
import unittest
from pathlib import Path

from mofft_compiler.portfolio import generate_portfolio
from mofft_compiler.profile import MachineProfile


ROOT = Path(__file__).resolve().parents[1]


class PortfolioGenerationTests(unittest.TestCase):
    def setUp(self):
        self.profile = MachineProfile.load(
            ROOT / "profiles" / "apple-m5-bootstrap.json")

    def test_emits_namespaced_family_banks_and_bank_dispatch_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            summary = generate_portfolio(self.profile, output, (2, 4, 16))

            self.assertEqual(summary["banks"], 4)
            self.assertEqual(summary["alternative_kernel_count"], 180)
            for bank in (1, 2, 3):
                bank_dir = output / f"bank{bank}"
                for name in ("mofft_kernels_fp32.c", "mofft_kernels_fp64.c",
                             "mofft_kernel_registry.c",
                             "mofft_generated_kernels.h", "manifest.json"):
                    self.assertTrue((bank_dir / name).is_file(),
                                    (bank, name))
                registry = (bank_dir / "mofft_kernel_registry.c").read_text()
                header = (bank_dir / "mofft_generated_kernels.h").read_text()
                self.assertIn(f"mofft_bank{bank}_dispatch_first_fp32", registry)
                self.assertIn(f"mofft_bank{bank}_kernel_r16_fp64_fwd_first",
                              header)
                self.assertIn("mofft_complex_f32", header)
                self.assertNotIn("void mofft_dispatch_first_fp32(", registry)
                self.assertNotIn("mofft_bank1_complex_f32", header)

            wrapper = (output / "mofft_kernel_portfolio.c").read_text()
            wrapper_header = (output / "mofft_kernel_portfolio.h").read_text()
            self.assertIn("mofft_generated_kernel_bank_count(void) { return 4U; }",
                          wrapper)
            for bank in (1, 2, 3):
                self.assertIn(f"case {bank}: mofft_bank{bank}_dispatch_other_fp64",
                              wrapper)
            self.assertIn("mofft_portfolio_dispatch_first_fp32(unsigned bank",
                          wrapper_header)
            self.assertIn("default: mofft_dispatch_first_fp32(", wrapper)

    def test_selection_metadata_is_model_forced_not_empirical_wisdom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            generate_portfolio(self.profile, output, (2, 3, 4, 16))

            for bank, family in ((1, "like_terms"), (2, "even_odd"),
                                 (3, "vector_reuse")):
                manifest = json.loads(
                    (output / f"bank{bank}" / "manifest.json").read_text())
                self.assertEqual(manifest["family"], family)
                self.assertIsNone(manifest["kernel_wisdom"])
                self.assertIn("no target kernel measurements",
                              manifest["selection_source"])
                self.assertNotIn("/Users/", json.dumps(manifest))
                for selection in manifest["selections"]:
                    self.assertIsNone(selection["measured_nanoseconds"])
                    self.assertTrue(selection["selection_reason"].startswith(
                        "model-forced-portfolio-"))

            evenodd = json.loads(
                (output / "bank2" / "manifest.json").read_text())
            odd_fallback = next(item for item in evenodd["selections"]
                                if item["radix"] == 3
                                and item["precision"] == "fp32"
                                and item["direction"] == "forward"
                                and item["stage"] == "first")
            self.assertEqual(odd_fallback["selection_reason"],
                             "model-forced-portfolio-fallback")
            self.assertEqual(odd_fallback["pattern"], "direct")

            vector_reuse = json.loads(
                (output / "bank3" / "manifest.json").read_text())
            legal_vector_reuse = [
                item for item in vector_reuse["selections"]
                if item["radix"] == 16 and item["precision"] == "fp64"]
            self.assertTrue(legal_vector_reuse)
            self.assertTrue(all(item["selection_reason"] ==
                                "model-forced-portfolio-family"
                                for item in legal_vector_reuse))
            self.assertTrue(all(item["pattern"] in (
                "vector_reuse", "like_terms_vector_reuse")
                for item in legal_vector_reuse))

    def test_generation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            generate_portfolio(self.profile, first, (2, 4))
            generate_portfolio(self.profile, second, (2, 4))
            first_files = {path.relative_to(first): path.read_bytes()
                           for path in first.rglob("*") if path.is_file()}
            second_files = {path.relative_to(second): path.read_bytes()
                            for path in second.rglob("*") if path.is_file()}
            self.assertEqual(first_files, second_files)


if __name__ == "__main__":
    unittest.main()
