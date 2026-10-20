"""SME1 source-generation regressions; no target instructions are executed.

Run alongside the unchanged upstream test_compiler.py on a login node.
This is not target numerical correctness or performance validation.
"""
import re
import unittest

from mofft_compiler.emitter import emit_candidate_source
from mofft_compiler.ir import Op
from mofft_compiler.patterns import enumerate_candidates
from mofft_compiler.targets.arm_sme import arm_sme_target


FORBIDDEN_SME2_SOURCE = re.compile(
    r"sv(?:mla|mls)_za\d+_\w+_vg1x[24]"
    r"|svread_(?:hor|ver)_za\d+_\w+_vg[24]"
    r"|svld1_\w+_x[24]\s*\("
    r"|svreinterpret_c\s*\(")


class SME1PortTests(unittest.TestCase):
    def test_fp64_all_power_radix_candidates_keep_outer_products(self):
        target = arm_sme_target(512, sme1=True)
        for radix in (2, 4, 8, 16, 32, 64):
            for stage in ("first", "other"):
                for direction in ("forward", "backward"):
                    candidates = enumerate_candidates(
                        radix, "fp64", stage, direction, target=target)
                    self.assertTrue(candidates)
                    for index, candidate in enumerate(candidates):
                        with self.subTest(radix=radix, stage=stage,
                                          direction=direction, index=index):
                            ops = [node.op for node in candidate.graph.nodes]
                            self.assertIn(Op.OUTER_PRODUCT_ACCUM, ops)
                            self.assertNotIn(Op.MATRIX_FMA, ops)
                            self.assertFalse(any(
                                node.op == Op.MATRIX_EXTRACT_H and
                                node.attr("phase") == "matrix_twiddle"
                                for node in candidate.graph.nodes))
                            if stage == "other":
                                self.assertIn(Op.FMLA_VECTOR, ops)
                            # Check normal, fused-first-transpose, and the
                            # other-stage repeated-twiddle/input variants.
                            variants = (("normal", "transposed")
                                        if stage == "first" else
                                        ("normal", "broadcast",
                                         "direct_broadcast"))
                            for context in variants:
                                source = emit_candidate_source(
                                    candidate, "port_probe", context=context)
                                self.assertIn("svmopa_za64_f64_m", source)
                                self.assertIsNone(
                                    FORBIDDEN_SME2_SOURCE.search(source))

    def test_default_still_exposes_sme2_other_stage(self):
        target = arm_sme_target()
        self.assertTrue(target.grouped_vector_ops)
        candidates = enumerate_candidates(16, "fp64", "other", "forward",
                                          target=target)
        self.assertTrue(any(
            node.op == Op.MATRIX_FMA
            for candidate in candidates for node in candidate.graph.nodes))
        self.assertTrue(any(
            FORBIDDEN_SME2_SOURCE.search(
                emit_candidate_source(candidate, "default_probe"))
            for candidate in candidates))

    def test_port_rejects_unreviewed_streaming_vector_lengths(self):
        for bits in (128, 256, 1024, 2048):
            with self.assertRaisesRegex(ValueError, "512-bit"):
                arm_sme_target(bits, sme1=True)


if __name__ == "__main__":
    unittest.main()
