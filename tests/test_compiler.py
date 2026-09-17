import json
import importlib.util
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mofft_compiler.calibrate import build_profile
from mofft_compiler.emitter import (_compiler_digest, emit,
                                    emit_candidate_source, profile_digest)
from mofft_compiler.model import estimate, rank
from mofft_compiler.ir import Op
from mofft_compiler.mapping import (MatrixMapping,
                                    evaluate_outer_product_mapping)
from mofft_compiler.patterns import (Pattern, enumerate_candidates,
                                     build_candidate, evaluate_pattern,
                                     legal_patterns)
from mofft_compiler.profile import LayoutCase, MachineProfile
from mofft_compiler.rewrites import algebraic_patterns
from mofft_compiler.scheduler import schedule
from mofft_compiler.targets.base import TargetCapabilities
from mofft_compiler.targets.arm_sme import arm_sme_target
from mofft_compiler.validate import _harness, _wrapper


ROOT = Path(__file__).resolve().parents[1]
WISDOM_SPEC = importlib.util.spec_from_file_location(
    "compile_wisdom", ROOT / "tools/compile_wisdom.py")
WISDOM_MODULE = importlib.util.module_from_spec(WISDOM_SPEC)
assert WISDOM_SPEC.loader is not None
WISDOM_SPEC.loader.exec_module(WISDOM_MODULE)
load_entries = WISDOM_MODULE.load_entries


class PatternTests(unittest.TestCase):
    def test_rewrites_are_equivalent(self):
        rng = random.Random(8137)
        for radix in (2, 3, 5, 8, 15, 16, 32, 64):
            values = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
                      for _ in range(radix)]
            for direction in ("forward", "backward"):
                reference = evaluate_pattern(values, direction, Pattern.DIRECT)
                patterns = set(legal_patterns(radix, "fp32")) | set(
                    legal_patterns(radix, "fp64"))
                for pattern in patterns:
                    actual = evaluate_pattern(values, direction, pattern)
                    self.assertLess(max(abs(a - b) for a, b in zip(actual, reference)),
                                    2e-10, (radix, direction, pattern))

    def test_pattern_preconditions_match_algorithm(self):
        self.assertNotIn(Pattern.EVEN_ODD, legal_patterns(15))
        self.assertIn(Pattern.EVEN_ODD, legal_patterns(16, "fp32"))
        self.assertIn(Pattern.EVEN_ODD, legal_patterns(16, "fp64"))
        self.assertIn(Pattern.LIKE_TERMS_VECTOR_REUSE,
                      legal_patterns(32, "fp64"))

    def test_algebraic_legality_is_independent_of_target_width(self):
        self.assertIn(Pattern.VECTOR_REUSE, algebraic_patterns(8))
        self.assertIn(Pattern.EVEN_ODD, algebraic_patterns(8))
        self.assertNotIn(Pattern.VECTOR_REUSE, legal_patterns(8, "fp32"))
        self.assertNotIn(Pattern.EVEN_ODD, algebraic_patterns(15))

    def test_target_capabilities_control_mapping_feasibility(self):
        narrow_outer = TargetCapabilities(
            name="test-outer", vector_bits=256, vector_registers=32,
            accumulator_tiles_fp32=4, accumulator_tiles_fp64=8,
            matrix_group_widths=(2, 4),
            mappings=(MatrixMapping.SPLIT_COMPLEX_OUTER,))
        self.assertIn(Pattern.VECTOR_REUSE,
                      legal_patterns(8, "fp32", narrow_outer))

        inner_only = TargetCapabilities(
            name="test-inner", vector_bits=512, vector_registers=32,
            accumulator_tiles_fp32=8, accumulator_tiles_fp64=8,
            matrix_group_widths=(4,),
            mappings=(MatrixMapping.COMPLEX_INNER,))
        self.assertIn(Pattern.LIKE_TERMS, algebraic_patterns(16))
        with self.assertRaisesRegex(ValueError, "does not support"):
            build_candidate(16, "fp64", "first", "forward",
                            Pattern.LIKE_TERMS, target=inner_only)

        candidate = build_candidate(
            16, "fp64", "first", "forward", Pattern.LIKE_TERMS,
            target=narrow_outer)
        self.assertIs(candidate.target, narrow_outer)
        self.assertEqual(candidate.mapping,
                         MatrixMapping.SPLIT_COMPLEX_OUTER)

    def test_split_and_interleaved_outer_mappings_are_equivalent(self):
        rng = random.Random(7301)
        coefficients = [[complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
                         for _ in range(7)] for _ in range(11)]
        inputs = [[complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
                  for _ in range(5)] for _ in range(7)]
        split = evaluate_outer_product_mapping(
            coefficients, inputs, MatrixMapping.SPLIT_COMPLEX_OUTER)
        interleaved = evaluate_outer_product_mapping(
            coefficients, inputs, MatrixMapping.INTERLEAVED_COMPLEX_OUTER)
        self.assertLess(max(abs(a - b)
                            for row_a, row_b in zip(split, interleaved)
                            for a, b in zip(row_a, row_b)), 1e-12)
        with self.assertRaisesRegex(ValueError, "not an outer-product"):
            evaluate_outer_product_mapping(
                coefficients, inputs, MatrixMapping.COMPLEX_INNER)

    def test_sme_advertises_only_lowered_mappings(self):
        target = arm_sme_target()
        self.assertTrue(target.supports(MatrixMapping.SPLIT_COMPLEX_OUTER))
        self.assertFalse(
            target.supports(MatrixMapping.INTERLEAVED_COMPLEX_OUTER))
        with self.assertRaisesRegex(ValueError, "does not support"):
            build_candidate(
                8, "fp64", "other", "forward", Pattern.DIRECT,
                target=target,
                mapping=MatrixMapping.INTERLEAVED_COMPLEX_OUTER)


class GraphTests(unittest.TestCase):
    def test_matrix_ir_does_not_name_an_instruction_set(self):
        candidate = enumerate_candidates(
            16, "fp64", "other", "forward")[0]
        operation_names = {node.op.value for node in candidate.graph.nodes}
        for architecture_word in ("sme", "sve", "za", "fmopa"):
            self.assertFalse(any(architecture_word in operation
                                 for operation in operation_names))

    def test_target_maps_neutral_ir_to_instruction_models(self):
        target = arm_sme_target()
        self.assertEqual(target.instruction_model_key(
            "outer_product_accumulate"), "fmopa")
        self.assertEqual(target.instruction_model_key(
            "matrix_fma", 4), "sme2_fmla_vg4")

    def test_schedule_is_topological_and_deterministic(self):
        candidate = enumerate_candidates(16, "fp32", "other", "forward")[0]
        first = schedule(candidate.graph)
        second = schedule(candidate.graph)
        self.assertEqual(first, second)
        position = {item.node.id: index for index, item in enumerate(first)}
        for item in first:
            for dep in item.node.inputs:
                self.assertLess(position[dep], position[item.node.id])

    def test_cse_digest_is_stable(self):
        one = enumerate_candidates(15, "fp64", "first", "forward")[0].graph
        two = enumerate_candidates(15, "fp64", "first", "forward")[0].graph
        self.assertEqual(one.digest(), two.digest())

    def test_outer_product_has_accumulator_coefficient_and_input_dependencies(self):
        graph = enumerate_candidates(15, "fp64", "first", "forward")[0].graph
        outer_products = [node for node in graph.nodes
                          if node.op == Op.OUTER_PRODUCT_ACCUM]
        self.assertTrue(outer_products)
        self.assertTrue(all(len(node.inputs) == 3
                            for node in outer_products))
        self.assertEqual({int(node.attr("tile")) for node in outer_products},
                         {0, 1, 2, 3})

    def test_other_stage_uses_matrix_fmla_when_a_tile_is_available(self):
        for precision in ("fp32", "fp64"):
            candidate = enumerate_candidates(
                16, precision, "other", "forward")[0]
            counts = candidate.graph.op_counts()
            self.assertGreater(counts.get("matrix_fma", 0), 0)
            self.assertEqual(counts.get("fmla_vector", 0), 0)

    def test_vg2_and_vg4_are_distinct_other_stage_candidates(self):
        candidates = enumerate_candidates(16, "fp64", "other", "forward")
        self.assertEqual({candidate.matrix_vg_width for candidate in candidates},
                         {2, 4})
        vg2 = next(candidate for candidate in candidates
                   if candidate.pattern == Pattern.DIRECT and
                   candidate.matrix_vg_width == 2)
        vg4 = next(candidate for candidate in candidates
                   if candidate.pattern == Pattern.DIRECT and
                   candidate.matrix_vg_width == 4)
        self.assertGreater(vg2.graph.op_counts()["matrix_fma"],
                           vg4.graph.op_counts()["matrix_fma"])

    def test_temp_tile_rotation_is_an_explicit_candidate_dimension(self):
        candidates = enumerate_candidates(16, "fp64", "other", "forward")
        direct_vg4 = [candidate for candidate in candidates
                      if candidate.pattern == Pattern.DIRECT and
                      candidate.matrix_vg_width == 4]
        self.assertEqual({candidate.rotate_temp_tiles
                          for candidate in direct_vg4}, {False, True})
        self.assertEqual(len({candidate.graph.digest()
                              for candidate in direct_vg4}), 2)
        off_only = enumerate_candidates(16, "fp64", "other", "forward",
                                        "off")
        self.assertTrue(all(not candidate.rotate_temp_tiles
                            for candidate in off_only))
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        self.assertTrue(rank(direct_vg4, profile)[0][0].rotate_temp_tiles)

    def test_batch_pipeline_is_an_empirical_candidate_dimension(self):
        for radix in (8, 16, 32):
            candidates = enumerate_candidates(
                radix, "fp64", "other", "forward", "off")
            direct_vg4 = [candidate for candidate in candidates
                          if candidate.pattern == Pattern.DIRECT and
                          candidate.matrix_vg_width == 4]
            self.assertEqual({candidate.batch_pipeline_depth
                              for candidate in direct_vg4}, {1, 2})
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        self.assertEqual(rank(direct_vg4, profile)[0][0].batch_pipeline_depth,
                         1)

    def test_moderate_radix_vg_tiebreak_matches_lowering_pressure(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        candidates = enumerate_candidates(32, "fp64", "other", "forward")
        expected_width = {
            Pattern.DIRECT: 4,
            Pattern.LIKE_TERMS: 4,
            Pattern.EVEN_ODD: 4,
            Pattern.LIKE_TERMS_EVEN_ODD: 2,
        }
        for pattern, width in expected_width.items():
            selected = rank(
                [candidate for candidate in candidates
                 if candidate.pattern == pattern], profile)[0][0]
            self.assertEqual(selected.matrix_vg_width, width)

        r64_like_terms = [
            candidate for candidate in enumerate_candidates(
                64, "fp64", "other", "forward")
            if candidate.pattern == Pattern.LIKE_TERMS]
        selected = rank(r64_like_terms, profile)[0][0]
        self.assertEqual(selected.matrix_vg_width, 4)
        self.assertTrue(selected.rotate_temp_tiles)

    def test_row_epilogue_cost_includes_even_odd_recombination(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        candidates = enumerate_candidates(16, "fp64", "other", "forward")
        like_terms = next(item for item in candidates
                          if item.pattern == Pattern.LIKE_TERMS and
                          item.matrix_vg_width == 4)
        even_odd = next(item for item in candidates
                        if item.pattern == Pattern.EVEN_ODD and
                        item.matrix_vg_width == 4)
        self.assertGreater(estimate(even_odd, profile).post_cost,
                           estimate(like_terms, profile).post_cost)

    def test_radix32_combined_pattern_stores_direct_accumulators(self):
        candidate = build_candidate(
            32, "fp64", "first", "forward",
            Pattern.LIKE_TERMS_EVEN_ODD)
        stores = [node for node in candidate.graph.nodes
                  if node.op == Op.STORE2]
        self.assertEqual(len(stores), 4)
        self.assertTrue(all(
            candidate.graph.nodes[value].op == Op.MATRIX_EXTRACT_H
            for store in stores for value in store.inputs))

        transposed = emit_candidate_source(
            candidate, "r32_lteo_transposed", "transposed")
        self.assertNotIn("vertical_low_r_", transposed)
        self.assertIn("svcreate2_f64(vertical_0_0, vertical_0_1)",
                      transposed)
        self.assertIn("svcreate2_f64(vertical_0_2, vertical_0_3)",
                      transposed)

    def test_radix32_later_like_terms_uses_chunked_input_order(self):
        candidate = build_candidate(
            32, "fp64", "other", "forward", Pattern.LIKE_TERMS,
            matrix_vg_width=4, rotate_temp_tiles=True)
        self.assertEqual(candidate.blocks_per_group, 2)
        self.assertEqual(candidate.matrix_iterations, 2)
        source = emit_candidate_source(candidate, "chunked_r32_lt")
        self.assertNotIn("(ptrdiff_t)30 * (ptrdiff_t)row_stride", source)
        self.assertNotIn("(ptrdiff_t)-29 * (ptrdiff_t)row_stride", source)

        direct = build_candidate(
            32, "fp64", "other", "forward", Pattern.DIRECT,
            matrix_vg_width=4, rotate_temp_tiles=True)
        self.assertEqual(direct.blocks_per_group, 2)
        self.assertEqual(direct.matrix_iterations, 2)

    def test_radix64_rotating_like_terms_uses_chunked_input_order(self):
        rotating = build_candidate(
            64, "fp64", "other", "forward", Pattern.LIKE_TERMS,
            matrix_vg_width=4, rotate_temp_tiles=True)
        rotating_source = emit_candidate_source(rotating, "chunked_r64_lt")
        self.assertNotIn("(ptrdiff_t)-61 * (ptrdiff_t)row_stride",
                         rotating_source)

        single_tile = build_candidate(
            64, "fp64", "other", "forward", Pattern.LIKE_TERMS,
            matrix_vg_width=4, rotate_temp_tiles=False)
        single_tile_source = emit_candidate_source(single_tile, "paired_r64_lt")
        self.assertIn("(ptrdiff_t)-61 * (ptrdiff_t)row_stride",
                      single_tile_source)

    def test_memory_model_accounts_for_working_set_and_bandwidth(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        small = build_candidate(3, "fp64", "first", "forward",
                                Pattern.DIRECT)
        large = build_candidate(64, "fp64", "first", "forward",
                                Pattern.DIRECT)
        small_cost = estimate(small, profile)
        large_cost = estimate(large, profile)
        self.assertAlmostEqual(
            small_cost.compute_cost,
            small_cost.matrix_cost + small_cost.post_cost +
            small_cost.register_pressure_cost)
        self.assertAlmostEqual(
            small_cost.total_cost,
            max(small_cost.compute_cost, small_cost.memory_cost))
        self.assertEqual(small_cost.memory_level, "l1d")
        self.assertEqual(profile.memory_hierarchy.level_for(1024 * 1024)[0],
                         "l2")
        self.assertEqual(profile.memory_hierarchy.level_for(64 * 1024 * 1024)[0],
                         "dram")
        self.assertGreater(large_cost.working_set_bytes,
                           small_cost.working_set_bytes)
        self.assertGreater(large_cost.traffic_bytes,
                           small_cost.traffic_bytes)
        self.assertGreater(large_cost.memory_cost,
                           small_cost.memory_cost)
        self.assertGreaterEqual(small_cost.cache_footprint_bytes,
                                small_cost.working_set_bytes)
        self.assertGreater(small_cost.cache_line_utilization, 0.0)
        self.assertLessEqual(small_cost.cache_line_utilization, 1.0)
        self.assertGreater(small_cost.arithmetic_intensity, 0.0)
        self.assertGreater(
            profile.memory_hierarchy.levels["l1d"].effective_mixed_bandwidth,
            0.0)

    def test_vector_reuse_negation_uses_matrix_resource_bound(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        candidate = build_candidate(64, "fp64", "first", "forward",
                                    Pattern.VECTOR_REUSE)
        cost = estimate(candidate, profile)
        # Coefficient sign reconstruction executes in the matrix phase.  Its
        # SVE work must be represented by matrix_cost's resource maximum, not
        # appended once more as a serial cost after that maximum.
        self.assertAlmostEqual(
            cost.compute_cost,
            (cost.matrix_cost + cost.post_cost) *
            candidate.matrix_iterations + cost.register_pressure_cost)


class CalibrationTests(unittest.TestCase):
    def test_legacy_memory_units_are_normalized_on_load(self):
        raw = MachineProfile.load(
            ROOT / "profiles/apple-m5-bootstrap.json").to_json()
        raw.pop("cost_unit")
        raw["measurement_metadata"] = {"throughput_unit": "instructions/ns"}
        for level in raw["memory_hierarchy"]["levels"].values():
            for kind in ("read", "write", "mixed"):
                current = f"{kind}_bandwidth_bytes_per_cost_unit"
                legacy = f"{kind}_bandwidth_bytes_per_cycle"
                level[legacy] = level.pop(current)
            level["latency_cycles"] = level.pop("latency_cost")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(raw, stream)
            stream.flush()
            profile = MachineProfile.load(stream.name)
        self.assertEqual(profile.cost_unit, "nanoseconds")
        self.assertGreater(
            profile.memory_hierarchy.levels["l1d"]
            .read_bandwidth_bytes_per_cost_unit, 0.0)

    def test_layout_efficiency_is_relative_to_nearest_copy_case(self):
        raw = {
            "cases": [],
            "memory_cases": [{
                "operation": "copy", "working_set_bytes": 8 * 1024 * 1024,
                "bytes_per_second": 40e9,
            }],
            "layout_cases": [{
                "operation": "blocked_transpose",
                "working_set_bytes": 8 * 1024 * 1024,
                "columns": 16, "batch": 2,
                "bytes_per_second": 30e9,
            }],
        }
        profile_json = build_profile(raw, json.dumps(raw).encode(), "test")
        self.assertEqual(profile_json["schema_version"], 3)
        self.assertEqual(profile_json["cost_unit"], "nanoseconds")
        case = profile_json["layout_model"]["cases"][0]
        self.assertAlmostEqual(case["relative_bandwidth"], 0.75)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(profile_json, stream)
            stream.flush()
            profile = MachineProfile.load(stream.name)
        self.assertEqual(profile.layout_cases[0].columns, 16)
        self.assertEqual(profile.to_json()["layout_model"]["cases"],
                         profile_json["layout_model"]["cases"])


class PlanWisdomTests(unittest.TestCase):
    def test_stage_layout_is_validated_and_defaults_compatibly(self):
        payload = {
            "schema_version": 1,
            "entries": [
                {"length": 64, "precision": "fp32", "radices": [8, 8]},
                {"length": 64, "precision": "fp64", "radices": [8, 8],
                 "stage_layout": "section"},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(payload, stream)
            stream.flush()
            entries, _, _ = load_entries(Path(stream.name))
        self.assertEqual([entry["stage_layout"] for entry in entries], [0, 1])

        payload["entries"][1]["stage_layout"] = "unknown"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(payload, stream)
            stream.flush()
            with self.assertRaisesRegex(ValueError, "invalid stage layout"):
                load_entries(Path(stream.name))


class EmissionTests(unittest.TestCase):
    def test_measured_layout_efficiency_is_emitted_for_plan_search(self):
        base = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        profile = replace(base, layout_cases=(
            LayoutCase("linear_transpose", 8 * 1024 * 1024, 16, 2, 0.5),
            LayoutCase("blocked_transpose", 8 * 1024 * 1024, 16, 2, 0.75),
        ))
        with tempfile.TemporaryDirectory() as output:
            emit(profile, output, (2,))
            registry = (Path(output) /
                        "mofft_kernel_registry.c").read_text()
            header = (Path(output) / "mofft_generated_kernels.h").read_text()
        self.assertIn("best_relative_bandwidth = 0.5", registry)
        self.assertIn("best_relative_bandwidth = 0.75", registry)
        self.assertIn("mofft_generated_transpose_cost", header)

    def test_batch_pipeline_only_changes_direct_input_lowering(self):
        candidate = build_candidate(
            16, "fp64", "other", "forward", Pattern.LIKE_TERMS,
            batch_pipeline_depth=2)
        direct = emit_candidate_source(
            candidate, "test_pipeline", "direct_broadcast")
        normal = emit_candidate_source(candidate, "test_normal", "normal")
        self.assertIn("group_offset += 2 * vl", direct)
        self.assertIn("pg_cols_b", direct)
        self.assertNotIn("pg_cols_b", normal)

    def test_repeat_wisdom_can_select_batch_pipeline(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        wisdom = {
            "compiler_sha256": _compiler_digest(),
            "profile_sha256": profile_digest(profile),
            "entries": [{
                "radix": 16, "precision": "fp64", "direction": "forward",
                "stage": "other", "context": "direct_broadcast",
                "batch": 256, "locality_repeat": 64,
                "pattern": Pattern.LIKE_TERMS.value,
                "matrix_vg_width": 4, "rotate_temp_tiles": False,
                "batch_pipeline_depth": 2, "median_nanoseconds": 1.0,
            }],
        }
        with tempfile.TemporaryDirectory() as output:
            manifest = emit(profile, output, (16,), wisdom)
            source = (Path(output) / "mofft_kernels_fp64.c").read_text()
        selection = next(item for item in manifest["selections"]
                         if item["precision"] == "fp64" and
                         item["direction"] == "forward" and
                         item["stage"] == "other")
        variant = selection["bucket_variants"][
            "direct_broadcast_default_rmedium"]
        self.assertEqual(variant["batch_pipeline_depth"], 2)
        self.assertIn("mofft_kernel_r16_fp64_fwd_other_"
                      "direct_broadcast_rmedium", source)
        self.assertIn("pg_cols_b", source)

    def test_broadcast_validator_uses_requested_locality(self):
        candidate = build_candidate(8, "fp64", "other", "forward",
                                    Pattern.DIRECT)
        broadcast = _wrapper(candidate, "test_kernel", "broadcast")
        direct = _wrapper(candidate, "test_kernel", "direct_broadcast")
        normal = _wrapper(candidate, "test_kernel", "normal")
        self.assertIn("test_kernel(input, batch, 0, twiddles, batch", normal)
        self.assertIn("test_kernel(input, batch, 0, twiddles", broadcast)
        self.assertIn("batch / repeat, repeat", broadcast)
        self.assertIn("test_kernel(input, repeat, repeat", direct)
        self.assertIn("batch / repeat, repeat", direct)
        self.assertNotIn("batch >= 64 ? 32", broadcast)

    def test_validator_harness_reports_repeat_and_accepts_runner_cli(self):
        harness = _harness([])
        self.assertIn("argc>=2 && argc<=4", harness)
        self.assertIn("benchmark_batch%benchmark_repeat", harness)
        self.assertIn('printf("benchmark,%s,%zu,%.3f', harness)

    def test_context_wisdom_overrides_only_the_measured_variant(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        ranked = rank(enumerate_candidates(
            15, "fp64", "other", "forward"), profile)
        forced = ranked[-1][0]
        wisdom = {
            "compiler_sha256": _compiler_digest(),
            "profile_sha256": profile_digest(profile),
            "formal_measurement": False,
            "entries": [{
                "radix": 15, "precision": "fp64", "direction": "forward",
                "stage": "other", "context": "broadcast",
                "pattern": forced.pattern.value,
                "matrix_vg_width": forced.matrix_vg_width,
                "rotate_temp_tiles": forced.rotate_temp_tiles,
                "median_nanoseconds": 1.0,
            }, {
                "radix": 15, "precision": "fp64", "direction": "forward",
                "stage": "other", "context": "broadcast", "batch": 16,
                "pattern": forced.pattern.value,
                "matrix_vg_width": forced.matrix_vg_width,
                "rotate_temp_tiles": forced.rotate_temp_tiles,
                "median_nanoseconds": 0.5,
            }],
        }
        with tempfile.TemporaryDirectory() as output:
            manifest = emit(profile, output, (15,), wisdom)
            registry = (Path(output) / "mofft_kernel_registry.c").read_text()
        selection = next(item for item in manifest["selections"] if
                         item["precision"] == "fp64" and
                         item["direction"] == "forward" and
                         item["stage"] == "other")
        self.assertEqual(selection["selection_reason"], "analytical-model")
        variant = selection["variants"]["broadcast"]
        self.assertEqual(variant["selection_reason"],
                         "empirical-kernel-wisdom")
        self.assertEqual((variant["pattern"], variant["matrix_vg_width"]),
                         (forced.pattern.value, forced.matrix_vg_width))
        self.assertEqual(variant["rotate_temp_tiles"],
                         forced.rotate_temp_tiles)
        small = selection["bucket_variants"]["broadcast_small"]
        self.assertEqual(small["max_batch"], 32)
        self.assertEqual(small["selection_reason"],
                         "empirical-kernel-wisdom")
        self.assertIn("mofft_kernel_r15_fp64_fwd_other_broadcast_small",
                      registry)
        self.assertIn("batch <= 32", registry)
        self.assertNotIn("mofft_kernel_r15_fp64_fwd_first_small", registry)

    def test_stale_kernel_wisdom_is_rejected(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "different compiler"):
                emit(profile, output, (15,), {
                    "compiler_sha256": "0" * 64,
                    "profile_sha256": profile_digest(profile), "entries": []})

    def test_emitter_rejects_a_profile_without_a_backend(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        unsupported = replace(profile, architecture="x86_64+amx")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "no matrix backend"):
                emit(unsupported, output, (2,))

    def test_direct_input_repeat_wisdom_emits_locality_dispatch(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        ranked = rank(enumerate_candidates(
            8, "fp64", "other", "forward"), profile)
        forced = ranked[-1][0]
        common = {
            "radix": 8, "precision": "fp64", "direction": "forward",
            "stage": "other", "batch": 512,
            "pattern": forced.pattern.value,
            "matrix_vg_width": forced.matrix_vg_width,
            "rotate_temp_tiles": forced.rotate_temp_tiles,
            "median_nanoseconds": 1.0,
        }
        wisdom = {
            "compiler_sha256": _compiler_digest(),
            "profile_sha256": profile_digest(profile),
            "entries": [
                dict(common, context=context, locality_repeat=repeat)
                for context in ("broadcast", "direct_broadcast")
                for repeat in (64, 512, 8192)
            ],
        }
        with tempfile.TemporaryDirectory() as output:
            manifest = emit(profile, output, (8,), wisdom)
            registry = (Path(output) / "mofft_kernel_registry.c").read_text()
        selection = next(item for item in manifest["selections"] if
                         item["precision"] == "fp64" and
                         item["direction"] == "forward" and
                         item["stage"] == "other")
        variants = selection["bucket_variants"]
        self.assertIn("direct_broadcast_default_rmedium", variants)
        self.assertIn("direct_broadcast_default_rlarge", variants)
        self.assertIn("direct_broadcast_default_rhuge", variants)
        self.assertIn("broadcast_default_rmedium", variants)
        self.assertIn("broadcast_default_rlarge", variants)
        self.assertIn("broadcast_default_rhuge", variants)
        self.assertIn("input_repeat > 32 && input_repeat <= 256", registry)
        self.assertIn("input_repeat <= 4096", registry)
        self.assertIn("tw_repeat > 32 && tw_repeat <= 256", registry)
        self.assertIn("tw_repeat <= 4096", registry)
        self.assertIn("mofft_kernel_r8_fp64_fwd_other_"
                      "direct_broadcast_rmedium", registry)
        self.assertIn("mofft_kernel_r8_fp64_fwd_other_"
                      "direct_broadcast_rlarge", registry)
        self.assertIn("mofft_kernel_r8_fp64_fwd_other_"
                      "direct_broadcast_rhuge", registry)

    def test_mismatched_kernel_wisdom_profile_is_rejected(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "different profile"):
                emit(profile, output, (15,), {
                    "compiler_sha256": _compiler_digest(),
                    "profile_sha256": "0" * 64, "entries": []})

    def test_selection_and_emission_are_deterministic(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ma = emit(profile, a, (2, 3, 16))
            mb = emit(profile, b, (2, 3, 16))
            self.assertEqual(ma["input_sha256"], mb["input_sha256"])
            self.assertEqual(ma["schema_version"], 4)
            for name in ("mofft_generated_kernels.h", "mofft_kernels_fp32.c",
                         "mofft_kernels_fp64.c", "mofft_kernel_registry.c",
                         "manifest.json"):
                self.assertEqual((Path(a) / name).read_bytes(),
                                 (Path(b) / name).read_bytes())
            self.assertEqual(len(ma["selections"]), 3 * 2 * 2 * 2)
            self.assertEqual(ma["emitted_kernel_count"], 3 * 2 * 2 * 5)
            self.assertEqual(len(ma["compiler_sha256"]), 64)
            for selection in ma["selections"]:
                if selection["stage"] == "other":
                    self.assertEqual(len(selection["broadcast_emission_sha256"]),
                                     64)
                    self.assertEqual(len(selection[
                        "direct_broadcast_emission_sha256"]), 64)
                else:
                    self.assertIsNone(selection["broadcast_emission_sha256"])
                    self.assertIsNone(selection[
                        "direct_broadcast_emission_sha256"])
                self.assertEqual(sorted(selection["emitted_node_ids"]),
                                 list(range(selection["schedule"]["nodes"])))
                expected_tiles = list(range(2 * min(
                    (selection["radix"] + (16 if selection["precision"] == "fp32" else 8) - 1) //
                    (16 if selection["precision"] == "fp32" else 8),
                    2 if selection["precision"] == "fp32" else 4)))
                self.assertEqual(selection["schedule"]["matrix_tiles"],
                                 expected_tiles)

    def test_vector_reuse_reduces_radix32_fp64_table(self):
        candidates = enumerate_candidates(32, "fp64", "first", "forward")
        candidate = next(item for item in candidates
                         if item.pattern == Pattern.LIKE_TERMS_VECTOR_REUSE)
        self.assertEqual((candidate.coefficient_rows,
                          candidate.coefficient_columns), (17, 8))
        self.assertIn("coeff_reconstruct", candidate.graph.op_counts())
        direct = next(item for item in candidates
                      if item.pattern == Pattern.DIRECT)
        self.assertLess(candidate.graph.op_counts()["load"],
                        direct.graph.op_counts()["load"])

    def test_vg_intrinsics_are_real_lowerings_with_fp64_lane_mapping(self):
        candidate = build_candidate(16, "fp64", "other", "forward",
                                    Pattern.DIRECT, 4)
        source = emit_candidate_source(candidate, "forced_vg4")
        self.assertIn("svmla_za64_f64_vg1x4", source)
        self.assertIn("svread_hor_za64_f64_vg4", source)
        # FP64 VG4 packs two logical complex inputs into each ZA slice.
        self.assertIn("svget4_f64", source)
        self.assertRegex(source, r"svget4_f64\([^,]+, 3\)")

    def test_full_coefficient_rows_use_group_loads(self):
        candidate = build_candidate(64, "fp64", "first", "forward",
                                    Pattern.DIRECT)
        source = emit_candidate_source(candidate, "forced_load_vg4")
        self.assertIn("svld1_f64_x2", source)
        self.assertNotIn("svld1_f64_x4", source)

    def test_bootstrap_profile_only_selects_connected_lowerings(self):
        profile = MachineProfile.load(ROOT / "profiles/apple-m5-bootstrap.json")
        for radix in (3, 15, 16, 64):
            selected = rank(enumerate_candidates(radix, "fp32", "first", "forward"),
                            profile)[0][0]
            self.assertTrue(selected.emittable)


if __name__ == "__main__":
    unittest.main()
