"""Generate precompiled, model-selected kernel alternatives for binaries.

The ordinary :func:`mofft_compiler.emitter.emit` path remains the default
kernel bank.  This module emits three additional, namespaced banks so a
runtime planner can compare code families without running a compiler on the
target device.  These are model-forced alternatives, not measured wisdom.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from . import __version__
from .backends import ARM_SME_BACKEND, backend_for_architecture
from .emitter import (DIRECTIONS, PRECISIONS, RADICES, STAGES,
                      _actual_vg_stats, _compiler_digest, _declaration,
                      _emit_kernel, _emit_registry, _kernel_name,
                      profile_digest)
from .ir import Op
from .model import rank
from .patterns import enumerate_candidates
from .profile import MachineProfile
from .scheduler import schedule


_BANK_FAMILIES = (
    (1, "like_terms", lambda c: c.like_terms),
    (2, "even_odd", lambda c: c.even_odd),
    (3, "vector_reuse", lambda c: c.vector_reuse),
)


def _radices(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    invalid = sorted(set(values) - set(RADICES))
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"radices must be selected from {RADICES}; invalid: {invalid}")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("radices must not contain duplicates")
    return values


def _namespace_symbols(text: str, bank: int) -> str:
    """Prefix generated function identifiers while retaining public types."""
    text = re.sub(
        r"\bmofft_(?!complex_)(?!generated_kernels\.h)([A-Za-z_]\w*)",
        rf"mofft_bank{bank}_\1", text)
    return text


def _header(declarations: list[str], bank: int) -> str:
    # This mirrors the default generated header's declarations, but uses a
    # bank-specific guard so the portfolio wrapper can include all banks in
    # one translation unit.  Public complex types are shared via mofft.h.
    header = """#ifndef MOFFT_GENERATED_KERNELS_H
#define MOFFT_GENERATED_KERNELS_H
#include <stddef.h>
#include <arm_sme.h>
#include "mofft.h"
""" + "\n".join(declarations) + """
__attribute__((aligned(128)))
void mofft_dispatch_first_fp32(int, int, const mofft_complex_f32 *, size_t,
                               mofft_complex_f32 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_first_fp64(int, int, const mofft_complex_f64 *, size_t,
                               mofft_complex_f64 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_first_transposed_fp32(
    int, int, const mofft_complex_f32 *, size_t,
    mofft_complex_f32 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_first_transposed_fp64(
    int, int, const mofft_complex_f64 *, size_t,
    mofft_complex_f64 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_other_fp32(int, int, const mofft_complex_f32 *, size_t,
                               size_t, const mofft_complex_f32 *, size_t,
                               size_t, mofft_complex_f32 *, size_t) __arm_streaming __arm_inout("za");
__attribute__((aligned(128)))
void mofft_dispatch_other_fp64(int, int, const mofft_complex_f64 *, size_t,
                               size_t, const mofft_complex_f64 *, size_t,
                               size_t, mofft_complex_f64 *, size_t) __arm_streaming __arm_inout("za");
double mofft_generated_radix_cost(int radix, int precision_bits);
double mofft_generated_radix_cost_stage(int radix, int precision_bits,
                                        int first_stage);
double mofft_generated_memory_cost(size_t working_set_bytes,
                                   size_t read_bytes, size_t write_bytes);
double mofft_generated_transpose_cost(size_t working_set_bytes,
                                      size_t data_bytes, size_t columns,
                                      size_t batch, size_t complex_bytes,
                                      int blocked);
size_t mofft_generated_cache_line_bytes(void);
#endif
"""
    header = _namespace_symbols(header, bank)
    return header.replace("MOFFT_GENERATED_KERNELS_H",
                          f"MOFFT_BANK{bank}_GENERATED_KERNELS_H")


def _portfolio_wrapper_header() -> str:
    lines = [
        "#ifndef MOFFT_KERNEL_PORTFOLIO_H",
        "#define MOFFT_KERNEL_PORTFOLIO_H",
        '#include "mofft_generated_kernels.h"',
        "unsigned mofft_generated_kernel_bank_count(void);",
        "",
    ]
    for precision in PRECISIONS:
        t = "mofft_complex_f32" if precision == "fp32" else "mofft_complex_f64"
        for stage in ("first", "first_transposed"):
            lines.extend([
                "__attribute__((aligned(128)))",
                f"void mofft_portfolio_dispatch_{stage}_{precision}(unsigned bank, int radix, int direction,",
                f"    const {t} *input, size_t stride, {t} *output, size_t batch)",
                '__arm_streaming __arm_inout("za");',
            ])
        lines.extend([
            "__attribute__((aligned(128)))",
            f"void mofft_portfolio_dispatch_other_{precision}(unsigned bank, int radix, int direction,",
            f"    const {t} *input, size_t stride, size_t input_repeat,",
            f"    const {t} *twiddles, size_t twiddle_stride, size_t twiddle_repeat,",
            f"    {t} *output, size_t batch) __arm_streaming __arm_inout(\"za\");",
        ])
        lines.append("")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def _portfolio_wrapper_source() -> str:
    lines = [
        '#include "mofft_kernel_portfolio.h"',
        '#include "bank1/mofft_generated_kernels.h"',
        '#include "bank2/mofft_generated_kernels.h"',
        '#include "bank3/mofft_generated_kernels.h"',
        "",
        "unsigned mofft_generated_kernel_bank_count(void) { return 4U; }",
        "",
    ]
    for precision in PRECISIONS:
        for stage in ("first", "first_transposed"):
            suffix = f"dispatch_{stage}_{precision}"
            args = "int radix, int direction, const mofft_complex_{p} *input, size_t stride, mofft_complex_{p} *output, size_t batch".format(
                p="f32" if precision == "fp32" else "f64")
            call_args = "radix, direction, input, stride, output, batch"
            lines.extend([
                '__attribute__((target("sme")))',
                f"void mofft_portfolio_{suffix}(unsigned bank, {args}) __arm_streaming __arm_inout(\"za\") {{",
                "  switch (bank) {",
                f"    case 1: mofft_bank1_{suffix}({call_args}); return;",
                f"    case 2: mofft_bank2_{suffix}({call_args}); return;",
                f"    case 3: mofft_bank3_{suffix}({call_args}); return;",
                f"    default: mofft_{suffix}({call_args}); return;",
                "  }",
                "}",
                "",
            ])
        suffix = f"dispatch_other_{precision}"
        p = "f32" if precision == "fp32" else "f64"
        lines.extend([
            '__attribute__((target("sme")))',
            f"void mofft_portfolio_{suffix}(unsigned bank, int radix, int direction,",
            f"    const mofft_complex_{p} *input, size_t stride, size_t input_repeat,",
            f"    const mofft_complex_{p} *twiddles, size_t twiddle_stride, size_t twiddle_repeat,",
            f"    mofft_complex_{p} *output, size_t batch) __arm_streaming __arm_inout(\"za\") {{",
            "  switch (bank) {",
            f"    case 1: mofft_bank1_{suffix}(radix, direction, input, stride, input_repeat, twiddles, twiddle_stride, twiddle_repeat, output, batch); return;",
            f"    case 2: mofft_bank2_{suffix}(radix, direction, input, stride, input_repeat, twiddles, twiddle_stride, twiddle_repeat, output, batch); return;",
            f"    case 3: mofft_bank3_{suffix}(radix, direction, input, stride, input_repeat, twiddles, twiddle_stride, twiddle_repeat, output, batch); return;",
            f"    default: mofft_{suffix}(radix, direction, input, stride, input_repeat, twiddles, twiddle_stride, twiddle_repeat, output, batch); return;",
            "  }",
            "}",
            "",
        ])
    return "\n".join(lines)


def _generate_bank(profile: MachineProfile, output: Path,
                   radices: tuple[int, ...], bank: int, family: str,
                   predicate, temp_tile_rotation: str) -> dict:
    selected_backend = backend_for_architecture(profile.architecture)
    if selected_backend is not ARM_SME_BACKEND:
        raise ValueError(
            f"{selected_backend.name} requires its own portfolio source generator")
    target = selected_backend.target(profile)
    output.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[str]] = {precision: [] for precision in PRECISIONS}
    declarations: list[str] = []
    costs: dict[tuple[int, str, str], float] = {}
    selections: list[dict] = []
    kernel_count = 0

    # Match default emit() ordering so alternatives with identical code keys
    # retain the established instruction-cache layout.
    emission_radices = tuple(r for r in radices if r & (r - 1) == 0) + tuple(
        r for r in radices if r & (r - 1) != 0)

    for precision in PRECISIONS:
        for direction in DIRECTIONS:
            for radix in emission_radices:
                for stage in STAGES:
                    candidates = enumerate_candidates(
                        radix, precision, stage, direction,
                        temp_tile_rotation, target)
                    ranked = rank(candidates, profile)
                    if not ranked:
                        raise ValueError(
                            f"no emittable candidates for r{radix}/{precision}/"
                            f"{direction}/{stage}")
                    family_candidates = [item for item in ranked
                                         if predicate(item[0])]
                    if family_candidates:
                        selected, cost = family_candidates[0]
                        reason = "model-forced-portfolio-family"
                    else:
                        selected, cost = ranked[0]
                        reason = "model-forced-portfolio-fallback"

                    normal_name = _kernel_name(radix, precision, direction,
                                               stage)
                    normal_schedule = schedule(selected.graph)
                    normal_source = _emit_kernel(selected, normal_schedule)
                    sources[precision].append(normal_source)
                    declarations.append(_declaration(selected))
                    kernel_count += 1
                    variants: dict[str, str] = {}

                    if stage == "first":
                        context_name = normal_name + "_t"
                        context_source = _emit_kernel(
                            selected, normal_schedule, context_name,
                            transposed_output=True,
                            coefficient_owner=normal_name)
                        sources[precision].append(context_source)
                        declarations.append(_declaration(selected,
                                                         context_name))
                        variants["transposed"] = sha256(
                            context_source.encode()).hexdigest()
                        kernel_count += 1
                    else:
                        for context, lowering in (
                                ("broadcast", {"twiddle_broadcast": True}),
                                ("direct_broadcast", {
                                    "twiddle_broadcast": True,
                                    "direct_input": True})):
                            context_name = normal_name + "_" + context
                            context_source = _emit_kernel(
                                selected, normal_schedule, context_name,
                                coefficient_owner=normal_name,
                                **lowering)
                            sources[precision].append(context_source)
                            declarations.append(_declaration(
                                selected, context_name))
                            variants[context] = sha256(
                                context_source.encode()).hexdigest()
                            kernel_count += 1

                    stats = _actual_vg_stats(selected, normal_schedule)
                    schedule_data = {
                        "nodes": len(normal_schedule),
                        "rounds": max((item.round for item in normal_schedule),
                                      default=0) + 1,
                        "spills": sum(item.spill for item in normal_schedule),
                        "vg2": stats["vg2"],
                        "vg4": stats["vg4"],
                        "matrix_tiles": sorted({
                            item.matrix_tile for item in normal_schedule
                            if item.node.op == Op.OUTER_PRODUCT_ACCUM}),
                    }
                    selections.append({
                        "radix": radix,
                        "precision": precision,
                        "direction": direction,
                        "stage": stage,
                        "portfolio_family": family,
                        "pattern": selected.pattern.value,
                        "matrix_vg_width": selected.matrix_vg_width,
                        "rotate_temp_tiles": selected.rotate_temp_tiles,
                        "batch_pipeline_depth": selected.batch_pipeline_depth,
                        "selection_reason": reason,
                        "measured_nanoseconds": None,
                        "cost": asdict(cost),
                        "graph_sha256": selected.graph.digest(),
                        "emission_sha256": sha256(
                            normal_source.encode()).hexdigest(),
                        "context_emission_sha256": variants,
                        "coefficient_shape": [selected.coefficient_rows,
                                              selected.coefficient_columns, 2],
                        "schedule": schedule_data,
                    })
                    key = (radix, precision, stage)
                    costs[key] = min(costs.get(key, float("inf")),
                                     cost.compute_cost)

    header = _header(declarations, bank)
    preamble = ('#include "mofft_generated_kernels.h"\n#include <stdint.h>\n'
                '#include <arm_sve.h>\n#include <arm_sme.h>\n')
    for precision, chunks in sources.items():
        body = preamble + "\n".join(chunks)
        (output / f"mofft_kernels_{precision}.c").write_text(
            _namespace_symbols(body, bank))
    (output / "mofft_generated_kernels.h").write_text(header)
    (output / "mofft_kernel_registry.c").write_text(
        _namespace_symbols(_emit_registry(radices, costs, set(), profile),
                           bank))

    manifest = {
        "schema_version": 1,
        "generator_version": __version__,
        "compiler_sha256": _compiler_digest(),
        "profile_name": profile.name,
        "profile_sha256": profile_digest(profile),
        "profile_measured": bool(profile.measured),
        "cost_unit": profile.cost_unit,
        "bank": bank,
        "family": family,
        "selection_policy": "best-ranked legal candidate in requested family; model winner when family is target-illegal",
        "selection_source": "analytical model; no target kernel measurements used",
        "kernel_wisdom": None,
        "radices": list(radices),
        "temp_tile_rotation": temp_tile_rotation,
        "emitted_kernel_count": kernel_count,
        "selections": selections,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["input_sha256"] = sha256(canonical.encode()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def generate_portfolio(profile: MachineProfile, output: str | Path,
                       radices: tuple[int, ...] = RADICES,
                       temp_tile_rotation: str = "auto") -> dict:
    """Generate the three namespaced alternative banks and wrapper API.

    The caller separately invokes the ordinary generator in ``output`` to
    create bank 0.  The returned summary contains no filesystem paths.
    """
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_radices = tuple(radices)
    invalid = sorted(set(normalized_radices) - set(RADICES))
    if not normalized_radices or invalid:
        raise ValueError(f"unsupported or empty radix set: {invalid}")
    if len(set(normalized_radices)) != len(normalized_radices):
        raise ValueError("radices must not contain duplicates")
    if temp_tile_rotation not in ("auto", "on", "off"):
        raise ValueError(f"unknown temporary-tile rotation mode: "
                         f"{temp_tile_rotation}")

    manifests = []
    for bank, family, predicate in _BANK_FAMILIES:
        manifests.append(_generate_bank(
            profile, output_path / f"bank{bank}", normalized_radices,
            bank, family, predicate, temp_tile_rotation))
    (output_path / "mofft_kernel_portfolio.h").write_text(
        _portfolio_wrapper_header())
    (output_path / "mofft_kernel_portfolio.c").write_text(
        _portfolio_wrapper_source())
    return {
        "banks": len(manifests) + 1,
        "alternative_kernel_count": sum(
            item["emitted_kernel_count"] for item in manifests),
        "bank_manifests": manifests,
    }


def generate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate precompiled MoFFT kernel portfolio banks")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radices", type=_radices,
                        default=",".join(str(x) for x in RADICES))
    parser.add_argument("--temp-tile-rotation",
                        choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args(argv)
    profile = MachineProfile.load(args.profile)
    radices = (args.radices if isinstance(args.radices, tuple)
               else _radices(args.radices))
    summary = generate_portfolio(profile, args.output, radices,
                                 args.temp_tile_rotation)
    print(json.dumps({
        "output": Path(args.output).name or ".",
        "profile": profile.name,
        "banks": summary["banks"],
        "alternative_kernels": summary["alternative_kernel_count"],
    }, indent=2))


if __name__ == "__main__":
    generate_main()

