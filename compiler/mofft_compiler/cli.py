from __future__ import annotations

import argparse
import json
from pathlib import Path

from .emitter import RADICES, emit
from .model import rank
from .patterns import enumerate_candidates
from .profile import MachineProfile
from .scheduler import schedule


def _radices(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    invalid = sorted(set(values) - set(RADICES))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported radices: {invalid}")
    return values


def generate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate MoFFT SME kernels")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel-wisdom")
    parser.add_argument("--temp-tile-rotation", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--radices", type=_radices,
                        default=",".join(str(x) for x in RADICES))
    args = parser.parse_args(argv)
    radices = args.radices if isinstance(args.radices, tuple) else _radices(args.radices)
    kernel_wisdom = (json.loads(Path(args.kernel_wisdom).read_text())
                      if args.kernel_wisdom else None)
    manifest = emit(MachineProfile.load(args.profile), args.output, radices,
                    kernel_wisdom, args.temp_tile_rotation)
    print(json.dumps({
        # CLI output is often captured in experiment logs.  Avoid recording a
        # developer's absolute workspace path in otherwise shareable output.
        "output": Path(args.output).name or ".",
        "profile": manifest["profile"]["name"],
        "kernels": manifest["emitted_kernel_count"],
        "input_sha256": manifest["input_sha256"],
    }, indent=2))


def inspect_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect MoFFT expression selection")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--radix", required=True, type=int, choices=RADICES)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--stage", choices=("first", "other"), default="first")
    parser.add_argument("--direction", choices=("forward", "backward"), default="forward")
    parser.add_argument("--temp-tile-rotation", choices=("auto", "on", "off"),
                        default="auto")
    args = parser.parse_args(argv)
    profile = MachineProfile.load(args.profile)
    ranked = rank(enumerate_candidates(args.radix, args.precision,
                                       args.stage, args.direction,
                                       args.temp_tile_rotation), profile)
    payload = []
    for candidate, cost in ranked:
        scheduled = schedule(candidate.graph)
        payload.append({
            "pattern": candidate.pattern.value,
            "matrix_vg_width": candidate.matrix_vg_width,
            "rotate_temp_tiles": candidate.rotate_temp_tiles,
            "cost_unit": profile.cost_unit,
            "total_cost": cost.total_cost,
            "matrix_cost": cost.matrix_cost,
            "post_cost": cost.post_cost,
            "register_pressure_cost": cost.register_pressure_cost,
            "compute_cost": cost.compute_cost,
            "memory_cost": cost.memory_cost,
            "working_set_bytes": cost.working_set_bytes,
            "cache_footprint_bytes": cost.cache_footprint_bytes,
            "cache_line_utilization": cost.cache_line_utilization,
            "traffic_bytes": cost.traffic_bytes,
            "arithmetic_intensity": cost.arithmetic_intensity,
            "memory_level": cost.memory_level,
            "bottlenecks": cost.bottlenecks,
            "ops": candidate.graph.op_counts(),
            "graph_sha256": candidate.graph.digest(),
            "schedule_rounds": max(item.round for item in scheduled) + 1,
            "spills": sum(item.spill for item in scheduled),
        })
    print(json.dumps({"profile": profile.name,
                      "cost_unit": profile.cost_unit,
                      "selected": {"pattern": payload[0]["pattern"],
                                   "matrix_vg_width": payload[0]["matrix_vg_width"],
                                   "rotate_temp_tiles": payload[0]["rotate_temp_tiles"]},
                      "candidates": payload}, indent=2))


if __name__ == "__main__":
    generate_main()
