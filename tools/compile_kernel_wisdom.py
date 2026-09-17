#!/usr/bin/env python3
"""Compile sharded candidate-runner logs into kernel wisdom JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

from mofft_compiler.emitter import _compiler_digest, profile_digest
from mofft_compiler.profile import MachineProfile


BENCHMARK = re.compile(
    r"^benchmark,r(\d+)/(fp(?:32|64))/(forward|backward)/"
    r"(first|other)/([^/]+)/vg([24])/rot([01])/"
    r"pipe([12])/"
    r"(normal|transposed|broadcast|direct_broadcast)"
    r",(\d+),([0-9.]+)$")


def measurement(value: str) -> tuple[int, Path]:
    try:
        batch_text, path_text = value.split(":", 1)
        batch = int(batch_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "measurement must be BATCH:LOG_PATH") from error
    if batch <= 0:
        raise argparse.ArgumentTypeError("measurement batch must be positive")
    return batch, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge iOS candidate benchmark logs into kernel wisdom")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--measurement", action="append", type=measurement,
                        required=True, metavar="BATCH:LOG_PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    samples: dict[tuple, list[float]] = {}
    source_logs = []
    for batch, path in args.measurement:
        # Preserve enough provenance to identify an input without embedding a
        # developer's absolute workspace path in a shareable wisdom file.
        source_logs.append(path.name)
        for raw_line in path.read_text(errors="replace").splitlines():
            match = BENCHMARK.match(raw_line.strip())
            if not match:
                continue
            (radix, precision, direction, stage, expression, width,
             rotate, pipeline_depth, context, repeat, nanoseconds) = match.groups()
            locality_repeat = (int(repeat) if context in
                               ("broadcast", "direct_broadcast") else 0)
            key = (batch, locality_repeat, int(radix), precision,
                   direction, stage, context, expression, int(width),
                   bool(int(rotate)), int(pipeline_depth))
            samples.setdefault(key, []).append(float(nanoseconds))
    if not samples:
        raise ValueError("no candidate benchmark records found")

    entries = []
    for key in sorted(samples):
        values = samples[key]
        if len(values) < 3 or len(values) % 3:
            raise ValueError(
                f"expected complete three-pass measurements for {key}; "
                f"found {len(values)}")
        (batch, locality_repeat, radix, precision, direction, stage, context,
         expression, width, rotate, pipeline_depth) = key
        entry = {
            "radix": radix,
            "precision": precision,
            "direction": direction,
            "stage": stage,
            "batch": batch,
            "context": context,
            "pattern": expression,
            "matrix_vg_width": width,
            "rotate_temp_tiles": rotate,
            "batch_pipeline_depth": pipeline_depth,
            "median_nanoseconds": statistics.median(values),
            "pass_nanoseconds": values,
        }
        if locality_repeat:
            entry["locality_repeat"] = locality_repeat
        entries.append(entry)

    profile = MachineProfile.load(args.profile)
    payload = {
        "schema_version": 4,
        "compiler_sha256": _compiler_digest(),
        "profile_sha256": profile_digest(profile),
        "formal_measurement": args.formal,
        "machine": args.machine,
        "profile": profile.name,
        "passes": 3,
        "pass_order": "forward,reverse,forward",
        "source_logs": source_logs,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
