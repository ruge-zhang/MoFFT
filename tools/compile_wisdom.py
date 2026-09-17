#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

RADICES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64}


def load_entries(path: Path | None) -> tuple[list[dict], bool, str]:
    if path is None:
        return [], False, "none"
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported wisdom schema")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("wisdom entries must be a list")
    seen: set[tuple[int, int]] = set()
    normalized = []
    for entry in entries:
        length = int(entry["length"])
        precision = {"fp32": 32, "fp64": 64}.get(entry["precision"])
        factors = tuple(int(value) for value in entry["radices"])
        product = 1
        for factor in factors:
            if factor not in RADICES:
                raise ValueError(f"unsupported wisdom radix {factor}")
            product *= factor
        if product != length or not factors:
            raise ValueError(f"wisdom factors do not multiply to {length}")
        key = (length, precision)
        if precision is None or key in seen:
            raise ValueError(f"invalid or duplicate wisdom entry {key}")
        transpose_name = entry.get("transpose_strategy", "linear")
        if transpose_name not in {"linear", "blocked", "mixed"}:
            raise ValueError(f"invalid transpose strategy {transpose_name!r}")
        if transpose_name == "mixed" and "blocked_transpose_mask" not in entry:
            raise ValueError("mixed transpose strategy requires an explicit mask")
        default_mask = (1 << (len(factors) - 1)) - 1 if transpose_name == "blocked" else 0
        transpose_mask = int(entry.get("blocked_transpose_mask", default_mask))
        if transpose_mask < 0 or transpose_mask >> max(0, len(factors) - 1):
            raise ValueError(f"invalid blocked transpose mask {transpose_mask}")
        seen.add(key)
        direct_input = entry.get("direct_input", True)
        if not isinstance(direct_input, bool):
            raise ValueError("direct_input must be a JSON boolean")
        stage_layout = entry.get("stage_layout", "transpose")
        if stage_layout not in {"transpose", "section"}:
            raise ValueError(f"invalid stage layout {stage_layout!r}")
        normalized.append({"length": length, "precision": precision,
                           "factors": factors,
                           "transpose_mask": transpose_mask,
                           "direct_input": int(direct_input),
                           "stage_layout": int(stage_layout == "section")})
    normalized.sort(key=lambda value: (value["length"], value["precision"]))
    # Generated sources and benchmark output may be shared.  Record only the
    # input filename so they do not disclose the build host's home directory.
    return normalized, bool(payload.get("formal_measurement")), path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile MoFFT plan wisdom")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--target-machine",
                        help="expected wisdom hw.model when cross-compiling")
    parser.add_argument("--output-c", required=True, type=Path)
    parser.add_argument("--output-h", required=True, type=Path)
    args = parser.parse_args()
    entries, formal, source = load_entries(args.input)
    if args.input is not None:
        if args.manifest is None:
            raise ValueError("a kernel manifest is required with wisdom")
        wisdom = json.loads(args.input.read_text())
        manifest = json.loads(args.manifest.read_text())
        expected_profile = manifest["profile"]["name"]
        expected_hash = manifest["input_sha256"]
        if wisdom.get("profile") != expected_profile:
            raise ValueError(
                f"wisdom profile {wisdom.get('profile')!r} does not match "
                f"generated profile {expected_profile!r}")
        if wisdom.get("kernel_input_sha256") != expected_hash:
            raise ValueError("wisdom was measured with different generated kernels")
        machine = args.target_machine or subprocess.run(
            ["sysctl", "-n", "hw.model"], check=True, capture_output=True,
            text=True).stdout.strip()
        if wisdom.get("machine") != machine:
            raise ValueError(
                f"wisdom machine {wisdom.get('machine')!r} does not match "
                f"build machine {machine!r}")
    args.output_h.write_text("""#ifndef MOFFT_GENERATED_WISDOM_H
#define MOFFT_GENERATED_WISDOM_H
#include <stddef.h>
#include <stdint.h>
size_t mofft_generated_wisdom_lookup(size_t length, int precision_bits,
                                     int *radices, size_t capacity,
                                     uint64_t *blocked_transpose_mask,
                                     int *direct_input,
                                     int *stage_layout);
int mofft_generated_wisdom_is_formal(void);
const char *mofft_generated_wisdom_source(void);
#endif
""")
    lines = [
        '#include "mofft_generated_wisdom.h"',
        "",
        "size_t mofft_generated_wisdom_lookup(size_t length, int precision_bits,",
        "                                     int *radices, size_t capacity,",
        "                                     uint64_t *blocked_transpose_mask,",
        "                                     int *direct_input,",
        "                                     int *stage_layout) {",
    ]
    for entry in entries:
        factors = entry["factors"]
        lines.append(
            f"  if (length == {entry['length']}u && precision_bits == "
            f"{entry['precision']}) {{")
        lines.append(f"    if (capacity < {len(factors)}u) return 0;")
        for index, factor in enumerate(factors):
            lines.append(f"    radices[{index}] = {factor};")
        lines.append(
            f"    *blocked_transpose_mask = UINT64_C({entry['transpose_mask']});")
        lines.append(f"    *direct_input = {entry['direct_input']};")
        lines.append(f"    *stage_layout = {entry['stage_layout']};")
        lines.append(f"    return {len(factors)}u;")
        lines.append("  }")
    lines.extend([
        "  (void)length; (void)precision_bits; (void)radices; (void)capacity;",
        "  (void)blocked_transpose_mask;",
        "  (void)direct_input;",
        "  (void)stage_layout;",
        "  return 0;",
        "}",
        "",
        f"int mofft_generated_wisdom_is_formal(void) {{ return {int(formal)}; }}",
        "",
        "const char *mofft_generated_wisdom_source(void) {",
        f"  return {json.dumps(source)};",
        "}",
        "",
    ])
    args.output_c.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
