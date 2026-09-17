from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median


RESOURCE = {
    "add": "sve-arithmetic-inferred",
    "sub": "sve-arithmetic-inferred",
    "fmla_v": "sve-fma-inferred",
    "fmopa": "sme-outer-product-inferred",
    "fmla_m_vg2": "sme2-matrix-fma-inferred",
    "fmla_m_vg4": "sme2-matrix-fma-inferred",
    "za_slice": "za-transfer-inferred",
    "streaming_transition": "streaming-mode-control-inferred",
}

PROFILE_NAME = {
    "add": "sve_add",
    "sub": "sve_sub",
    "fmla_v": "sve_fmla",
    "fmopa": "fmopa",
    "fmla_m_vg2": "sme2_fmla_vg2",
    "fmla_m_vg4": "sme2_fmla_vg4",
    "za_slice": "za_extract",
    "streaming_transition": "smstart_stop",
}


def _memory_hierarchy(raw: dict) -> dict:
    cases = raw.get("memory_cases", [])
    cache = raw.get("cache_info", {})
    l1_capacity = int(cache.get("l1d_bytes", 64 * 1024))
    l2_capacity = int(cache.get("l2_bytes", 4 * 1024 * 1024))

    def bandwidth(operation: str, capacity: int, fallback: float) -> float:
        eligible = [case for case in cases
                    if case.get("operation") == operation and
                    int(case["working_set_bytes"]) <= capacity]
        if not eligible:
            return fallback
        chosen = max(eligible, key=lambda case: int(case["working_set_bytes"]))
        # Instruction rates use operations/ns, so bytes/ns is the compatible
        # wall-clock unit for the model despite the historical "cycles" name.
        return float(chosen["bytes_per_second"]) / 1e9

    def mixed_bandwidth(capacity: int, read: float, write: float) -> float:
        measured = bandwidth("copy", capacity, 0.0)
        # Legacy calibration files have no copy stream. The harmonic bound is
        # explicit inference from the independently measured read/write rates.
        return measured if measured > 0.0 else 2.0 / (1.0 / read + 1.0 / write)

    largest = max((int(case["working_set_bytes"]) for case in cases),
                  default=1 << 30)
    l1_read = bandwidth("read", l1_capacity, 64.0)
    l1_write = bandwidth("write", l1_capacity, 48.0)
    l2_read = bandwidth("read", l2_capacity, 32.0)
    l2_write = bandwidth("write", l2_capacity, 24.0)
    dram_read = bandwidth("read", largest, 12.0)
    dram_write = bandwidth("write", largest, 10.0)
    levels = {
        "l1d": {
            "capacity_bytes": l1_capacity,
            "read_bandwidth_bytes_per_cost_unit": l1_read,
            "write_bandwidth_bytes_per_cost_unit": l1_write,
            "mixed_bandwidth_bytes_per_cost_unit":
                mixed_bandwidth(l1_capacity, l1_read, l1_write),
            "latency_cost": 4.0, "inferred": not bool(cases),
        },
        "l2": {
            "capacity_bytes": l2_capacity,
            "read_bandwidth_bytes_per_cost_unit": l2_read,
            "write_bandwidth_bytes_per_cost_unit": l2_write,
            "mixed_bandwidth_bytes_per_cost_unit":
                mixed_bandwidth(l2_capacity, l2_read, l2_write),
            "latency_cost": 16.0, "inferred": not bool(cases),
        },
        "dram": {
            "capacity_bytes": 1 << 62,
            "read_bandwidth_bytes_per_cost_unit": dram_read,
            "write_bandwidth_bytes_per_cost_unit": dram_write,
            "mixed_bandwidth_bytes_per_cost_unit":
                mixed_bandwidth(largest, dram_read, dram_write),
            "latency_cost": 100.0, "inferred": not bool(cases),
        },
    }
    # A lower memory tier cannot provide more sustainable bandwidth than the
    # tier above it in this model. Clamp noisy non-exclusive sweeps to preserve
    # that physical monotonicity while retaining the raw cases in the input.
    for operation in ("read_bandwidth_bytes_per_cost_unit",
                      "write_bandwidth_bytes_per_cost_unit",
                      "mixed_bandwidth_bytes_per_cost_unit"):
        levels["l2"][operation] = min(levels["l1d"][operation],
                                      levels["l2"][operation])
        levels["dram"][operation] = min(levels["l2"][operation],
                                        levels["dram"][operation])
    return {"cache_line_bytes": int(cache.get("line_bytes", 128)),
            "levels": levels}


def _layout_model(raw: dict) -> dict | None:
    layout_cases = raw.get("layout_cases", [])
    copy_cases = [case for case in raw.get("memory_cases", [])
                  if case.get("operation") == "copy"]
    if not layout_cases or not copy_cases:
        return None

    def closest_copy_bandwidth(working_set_bytes: int) -> float:
        chosen = min(
            copy_cases,
            key=lambda case: abs(math.log2(
                max(int(case["working_set_bytes"]), 1) /
                max(working_set_bytes, 1))))
        return float(chosen["bytes_per_second"])

    cases = []
    for case in layout_cases:
        operation = str(case["operation"])
        if operation not in {"linear_transpose", "blocked_transpose"}:
            raise ValueError(f"unknown layout operation {operation!r}")
        dimensions = (int(case["working_set_bytes"]), int(case["columns"]),
                      int(case["batch"]))
        if any(value <= 0 for value in dimensions):
            raise ValueError("layout dimensions must be positive")
        bandwidth = float(case["bytes_per_second"])
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("layout bandwidth must be positive and finite")
        reference = closest_copy_bandwidth(dimensions[0])
        # The ratio captures access-pattern efficiency while the memory model
        # supplies the bandwidth of the live cache/DRAM tier. Broad bounds
        # retain legitimate vectorization gains but reject broken counters.
        relative = min(2.0, max(0.05, bandwidth / max(reference, 1.0)))
        cases.append({
            "operation": operation,
            "working_set_bytes": dimensions[0],
            "columns": dimensions[1],
            "batch": dimensions[2],
            "relative_bandwidth": relative,
            "inferred": False,
        })
    return {
        "bandwidth_basis": "relative-to-sequential-copy",
        "cases": cases,
    }


def build_profile(raw: dict, raw_bytes: bytes, name: str) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for case in raw["cases"]:
        grouped.setdefault((case["operation"], case["precision"]), []).append(case)
    instructions = {}
    for (operation, precision), cases in sorted(grouped.items()):
        if operation.startswith("mixed_"):
            continue
        cases.sort(key=lambda x: x["ilp"])
        rates = [float(x["instructions_per_second"]) for x in cases]
        peak = max(rates)
        saturation = next(x["ilp"] for x, rate in zip(cases, rates) if rate >= .95 * peak)
        latency_proxy_ns = 1e9 / rates[0]
        base = PROFILE_NAME.get(operation, operation)
        key = base if precision == "mode" else f"{base}_{precision}"
        instructions[key] = {
            "throughput": peak / 1e9,
            "latency": latency_proxy_ns,
            "saturation_ilp": saturation,
            "resource": RESOURCE.get(operation, "unknown-inferred"),
            "inferred": True,
        }
    sharing = []
    for precision in ("fp32", "fp64"):
        add = grouped.get(("add", precision), [])
        fmopa = grouped.get(("fmopa", precision), [])
        mixed = grouped.get(("mixed_add_fmopa", precision), [])
        if add and fmopa and mixed:
            ra = max(float(x["instructions_per_second"]) for x in add)
            rm = max(float(x["instructions_per_second"]) for x in fmopa)
            observed = max(float(x["instructions_per_second"]) for x in mixed)
            independent = 8.0 / max(4.0 / ra, 4.0 / rm)
            shared = 8.0 / (4.0 / ra + 4.0 / rm)
            relation = "likely-shared" if abs(observed-shared) < abs(observed-independent) else "likely-independent"
            sharing.append({"pair": ["sve_add", "fmopa"], "precision": precision,
                            "inference": relation, "observed_instructions_per_second": observed,
                            "shared_prediction": shared, "independent_prediction": independent})
    result = {
        "schema_version": 3,
        "name": name,
        "architecture": "arm64+sme+sme2",
        "streaming_vector_bits": 512,
        "instructions": instructions,
        "source": (
            "mofft_microbench; timestamp="
            + datetime.now(timezone.utc).isoformat()
            + "; input_sha256="
            + hashlib.sha256(raw_bytes).hexdigest()
        ),
        "measured": True,
        "cost_unit": "nanoseconds",
        "memory_hierarchy": _memory_hierarchy(raw),
        "measurement_metadata": {
            "machine": raw.get("machine", "unknown"),
            "os": raw.get("os", "unknown"),
            "samples_per_case": raw.get("samples_per_case"),
            "statistics": "median",
            "formal_measurement": bool(raw.get("formal_measurement", False)),
            "load_average": raw.get("load_average"),
            "latency_definition": "ILP=1 reciprocal issue-rate proxy, not architectural latency",
            "resource_claims": "inferred; no unpublished Apple port number is claimed",
            "resource_sharing_inferences": sharing,
            "throughput_unit": "instructions/ns (wall-clock rate; relative cost input)",
            "memory_bandwidth_unit": "bytes/ns (wall-clock rate)",
            "memory_latency_note": "cache capacities are sysctl-reported when available; tier latency remains an explicit inferred penalty",
        },
    }
    layout_model = _layout_model(raw)
    if layout_model is not None:
        result["layout_model"] = layout_model
        result["measurement_metadata"]["layout_bandwidth_unit"] = (
            "ratio to sequential copy at the nearest working-set size")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MoFFT microbenchmark JSON to a profile")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="apple-m5")
    args = parser.parse_args()
    raw_bytes = args.input.read_bytes()
    profile = build_profile(json.loads(raw_bytes), raw_bytes, args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
