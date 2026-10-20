#!/usr/bin/env python3
"""Write a small, privacy-filtered description of an installed binary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def safe_profile_name(value: object) -> str:
    if not isinstance(value, str):
        return "custom"
    # Public package metadata must not copy arbitrary labels that could contain
    # a person's name, machine identifier, or local path. Retain only common
    # generic Apple target labels; report every other profile as custom.
    if re.fullmatch(
            r"apple-(?:m[0-9]+(?:-(?:bootstrap|local|measured))?|native-local|sme2)",
            value):
        return value
    return "custom"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sdk", choices=("macos", "ios"), required=True)
    parser.add_argument("--minimum-os")
    parser.add_argument("--build-id")
    parser.add_argument("--plan-wisdom", type=Path)
    parser.add_argument("--kernel-wisdom", type=Path)
    parser.add_argument("--kernel-bank-count", type=int, choices=(1, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.build_id and not re.fullmatch(r"[0-9a-f]{64}", args.build_id):
        parser.error("--build-id must be a lowercase SHA-256 digest")

    profile = json.loads(args.profile.read_text())
    # Read the manifest to ensure the metadata belongs to an actual generated
    # artifact, but intentionally expose no raw model data or compiler hashes.
    json.loads(args.manifest.read_text())
    plans = json.loads(args.plan_wisdom.read_text()) if args.plan_wisdom else {}
    kernels = json.loads(args.kernel_wisdom.read_text()) if args.kernel_wisdom else {}
    plan_entries = plans.get("entries", [])
    kernel_entries = kernels.get("entries", [])
    minimum_os = args.minimum_os or "unspecified"
    if minimum_os != "unspecified" and not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", minimum_os):
        minimum_os = "unspecified"
    sdk_name = "iphoneos" if args.sdk == "ios" else "macosx"
    sdk_result = subprocess.run(
        ["xcrun", "--sdk", sdk_name, "--show-sdk-version"],
        capture_output=True, text=True, check=False, timeout=10)
    sdk_version = sdk_result.stdout.strip() if sdk_result.returncode == 0 else "unknown"
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", sdk_version):
        sdk_version = "unknown"
    metadata = {
        "schema_version": 1,
        "target": "apple-arm64-sme2",
        "platform": {
            "name": args.sdk,
            "minimum_os": minimum_os,
            "sdk_version": sdk_version,
        },
        "profile": {
            "name": safe_profile_name(profile.get("name")),
            "measured": bool(profile.get("measured", False)),
            "formal": bool(profile.get("formal_measurement", False)),
        },
        "preset_plan_wisdom": {
            "available": bool(plan_entries),
            "formal": bool(plans.get("formal_measurement", False)),
            "entry_count": len(plan_entries) if isinstance(plan_entries, list) else 0,
        },
        "kernel_selection_wisdom": {
            "available": bool(kernel_entries),
            "entry_count": len(kernel_entries) if isinstance(kernel_entries, list) else 0,
        },
        "kernel_bank_count": args.kernel_bank_count,
    }
    if args.build_id:
        metadata["build_id"] = args.build_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

