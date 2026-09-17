#!/usr/bin/env python3
"""Merge independently measured plan-search shards with strict provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROVENANCE_FIELDS = (
    "schema_version",
    "formal_measurement",
    "machine",
    "generator_version",
    "profile",
    "kernel_input_sha256",
)


def load_payload(path: Path) -> dict:
    text = path.read_text()
    begin = text.find("WISDOM_JSON_BEGIN")
    if begin != -1:
        begin += len("WISDOM_JSON_BEGIN")
        end = text.find("WISDOM_JSON_END", begin)
        if end == -1:
            raise ValueError(f"{path}: missing WISDOM_JSON_END marker")
        text = text[begin:end]
    return json.loads(text)


def merge(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one input shard is required")
    payloads = [load_payload(path) for path in paths]
    reference = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        for field in PROVENANCE_FIELDS:
            if payload.get(field) != reference.get(field):
                raise ValueError(f"{path}: mismatched {field}")

    entries: dict[tuple[int, str], dict] = {}
    for path, payload in zip(paths, payloads):
        shard_entries = payload.get("entries")
        if not isinstance(shard_entries, list):
            raise ValueError(f"{path}: entries must be a list")
        for entry in shard_entries:
            key = (int(entry["length"]), str(entry["precision"]))
            if key in entries:
                raise ValueError(f"duplicate plan entry {key}")
            entries[key] = entry

    result = {field: reference.get(field) for field in PROVENANCE_FIELDS}
    # Filenames retain useful provenance without publishing local paths.
    result["source_shards"] = [path.name for path in paths]
    result["entries"] = [entries[key] for key in sorted(entries)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(merge(args.inputs), indent=2) + "\n")


if __name__ == "__main__":
    main()
