#!/usr/bin/env python3
"""Extract one JSON object from devicectl console output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = args.input.read_text()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return
    raise ValueError(f"{args.input}: no complete JSON object found")


if __name__ == "__main__":
    main()
