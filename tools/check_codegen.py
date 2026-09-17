from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


def disassemble(path: Path) -> str:
    return subprocess.run(["xcrun", "llvm-objdump", "-d", str(path)],
                          check=True, text=True, capture_output=True).stdout


def require(text: str, pattern: str, description: str) -> int:
    count = len(re.findall(pattern, text, re.MULTILINE))
    if count == 0:
        raise SystemExit(f"missing code-generation feature: {description}")
    return count


def forbid(text: str, pattern: str, description: str) -> None:
    if re.search(pattern, text, re.MULTILINE):
        raise SystemExit(f"unsupported code-generation feature: {description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that ACLE groups survive as SME2/SVE instructions")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--microbench", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    kernels = disassemble(args.archive)
    micro = disassemble(args.microbench)
    manifest = json.loads(args.manifest.read_text())
    if any(selection["schedule"]["spills"] for selection in manifest["selections"]):
        raise SystemExit("selected schedule contains modeled vector spills")
    selected_widths = {
        (selection["precision"], selection["matrix_vg_width"])
        for selection in manifest["selections"]
        if selection["stage"] == "other"
    }
    if any(not selection.get("broadcast_emission_sha256")
           for selection in manifest["selections"]
           if selection["stage"] == "other"):
        raise SystemExit("missing specialized repeated-twiddle broadcast kernel")
    if any(not selection.get("direct_broadcast_emission_sha256")
           for selection in manifest["selections"]
           if selection["stage"] == "other"):
        raise SystemExit("missing direct-input/broadcast other-stage kernel")
    # The current Apple M5 streaming-SVE implementation traps on gather loads
    # even though AppleClang accepts the ACLE intrinsic and emits an opcode.
    # Keep layout conversion and generated kernels inside the supported
    # contiguous-load subset until calibration proves otherwise.
    forbid(kernels, r"\bld1[wd]\b[^\n]*\[[^\]]*,\s*z\d+",
           "streaming-SVE gather load")
    result = {
        "fp32_fmla_m_vg4_probe": require(
            micro, r"\bfmla\s+za\.s\[[^\n]*vgx4", "FP32 FMLA-M VG4 probe"),
        "fp64_fmla_m_vg4_probe": require(
            micro, r"\bfmla\s+za\.d\[[^\n]*vgx4", "FP64 FMLA-M VG4 probe"),
        "fp32_fmla_m_vg2_probe": require(
            micro, r"\bfmla\s+za\.s\[[^\n]*vgx2", "FP32 FMLA-M VG2 probe"),
        "fp64_fmla_m_vg2_probe": require(
            micro, r"\bfmla\s+za\.d\[[^\n]*vgx2", "FP64 FMLA-M VG2 probe"),
        "grouped_fp64_load": require(
            kernels, r"\bld1d\s+\{\s*z\d+\.d,\s*z\d+\.d\s*\},\s*pn",
            "two-vector FP64 coefficient load"),
        "fmopa": require(kernels, r"\bfmopa\s+za", "FMOPA"),
        "fp32_vertical_za_read": require(
            kernels, r"\bmov\s+z\d+\.s[^\n]*za\d+v\.s",
            "FP32 vertical ZA read"),
        "fp64_vertical_za_read": require(
            kernels, r"\bmov\s+z\d+\.d[^\n]*za\d+v\.d",
            "FP64 vertical ZA read"),
    }
    for precision, suffix in (("fp32", "s"), ("fp64", "d")):
        for width in (2, 4):
            if (precision, width) in selected_widths:
                result[f"selected_{precision}_fmla_m_vg{width}"] = require(
                    kernels, rf"\bfmla\s+za\.{suffix}\[[^\n]*vgx{width}",
                    f"selected {precision.upper()} FMLA-M VG{width} kernel")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
