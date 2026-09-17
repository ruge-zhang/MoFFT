#!/usr/bin/env python3
"""Plot one or two benchmark result files without bundling private data.

The default layout is the compact two-column view used by the original
command line interface.  ``--layout overall`` adds the four-column physical
platform view used by the README: FP64 powers of two, FP64 non-powers of two,
FP32 powers of two, and FP32 non-powers of two.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


GFLOPS_RE = re.compile(
    r"GFLOPS\s+(fp\d+)\s+(\S+)\s+n=(\d+)\s+gflops=([0-9.eE+-]+)"
)


def _pick_implementation(payload: dict, precision: str, candidates: tuple[str, ...]) -> str:
    available = {
        result["implementation"]
        for result in payload["results"]
        if result["precision"] == precision
    }
    for candidate in candidates:
        if candidate in available:
            return candidate
    names = ", ".join(sorted(available)) or "none"
    raise ValueError(
        f"no supported implementation for {precision}; available: {names}"
    )


def _plot_overall(payloads: list[dict], labels: list[str], output: Path) -> None:
    """Render the four-column M4/M5-style physical-platform figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    if not payloads:
        raise ValueError("--layout overall requires at least one --results file")

    columns = (
        ("fp64", False, "FP64, powers of two"),
        ("fp64", True, "FP64, non-powers"),
        ("fp32", False, "FP32, powers of two"),
        ("fp32", True, "FP32, non-powers"),
    )
    figure, axes = plt.subplots(
        len(payloads), len(columns),
        figsize=(10.6, 4.8 if len(payloads) == 2 else 2.7),
        constrained_layout=False,
        squeeze=False,
    )
    colors = {"mofft": "#B3574B", "fftw": "#627EAC"}
    markers = {"mofft": "D", "fftw": "^"}
    line_styles = {"mofft": "-", "fftw": "--"}
    handles = {}

    for row, (payload, label) in enumerate(zip(payloads, labels)):
        for col, (precision, non_power, title) in enumerate(columns):
            ax = axes[row, col]
            mofft = _pick_implementation(payload, precision, ("mofft", "MoFFT"))
            baseline_candidates = (
                ("fftw-3.3.11-neon", "fftw-3.3.11", "fftw-neon", "fftw")
                if precision == "fp32" else
                ("fftw-3.3.11", "fftw", "fftw-scalar")
            )
            fftw = _pick_implementation(payload, precision, baseline_candidates)
            # Keep only sizes for which both series are present.  This makes
            # independently captured M4/M5 files directly comparable while
            # still allowing a partial smoke run to be plotted.
            mofft_ns = {
                int(result["n"]): float(result["gflops"])
                for result in payload["results"]
                if result["precision"] == precision
                and result["implementation"] == mofft
            }
            fftw_ns = {
                int(result["n"]): float(result["gflops"])
                for result in payload["results"]
                if result["precision"] == precision
                and result["implementation"] == fftw
            }
            sizes = sorted(set(mofft_ns) & set(fftw_ns))
            sizes = [
                n for n in sizes
                if n != 144 and ((n & (n - 1)) != 0) == non_power
            ]
            if not sizes:
                ax.set_visible(False)
                continue
            for role, implementation in (("mofft", mofft), ("fftw", fftw)):
                handle = ax.plot(
                    np.arange(len(sizes)),
                    [
                        (mofft_ns if role == "mofft" else fftw_ns)[n]
                        for n in sizes
                    ],
                    label="MoFFT" if role == "mofft" else "FFTW 3.3.11",
                    color=colors[role], linestyle=line_styles[role],
                    linewidth=0.9, marker=markers[role], markersize=3.0,
                    markeredgecolor="black", markeredgewidth=0.25,
                )[0]
                handles[role] = handle
            all_values = [mofft_ns[n] for n in sizes] + [fftw_ns[n] for n in sizes]
            ymax = max(all_values)
            ax.set_ylim(0, max(1.0, ymax * 1.12))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
            ax.set_xticks(
                np.arange(len(sizes)), [str(n) for n in sizes],
                rotation=58, ha="right", rotation_mode="anchor",
            )
            ax.grid(axis="y", linestyle="--", linewidth=0.48,
                    color="#BEBEBE", alpha=0.78)
            ax.tick_params(direction="out", labelsize=7, pad=1.0)
            if row == 0:
                ax.set_title(title, pad=3.0, fontsize=8)
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("N", fontsize=8)
            if col == 0:
                ax.set_ylabel("GFLOPS", fontsize=8)
                ax.text(0.035, 0.955, f"({chr(97 + row)}) {label}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    order = [("mofft", "MoFFT"), ("fftw", "FFTW 3.3.11")]
    figure.legend(
        [handles[key] for key, _ in order if key in handles],
        [name for key, name in order if key in handles],
        loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.995), fontsize=8,
    )
    figure.subplots_adjust(
        left=0.055, right=0.995, top=0.90, bottom=0.18,
        wspace=0.12, hspace=0.10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", metadata={})
    plt.close(figure)


def load_results(path: Path) -> dict:
    text = path.read_text(errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        results = [
            {
                "precision": match.group(1),
                "implementation": match.group(2),
                "n": int(match.group(3)),
                "gflops": float(match.group(4)),
            }
            for match in GFLOPS_RE.finditer(text)
        ]
        if not results:
            raise ValueError(f"{path} is neither benchmark JSON nor a GFLOPS log")
        return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, action="append", required=True,
                        help="benchmark JSON or captured GFLOPS log; pass twice to compare platforms")
    parser.add_argument("--label", action="append",
                        help="panel label matching each --results argument")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("compact", "overall"), default="compact",
        help="plot layout; default preserves the original two-column view",
    )
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    if len(args.results) > 2:
        parser.error("at most two --results files are supported")
    labels = args.label or [path.stem for path in args.results]
    if len(labels) != len(args.results):
        parser.error("provide either zero labels or one --label per --results")
    payloads = [load_results(path) for path in args.results]
    if args.layout == "overall":
        _plot_overall(payloads, labels, args.output)
        return

    fig, axes = plt.subplots(len(payloads), 2, figsize=(12, 3.5 * len(payloads)),
                             constrained_layout=True, squeeze=False)
    for row, (payload, panel_label) in enumerate(zip(payloads, labels)):
        for col, precision in enumerate(("fp32", "fp64")):
            ax = axes[row, col]
            implementations = sorted({result["implementation"]
                                      for result in payload["results"]})
            for implementation in implementations:
                values = sorted(
                    (result for result in payload["results"]
                     if result["precision"] == precision and
                     result["implementation"] == implementation),
                    key=lambda result: result["n"])
                if values:
                    ax.plot([result["n"] for result in values],
                            [result["gflops"] for result in values], "o-",
                            label=implementation)
            ax.set(title=f"{panel_label} — {precision.upper()}", xscale="log",
                   xlabel="FFT length", ylabel="GFLOP/s")
            ax.grid(True, alpha=.25)
            ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
