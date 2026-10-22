#!/usr/bin/env python3
"""Replot accepted Linux CPU results; no FFT execution or new statistics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/lingsheng-linux-reference.json"
MANIFEST = ROOT / "benchmarks/results/lingsheng-linux-reference.manifest.json"
POWERS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
NONPOWERS = [169, 196, 225, 1728, 2197, 2744, 3375, 20736, 28561, 38416,
             50625, 248832, 371293, 537824, 759375]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_results():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = RESULTS.read_bytes()
    pin = manifest["files"][RESULTS.relative_to(ROOT).as_posix()]
    require(len(body) == pin["bytes"], "Result size differs from manifest")
    require(hashlib.sha256(body).hexdigest() == pin["sha256"], "Result hash mismatch")
    data = json.loads(body)
    require(data["run_id"] == "lingsheng-linux-512" and data["accepted"] is True, "Wrong/unaccepted run")
    require(data["sample_count"] == 3240, "Wrong total sample count")
    rows = data["results"]
    require(len(rows) == 108, "Expected 108 library/precision/length groups")
    expected = {(p, n, lib) for p in ("fp32", "fp64")
                for n in POWERS + [144] + NONPOWERS for lib in ("mofft", "fftw")}
    indexed = {}
    for row in rows:
        implementation = row["implementation"]
        require(implementation == "mofft" or implementation.startswith("fftw-3.3.11"),
                "Unexpected implementation")
        key = (row["precision"], row["n"], "mofft" if implementation == "mofft" else "fftw")
        require(key not in indexed, "Duplicate group")
        samples = row["raw_seconds"]
        require(len(samples) == 30 and all(math.isfinite(x) and x > 0 for x in samples),
                "Expected all 30 finite positive samples")
        require(0 <= row["cv"] <= .05, "Original 5% CV gate not satisfied")
        median = row["median_seconds"]
        lower, upper = row["ci95"]
        require(0 < lower <= median <= upper and math.isfinite(upper), "Invalid frozen interval")
        require(math.isclose(row["gflops"], 5 * row["n"] * math.log2(row["n"]) / median / 1e9,
                             rel_tol=2e-10), "Throughput differs from frozen median")
        indexed[key] = row
    require(set(indexed) == expected, "Incomplete precision/size/library coverage")
    return data, indexed


def plot(indexed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output.mkdir(parents=True, exist_ok=False)
    styles = [("mofft", "MoFFT", "#D55E00", "D", "-"),
              ("fftw", "FFTW 3.3.11", "#0072B2", "^", "--")]
    panels = [("fp64", POWERS, "FP64 · powers of two"),
              ("fp64", NONPOWERS, "FP64 · non-powers"),
              ("fp32", POWERS, "FP32 · powers of two"),
              ("fp32", NONPOWERS, "FP32 · non-powers")]
    plotted = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    }):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained")
        for panel, (ax, (precision, sizes, title)) in enumerate(zip(axes, panels), 1):
            for implementation, label, color, marker, line in styles:
                selected = []
                for n in sizes:
                    row = indexed[precision, n, implementation]
                    cost = 5 * n * math.log2(n) / 1e9
                    lo, hi = cost / row["ci95"][1], cost / row["ci95"][0]
                    selected.append((row["gflops"], lo, hi))
                    plotted.append(dict(panel=panel, precision=precision, n=n,
                                        implementation=row["implementation"], gflops=row["gflops"],
                                        bootstrap95_lower_gflops=lo, bootstrap95_upper_gflops=hi))
                values = [s[0] for s in selected]
                errors = [[max(0, s[0] - s[1]) for s in selected],
                          [max(0, s[2] - s[0]) for s in selected]]
                ax.errorbar(range(len(sizes)), values, yerr=errors, color=color,
                            marker=marker, markersize=4, linestyle=line,
                            linewidth=1.1, capsize=2, label=label)
            ax.set_title(title)
            ax.set_xticks(range(len(sizes)), [str(n) for n in sizes], rotation=60, ha="right")
            ax.set_xlabel("N")
            ax.grid(axis="y", linestyle=":", alpha=.45)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.spines[["top", "right"]].set_visible(False)
            if panel in (1, 3):
                ax.set_ylabel("GFLOP/s")
        for first, second in ((0, 1), (2, 3)):
            top = max(axes[first].get_ylim()[1], axes[second].get_ylim()[1])
            axes[first].set_ylim(0, top)
            axes[second].set_ylim(0, top)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output / ("overall_linux." + suffix), dpi=220)
        plt.close(fig)
    with (output / "figure_data.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    provenance = {
        "run_id": "lingsheng-linux-512", "mode": "replot_accepted_frozen_results",
        "new_measurement": False, "new_bootstrap": False, "displayed_points": len(plotted),
        "retained_conditions": 54, "predefined_display_exclusion": [144],
        "estimator": "5*N*log2(N)/median_seconds/1e9",
        "interval": "inverse of frozen per-library 2000 MT19937_64 median bootstrap 95% CI",
        "matplotlib_version": matplotlib.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "note": "No sample removed; intervals are not paired speedup or familywise intervals.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate included data without plotting")
    parser.add_argument("--output", type=Path, default=ROOT / "build/figures/linux-reference",
                        help="New output directory; existing directories are refused")
    args = parser.parse_args()
    data, indexed = load_results()
    if args.check_only:
        print(json.dumps({"run_id": data["run_id"], "groups": len(indexed), "samples": 3240,
                          "accepted": True, "mode": "included_data_check"}))
    else:
        print(json.dumps(plot(indexed, args.output)))


if __name__ == "__main__":
    main()
