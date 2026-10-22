"""One platform, four panels. Plot only a fully accepted formal measurement."""
import csv
from pathlib import Path
import sys

from common import login_only, now, put, read, sha


def main(root):
    login_only()
    root = Path(root)
    assert read(root / "P_acceptance/completion.json")["accepted"] is True
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cfg = read(root / "configuration.json")
    data = read(root / "P_acceptance/frozen_summary.json")
    rows = data["results"]
    expected = [int(x) for x in cfg["extended_sizes"].split(",") if int(x) != 144]
    assert len(set(expected)) == 26
    out = root / "figures"
    out.mkdir(exist_ok=False)
    pow2 = [n for n in expected if n & (n - 1) == 0]
    nonpow = [n for n in expected if n & (n - 1)]
    styles = [
        ("mofft", "MoFFT", "#D55E00", "D", "-"),
        ("fftw", "FFTW 3.3.11", "#0072B2", "^", "--")]
    dataset = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained")
        panels = [("fp64", pow2, "FP64 · powers of two"),
                  ("fp64", nonpow, "FP64 · non-powers"),
                  ("fp32", pow2, "FP32 · powers of two"),
                  ("fp32", nonpow, "FP32 · non-powers")]
        for index, (ax, (precision, ns, title)) in enumerate(zip(axes, panels)):
            for key, label, color, marker, line in styles:
                selected = []
                for n in ns:
                    candidates = [r for r in rows if r["precision"] == precision and r["n"] == n
                                  and (r["implementation"] == "mofft" if key == "mofft"
                                       else r["implementation"].startswith("fftw-3.3.11"))]
                    assert len(candidates) == 1
                    r = candidates[0]
                    cost = 5 * n * __import__("math").log2(n) / 1e9
                    lo, hi = cost / r["ci95"][1], cost / r["ci95"][0]
                    selected.append((r["gflops"], lo, hi))
                    dataset.append(dict(panel=index + 1, precision=precision, n=n,
                                        implementation=r["implementation"], gflops=r["gflops"],
                                        bootstrap95_lower_gflops=lo, bootstrap95_upper_gflops=hi))
                y = [s[0] for s in selected]
                errors = [[max(0, s[0] - s[1]) for s in selected],
                          [max(0, s[2] - s[0]) for s in selected]]
                ax.errorbar(range(len(ns)), y, yerr=errors, color=color,
                            marker=marker, markersize=4, linestyle=line,
                            linewidth=1.1, capsize=2, label=label)
            ax.set_title(title)
            ax.set_xticks(range(len(ns)), [str(n) for n in ns], rotation=60, ha="right")
            ax.set_xlabel("N")
            ax.grid(axis="y", linestyle=":", alpha=.45)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.spines[["top", "right"]].set_visible(False)
            if index in (0, 2):
                ax.set_ylabel("GFLOP/s")
        for a, b in ((0, 1), (2, 3)):
            top = max(axes[a].get_ylim()[1], axes[b].get_ylim()[1])
            axes[a].set_ylim(0, top)
            axes[b].set_ylim(0, top)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out / ("overall_linux." + suffix), dpi=220)
        plt.close(fig)
    with (out / "figure_data.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dataset[0]))
        writer.writeheader()
        writer.writerows(dataset)
    put(out / "provenance.json", dict(
        utc=now(), node=cfg["node"], cpu=0, thread_count=1, exclusive=True,
        source=str(root / "P_acceptance/frozen_summary.json"),
        source_sha256=sha(root / "P_acceptance/frozen_summary.json"),
        estimator="GFLOP/s = 5*N*log2(N)/median_seconds/1e9",
        intervals="inverse transform of original per-library 2000 MT19937_64 median bootstrap 95% intervals",
        intervals_are_not_paired_speedup_intervals=True,
        n_samples_per_library_condition=30, plots=104, numeric_conditions_retained=54,
        predefined_display_exclusion=[144], filtering=False,
        x_axis="discrete tested lengths with equal spacing, as the upstream display",
        matplotlib_version=matplotlib.__version__,
        caption="Single-thread warm execution on " + cfg["node"] +
                "; plan/input/warmup/destroy outside timer; all 30 batches retained.",
        intended_destination="local preview; no journal-specific compliance claim; no upload"))


if __name__ == "__main__":
    main(sys.argv[1])
