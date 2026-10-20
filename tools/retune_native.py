#!/usr/bin/env python3
"""Measure, validate, freeze, and package a native macOS arm64 build.

Consumers of the resulting archive need neither Python nor the generator.
All measurement files remain in the explicitly selected local work directory.
This workflow does not publish a release or label measurements as formal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PAPER_SIZES = (
    "256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,"
    "169,196,225,1728,2197,2744,3375,20736,28561,38416,50625,"
    "248832,371293,537824,759375"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--sizes", help="comma-separated supported C2C lengths")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="short nonformal pipeline test, not release tuning")
    parser.add_argument("--exclusive-confirmed", action="store_true",
                        help="confirm a powered, thermally stable exclusive window")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--kernel-wisdom", type=Path)
    parser.add_argument("--plan-wisdom", type=Path)
    args = parser.parse_args(argv)
    supplied = (args.profile, args.kernel_wisdom, args.plan_wisdom)
    if any(supplied) and not all(supplied):
        parser.error("reuse requires --profile, --kernel-wisdom and --plan-wisdom together")
    if not all(supplied) and not args.smoke and not args.exclusive_confirmed:
        parser.error("measurement requires --exclusive-confirmed, or --smoke for a flow test")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    sizes = args.sizes or ("256,169,1024" if args.smoke else PAPER_SIZES)
    try:
        if any(int(n) < 2 for n in sizes.split(",")):
            raise ValueError
    except ValueError:
        parser.error("--sizes must be a comma-separated list of lengths >= 2")
    root = Path(__file__).resolve().parents[1]
    work = args.workdir.resolve()
    if work == root:
        parser.error("the work directory must not be the source root")
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "compiler"))

    def run(command: list[str], output: Path | None = None) -> None:
        print("Running:", Path(command[0]).name, *command[1:], flush=True)
        if output is None:
            subprocess.run(command, env=env, check=True)
        else:
            with output.open("w") as stream:
                subprocess.run(command, env=env, stdout=stream, check=True)

    def configure(build: Path, *extra: str) -> None:
        run(["cmake", "-S", str(root), "-B", str(build),
             "-DCMAKE_BUILD_TYPE=Release", "-DMOFFT_BUILD_KERNEL_PORTFOLIO=ON",
             *extra])

    def build(directory: Path, *targets: str) -> None:
        command = ["cmake", "--build", str(directory), "--parallel", str(args.jobs)]
        if targets:
            command.extend(["--target", *targets])
        run(command)

    profile = args.profile.resolve() if args.profile else work / "target-profile.json"
    kernel = (args.kernel_wisdom.resolve() if args.kernel_wisdom else
              work / "kernel-wisdom.json")
    plans = args.plan_wisdom.resolve() if args.plan_wisdom else work / "plan-wisdom.json"
    if all(supplied):
        for path in (profile, kernel, plans):
            if not path.is_file():
                parser.error("one of the reusable input files does not exist")
    else:
        bootstrap = work / "bootstrap"
        configure(bootstrap, "-DMOFFT_BUILD_TESTS=OFF", "-DMOFFT_BUILD_TOOLS=ON")
        build(bootstrap, "mofft_microbench")
        raw = work / "calibration.json"
        run([str(bootstrap / "mofft_microbench"), "--iterations",
             "2000" if args.smoke else "50000", "--samples",
             "3" if args.smoke else "9"], raw)
        run([sys.executable, "-m", "mofft_compiler.calibrate", str(raw),
             "--output", str(profile), "--name", "apple-native-local"])
        validation = [sys.executable, "-m", "mofft_compiler.validate",
                      "--root", str(root), "--profile", str(profile),
                      "--benchmark", "--benchmark-batches", "256",
                      "--benchmark-repeats", "32", "--benchmark-output", str(kernel),
                      "--keep", str(work / "candidate-validation")]
        if args.smoke:
            validation.extend(["--radices", "8,16"])
        run(validation, work / "kernel-validation.log")
        search = work / "search"
        configure(search, f"-DMOFFT_PROFILE={profile}",
                  f"-DMOFFT_KERNEL_WISDOM={kernel}", "-DMOFFT_WISDOM=",
                  "-DMOFFT_BUILD_TESTS=ON", "-DMOFFT_BUILD_TOOLS=ON")
        build(search)
        run(["ctest", "--test-dir", str(search), "--output-on-failure"])
        run([str(search / "mofft_plan_search"), "--sizes", sizes,
             "--precision", "both", "--candidates", "4" if args.smoke else "32",
             "--samples", "3" if args.smoke else "9",
             "--steady-finalists", "1" if args.smoke else "4",
             "--steady-warmup", "16" if args.smoke else "1000",
             "--compact", "--output", str(plans)], work / "plan-search.log")

    final = work / "frozen"
    install = work / "install"
    configure(final, f"-DMOFFT_PROFILE={profile}", f"-DMOFFT_KERNEL_WISDOM={kernel}",
              f"-DMOFFT_WISDOM={plans}", "-DMOFFT_BUILD_TESTS=ON",
              "-DMOFFT_BUILD_TOOLS=OFF", f"-DCMAKE_INSTALL_PREFIX={install}")
    build(final)
    run(["ctest", "--test-dir", str(final), "--output-on-failure"])
    run(["cmake", "--install", str(final)])
    package = root / "tools/packaging/package.py"
    run([sys.executable, str(package), "verify-macos", "--prefix", str(install)])
    archive = work / ("MoFFT-smoke-macos-arm64.tar.gz" if args.smoke else
                      "MoFFT-local-tuned-macos-arm64.tar.gz")
    run([sys.executable, str(package), "package-macos", "--prefix", str(install),
         "--output", str(archive)])
    summary = {"schema_version": 1, "smoke": args.smoke,
               "formal_measurement": False, "reused_inputs": bool(all(supplied)),
               "sizes": sizes.split(","), "archive": archive.name,
               "selection": "measured bounded search; no universal optimum claim",
               "consumer_requires_codegen": False}
    (work / "workflow-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Ready:", archive, flush=True)


if __name__ == "__main__":
    main()
