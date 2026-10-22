"""Bounded login protocol review; mock results never qualify native science."""
import json
import math
import os
from pathlib import Path
import sys
import traceback

from common import Commands, login_only, now, put, read, sha
from defer_statistics import derive as defer_statistics


def main(root):
    login_only()
    root = Path(root)
    cfg = read(root / "configuration.json")
    out = root / "login_protocol_review_01"
    out.mkdir(exist_ok=False)
    log = Commands(out / "logs")
    try:
        sys.path.insert(0, str(root / "overall"))
        from derive import derive
        derived = out / "derived"
        derive(root / "repo", derived)
        source = derived / "benchmark_linux.cpp"
        text = source.read_text()
        for bits, eps in ((32, "FLT_EPSILON"), (64, "DBL_EPSILON")):
            needle = f"  const double tolerance = correctness_reference_tolerance(n, {eps});"
            assert text.count(needle) == 1
            text = text.replace(needle, needle + "\n" +
                f'  std::cout << std::setprecision(17) << "correctness_case,fp{bits}," << n'
                ' << "," << relative_error << "," << tolerance << "," << (int)status << "\\n";')
        source.write_text(text)
        put(out / "deferred_statistics_identity.json", defer_statistics(source))
        prefix = Path(cfg["fftw_prefix"])
        executable = out / "mock_benchmark"
        log.run("compile_mock_benchmark", [
            cfg["cxx"], "-std=c++17", "-O3", "-D_GNU_SOURCE=1",
            "-DMOFFT_LOGIN_MOCK=1", "-DMOFFT_HAVE_FFTW=1",
            "-I" + str(root / "repo/include"), "-I" + str(root / "overall"),
            "-I" + str(prefix / "include"), str(source), str(root / "overall/mock_mofft.cpp"),
            str(prefix / "lib/libfftw3f.a"), str(prefix / "lib/libfftw3.a"),
            "--rtlib=compiler-rt", "-lgcc_s", "-lm", "-o", str(executable)])
        text = log.run("mock_correctness_trace", [
            executable, "--correctness", "--sizes", "16,169", "--precision", "both"])
        assert "overall correctness passed: 4 " in text
        assert len([l for l in text.splitlines() if l.startswith("correctness_case,")]) == 4
        ns = [int(x) for x in cfg["extended_sizes"].split(",")]
        def create(path, bad_cv=False, short=False):
            lines = ["mock-machine", "mock-os", "cn-mock-only", "0 0 0", "0 0 0",
                     "107" if short else "108"]
            expected = []
            for n in ns:
                for precision in ("fp32", "fp64"):
                    for implementation in ("mofft", "fftw-3.3.11"):
                        scale = 1e-6 * n / 256
                        values = [scale * (1 + .001 * i) for i in range(30)]
                        if bad_cv:
                            values[-1] = scale * 3
                        lines.append(" ".join(map(str, [
                            precision, implementation, n, 10, 1000, .25, 30, *values])))
                        expected.append((n, precision, implementation, values, scale * 1.0145))
            path.write_text("\n".join(lines) + "\n")
            return expected
        good = out / "known_statistics_input.txt"
        expected = create(good)
        result = out / "known_statistics.json"
        log.run("known_statistics", [executable, "--login-statistics", good, result])
        data = read(result)
        assert data["evidence_class"] == "login_mock_not_target_evidence"
        assert len(data["results"]) == 108
        for r, (n, precision, implementation, values, median) in zip(data["results"], expected):
            assert r["n"] == n and r["precision"] == precision and r["implementation"] == implementation
            assert math.isclose(r["median_seconds"], median, rel_tol=2e-10)
            assert r["ci95"][0] <= median <= r["ci95"][1]
            assert all(math.isclose(a, b, rel_tol=2e-10) for a, b in zip(values, r["raw_seconds"]))
        bad = out / "unstable_input.txt"
        create(bad, bad_cv=True)
        rejected = out / "unstable_statistics_retained.json"
        log.run("quality_rejection_keeps_all_samples", [
            executable, "--login-statistics", bad, rejected], expected=2)
        assert len(read(rejected)["results"]) == 108
        short = out / "short_input.txt"
        create(short, short=True)
        log.run("incomplete_shape_rejected", [
            executable, "--login-statistics", short, out / "must_not_exist.json"], expected=2)
        assert not (out / "must_not_exist.json").exists()
        log.run("compute_statistics_refused", [
            executable, "--login-statistics", good, out / "forbidden.json"],
            expected=2, env=dict(os.environ, CCS_JOB_ID="1234"))
        assert not (out / "forbidden.json").exists()
        # Compile lifecycle separately against original real library, but never
        # execute it here. Genuine target admission must reject on login.
        original_lib = root / "build_bootstrap/libmofft.a"
        lifecycle = out / "lifecycle_login_refusal"
        log.run("compile_lifecycle", [
            cfg["cxx"], "-std=c++17", "-O3", "-march=armv9.2-a+nosve",
            "-I" + str(root / "repo/include"), "-I" + str(root / "overall"),
            str(Path(__file__).parent / "lifecycle.cpp"), str(original_lib),
            "--rtlib=compiler-rt", "-lgcc_s", "-lm", "-o", str(lifecycle)])
        log.run("genuine_lifecycle_login_refusal", [lifecycle], expected=2)
        put(out / "completion.json", dict(
            utc=now(), status="LOGIN_PROTOCOL_REVIEW_PASS_NO_NATIVE_QUALIFICATION",
            known_statistics_rows=108, original_clock_samples_preserved=3240,
            mock_correctness_cases=4, numeric_limits_changed=False,
            original_source_math_identity=True, native_SME_executed=False))
    except Exception:
        put(out / "failure.json", dict(utc=now(), traceback=traceback.format_exc(),
                                      last_command=log.rows[-1] if log.rows else None))
        raise


if __name__ == "__main__":
    main(sys.argv[1])
