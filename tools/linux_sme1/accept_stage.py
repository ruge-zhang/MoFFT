"""Full login acceptance. Raw clocks stay intact; no sample selection or rerun."""
import csv
import json
import math
from pathlib import Path
import re
import statistics as st
import sys
import traceback

from common import Commands, check_manifest, file_manifest, login_only, now, put, read, sha
from submit import decoded, query
from target import require_identity, same_environment


def expected_cells(cfg):
    return {(int(n), p) for n in cfg["extended_sizes"].split(",")
            for p in ("fp32", "fp64")}


def correctness(root, phase, cfg, out):
    path = root / (phase + "_capture/correctness.stdout")
    lines = path.read_text().splitlines()
    assert any("overall correctness passed: 54 " in x for x in lines)
    rows = []
    for text in lines:
        if not text.startswith("correctness_case,"):
            continue
        _, precision, n, error, limit, status = text.split(",")
        n, error, limit, status = int(n), float(error), float(limit), int(status)
        epsilon = 2**-23 if precision == "fp32" else 2**-52
        assert status == 0 and math.isfinite(error) and 0 <= error <= limit
        assert math.isclose(limit, 256 * epsilon * max(1, math.log2(n)), rel_tol=2e-12)
        rows.append(dict(n=n, precision=precision, relative_error=error,
                         original_limit=limit, status=status))
    assert len(rows) == 54 and {(r["n"], r["precision"]) for r in rows} == expected_cells(cfg)
    put(out / "all_correctness_errors.json", rows)
    return rows


def percentile(values, q):
    a = sorted(values)
    p = (len(a) - 1) * q
    i = int(p)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (p - i)


def main(root, phase):
    login_only()
    root = Path(root)
    cfg = read(root / "configuration.json")
    out = root / (phase + "_acceptance")
    out.mkdir(exist_ok=False)
    log = Commands(out / "logs")
    try:
        job = read(root / (phase + "_job.json"))["job"]
        packet = query(["djob", "-J", "-l", job])
        put(out / "scheduler.json", packet)
        j = decoded(packet)["successData"][0]
        assert j["state"] == "SUCCEEDED" and j["jobExitCode"] == 0
        assert j["exclusive"] is True and j["taskExecNodes"] == cfg["node"]
        assert j["resGroups"][0]["nodeResUsages"][0]["hwThreads"] == "0"
        seal = root / (phase + "_seal.json")
        check_manifest(read(seal)["objects"])
        check_manifest(read(root / "prepared_C0_seal.json")["objects"])
        capture = root / (phase + "_capture")
        end = read(capture / "completion.json")
        assert end["job"] == job and all(c["exit_code"] == 0 for c in end["calls"])
        before = read(capture / "environment_before.json")
        after = read(capture / "environment_after.json")
        require_identity(before, cfg["node"])
        same_environment(before, after)
        same_environment(read(root / "calibration/target_start.json"), before)
        if phase in ("SEARCH", "CHECK"):
            correctness(root, phase, cfg, out)
        if phase == "SEARCH":
            data = read(capture / "plan_wisdom.json")
            manifest = read(root / "build_search/generated/manifest.json")
            assert data["formal_measurement"] is True and data["quality_rejected"] is False
            assert data["profile"] == read(root / "C0_acceptance/accepted_profile.json")["name"]
            assert data["kernel_input_sha256"] == manifest["input_sha256"]
            assert data["steady_finalists"] == 4 and data["steady_warmup_executions"] == 1000
            entries = data["entries"]
            assert len(entries) == 54
            assert {(e["length"], e["precision"]) for e in entries} == expected_cells(cfg)
            clocks = 0
            plans = {}
            for line in (capture / "plan_wisdom.json.raw.jsonl").open():
                r = json.loads(line)
                key = (r["length"], r["precision_bits"], r["candidate_index"], r["phase"])
                if r["kind"] == "plan":
                    assert math.prod(r["radices"]) == r["length"]
                    plans[key] = r
                else:
                    assert r["kind"] == "clock" and key in plans
                    assert math.isfinite(r["seconds"]) and r["seconds"] > 0
                    assert type(r["repetitions"]) is int and r["repetitions"] > 0
                    clocks += 1
            assert clocks > 0
            for e in entries:
                assert e["candidates"] and math.prod(e["radices"]) == e["length"]
                best = e["candidates"][0]
                values = best["relative_samples"] if best["steady_finalist"] else best["raw_seconds"]
                assert len(values) >= 15 and all(math.isfinite(v) and v > 0 for v in values)
                assert st.pstdev(values) / st.mean(values) <= .05 + 1e-10
            put(out / "plan_validation.json", dict(
                conditions=54, raw_clock_records=clocks, descriptors=len(plans),
                profile_and_kernel_binding=True, original_cv_limit=.05))
        elif phase == "CHECK":
            text = (capture / "lifecycle.stdout").read_text()
            assert "PUBLIC_API_LIFECYCLE_PASS:54_CONDITIONS_216_TRANSITIONS" in text
            states = [line.split(",") for line in text.splitlines() if line.startswith("state,")]
            assert len(states) == 216
            seen = set()
            for _, bits, n, operation, a, b in states:
                assert int(a) & 3 == int(b) & 3
                seen.add((int(bits), int(n), operation))
            assert len(seen) == 216
            put(out / "lifecycle_validation.json", dict(
                conditions=54, observed_public_api_transitions=216,
                caller_state="normal non-streaming private-ZA interface",
                active_ZA_caller_contents_not_claimed=True))
        elif phase == "P":
            data = read(capture / "measurements.json")
            assert data["evidence_class"] == "target_capture_requires_external_acceptance"
            assert data["raw_only"] is True and data["formal_run"] is True
            assert data["wisdom_formal"] is True and data["node"] == cfg["node"]
            assert data["machine"] == read(root / "SEARCH_capture/plan_wisdom.json")["machine"]
            rows = data["results"]
            assert len(rows) == 108
            assert {(r["n"], r["precision"]) for r in rows} == expected_cells(cfg)
            grouped = {}
            for r in rows:
                key = (r["n"], r["precision"])
                grouped.setdefault(key, []).append(r)
                a = r["raw_seconds"]
                assert len(a) == 30 and all(math.isfinite(v) and v > 0 for v in a)
                assert r["warmup_executions"] >= 1000 and r["warmup_seconds"] > 0
                assert type(r["repetitions_per_sample"]) is int and r["repetitions_per_sample"] >= 1
            assert len(grouped) == 54
            for pair in grouped.values():
                assert len(pair) == 2 and sum(r["implementation"] == "mofft" for r in pair) == 1
                assert any(r["implementation"].startswith("fftw-3.3.11") for r in pair)
                assert pair[0]["repetitions_per_sample"] == pair[1]["repetitions_per_sample"]
            put(out / "raw_acceptance.json", dict(
                accepted=True, conditions=54, rows=108, clocks=3240,
                all_samples_preserved=True, original_pairing_and_timer_source_verified=True))
            flat = [
                data["machine"], data["os"], data["node"],
                " ".join(str(v) for v in data["load_average"]),
                " ".join(str(v) for v in data["ending_load_average"]), str(len(rows))]
            for r in rows:
                flat.append(" ".join(str(v) for v in (
                    r["precision"], r["implementation"], r["n"], r["repetitions_per_sample"],
                    r["warmup_executions"], r["warmup_seconds"], len(r["raw_seconds"]),
                    *r["raw_seconds"])))
            source = out / "frozen_statistics_input.txt"
            source.write_text("\n".join(flat) + "\n")
            summary = out / "frozen_summary.json"
            log.run("original_frozen_statistics", [
                root / "programs/benchmark_final", "--login-statistics", source, summary])
            frozen = read(summary)
            assert frozen["evidence_class"] == "target_capture_requires_external_acceptance"
            assert frozen["formal_run"] is True and frozen["wisdom_formal"] is True
            assert frozen["node"] == cfg["node"] and len(frozen["results"]) == 108
            descriptions = []
            for original, result in zip(rows, frozen["results"]):
                assert (original["n"], original["precision"], original["implementation"]) == (
                    result["n"], result["precision"], result["implementation"])
                a = original["raw_seconds"]
                assert math.isclose(result["median_seconds"], st.median(a), rel_tol=2e-10)
                assert result["cv"] <= .05 + 1e-10
                assert all(math.isclose(x, y, rel_tol=2e-10) for x, y in zip(a, result["raw_seconds"]))
                expected_gflops = 5 * original["n"] * math.log2(original["n"]) / st.median(a) / 1e9
                assert math.isclose(result["gflops"], expected_gflops, rel_tol=2e-10)
                lo, hi = result["ci95"]
                assert 0 < lo <= result["median_seconds"] <= hi
                descriptions.append(dict(
                    n=original["n"], precision=original["precision"],
                    implementation=original["implementation"], samples=30,
                    mean_seconds=st.mean(a), total_seconds=sum(a), median_seconds=st.median(a),
                    p95_seconds=percentile(a, .95), sample_stddev_seconds=st.stdev(a),
                    iqr_seconds=percentile(a, .75) - percentile(a, .25)))
            put(out / "descriptive_statistics.json", descriptions)
        else:
            raise ValueError("UNKNOWN_PHASE")
        put(out / "completion.json", dict(
            accepted=True, status=phase + "_FULLY_ACCEPTED",
            job=job, node=cfg["node"], utc=now(), all_samples_preserved=True))
    except Exception:
        put(out / "failure.json", dict(
            accepted=False, utc=now(), phase=phase,
            traceback=traceback.format_exc(), later_stages_blocked=True,
            automatic_retry=0))
        raise


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
