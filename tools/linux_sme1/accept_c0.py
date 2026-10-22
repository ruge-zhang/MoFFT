"""Frozen CAL03 statistics plus complete native candidate acceptance on login."""
import json
from pathlib import Path
import shutil
import sys
import traceback

from common import Commands, check_manifest, login_only, now, put, read, sha
from submit import decoded, query
from target import require_identity, same_environment


def main(root):
    login_only()
    root = Path(root)
    out = root / "C0_acceptance"
    out.mkdir(exist_ok=False)
    cfg = read(root / "configuration.json")
    log = Commands(out / "logs")
    try:
        job = read(root / "C0_job.json")["job"]
        scheduler = query(["djob", "-J", "-l", job])
        put(out / "scheduler.json", scheduler)
        j = decoded(scheduler)["successData"][0]
        assert j["state"] == "SUCCEEDED" and j["jobExitCode"] == 0
        assert j["exclusive"] is True and j["taskExecNodes"] == cfg["node"]
        assert j["resGroups"][0]["nodeResUsages"][0]["hwThreads"] == "0"
        complete = read(root / "C0_capture/completion.json")
        assert complete["job"] == job and len(complete["calls"]) == 6
        first = read(root / "C0_capture/environment_before.json")
        last = read(root / "C0_capture/environment_after.json")
        require_identity(first, cfg["node"])
        same_environment(first, last)
        assert first["job"] == job
        check_manifest(read(root / "prepared_C0_seal.json")["objects"])
        cal = root / "calibration"
        sys.path.insert(0, str(cal))
        from analyze import analyze
        accepted, original_profile = analyze(cal)
        put(out / "frozen_calibration_acceptance.json", accepted)
        put(out / "frozen_analyzer_profile.json", original_profile)
        assert accepted["samples"] == 3699 and accepted["warmups"] == 411
        if accepted["status"] != "CALIBRATION_ACCEPTED":
            raise ValueError("CALIBRATION_QUALITY_REJECTED_NO_RETRY:" +
                             str(accepted["blocking_unstable_cases"]))
        profile = json.loads(json.dumps(original_profile))
        profile["name"] = "linux-sme1-cal03-" + cfg["node"] + "-" + job
        profile["measurement_metadata"].update(
            nonexclusive=False, exclusive=True, native_job=job,
            binding_reference="same-job pre-capture CPU identity, not a previous-node profile",
            binding_note="Only metadata below frozen analysis changes; all numerical model values unchanged")
        profilepath = out / "accepted_profile.json"
        put(profilepath, profile)
        sys.path.insert(0, str(root / "tuning"))
        from candidate_results import parse
        old_manifest = read(root / "candidates/manifest.json")
        all_entries = []
        summaries = []
        for batch, repeat in cfg["candidate_configurations"]:
            path = root / "C0_capture" / f"candidate_b{batch}_r{repeat}.stdout"
            parsed = parse(path.read_text(), old_manifest["cases"])
            assert len(parsed["entries"]) == 1076
            summaries.append(dict(batch=batch, repeat=repeat, cases=1076,
                                  raw_sha256=sha(path), raw_path=str(path)))
            for e in parsed["entries"]:
                item = {k: e[k] for k in (
                    "radix", "precision", "direction", "stage", "context", "pattern",
                    "matrix_vg_width", "rotate_temp_tiles", "batch_pipeline_depth",
                    "median_nanoseconds", "pass_nanoseconds", "raw_round_nanoseconds")}
                item["batch"] = batch
                if e["context"] in ("broadcast", "direct_broadcast"):
                    item["locality_repeat"] = repeat
                all_entries.append(item)
        # Explicitly demonstrate that the fresh accepted profile generates the
        # identical forced candidates before binding its digest to measured bytes.
        fresh = root / "candidates_accepted_profile"
        log.run("accepted_profile_candidate_identity", [
            sys.executable, "-B", str(root / "tuning/candidate_prepare.py"),
            str(root / "repo"), str(profilepath), str(fresh)])
        manifest = read(fresh / "manifest.json")
        assert manifest["cases"] == old_manifest["cases"]
        for e in manifest["source_files"]:
            assert sha(fresh / e["path"]) == sha(root / "candidates" / e["path"]) == e["sha256"]
        wisdom = dict(
            schema_version=4, compiler_sha256=manifest["upstream_compiler_sha256"],
            profile_sha256=manifest["profile_sha256"], formal_measurement=True,
            machine=cfg["node"], native_job=job, passes=3,
            pass_order="forward,reverse,forward", rounds_per_pass=9,
            repetitions_per_round=100, entries=all_entries, configurations=summaries,
            scope="all forced candidates at five predeclared batch/repeat settings; not global optimality",
            candidate_source_identity_proved_under_accepted_profile=True)
        put(out / "kernel_wisdom.json", wisdom)
        put(out / "completion.json", dict(
            accepted=True, status="C0_FULL_CALIBRATION_AND_CANDIDATES_ACCEPTED",
            utc=now(), job=job, node=cfg["node"], candidate_cases=1076,
            configurations=5, retained_raw_rounds=5 * 1076 * 27,
            profile_sha256=sha(profilepath), wisdom_sha256=sha(out / "kernel_wisdom.json"),
            frozen_analyzer_sha256=sha(cal / "cal02_analyze.py"),
            raw_samples_preserved=True, automatic_retry=0))
    except Exception:
        put(out / "failure.json", dict(
            accepted=False, utc=now(), traceback=traceback.format_exc(),
            remaining_stages_blocked=True, automatic_retry=0))
        raise


if __name__ == "__main__":
    main(sys.argv[1])
