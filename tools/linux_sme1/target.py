"""Compute node: native capture and resource observations only."""
import ctypes
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import traceback

from common import append, now, put, read


def observation():
    affinity = sorted(os.sched_getaffinity(0))
    caches = []
    for i in range(8):
        p = Path(f"/sys/devices/system/cpu/cpu0/cache/index{i}")
        if (p / "level").exists():
            caches.append({k: (p / k).read_text().strip() for k in
                           ("level", "type", "size", "coherency_line_size",
                            "shared_cpu_list")})
    frequency = {}
    for k in ("scaling_governor", "scaling_cur_freq",
              "scaling_min_freq", "scaling_max_freq"):
        p = Path("/sys/devices/system/cpu/cpu0/cpufreq") / k
        frequency[k] = p.read_text().strip() if p.exists() else None
    binding = {}
    for k, path in (
        ("midr", "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1"),
        ("boot_id", "/proc/sys/kernel/random/boot_id"),
        ("cpu_online", "/sys/devices/system/cpu/online"),
    ):
        p = Path(path)
        binding[k] = p.read_text().strip() if p.exists() else None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getauxval.restype = ctypes.c_ulong
    sve, sme = libc.prctl(51, 0, 0, 0, 0), libc.prctl(64, 0, 0, 0, 0)
    return dict(
        utc=now(), host=socket.gethostname(), job=os.getenv("CCS_JOB_ID", ""),
        original_affinity=affinity, affinity=affinity, cache=caches,
        frequency=frequency, uname=list(os.uname()), cpu_binding=binding,
        cpuinfo=Path("/proc/cpuinfo").read_text(),
        environment={k: os.getenv(k) for k in (
            "CCS_JOB_ID", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        capability=dict(sve_bytes=sve & 65535 if sve >= 0 else None,
                        sme_bytes=sme & 65535 if sme >= 0 else None,
                        hwcap=libc.getauxval(16), hwcap2=libc.getauxval(26)),
        runtime_limit=None)


def require_identity(x, node):
    assert x["job"].isdecimal() and int(x["job"]) > 0
    assert x["host"] == node
    assert x["original_affinity"] == x["affinity"] == [0], "CPU0_AFFINITY_REQUIRED"
    assert x["cpu_binding"]["midr"] is not None, "MIDR_UNAVAILABLE"
    c = x["capability"]
    mask = (1 << 23) | (1 << 25) | (1 << 29)
    assert c["hwcap"] & (1 << 22) and c["hwcap2"] & mask == mask, "SME_CAPABILITY_REQUIRED"
    assert c["sve_bytes"] == c["sme_bytes"] == 64, "VECTOR_LENGTH_64_BYTES_REQUIRED"
    assert all(x["environment"][k] == "1" for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))


def same_environment(a, b):
    for k in ("host", "affinity", "cache", "cpu_binding", "capability"):
        assert a[k] == b[k], ("ENVIRONMENT_CHANGED", k)
    assert a["uname"][2:] == b["uname"][2:]
    for k in ("scaling_governor", "scaling_min_freq", "scaling_max_freq"):
        assert a["frequency"][k] == b["frequency"][k], ("FREQUENCY_POLICY_CHANGED", k)


def main(root, phase):
    root = Path(root)
    cfg = read(root / "configuration.json")
    out = root / (phase + "_capture")
    out.mkdir(exist_ok=False)
    started = now()
    calls = []

    def execute(label, args, stdout_path=None, stderr_path=None, cwd=None):
        row = dict(label=label, argv=[str(a) for a in args], start_utc=now())
        append(out / "progress.jsonl", dict(**row, state="STARTED"))
        with Path(stdout_path or out / (label + ".stdout")).open("xb") as f:
            with Path(stderr_path or out / (label + ".stderr")).open("xb") as e:
                p = subprocess.run(args, stdout=f, stderr=e, cwd=cwd or root)
        row.update(end_utc=now(), exit_code=p.returncode)
        calls.append(row)
        put(out / (label + ".receipt.json"), row)
        append(out / "progress.jsonl", dict(**row, state="FINISHED"))
        if p.returncode:
            raise RuntimeError(f"NATIVE_CAPTURE_FAILED:{label}:{p.returncode}")

    try:
        first = observation()
        put(out / "environment_before.json", first)
        require_identity(first, cfg["node"])
        if phase == "C0":
            cal = root / "calibration"
            # A new CPU binding observed before this capture, not an old CPU profile.
            put(cal / "cpu_reference.json", first)
            put(cal / "target_start.json", first)
            execute("calibration", [str(cal / "calibrate"), "--capture"],
                    cal / "samples.jsonl", cal / "target.stderr")
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            put(cal / "target_end.json", dict(
                utc=now(), returncode=0, user_seconds=usage.ru_utime,
                system_seconds=usage.ru_stime, maxrss_kib=usage.ru_maxrss))
            for batch, repeat in cfg["candidate_configurations"]:
                execute(f"candidate_b{batch}_r{repeat}", [
                    cfg["candidate_executable"], "--benchmark", str(batch), str(repeat)])
        else:
            reference = read(root / "calibration/target_start.json")
            same_environment(reference, first)
            if phase in ("SEARCH", "CHECK"):
                stage = "search" if phase == "SEARCH" else "final"
                execute("correctness", [
                    str(root / "programs" / ("benchmark_" + stage)),
                    "--correctness", "--sizes", cfg["extended_sizes"],
                    "--precision", "both"])
                if phase == "SEARCH":
                    execute("plan_search", [
                        str(root / "programs/plan_search"), "--formal",
                        "--exclusive-confirmed", "--precision", "both",
                        "--sizes", cfg["extended_sizes"], "--candidates", "32",
                        "--samples", "15", "--steady-finalists", "4",
                        "--steady-warmup", "1000", "--manifest",
                        str(root / "build_search/generated/manifest.json"),
                        "--output", str(out / "plan_wisdom.json")])
                else:
                    execute("lifecycle", [str(root / "programs/lifecycle_final")])
            elif phase == "P":
                execute("formal", [
                    str(root / "programs/benchmark_final"), "--formal",
                    "--exclusive-confirmed", "--precision", "both",
                    "--output", str(out / "measurements.json")])
            else:
                raise ValueError("UNKNOWN_PHASE")
        last = observation()
        put(out / "environment_after.json", last)
        same_environment(first, last)
        put(out / "completion.json", dict(
            status="NATIVE_CAPTURE_COMPLETE_REQUIRES_LOGIN_ACCEPTANCE",
            phase=phase, start_utc=started, end_utc=now(), job=first["job"],
            node=first["host"], calls=calls, runtime_limit=None))
    except Exception:
        put(out / "failure.json", dict(
            status="NATIVE_CAPTURE_FAILED", phase=phase, start_utc=started,
            end_utc=now(), job=os.getenv("CCS_JOB_ID"), node=socket.gethostname(),
            traceback=traceback.format_exc(), calls=calls))
        raise


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
