"""Durable, finite-stage control helpers; no execution deadline or auto retry."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def login_only():
    assert socket.gethostname().startswith("login") and not os.getenv("CCS_JOB_ID")


class Commands:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.rows = []

    def run(self, label, argv, cwd=None, expected=0, env=None):
        argv = [str(x) for x in argv]
        put(self.directory / (label + ".argv.json"), argv)
        row = dict(label=label, argv=argv, cwd=str(cwd), start_utc=now())
        append(self.directory / "progress.jsonl", dict(**row, state="STARTED"))
        with (self.directory / (label + ".stdout")).open("xb") as out:
            with (self.directory / (label + ".stderr")).open("xb") as err:
                p = subprocess.run(argv, cwd=cwd, env=env, stdout=out, stderr=err)
        row.update(end_utc=now(), exit_code=p.returncode, expected=expected)
        self.rows.append(row)
        put(self.directory / (label + ".receipt.json"), row)
        append(self.directory / "progress.jsonl", dict(**row, state="FINISHED"))
        if p.returncode != expected:
            raise RuntimeError(f"{label}: exit {p.returncode}, expected {expected}")
        return (self.directory / (label + ".stdout")).read_text()


def file_manifest(paths):
    return [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(set(map(Path, paths))) if p.is_file()]


def check_manifest(entries):
    for e in entries:
        p = Path(e["path"])
        assert p.is_file() and not p.is_symlink(), e["path"]
        assert p.stat().st_size == e["bytes"] and sha(p) == e["sha256"], e["path"]
