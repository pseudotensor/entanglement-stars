#!/usr/bin/env python3
"""Stdlib launcher; invoke under GNU timeout --signal=KILL 600s nice -n 10."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--producer-sha256", required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--source-sha256", required=True)
    args = ap.parse_args()
    producer = Path(__file__).with_name("check_WEB40I_static_scalar_v1.py")
    parent_argv = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    parent_text = [x.decode(errors="replace") for x in parent_argv if x]
    if not parent_text or Path(parent_text[0]).name != "timeout" or "600s" not in parent_text or "--signal=KILL" not in parent_text:
        raise SystemExit("external timeout --signal=KILL 600s required")
    if sha(producer) != args.producer_sha256 or sha(args.source) != args.source_sha256:
        raise SystemExit("producer/source identity mismatch")
    if args.output_dir.exists():
        raise SystemExit("fresh output directory required; never retry into an existing run")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    free = shutil.disk_usage("/").free
    if free < 100 * GIB or available < 64 * GIB:
        raise SystemExit("root-free or MemAvailable guard failed")
    affinity = sorted(os.sched_getaffinity(0))[:4]
    if not affinity:
        raise SystemExit("no allowed CPUs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               NUMEXPR_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4",
               CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")

    def child_limits():
        resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, 4 * GIB))
        resource.setrlimit(resource.RLIMIT_CPU, (2400, 2400))
        os.sched_setaffinity(0, affinity)
        priority = os.getpriority(os.PRIO_PROCESS, 0)
        if priority < 10:
            os.nice(10 - priority)

    cmd = [sys.executable, str(producer), "--output-dir", str(args.output_dir),
           "--source", str(args.source), "--source-sha256", args.source_sha256,
           "--outer-start", str(started)]
    receipt = dict(status="launching", command=cmd, external_parent_argv=parent_text,
                   producer_sha256=args.producer_sha256, source_sha256=args.source_sha256,
                   launcher_sha256=sha(__file__), root_free_bytes=free,
                   MemAvailable_bytes=available, cpu_affinity=affinity, nice_at_least=10,
                   CUDA_VISIBLE_DEVICES="", threads_at_most=4,
                   address_space_bytes=4 * GIB, external_cap_seconds=600,
                   producer_graceful_seconds=580, launcher_child_kill_seconds=590,
                   attempt_count=1, no_automatic_retry=True)
    path = args.output_dir / "execution.json"
    atomic(path, receipt)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    forced = False
    with (args.output_dir / "process.log").open("wb") as stream:
        child = subprocess.Popen(cmd, env=env, preexec_fn=child_limits,
                                 stdout=stream, stderr=subprocess.STDOUT)
        receipt["child_pid"] = child.pid
        atomic(path, receipt)
        try:
            rc = child.wait(timeout=max(.01, 590 - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            forced = True
            os.kill(child.pid, signal.SIGKILL)
            rc = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt.update(status="terminal", returncode=rc, child_forced_kill=forced,
        wall_before_final_receipt_seconds=time.monotonic() - started,
        user_CPU_seconds=after.ru_utime - before.ru_utime,
        system_CPU_seconds=after.ru_stime - before.ru_stime,
        child_peak_RSS_KiB=after.ru_maxrss)
    summary = args.output_dir / "summary.json"
    if summary.exists():
        receipt["summary_sha256"] = sha(summary)
        receipt["producer_status"] = json.loads(summary.read_text()).get("status")
    atomic(path, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "returncode", "wall_before_final_receipt_seconds", "child_peak_RSS_KiB")}))
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
