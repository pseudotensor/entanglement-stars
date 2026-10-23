#!/usr/bin/env python3
"""Run the single X4 scalar attempt and retain its bounded execution receipt.

Requires the box qi2 interpreter and an external GNU timeout --signal=KILL
30s parent. This is process control only; the separately reviewed producer
contains every scientific formula. Do not execute before the original pair
has read both actual files and the final producer identity.
"""

from __future__ import annotations

import time

LAUNCHER_STARTED = time.monotonic()

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

BOX_PYTHON = Path("<internal>/python")
BULK_ROOT = Path("<internal>/solve_b_bulk")
MEMORY_BYTES = 4 * 1024**3
HARD_WALL_SECONDS = 30
GRACEFUL_SECONDS = 28
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def child_limits():
    """All limits are applied before exec starts the producer interpreter."""
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--bound-module-sha256", required=True)
    parser.add_argument("--marginal-json", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != BOX_PYTHON.resolve():
        parser.error("execution requires the box qi2 interpreter")
    parent = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    parent_args = [part.decode(errors="replace") for part in parent if part]
    if (not parent_args or Path(parent_args[0]).name != "timeout"
            or "--signal=KILL" not in parent_args or "30s" not in parent_args):
        parser.error("run under GNU timeout --signal=KILL 30s")
    output = args.output_dir.resolve()
    if not output.is_relative_to(BULK_ROOT.resolve()) or output == BULK_ROOT.resolve():
        parser.error("output must be a run directory under <internal>/solve_b_bulk")
    producer = Path(__file__).with_name("check_WEB40I_formation_scalar_v1.py").resolve()
    if digest(producer) != args.producer_sha256:
        parser.error("producer differs from the reviewed identity")
    bound_module = producer.with_name("WEB40I_bound_ball_scalar_v1.py")
    if digest(bound_module) != args.bound_module_sha256:
        parser.error("bound module differs from the reviewed identity")
    if any((output / name).exists() for name in ("scalar.json", "execution.json")):
        parser.error("existing results are neither overwritten nor recomputed")
    root_free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    mem_available = int(meminfo["MemAvailable"].split()[0]) * 1024
    if root_free <= 100 * 1024**3 or mem_available < 64 * 1024**3:
        parser.error("root free space or available memory is below the lane guard")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_ENV})
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["QI2_X4_OUTER_STARTED_MONOTONIC"] = repr(LAUNCHER_STARTED)
    environment["QI2_X4_BOUND_MODULE_SHA256"] = args.bound_module_sha256
    command = [str(BOX_PYTHON), str(producer), "--output-dir", str(output),
               "--marginal-json", str(args.marginal_json.resolve())]
    receipt_path = output / "execution.json"
    receipt = {
        "schema": "WEB40I_X4_scalar_execution_v1", "status": "started",
        "purpose": "Y3 point/sine scalars and Y1 bound-rest ray-map comparison",
        "command": command, "external_watchdog_parent": parent_args,
        "launcher": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "producer": {"path": str(producer), "sha256_before": args.producer_sha256},
        "bound_module": {"path": str(bound_module), "sha256_before": args.bound_module_sha256},
        "hard_outer_wall_seconds": HARD_WALL_SECONDS,
        "graceful_stop_seconds": GRACEFUL_SECONDS,
        "whole_X4_budget_seconds": 1200,
        "aggregate_budget_debit_seconds": HARD_WALL_SECONDS,
        "debit_kind": "conservative full scalar allowance, not a reset for later X4 stages",
        "memory_limit_bytes_before_producer_exec": MEMORY_BYTES,
        "thread_environment": {name: environment[name] for name in THREAD_ENV},
        "CUDA_VISIBLE_DEVICES": "", "root_free_bytes": root_free,
        "MemAvailable_bytes": mem_available, "automatic_retry": False,
    }
    atomic_save(receipt_path, receipt)
    spawned = time.monotonic()
    process = subprocess.Popen(command, env=environment, preexec_fn=child_limits,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    receipt["producer_pid"] = process.pid
    forced_stop = None
    try:
        # The producer alone raises its graceful exception at outer second 28.
        # Leave its checkpoint/finalization free of a competing SIGINT.
        stdout, stderr = process.communicate(timeout=max(0.001, LAUNCHER_STARTED + 29 - time.monotonic()))
    except subprocess.TimeoutExpired:
        forced_stop = "SIGKILL_at_29_seconds_before_external_watchdog"
        process.kill()
        stdout, stderr = process.communicate(timeout=0.25)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = output / "scalar.json"
    receipt.update(
        status="complete" if process.returncode == 0 else "failed_or_partial",
        returncode=process.returncode, forced_stop=forced_stop,
        producer_wall_seconds_including_interpreter_startup_and_finalization=time.monotonic() - spawned,
        launcher_wall_seconds_before_final_receipt_write=time.monotonic() - LAUNCHER_STARTED,
        user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
        child_peak_rss_kib=usage.ru_maxrss,
        stdout_tail=stdout[-8192:], stderr_tail=stderr[-8192:],
        stdout_characters=len(stdout), stderr_characters=len(stderr),
    )
    receipt["producer"]["sha256_after"] = digest(producer)
    receipt["producer"]["identity_unchanged"] = digest(producer) == args.producer_sha256
    receipt["bound_module"]["sha256_after"] = digest(bound_module)
    receipt["bound_module"]["identity_unchanged"] = digest(bound_module) == args.bound_module_sha256
    receipt["result"] = ({"path": str(result), "bytes": result.stat().st_size,
                          "sha256": digest(result)} if result.exists() else None)
    if not receipt["producer"]["identity_unchanged"] or not receipt["bound_module"]["identity_unchanged"]:
        receipt["status"] = "producer_identity_changed"
    atomic_save(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
