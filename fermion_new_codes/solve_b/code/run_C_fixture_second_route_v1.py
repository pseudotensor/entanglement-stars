#!/usr/bin/env python3
"""Stdlib-only launcher; external GNU timeout --signal=KILL 1800s required."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

START = time.monotonic()
BASE = Path("<internal>/C_fixture_second_route_v1")
GIB = 2**30


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def atomic(path, obj):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def finish_locator(run):
    """Only filesystem metadata; the child and its log writer are now terminal."""
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    locator = run / "LARGE_FILES.tsv"
    old_identity = identity(locator) if locator.exists() else None
    bound = bool(summary is not None and old_identity and
                 summary.get("raw_locator_sha256") == old_identity["sha256"])
    if bound:
        contents = locator.read_text()
        mode = "producer raw-file hashes plus terminal log hash"
    else:
        # A timeout may have interrupted the producer's final file hashing.
        # The owner permits a file count instead of sha256 for box-only bulk.
        raw_files = [p for p in (run / "raw").rglob("*") if p.is_file()]
        contents = "path\tsize_bytes\tsha256\tbox_location\n"
        contents += f"raw/\t{sum(p.stat().st_size for p in raw_files)}\tfile_count:{len(raw_files)}\tcompute-host:{run / 'raw'}\n"
        mode = "directory count after incomplete producer finalization"
    log = identity(run / "process.log")
    contents += f"process.log\t{log['bytes']}\t{log['sha256']}\tcompute-host:{run / 'process.log'}\n"
    temporary = locator.with_suffix(".tsv.tmp")
    temporary.write_text(contents)
    temporary.replace(locator)
    if summary is not None:
        summary["raw_locator_finalization"] = dict(mode=mode,
            producer_raw_only_listing_sha256=old_identity["sha256"] if bound else None,
            terminal_log_bound_after_child_exit=True)
        summary["raw_locator_sha256"] = identity(locator)["sha256"]
        atomic(summary_path, summary)
    return dict(mode=mode, producer_raw_listing_was_bound=bound,
                terminal_log=log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--specification-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    if Path(__file__).resolve().parent != BASE / "code":
        raise SystemExit("Wrong launcher location")
    producer = BASE / "code/check_C_fixture_second_route_v1.py"
    specification = BASE / "code/C_fixture_second_route_v1.tex"
    review = BASE / "code/redteam_C_fixture_second_route_v1.tex"
    expected = {"producer": args.producer_sha256,
                "specification": args.specification_sha256, "review": args.review_sha256}
    inputs = {key: identity(path) for key, path in
              (("producer", producer), ("specification", specification), ("review", review))}
    if any(inputs[key]["sha256"] != value for key, value in expected.items()):
        raise SystemExit("Actual approved producer/specification/review changed")
    parent = [os.fsdecode(x) for x in Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0") if x]
    if not parent or Path(parent[0]).name != "timeout" or "--signal=KILL" not in parent or "1800s" not in parent:
        raise SystemExit("External hard timeout --signal=KILL 1800s is required")
    run = BASE / "run"
    if run.exists():
        raise SystemExit("Exclusive run directory already exists; never retry")
    memory = dict(s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0])*1024
    root_free, data_free = shutil.disk_usage("/").free, shutil.disk_usage("/data").free
    if available < 64*GIB or root_free < 100*GIB or data_free < 5*GIB:
        raise SystemExit("Memory<internal>/raw-storage reserve failed")
    packages = {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "threadpoolctl")}
    if packages != {"numpy": "2.2.6", "scipy": "1.17.0", "threadpoolctl": "3.6.0"}:
        raise SystemExit("Declared installed numerical environment changed")
    affinity = sorted(os.sched_getaffinity(0))[:8]
    if len(affinity) != 8:
        raise SystemExit("Eight allowed CPU cores are required")
    environment = os.environ.copy()
    settings = {key: "8" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
    settings.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", MALLOC_ARENA_MAX="2")
    environment.update(settings)
    environment.pop("PYTHONPATH", None)

    def child_limits():
        resource.setrlimit(resource.RLIMIT_AS, (4*GIB, 4*GIB))
        resource.setrlimit(resource.RLIMIT_CPU, (14400, 14400))
        os.sched_setaffinity(0, affinity)
        actual_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if actual_nice < 10:
            os.nice(10-actual_nice)

    run.mkdir()
    command = [sys.executable, str(producer), "--output-dir", str(run),
               "--outer-start", repr(START)]
    receipt = dict(status="launching", purpose="Computation C prescribed dense fixture control",
        inputs=inputs, launcher=identity(__file__), installed_versions=packages,
        command=command, parent_command=parent, cpu_affinity=affinity,
        requested_threads=8, address_space_bytes=4*GIB, requested_nice_at_least=10,
        environment_settings=settings, MemAvailable_bytes=available,
        root_free_bytes=root_free, data_free_bytes=data_free,
        outer_hard_seconds=1800, producer_save_deadline_seconds=1740,
        no_new_dense_after_seconds=1680, own_child_term_seconds=1790,
        own_child_kill_seconds=1795, no_automatic_retry=True,
        wrapper_pid=os.getpid(), started_unix=time.time())
    receipt_path = run / "execution.json"
    atomic(receipt_path, receipt)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (run / "process.log").open("xb") as stream:
        child = subprocess.Popen(command, env=environment, preexec_fn=child_limits,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        receipt.update(status="running", child_pid=child.pid)
        atomic(receipt_path, receipt)
        try:
            returncode = child.wait(timeout=max(.01, 1790-(time.monotonic()-START)))
        except subprocess.TimeoutExpired:
            receipt["own_child_SIGTERM"] = True
            child.send_signal(signal.SIGTERM)
            try:
                returncode = child.wait(timeout=max(.01, 1795-(time.monotonic()-START)))
            except subprocess.TimeoutExpired:
                receipt["own_child_SIGKILL"] = True
                child.kill()
                returncode = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt.update(status="terminal", child_returncode=returncode,
        wall_seconds=time.monotonic()-START, child_peak_RSS_KiB=after.ru_maxrss,
        child_user_CPU_seconds=after.ru_utime-before.ru_utime,
        child_system_CPU_seconds=after.ru_stime-before.ru_stime)
    try:
        receipt["locator_finalization"] = finish_locator(run)
    except Exception as exc:
        receipt["locator_finalization_error"] = f"{type(exc).__name__}: {exc}"
    for name in ("summary.json", "LARGE_FILES.tsv", "process.log"):
        path = run / name
        if path.exists():
            receipt[name] = identity(path)
    if (run / "summary.json").exists():
        receipt["producer_complete"] = json.loads((run / "summary.json").read_text()).get("complete", False)
    receipt["execution_complete"] = bool(returncode == 0 and receipt.get("producer_complete")
        and receipt.get("locator_finalization", {}).get("producer_raw_listing_was_bound"))
    receipt["wall_seconds"] = time.monotonic()-START
    atomic(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in
        ("status", "child_returncode", "wall_seconds", "child_peak_RSS_KiB")}))
    return returncode if returncode >= 0 else 128-returncode


if __name__ == "__main__":
    sys.exit(main())
