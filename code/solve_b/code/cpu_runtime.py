"""Admission checks for this lane's remote CPU-only recomputations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def require_remote_cpu(output_directory: str | Path) -> dict:
    """Fail before numerical work if the owner's execution limits are unmet."""
    if os.environ.get("QI2_REMOTE_CPU") != "1":
        raise RuntimeError("Set QI2_REMOTE_CPU=1 only in the SSH job on compute-host")
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    if "compute-host" not in addresses:
        raise RuntimeError("Numerical work is restricted to compute-host")
    if Path(sys.executable).parent != Path("<internal>/bin"):
        raise RuntimeError("Use the qi2 conda environment's Python")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("Numerical jobs require nice -n 10")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = os.environ.get(name, "")
        if not value.isdigit() or not 1 <= int(value) <= 16:
            raise RuntimeError(f"Set {name} to a value from 1 to 16")
    threads = len(list(Path("/proc/self/task").iterdir()))
    if threads > 16:
        raise RuntimeError(f"Process has {threads} threads; maximum is 16")
    output_directory = Path(output_directory).resolve()
    bulk = Path("<internal>/solve_b_bulk").resolve()
    if not output_directory.is_relative_to(bulk):
        raise RuntimeError("Numerical outputs belong under <internal>/solve_b_bulk")
    stats = os.statvfs("/")
    if stats.f_bavail * stats.f_frsize < 100 * 10**9:
        raise RuntimeError("Root filesystem must retain at least 100 GB free")
    output_directory.mkdir(parents=True, exist_ok=True)
    return {"host": "compute-host", "pid": os.getpid(), "threads": threads,
            "niceness": os.getpriority(os.PRIO_PROCESS, 0), "device": "CPU"}
