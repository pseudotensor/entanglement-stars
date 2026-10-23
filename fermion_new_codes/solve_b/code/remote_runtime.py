"""Remote qi2 launch admission and an in-process visible CUDA allocation.

The device must be free at launch, and this job must be its sole compute
PID immediately after allocation, before work. The lane must keep one
own compute process on that device until this job finishes. Admission
does not assert exclusive ownership throughout a correctly admitted run.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID


def hold_cuda_slot(expected_uuid: str):
    """Use the same checked Driver API as the executed T10 certificate."""
    driver = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
        "cuCtxCreate_v2": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
        "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t],
        "cuMemFree_v2": [ctypes.c_ulonglong],
        "cuCtxDestroy_v2": [ctypes.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = arguments, ctypes.c_int

    def call(name, *arguments):
        error = getattr(driver, name)(*arguments)
        if error:
            raise RuntimeError(f"CUDA Driver API {name} failed with code {error}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    selected = None
    for ordinal in range(count.value):
        device = ctypes.c_int()
        identifier = (ctypes.c_ubyte * 16)()
        call("cuDeviceGet", ctypes.byref(device), ordinal)
        call("cuDeviceGetUuid", ctypes.byref(identifier), device.value)
        if "GPU-" + str(UUID(bytes=bytes(identifier))) == expected_uuid:
            selected = device.value
            break
    if selected is None:
        raise RuntimeError("The assigned GPU UUID is unavailable to this process")
    context, memory = ctypes.c_void_p(), ctypes.c_ulonglong()
    call("cuCtxCreate_v2", ctypes.byref(context), 0, selected)
    allocation_bytes = 1 << 20
    call("cuMemAlloc_v2", ctypes.byref(memory), allocation_bytes)

    def release():
        call("cuMemFree_v2", memory)
        call("cuCtxDestroy_v2", context)

    atexit.register(release)
    return {"assigned_gpu_uuid": expected_uuid, "cuda_allocation_bytes": allocation_bytes,
            "cuda_context_in_this_single_job_process": True}


def remote_execution(max_threads=8):
    if os.environ.get("QI2_REMOTE_RUN") != "1":
        raise RuntimeError("Execution requires the explicitly selected remote qi2 job")
    if Path(sys.executable).parent != Path("<internal>/bin"):
        raise RuntimeError("The qi2 Python interpreter is required")
    niceness = os.getpriority(os.PRIO_PROCESS, 0)
    if niceness < 10:
        raise RuntimeError("The compute job must run at nice 10 or lower priority")
    expected_uuid = os.environ.get("QI2_GPU_UUID")
    if not expected_uuid:
        raise RuntimeError("The freshly checked GPU UUID is required")
    owners = cuda_device_owners(expected_uuid)
    if owners:
        raise RuntimeError(f"Assigned GPU is occupied before allocation: {sorted(owners)}")
    result = hold_cuda_slot(expected_uuid)
    result["same_process_fresh_device_query_before_allocation"] = True
    result.update(verify_cuda_ownership(expected_uuid))
    threads = len(list(Path("/proc/self/task").iterdir()))
    if threads > max_threads:
        raise RuntimeError(f"Job has {threads} threads; permitted maximum is {max_threads}")
    result.update({"single_process_pid": os.getpid(), "niceness": niceness,
                   "threads_at_launch": threads, "maximum_threads": max_threads})
    return result


def cuda_device_owners(expected_uuid):
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    owners = set()
    for line in listing.splitlines():
        identifier, pid = [field.strip() for field in line.split(",", 1)]
        if identifier == expected_uuid:
            owners.add(int(pid))
    return owners


def verify_cuda_ownership(expected_uuid):
    """At launch, check immediately after allocation and before work."""
    owners = cuda_device_owners(expected_uuid)
    if owners != {os.getpid()}:
        raise RuntimeError(f"Assigned GPU ownership changed before numerical work: {sorted(owners)}")
    return {"post_allocation_single_owned_compute_pid_verified": True}
