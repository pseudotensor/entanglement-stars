"""h3ab_ptracer_v1.py (solve 8a7b5274, PERF H3 A/B on h100-f): run an engine file unchanged after prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY),
so that an unprivileged py-spy (not an ancestor) can attach under yama ptrace_scope 1. No numerics, no I/O patch; used identically in every arm.
Must sit beside the engines (sys.path[0] is then the engine's directory, as for a direct `python engine.py`).
Usage: python h3ab_ptracer_v1.py ENGINE.py ARGS..."""
import ctypes
import os
import runpy
import sys

try:
    rc = ctypes.CDLL(None, use_errno=True).prctl(0x59616D61, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)
except Exception as exc:  # the run goes on without py-spy
    rc = "error %s" % exc
engine = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(engine)
print("h3ab_ptracer_v1: prctl(PR_SET_PTRACER_ANY) -> %s; running %s" % (rc, engine), flush=True)
runpy.run_path(engine, run_name="__main__")
