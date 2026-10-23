# sitecustomize.py (busi, MERGE-1): at interpreter exit, print the process's torch CUDA peak (max_memory_allocated and max_memory_reserved,
# since the process start) to stderr. Reads counters only; nothing computed is changed. Put its directory on PYTHONPATH for an arm.
import atexit
import os
import sys


def _peak_v1():
    t = sys.modules.get("torch")
    try:
        if t is not None and t.cuda.is_initialized():
            d = t.cuda.current_device()
            sys.stderr.write("peakhook_v1: pid %d device %d max_memory_allocated %.3f GiB max_memory_reserved %.3f GiB\n" % (
                os.getpid(), d, t.cuda.max_memory_allocated(d) / 2**30, t.cuda.max_memory_reserved(d) / 2**30))
            sys.stderr.flush()
    except Exception as e:
        sys.stderr.write("peakhook_v1: %r\n" % (e,))


atexit.register(_peak_v1)
