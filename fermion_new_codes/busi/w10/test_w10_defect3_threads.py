#!/usr/bin/env python
"""test_w10_defect3_threads.py (busi W10): defect 3 of REVIEWA_REVIEW_web40i section 5. The eigensolve's thread join
(engine_collapse_v36d.eigh_all, the --streams 1 branch) does not re-raise a thread's exception: a failed channel leaves the dictionary
partial (the caller's eig.pop raises KeyError far from the cause, or the Newton wrapper records "stopped: <kappa>") and the counter
TM["eigh_n"] += len(chs) overcounts. Gate: a test injecting a failure in one thread.
(1) the v36d pattern reproduced: a ValueError injected in one of three threads is swallowed, the output lacks the key, the counter counts 3;
(2) w10_lib_v1.run_threads_reraise (used by engine_collapse_v36e.eigh_all): the exception is re-raised after the join naming the channel,
    the output holds the two completed channels, the count is 2; (3) the no-failure path: every key, no exception;
(4) --gpu N (the A6000, with W10_ENGINE_DIR holding engine_collapse_v36e.py and its imports): engine_collapse_v36e.eigh_all with streams
    on 3 channels and torch.linalg.eigh patched to fail on one call: a RuntimeError naming the channel and TM["eigh_n"] 2.
Run: qi2 python test_w10_defect3_threads.py [--gpu 0]"""
import argparse
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def v36d_join(jobs):
    """engine_collapse_v36d.eigh_all's pattern: the threads started and joined, nothing re-raised, the counter adds len(jobs)."""
    out = {}

    def work(k):
        out[k] = jobs[k]()
    th = [threading.Thread(target=work, args=(k,)) for k in jobs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out, len(jobs)


def compute(k):
    return ("e%d" % k, "U%d" % k)


def fail():
    raise ValueError("injected failure in channel 2")


def gpu_part(gpu):
    for d in (os.environ.get("W10_ENGINE_DIR"),):
        if d and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import torch
        import engine_collapse_v36e as E
    except Exception as ex:  # noqa: BLE001
        check(False, "GPU part: engine_collapse_v36e not importable (%s): run w10_apply_fixes_v1.py and set W10_ENGINE_DIR" % ex)
        return
    E.C5.backend(gpu)
    chs = [E.Chan(k, 0.2, 256, False) for k in (1, 2, 3)]
    N = np.ones(256); v = np.zeros(256)
    E.tm_reset()
    real_eigh = torch.linalg.eigh
    calls = {"n": 0}
    lock = threading.Lock()

    def bad_eigh(M, *args, **kw):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n == 2:
            raise RuntimeError("injected eigh failure")
        return real_eigh(M, *args, **kw)
    torch.linalg.eigh = bad_eigh
    raised = None
    try:
        E.eigh_all(chs, N, v, streams=True)
    except RuntimeError as ex:
        raised = ex
    finally:
        torch.linalg.eigh = real_eigh
    check(raised is not None and "injected eigh failure" in str(raised) and "eigh_all (streams)" in str(raised),
          "GPU: engine_collapse_v36e.eigh_all (streams) re-raises the injected failure: %s" % (str(raised)[:160] if raised else "nothing raised"))
    check(E.TM["eigh_n"] == 2, "GPU: TM['eigh_n'] counts the 2 completed decompositions (v36d counted 3): %d" % E.TM["eigh_n"])
    E.tm_reset()
    out = E.eigh_all(chs, N, v, streams=True)
    check(set(out) == {1, 2, 3} and E.TM["eigh_n"] == 3, "GPU: without a failure every channel is present and the count is 3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None, help="A6000: also run engine_collapse_v36e.eigh_all with streams and an injected eigh failure")
    a = ap.parse_args()
    jobs = {1: lambda: compute(1), 2: fail, 3: lambda: compute(3)}
    hook = threading.excepthook
    threading.excepthook = lambda args: None                  # the swallowed traceback is the defect; the log is kept clean here
    try:
        out, counter = v36d_join(jobs)
    finally:
        threading.excepthook = hook
    check(set(out) == {1, 3} and counter == 3, "defect 3 reproduced (the v36d join): the injected failure in channel 2 is swallowed, the dictionary holds %s (partial), the counter counts %d of 2 completed (overcount)" % (sorted(out), counter))
    out2 = {}
    jobs2 = {1: (lambda: out2.__setitem__(1, compute(1))), 2: fail, 3: (lambda: out2.__setitem__(3, compute(3)))}
    raised = None
    try:
        W10.run_threads_reraise(jobs2, out2, "eigh_all (streams)")
    except RuntimeError as ex:
        raised = ex
    check(raised is not None and "injected failure in channel 2" in str(raised) and isinstance(raised.__cause__, ValueError),
          "gate (a test injecting a failure in one thread): run_threads_reraise re-raises after the join: %s" % (str(raised)[:170] if raised else "nothing raised"))
    check(set(out2) == {1, 3}, "the output holds the 2 completed channels %s; the failed channel 2 is named in the message, the count is 2 not 3" % sorted(out2))
    out3 = {}
    n3 = W10.run_threads_reraise({k: (lambda k=k: out3.__setitem__(k, compute(k))) for k in (1, 2, 3)}, out3)
    check(n3 == 3 and set(out3) == {1, 2, 3}, "no-failure path: 3 of 3 channels, no exception (the fix changes nothing when no thread fails)")
    if a.gpu is not None:
        gpu_part(a.gpu)
    print("test_w10_defect3_threads: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
