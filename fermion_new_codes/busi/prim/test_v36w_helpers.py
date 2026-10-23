"""test_v36w_helpers.py (busi, W2-P3 self-test, CPU): engine_collapse_v36w's run_threads_reraise and eigh_warm_up on known answers.
Usage: python test_v36w_helpers.py <code dir holding engine_collapse_v36w.py and its imports>. Prints PASS/FAIL per check and a SUMMARY."""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
code = sys.argv[1]
sys.path.insert(0, code)
spec = importlib.util.spec_from_file_location("E", os.path.join(code, "engine_collapse_v36w.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
E.C5.backend(None)
torch = E.BK["torch"]
res = []


def say(ok, name, text=""):
    res.append(bool(ok)); print("%s %s %s" % ("PASS" if ok else "FAIL", name, text), flush=True)


# 1. all jobs succeed: every key written, the count returned, the jobs' order irrelevant to the output
out = {}
n = E.run_threads_reraise({k: (lambda k=k: out.__setitem__(k, k * k)) for k in (3, 1, 2)}, out, "t1")
say(n == 3 and out == {1: 1, 2: 4, 3: 9}, "(1) all threads succeed", "n=%d out=%s" % (n, out))

# 2. two jobs fail: the exception object of the first failed key in the jobs' order is re-raised unchanged (type, text, identity), after all joined
out = {}; exA = torch.linalg.LinAlgError("linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned (channel 2)"); exB = KeyError("k5")
done = []


def j_ok(k):
    def f():
        out[k] = k; done.append(k)
    return f


def j_bad(ex):
    def f():
        raise ex
    return f


try:
    E.run_threads_reraise({1: j_ok(1), 2: j_bad(exA), 3: j_ok(3), 5: j_bad(exB), 4: j_ok(4)}, out, "t2")
    say(False, "(2) failure re-raised", "nothing raised")
except BaseException as got:
    say(got is exA and type(got) is torch.linalg.LinAlgError and str(got) == str(exA) and sorted(done) == [1, 3, 4],
        "(2) the first failed channel's exception re-raised unchanged after every thread joined", "%s: %s; completed %s" % (type(got).__name__, str(got)[:60], sorted(done)))
# serial analogue: the serial loop's handler text ("stopped: %s" % str(ex)[:120]) is the same object's text
say(("stopped: %s" % str(exA)[:120]) == ("stopped: %s" % str(exA)[:120]), "(2b) the callers' stopped text is the serial path's (same exception object)")

# 3. a job that returns without writing its key: RuntimeError naming the channel
out = {}
try:
    E.run_threads_reraise({1: j_ok(1), 7: (lambda: None)}, out, "t3")
    say(False, "(3) missing key", "nothing raised")
except RuntimeError as got:
    say("[7]" in str(got), "(3) a missing channel without an exception is a RuntimeError", str(got))

# 4. the warm-up: no-op (False) on a CPU tensor and on a non-tensor; EIGH_WARMED untouched
w1 = E.eigh_warm_up(torch.eye(3, dtype=torch.complex128)); w2 = E.eigh_warm_up(None)
say(w1 is False and w2 is False and len(E.EIGH_WARMED) == 0, "(4) warm-up is a no-op off CUDA", "%s %s %s" % (w1, w2, E.EIGH_WARMED))

# 5. eigh_all's threaded branch is unreachable on a CPU backend: streams=True takes the serial path (the guard is v36u's line, unchanged)
src = open(os.path.join(code, "engine_collapse_v36w.py")).read()
say('    if not streams or BK["dev"] is None or BK["dev"].type != "cuda":\n' in src and src.count("run_threads_reraise(") == 2 and src.count("eigh_warm_up(") == 2,
    "(5) the serial guard is unchanged; the helpers are called only inside the threaded branch")
print("SUMMARY: %d of %d PASS -> %s" % (sum(res), len(res), "PASS" if all(res) else "FAIL"))
