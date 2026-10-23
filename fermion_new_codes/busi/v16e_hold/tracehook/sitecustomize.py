# sitecustomize.py (busi, v16e hold readings, ReviewA_v16e_v1.md sec. 2 (i)): a read-only trace of every field solve on the
# collocation route. When engine_fields_v16 is imported, its module-level _rd2_solve (the RD-2 dispatch, looked up by name at each
# call from solve_constraints_ivp) is wrapped: after the original returns, one JSON line per call goes to
# $QI2_TRACE_DIR/fieldtrace_<pid>.jsonl with the call's sequence number, the caller chain (function names), the call's rtol/atol,
# the route ("collocation" or "ivp" with why), the branch (negative centre or regular), r_*, U'(r_*), the centre's regular_start flag,
# the solve's Newton iterations and failed flag. The arguments and the returned objects are not touched, so nothing computed changes.
# Also prints the torch CUDA peak at exit (MERGE-1's peakhook_v1, unchanged).
import atexit
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time

_DIR = os.environ.get("QI2_TRACE_DIR")
_SEQ = [0]
_FH = [None]


def _out():
    if _FH[0] is None:
        _FH[0] = open(os.path.join(_DIR, "fieldtrace_%d.jsonl" % os.getpid()), "a", buffering=1)
    return _FH[0]


def _wrap(orig):
    def _rd2_solve(eps_tot, J_tot, g, R, ell, kw):
        t0 = time.time()
        res = orig(eps_tot, J_tot, g, R, ell, kw)
        try:
            _SEQ[0] += 1
            info = res[3]
            fr = info.get("fields_route") or {}
            csc = info.get("centre_series_check") or {}
            names = []
            f = sys._getframe(1)
            while f is not None and len(names) < 12:
                names.append(f.f_code.co_name)
                f = f.f_back
            rec = {"seq": _SEQ[0], "t": t0, "dt": time.time() - t0, "stack": names, "rtol": kw.get("rtol"), "atol": kw.get("atol"),
                   "route": fr.get("route"), "why": fr.get("why"), "reason": (fr.get("reason") or "")[:160] or None,
                   "branch": ("negative" if str(fr.get("branch", "")).startswith("negative") else ("regular" if fr.get("route") == "collocation" else None)),
                   "r_star": fr.get("r_star"), "U_prime_at_r_star": fr.get("U_prime_at_r_star"), "regular_start": csc.get("regular_start"),
                   "newton_iterations": fr.get("newton_iterations"), "failed": info.get("failed")}
            _out().write(json.dumps(rec, default=str) + "\n")
        except Exception as e:
            try:
                _out().write(json.dumps({"seq": _SEQ[0], "trace_error": repr(e)}) + "\n")
            except Exception:
                pass
        return res
    return _rd2_solve


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        if hasattr(module, "_rd2_solve"):
            module._rd2_solve = _wrap(module._rd2_solve)
            sys.stderr.write("tracehook_v1: engine_fields_v16._rd2_solve wrapped (pid %d, %s)\n" % (os.getpid(), getattr(module, "__file__", "?")))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "engine_fields_v16":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if _DIR:
    os.makedirs(_DIR, exist_ok=True)
    sys.meta_path.insert(0, _Finder())


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
