#!/usr/bin/env python
"""v37_inject_rc.py (busi builder, analyst; the self-test harness of engine_collapse_v37_selftest.py's AE case, never for a member): runs an
engine's main() with confirm_v32 wrapped so that, from its K-th call on, the returned confirming residual r_c has ADD added (after the engine's
own stop-rule test has been computed from the true r_c). The stored solve_v32.confirming_residual and the record line then carry the injected
value. On the 256-site fixture every record's confirming residual is below its kept (ratio 0.02 to 0.08), so the end rule's reading of the
larger residual cannot be exercised on the confirming residual alone without it.
    python v37_inject_rc.py ENGINE --rc-add 5e-3 --rc-from-call 2 -- <the engine's arguments>"""
import importlib.util
import os
import sys


def main():
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    k = sys.argv.index("--")
    head, rest = sys.argv[1:k], sys.argv[k + 1:]
    eng = head[0]
    add = float(head[head.index("--rc-add") + 1]); frm = int(head[head.index("--rc-from-call") + 1])
    sys.path.insert(0, os.path.dirname(os.path.abspath(eng)))
    spec = importlib.util.spec_from_file_location("engine_under_injection", eng)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    orig = E.confirm_v32
    n = [0]

    def wrapped(a, b, evalG, it_hist, ctr):
        out = orig(a, b, evalG, it_hist, ctr)
        n[0] += 1
        if n[0] >= frm and out.get("r_c") is not None:
            print("  INJECT (self-test harness) confirm_v32 call %d: r_c %.3e -> %.3e (conv as the engine computed it: %s)" % (n[0], out["r_c"], out["r_c"] + add, out["conv"]), flush=True)
            out["r_c"] = out["r_c"] + add
        return out
    E.confirm_v32 = wrapped
    sys.argv = [eng] + rest
    E.main()


if __name__ == "__main__":
    main()
