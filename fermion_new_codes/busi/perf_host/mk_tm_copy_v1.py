"""mk_tm_copy_v1.py (busi; measurement only, never shipped): engine_fields_v16b_tm.py = engine_fields_v16b.py with perf_counter accumulators
around the final W integration and the two residuals_at calls (dict _H2_TM); the arithmetic is untouched."""
import sys
s = open(sys.argv[1]).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
rep("import os\n", "import os\nimport time as _tm_time\n_H2_TM = {}\n")
rep("    solW = solve_ivp(", "    _tm0 = _tm_time.perf_counter()\n    solW = solve_ivp(")
rep("    if not solW.success:\n", "    _H2_TM['W'] = _tm_time.perf_counter() - _tm0\n    if not solW.success:\n")
rep("        rW, rN, rW8, rN8 = residuals_at(ell / 64.0)\n" if "_H2_RESIDUAL_PAIR" in s else "    rW, rN, rW8, rN8 = residuals_at(ell / 64.0)\n",
    "    _tm1 = _tm_time.perf_counter()\n    rW, rN, rW8, rN8 = residuals_at(ell / 64.0)\n    _H2_TM['res64'] = _tm_time.perf_counter() - _tm1\n")
rep("    rW2, rN2, _, _ = residuals_at(ell / 32.0)\n", "    _tm2 = _tm_time.perf_counter()\n    rW2, rN2, _, _ = residuals_at(ell / 32.0)\n    _H2_TM['res32'] = _tm_time.perf_counter() - _tm2\n    _H2_TM['res'] = _H2_TM['res64'] + _H2_TM['res32']\n")
open(sys.argv[2], "w").write(s)
