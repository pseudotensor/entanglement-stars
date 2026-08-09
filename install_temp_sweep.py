"""Install the 0.027 / 0.135 temperature sweeps into exact_solutions.npz.

The sweeps come from temp_sweep_probe.py, which calls the SAME factories as
PHASE B2 -- make_residual_tempsweep(lam=LAM_CLOSURE) + make_newton_solver_3arg,
same N/t0/n_core/cstar_sq, same seed procedure -- so this is the Phase-B2
computation, run standalone.  compute_exact_data_gpu.py has been updated to
V0_TEMP_SPEC = ((0.027, False), (0.135, True)) and will reproduce these keys on
its next full run.

Every non-temp key is copied through unchanged; only the temp_* family is
replaced.  The old 0.001/0.005 tags are dropped, not left behind, so nothing can
read a stale pair.
"""
import sys, re
import numpy as np

src, out = sys.argv[1], sys.argv[2]
probes = sys.argv[3:]
d = np.load(src, allow_pickle=True)
keep = {k: d[k] for k in d.files if not k.startswith("temp")}
n_before = len(d.files)
n_temp_before = sum(1 for k in d.files if k.startswith("temp"))

new = {}
V0s, sh, rev, e_bt0, e_minN, e_reason = [], [], [], [], [], []
grid = None
for p in probes:
    q = np.load(p, allow_pickle=True)
    V0 = float(q["V0_derived"]); tag = f"temp_{V0:.8f}"
    bt = np.asarray(q["bt0"], float)
    order = np.argsort(bt)          # store ascending regardless of sweep direction
    new[f"{tag}_bt0"] = bt[order]
    new[f"{tag}_rs"] = np.asarray(q["rs"], float)[order]
    new[f"{tag}_proxy_rs"] = np.asarray(q["rs_proxy"], float)[order]
    new[f"{tag}_minN"] = np.asarray(q["minN"], float)[order]
    new[f"{tag}_F"] = np.asarray(q["F"], float)[order]
    new[f"{tag}_conv"] = np.asarray(q["conv"], bool)[order]
    new[f"{tag}_Phi"] = np.asarray(q["Phi"], float)[order]
    new[f"{tag}_eps_b"] = np.asarray(q["eps_b"], float)[order]
    # GM_est plateau, same estimator as PHASE B2 (mean over r in [60,150])
    r = np.asarray(q["r_arr"], float); R = float(r[-1])
    m = (r >= 60) & (r <= 150)
    P = new[f"{tag}_Phi"]
    G = -P[:, m] * r[m] * R / (R - r[m])
    new[f"{tag}_GM"] = G.mean(axis=1)
    new[f"{tag}_GM_std"] = G.std(axis=1) / np.maximum(G.mean(axis=1), 1e-30)
    new[f"{tag}_stagger"] = np.array([
        np.max(np.abs(np.diff(p_[:50]))) / max(np.max(np.abs(p_)), 1e-15)
        for p_ in P])
    V0s.append(V0); sh.append(float(q["V0_shorthand"]))
    reverse = bool(np.asarray(q["bt0"])[0] > np.asarray(q["bt0"])[-1])
    rev.append(reverse)
    if bool(q["fold_reached"]):
        # the branch stopped one step past the last stored point, in the
        # direction the sweep was travelling
        step = float(np.abs(np.diff(np.sort(bt))).min())
        e = (bt.min() - step) if reverse else (bt.max() + step)
        i = int(np.argmin(bt)) if reverse else int(np.argmax(bt))
        e_bt0.append(float(bt[i])); e_minN.append(float(np.asarray(q["minN"])[i]))
        e_reason.append("branch ends (last converged point)")
    else:
        e_bt0.append(np.nan); e_minN.append(np.nan); e_reason.append("")
    if grid is None or len(bt) > len(grid):
        grid = np.sort(bt)

new["temp_V0s"] = np.array(V0s)
new["temp_bt0"] = grid
new["temp_V0_shorthand"] = np.array(sh)
new["temp_reverse"] = np.array(rev)
new["temp_end_bt0"] = np.array(e_bt0)
new["temp_end_minN"] = np.array(e_minN)
new["temp_end_reason"] = np.array(e_reason)

merged = {**keep, **new}
# integrity: nothing outside the temp_ family may change
assert all(k in merged for k in keep), "dropped a non-temp key"
np.savez(out, **merged)
chk = np.load(out, allow_pickle=True)
print(f"in  : {n_before} keys ({n_temp_before} temp)")
print(f"out : {len(chk.files)} keys ({sum(1 for k in chk.files if k.startswith('temp'))} temp)")
print(f"temp_V0s        = {chk['temp_V0s']}")
print(f"temp_V0_shorthand = {chk['temp_V0_shorthand']}")
print(f"temp_end_bt0    = {chk['temp_end_bt0']}")
for k in keep:
    a, b = np.asarray(keep[k]), np.asarray(chk[k])
    assert a.shape == b.shape and (a == b).all() if a.dtype.kind not in 'fc' \
        else np.allclose(a, b, equal_nan=True), f"non-temp key changed: {k}"
print("all non-temp keys identical: OK")
