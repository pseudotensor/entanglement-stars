#!/usr/bin/env python
"""s16_rootgate_test_v1.py (busi; ReviewA_s4cost_v1 item 4 = inbox #438 (a)): the stored flat 8a member F256_s8's six-term root gate on
r >= R (the trapping root R_t) at T 4.375 to 4.5, doubled; under 0.1 -> F256_s16 GO, else NO-GO.
The number is the POSTGATE STOPPER'S OWN root_row (stop_flat_h100f_v2.py, --mode postgate, --sigma 8): the stopper's definitions are
executed from its source (everything above its main loop) with its own argparse, so root_row, six (gate_common_v1.load_sixterm of the
field module dir) and kop_rows are the stopper's, unchanged. A per-term breakdown re-runs the same lines of root_row with the six terms
kept apart (asserted equal to root_row's maximum). Reported beside, never deciding: the six-term max on r >= R_t out to 4 R_t, and the same
root_row evaluated with sigma 16 on the 8a's fields (nodes 16 apart, h = 16/32, times 16).
  s16_rootgate_test_v1.py STOPPER GC_CODE_DIR MEMBER_DIR TAG FM_DIR OUT_JSON
"""
import json, os, sys
import numpy as np

STOPPER, GCD, M, TAG, FM, OUTJ = sys.argv[1:7]
TS = [4.375, 4.4375, 4.5]
src = open(STOPPER).read()
head = src.split("\nt_start = dt.datetime.utcnow()")[0]


def defs(sigma):
    g = {"__file__": os.path.join(GCD, "stop_flat_h100f_v2.py"), "__name__": "stopper_defs"}
    sys.argv = ["stop_flat_h100f_v2.py", M, TAG, FM, "--sigma", str(sigma), "--mode", "postgate", "--from-T", "4.375", "--dry", "--once"]
    exec(compile(head, STOPPER, "exec"), g)
    return g


G = defs(8.0)
GC = G["GC"]; six = G["six"]; root_row = G["root_row"]; rs = 256.0
RUN = os.path.join(M, "run")
J = json.load(open(os.path.join(RUN, "collapse_%s.json" % TAG)))
F = np.load(os.path.join(RUN, "collapse_%s_fields.npz" % TAG), allow_pickle=True)
Rg = np.asarray(F["R"], float); RB = GC.recs_by_T(J, rs); IB = GC.rows_by_T(F, rs)
NAMES = ["|v'/N|", "|v/(N r)|", "|N'/N|", "sqrt|D k_r|", "sqrt|D k_t|", "sqrt|N''/N|"]


def terms(sigma, t, rmax_fac):
    """the six terms on root_row's own nodes (ceil(R_t/sigma) sigma ... rmax_fac R_t, sigma apart), same lines as root_row."""
    rec = RB[t]; i = IB[t]; Rt = rec.get("R_t")
    S0 = six.solve(F, i); n_ = len(F["T"])
    Sp = six.solve(F, i - 1) if i > 0 else None; Sn = six.solve(F, i + 1) if i + 1 < n_ else None
    r = np.arange(np.ceil(Rt / sigma) * sigma, rmax_fac * Rt + 1e-9, sigma)
    GP = six.GP; h = sigma / 32.0; Td = np.asarray(F["T"], float)
    k0 = GP.kfields(S0[0], S0[1], r, h)
    kp = GP.kfields(Sp[0], Sp[1], r, h) if Sp else None; kn = GP.kfields(Sn[0], Sn[1], r, h) if Sn else None
    if kp is not None and kn is not None:
        dTr = (kn["kr"] - kp["kr"]) / (Td[i + 1] - Td[i - 1]); dTt = (kn["kt"] - kp["kt"]) / (Td[i + 1] - Td[i - 1]); mode = "central"
    elif kp is not None:
        dTr = (k0["kr"] - kp["kr"]) / (Td[i] - Td[i - 1]); dTt = (k0["kt"] - kp["kt"]) / (Td[i] - Td[i - 1]); mode = "backward"
    else:
        dTr = (kn["kr"] - k0["kr"]) / (Td[i + 1] - Td[i]); dTt = (kn["kt"] - k0["kt"]) / (Td[i + 1] - Td[i]); mode = "forward"
    Dkr = (dTr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dTt + k0["v"] * k0["kt1"]) / k0["N"]
    T6 = np.array([np.abs(k0["kr"]), np.abs(k0["kt"]), np.abs(k0["N1"] / k0["N"]), np.sqrt(np.abs(Dkr)), np.sqrt(np.abs(Dkt)), np.sqrt(np.abs(k0["N2"] / k0["N"]))])
    per = {NAMES[j]: {"sigma_max": float(sigma * T6[j].max()), "r_at": float(r[int(np.argmax(T6[j]))])} for j in range(6)}
    L6 = T6.max(axis=0); j = int(np.argmax(L6))
    bind = NAMES[int(np.argmax(T6[:, j]))]
    prof = [[float(r[q]), float(sigma * L6[q])] + [float(sigma * T6[jj, q]) for jj in range(6)] for q in range(len(r))]
    return {"R_t": Rt, "nodes": [float(r[0]), float(r[-1]), len(r)], "six": float(sigma * L6[j]), "r_six": float(r[j]), "binding": bind,
            "per_term": per, "dT_mode": mode, "profile_first8": prof[:8]}


out = {"member": M, "tag": TAG, "fm_dir": FM, "stopper": STOPPER, "sigma": 8.0, "records": {}}
for t in TS:
    rr = root_row(F, RB, IB, Rg, t)          # (t, R_t, sigma||K||_op [R_t,2R_t], sigma|N'/N| max, six-term max, r_six): the stopper's own
    b = terms(8.0, t, 2.0)
    assert abs(b["six"] - rr[4]) < 1e-12 * max(1.0, rr[4]), (b["six"], rr[4])
    ext = terms(8.0, t, 4.0)
    out["records"]["%.4f" % t] = {"root_row": list(rr), "six_R_2R": rr[4], "doubled": 2 * rr[4], "breakdown_R_2R": b,
                                  "six_R_4R_reported": {"six": ext["six"], "r_six": ext["r_six"], "binding": ext["binding"]}}
    print("T %.4f R_t %.2f | stopper root_row: sKop %.4f sN'/N %.4f six %.4f at r %.1f | doubled %.4f | binds %s (%s) | R..4R %.4f at %.1f"
          % (t, rr[1], rr[2], rr[3], rr[4], rr[5], 2 * rr[4], b["binding"], b["dT_mode"], ext["six"], ext["r_six"]), flush=True)
    for k, v in b["per_term"].items():
        print("    %-12s sigma*max %.5f at r %.1f" % (k, v["sigma_max"], v["r_at"]), flush=True)
mx = max(v["six_R_2R"] for v in out["records"].values())
out["max_six"] = mx; out["doubled_max"] = 2 * mx; out["verdict"] = "GO" if 2 * mx < 0.1 else "NO-GO"
# beside (reported, not deciding): root_row itself at sigma 16 on the 8a's fields
G16 = defs(16.0); G16["six"].cache.clear()
G16["six"] = six  # share the re-solved fields (the solve is sigma-free)
out["sigma16_direct_reported"] = {}
for t in TS:
    rr = G16["root_row"](F, RB, IB, Rg, t)
    out["sigma16_direct_reported"]["%.4f" % t] = list(rr)
    print("beside: sigma 16 root_row on the 8a fields, T %.4f: sKop %.4f six %.4f at r %.1f" % (t, rr[2], rr[4], rr[5]), flush=True)
print("max six %.5f, doubled %.5f -> %s" % (mx, 2 * mx, out["verdict"]), flush=True)
json.dump(out, open(OUTJ, "w"), indent=1, default=str)
