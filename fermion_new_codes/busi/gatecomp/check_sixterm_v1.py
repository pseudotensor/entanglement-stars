#!/usr/bin/env python
"""check_sixterm_v1.py (busi, GATECOMP (iii); ReviewA_gatecomp_v1 sec. 3.8 'gate diagnostics by two codes'). The second code: the solve
lane's engine_core_slices_v3.kfields (imported read-only from a byte copy, md5 printed) with that script's own six-term lines (its
lines 192-205, applied here unchanged in form) at its nodes r = ell, 2 ell, ... up to --rmax; set beside gate_profiles_v1's
profile_every_sigma (the same nodes) for the named records. Prints the largest relative difference of sigma Lambda_geom per record."""
import argparse
import hashlib
import importlib
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True); ap.add_argument("--fm-dir", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--gate-json", required=True); ap.add_argument("--T", required=True); ap.add_argument("--rs", type=float, required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.fm_dir); sys.path.insert(0, a.ext)
    CS = importlib.import_module("engine_core_slices_v3"); RW = importlib.import_module("engine_rows_v2")
    FM = importlib.import_module("engine_fields_v16")
    print("engine_core_slices_v3 md5", hashlib.md5(open(CS.__file__, "rb").read()).hexdigest(), "engine_rows_v2 md5", hashlib.md5(open(RW.__file__, "rb").read()).hexdigest())
    Z = np.load(a.data, allow_pickle=True); Td = np.asarray(Z["T"], float); Rd = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"])
    G = {round(r["T_over_rs"], 6): r for r in json.load(open(a.gate_json))["rows"]}
    out = []
    for Ts in [float(x) for x in a.T.split(",")]:
        kd = int(np.argmin(np.abs(Td / a.rs - Ts)))
        prof = G[round(Ts, 6)]["results"]["h32"]["profile_every_sigma"]
        rr = np.array(prof["r"]); mine = np.array(prof["sigma_six"]); mineK = np.array(prof["sigma_Kop"])
        nod = rr[(rr >= ell - 1e-9)]
        hh = ell / 32.0
        k0, _ = CS.kfields(FM, Z, kd, g, Rd, ell, nod, hh)
        kp = CS.kfields(FM, Z, kd - 1, g, Rd, ell, nod, hh)[0] if kd - 1 >= 0 else None
        kn = CS.kfields(FM, Z, kd + 1, g, Rd, ell, nod, hh)[0] if kd + 1 < len(Td) else None
        if kp is not None and kn is not None:
            dT_kr = (kn["kr"] - kp["kr"]) / (Td[kd + 1] - Td[kd - 1]); dT_kt = (kn["kt"] - kp["kt"]) / (Td[kd + 1] - Td[kd - 1]); mode = "central"
        else:
            dT_kr = (k0["kr"] - kp["kr"]) / (Td[kd] - Td[kd - 1]); dT_kt = (k0["kt"] - kp["kt"]) / (Td[kd] - Td[kd - 1]); mode = "backward"
        Dkr = (dT_kr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dT_kt + k0["v"] * k0["kt1"]) / k0["N"]
        terms = {"abs(v'/N)": np.abs(k0["kr"]), "abs(v/(N r))": np.abs(k0["kt"]), "abs(N'/N)": np.abs(k0["N1"] / k0["N"]),
                 "sqrt(abs(D k_r))": np.sqrt(np.abs(Dkr)), "sqrt(abs(D k_t))": np.sqrt(np.abs(Dkt)), "sqrt(abs(N''/N))": np.sqrt(np.abs(k0["N2"] / k0["N"]))}
        lam = ell * np.max(np.array(list(terms.values())), axis=0)
        kop = ell * np.maximum(np.abs(k0["kr"]), np.abs(k0["kt"]))
        m = mine[rr >= ell - 1e-9]; mK = mineK[rr >= ell - 1e-9]
        rel = float(np.max(np.abs(lam - m) / np.abs(lam))); relK = float(np.max(np.abs(kop - mK) / np.abs(kop)))
        print("T/rs %.4f (%s, %d nodes): max rel diff sigma*Lambda_geom %.2e, sigma*||K|| %.2e; max over nodes: theirs %.6f mine %.6f" % (
            Ts, mode, len(nod), rel, relK, lam.max(), m.max()), flush=True)
        out.append({"T_over_rs": Ts, "dT_mode": mode, "nodes": len(nod), "rel_six": rel, "rel_Kop": relK, "max_theirs": float(lam.max()), "max_mine": float(m.max())})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
