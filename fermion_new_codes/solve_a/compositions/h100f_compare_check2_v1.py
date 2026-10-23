#!/usr/bin/env python
"""h100f_compare_check2_v1.py (solve 8a7b5274, COMP1 h100-f checks, Check 2: the replicate).

The control's 50 to 60 rerun on engine_twofield_v3 on h100-f (H100) from the control's stored v3-written T/r_s 50 slice (pod 1), against R2's
stored control records (engine_twofield_v8 implicit on R2) over the same clocks 50.0625 to 60, and the h100-f T/r_s 60 (and 55) orbitals slice
against R2's v8-written slice at the same clock (60 only on R2's side as copied; 55 if given).

Prints and writes (json, argv[1] or default) per record:
  - the json leaves: count, number differing, worst relative difference (leaf name), and the leaf percentiles;
  - named quantities: E, E/E0, H_in, the heavy field's collar quantities (s11 ball_beyond_{8,16,24} weight and energy_density_route), the light
    channels' E_inv / H_in totals, M9 cut currents; their absolute and relative differences;
  - the fields npz rows (N, v, eps_tot, J_tot, eps_L*, J_L*, N_applied, v_applied): max |d| and max |d| / max |x|.
Slices: N_applied, v_applied, eigenvalues per channel, mask bits, sea matrices (propagated, same gauge: direct difference), and the R2 reference
columns' subspace distance ||X - X' (X'^H X)||_F (gauge free, stable form; v1 first used 2n - 2||X^H X'||^2, cancellation-limited at
~1e-9 absolute, replaced before any number was reported), on the GPU.
Read-only on its inputs."""
import json, os, sys
import numpy as np

W = "<internal>/comp1"
TAG = "C32_s4_ell4_tau2_twofield_v3_off"
R2J = W + "/in/R2run/collapse_%s.json" % TAG
R2F = W + "/in/R2run/collapse_%s_fields.npz" % TAG
HJ = W + "/out/check2_v3/collapse_%s.json" % TAG
HF = W + "/out/check2_v3/collapse_%s_fields.npz" % TAG
SL_H = W + "/out/check2_v3/orbitals_%s_T%07.2f.npz"
SL_R = W + "/in/slices/orbitals_%s_T%07.2f.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else W + "/out/check2_compare_v1.json"
SKIP = ("seconds", "timing", "slice_saved", "light_slice_saved", "engine_v34", "lag_s", "jvp_v35", "kernel_cache")


def leaves(o, path=(), out=None):
    if out is None:
        out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in str(k) for s in SKIP):
                continue
            leaves(v, path + (str(k),), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            leaves(v, path + (str(i),), out)
    else:
        out[".".join(path)] = o
    return out


def get(r, path):
    o = r
    for p in path.split("."):
        if isinstance(o, dict) and p in o:
            o = o[p]
        else:
            return None
    return o


NAMED = ["E", "E_over_E0", "H_in", "M1_I_E", "R_c", "N_min", "min_N_minus_absv",
         "s11.ball_beyond_8.weight", "s11.ball_beyond_8.energy_density_route",
         "s11.ball_beyond_16.weight", "s11.ball_beyond_16.energy_density_route",
         "s11.ball_beyond_24.weight", "s11.ball_beyond_24.energy_density_route",
         "light_v36.E_inv_heavy", "light_v36.H_in_light", "light_v36.I_E_light", "light_v36.I_E_light_sourcing",
         "M9.collar.I_E_cut", "M9.0.5 R_sh.I_E_cut"]


def main():
    JH = json.load(open(HJ)); JR = json.load(open(R2J))
    print("scripts:", JH["script"], "|", JR["script"])
    RH = {round(r["T_over_rs"], 6): r for r in JH["records"] if r["T_over_rs"] > 50 + 1e-9}
    RR = {round(r["T_over_rs"], 6): r for r in JR["records"]}
    clocks = sorted(t for t in RH if t in RR)
    print("h100-f new records %d (%.4f to %.4f); matched with R2: %d" % (len(RH), min(RH), max(RH), len(clocks)))
    res = {"clocks": clocks, "per_record": [], "named": {n: [] for n in NAMED}, "fields": {}, "slices": {}}
    allrel = []
    for t in clocks:
        la, lb = leaves(RH[t]), leaves(RR[t])
        nnum = ndiff = 0; worst = (0.0, None); rels = []
        for k in set(la) & set(lb):
            a, b = la[k], lb[k]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
                nnum += 1
                if a != b:
                    ndiff += 1
                    rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
                    rels.append(rel)
                    if rel > worst[0]:
                        worst = (rel, k)
        q = np.sort(np.array(rels)) if rels else np.zeros(1)
        res["per_record"].append({"T_over_rs": t, "numbers": nnum, "differ": ndiff, "worst_rel": worst[0], "worst_leaf": worst[1],
                                  "median_rel": float(q[len(q) // 2]), "p90_rel": float(q[int(0.9 * (len(q) - 1))])})
        for n in NAMED:
            a, b = get(RH[t], n), get(RR[t], n)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                res["named"][n].append([t, a, b, a - b, abs(a - b) / max(abs(a), abs(b), 1e-300)])
            else:
                res["named"][n].append([t, a, b, None, None])
    print("\nper record (json leaves, h100-f v3 vs R2 stored v8):")
    for p in res["per_record"]:
        if abs(p["T_over_rs"] * 4 - round(p["T_over_rs"] * 4)) < 1e-9 or p is res["per_record"][0]:
            print("  T/rs %8.4f: %d numbers, %d differ, median rel %.2e, 90th %.2e, worst %.2e (%s)" % (
                p["T_over_rs"], p["numbers"], p["differ"], p["median_rel"], p["p90_rel"], p["worst_rel"], p["worst_leaf"]))
    print("\nnamed quantities: relative |d| at 50.0625, 51, 52.5, 55, 57.5, 60 and the max over 50.0625 to 60")
    for n in NAMED:
        rows = [r for r in res["named"][n] if r[4] is not None]
        if not rows:
            print("  %-45s n/a" % n); continue
        pick = {round(r[0], 4): r for r in rows}
        s = "  ".join("%.2e" % pick[c][4] if c in pick else "  -   " for c in (50.0625, 51.0, 52.5, 55.0, 57.5, 60.0))
        mx = max(rows, key=lambda r: r[4])
        print("  %-45s %s | max %.2e at %.4f; value at 60 h100 %r R2 %r d %.3e" % (n, s, mx[4], mx[0], pick.get(60.0, [0, None])[1], pick.get(60.0, [0, 0, None])[2], pick.get(60.0, [0, 0, 0, float('nan')])[3]))
    # fields npz rows
    FH = np.load(HF, allow_pickle=True); FR = np.load(R2F, allow_pickle=True)
    TH, TR = np.asarray(FH["T"], float), np.asarray(FR["T"], float)
    rs = 32.0
    keys = [k for k in FH.files if k in FR.files and FH[k].dtype != object and FH[k].ndim == 2 and FH[k].shape[0] == len(TH)]
    print("\nfields npz rows: max |d| / max|x| at selected clocks (h100-f vs R2)")
    for k in keys:
        A, B = FH[k], FR[k]
        ser = []
        for t in clocks:
            iH = int(np.argmin(np.abs(TH - t * rs))); iR = int(np.argmin(np.abs(TR - t * rs)))
            if abs(TH[iH] - t * rs) > 1e-6 or abs(TR[iR] - t * rs) > 1e-6:
                continue
            d = float(np.max(np.abs(A[iH] - B[iR]))); s = float(np.max(np.abs(B[iR])))
            ser.append([t, d, d / s if s > 0 else 0.0])
        res["fields"][k] = ser
        pick = {round(r[0], 4): r for r in ser}
        print("  %-12s %s | max %.2e" % (k, "  ".join("%.2e" % pick[c][2] if c in pick else "   -    " for c in (50.0625, 51.0, 52.5, 55.0, 57.5, 60.0)), max(r[2] for r in ser) if ser else float("nan")))
    # slices
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for T in (55.0, 60.0):
        ph, pr = SL_H % (TAG, T), SL_R % (TAG, T)
        if not (os.path.exists(ph) and os.path.exists(pr)):
            print("\nslice %g: not both present (%s, %s)" % (T, os.path.exists(ph), os.path.exists(pr))); continue
        ZH, ZR = np.load(ph), np.load(pr)
        out = {"T_h100": float(ZH["T"]) / rs, "T_R2": float(ZR["T"]) / rs}
        for k in ("N_applied", "v_applied"):
            a, b = np.asarray(ZH[k]), np.asarray(ZR[k]); out[k] = [float(np.max(np.abs(a - b))), float(np.max(np.abs(a - b)) / np.max(np.abs(b)))]
        chans = [str(i) for i in range(1, 11)] + ["L1", "L2", "L3"]
        out["channels"] = {}
        for c in chans:
            q = {}
            e1, e2 = np.asarray(ZH["e_%s" % c], float), np.asarray(ZR["e_%s" % c], float)
            q["e_max_abs"] = float(np.max(np.abs(e1 - e2)))
            q["mask_equal"] = bool(np.array_equal(np.asarray(ZH["mask_%s" % c]), np.asarray(ZR["mask_%s" % c])))
            S1 = torch.as_tensor(np.asarray(ZH["sea_%s" % c]), device=dev); S2 = torch.as_tensor(np.asarray(ZR["sea_%s" % c]), device=dev)
            q["sea_max_abs"] = float((S1 - S2).abs().max()); q["sea_rel_fro"] = float(torch.linalg.norm(S1 - S2) / torch.linalg.norm(S2))
            # gauge-free distance of the sea subspaces, stable form (2n - 2||S1^H S2||^2 cancels at n = 2048): ||S1 - S2 (S2^H S1)||_F
            q["sea_proj_fro"] = float(torch.linalg.norm(S1 - S2 @ (S2.conj().T @ S1)))
            del S1, S2
            R1 = torch.as_tensor(np.asarray(ZH["R2_%s" % c]), device=dev); R2 = torch.as_tensor(np.asarray(ZR["R2_%s" % c]), device=dev)
            q["ref_proj_fro"] = float(torch.linalg.norm(R1 - R2 @ (R2.conj().T @ R1)))   # stable gauge-free subspace distance
            del R1, R2
            out["channels"][c] = q
        res["slices"][str(T)] = out
        print("\nslice T/rs %g (h100-f v3 vs R2 v8): N_applied max|d| %.2e (rel %.2e), v_applied %.2e (rel %.2e)" % (T, out["N_applied"][0], out["N_applied"][1], out["v_applied"][0], out["v_applied"][1]))
        for c, q in out["channels"].items():
            print("  ch %-3s e max|d| %.2e  mask equal %s  sea max|d| %.2e rel_F %.2e  sea subspace dist %.2e  ref subspace dist %.2e" % (
                c, q["e_max_abs"], q["mask_equal"], q["sea_max_abs"], q["sea_rel_fro"], q["sea_proj_fro"], q["ref_proj_fro"]))
    json.dump(res, open(OUT, "w"), default=float)
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
