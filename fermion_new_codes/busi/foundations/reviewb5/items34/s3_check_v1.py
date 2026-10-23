#!/usr/bin/env python
"""s3_check_v1.py (busi, reviewb5 item 9 for items 3/4). Independent checks of the s2 outputs, box CPU.

 (1) fact 1 (the periodic no-go), exact in sympy: 16 pi G N^2 rho_g = div F - (1 + b)/2 |curl v|^2 pointwise,
     F = v div v - (v . grad) v; so on a torus 16 pi G int N^2 rho_g = -(1 + b)/2 int |curl v|^2.
 (2) the Gram lower bound by a second, independent method: interval Cholesky (mpmath iv) of Gamma - lam_lo I, with Gamma's stored
     entries widened by their stored rounding radius; success proves lambda_min(Gamma_exact) >= lam_lo (d = 4 and d = 3).
 (3) the neighbourhood arithmetic redone in interval arithmetic from the stored parts.
Output: <out>/s3_check_v1.json."""
import json
import os
import sys

import numpy as np
import sympy as S
from mpmath import iv, mpf

OUT = "<internal>/items34"


def fact1():
    x = S.symbols("x1:4", real=True)
    b, G = S.symbols("b G", real=True)
    v = [S.Function("v%d" % i)(*x) for i in range(3)]
    gv = S.Matrix(3, 3, lambda i, j: S.diff(v[j], x[i]))
    Sm = (gv + gv.T) / 2; Am = (gv - gv.T) / 2
    N2rho16piG = Sm.trace() ** 2 - (Sm * Sm).trace() - b * (Am.T * Am).trace()
    div = sum(S.diff(v[i], x[i]) for i in range(3))
    F = [v[i] * div - sum(v[j] * S.diff(v[i], x[j]) for j in range(3)) for i in range(3)]
    divF = sum(S.diff(F[i], x[i]) for i in range(3))
    curl = [S.diff(v[2], x[1]) - S.diff(v[1], x[2]), S.diff(v[0], x[2]) - S.diff(v[2], x[0]), S.diff(v[1], x[0]) - S.diff(v[0], x[1])]
    c2 = sum(c ** 2 for c in curl)
    return bool(S.simplify(S.expand(N2rho16piG - (divF - (1 + b) / 2 * c2))) == 0)


def iv_cholesky_ok(Gm, rad, lam_lo):
    n = Gm.shape[0]
    A = [[iv.mpf([mpf(float(Gm[i, j])) - mpf(float(rad[i, j])), mpf(float(Gm[i, j])) + mpf(float(rad[i, j]))]) for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] = A[i][i] - iv.mpf(lam_lo)
    Lm = [[iv.mpf(0)] * n for _ in range(n)]
    for j in range(n):
        s = A[j][j] - sum((Lm[j][k] * Lm[j][k] for k in range(j)), iv.mpf(0))
        if not s.a > 0:
            return False, j
        Lm[j][j] = iv.sqrt(s)
        for i in range(j + 1, n):
            t = A[i][j] - sum((Lm[i][k] * Lm[j][k] for k in range(j)), iv.mpf(0))
            Lm[i][j] = t / Lm[j][j]
    return True, n


def jets_crosscheck(Z, d):
    """agent A's independent background splines (pm1_common_v1.RadialBG on the shared cache) and laws against B's FD jets and laws."""
    sys.path.insert(0, "<internal>/common")
    sys.path.insert(0, "<internal>/items34")
    import pm1_common_v1 as PM
    import items34_lib as L
    C = np.load("<internal>/pm1_background_v1.npz")
    bg = PM.RadialBG(C["r"], C["N"], C["v"], C["rho"], C["J"])
    Y = Z["Y"]; J0 = Z["jet0"]
    N, dN, dv, ddv = bg.jets3d(Y)
    mine = {"N": J0[:, 0], "dN": J0[:, 1:4], "dv": J0[:, 4:13].reshape(-1, 3, 3)}
    ddm = np.zeros_like(ddv)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                ddm[:, k, i, j] = J0[:, L.jet_index(L.ddv_name(k, i, j))]
    rho, Jv = PM.laws(N, dN, dv, ddv, 0.0)
    sg = Z["sg0"].reshape(-1, 4)
    return {"N_max_abs_diff": float(np.abs(N - mine["N"]).max()), "dN_max_abs_diff": float(np.abs(dN - mine["dN"]).max()),
            "dv_max_abs_diff": float(np.abs(dv - mine["dv"]).max()), "ddv_max_abs_diff": float(np.abs(ddv - ddm).max()),
            "rho_g_max_rel_diff": float(np.abs(rho - sg[:, 0]).max() / np.abs(sg[:, 0]).max()),
            "J_g_max_abs_diff": float(np.abs(Jv - sg[:, 1:]).max()), "rho_g_max": float(np.abs(sg[:, 0]).max())}


def main():
    iv.dps = 40
    res = {"fact1_pointwise_identity_exact": fact1()}
    for d in (4, 3, 2):
        f = os.path.join(OUT, "s2_items34_d%d" % d)
        if not os.path.exists(f + ".json"):
            continue
        R = json.load(open(f + ".json")); Z = np.load(f + ".npz")
        g = R["gram"]; out = {"lam_min_lower_s2": g["lam_min_lower"]}
        if d in (4, 3) and g["lam_min_lower"] > 0:
            out["interval_cholesky"] = []
            Gm, Gr = Z["Gram"], Z["Gram_rad"]
            off = max(float(np.abs(Gm[t1::4, t2::4]).max()) for t1 in range(4) for t2 in range(4) if t1 != t2)
            out["interval_cholesky_offblock_max_abs"] = off          # exactly 0: the blocks decouple (spin traces)
            for fr in (0.999, 0.9, 0.5):
                okall = True; rows = []
                for tau in range(4):
                    ok, j = iv_cholesky_ok(Gm[tau::4, tau::4], Gr[tau::4, tau::4], fr * g["lam_min_lower"])
                    rows.append(j); okall = okall and ok
                out["interval_cholesky"].append({"per_block": True, "shift_over_lam_lo": fr, "ok": okall, "rows_done": rows})
                if okall:
                    break
        # second, independent certificate: floating Cholesky of Gamma - c I per block with Higham's backward error
        # (R^T R = A' + dA, |dA| <= gamma_{n+1} |R^T||R|  =>  lambda_min(Gamma_exact) >= c - gamma_{n+1} ||R||_F^2 - u max|a_ii - c| - ||rad||_F)
        Gm, Gr = Z["Gram"], Z["Gram_rad"]
        cb = []
        for tau in range(4):
            A = Gm[tau::4, tau::4]; rd = Gr[tau::4, tau::4]; nb_ = A.shape[0]
            lf = float(np.linalg.eigvalsh(A)[0])
            c = 0.9 * lf
            try:
                Rc = np.linalg.cholesky(A - c * np.eye(nb_))
                gm = (nb_ + 1) * 2.0 ** -53 / (1 - (nb_ + 1) * 2.0 ** -53)
                lo = c - gm * float(np.sum(Rc ** 2)) * (1 + 1e-10) - 2.0 ** -53 * float(np.abs(np.diag(A) - c).max()) - float(np.linalg.norm(rd))
                cb.append({"block": tau, "lam_float": lf, "lam_lo": lo, "ok": True})
            except np.linalg.LinAlgError:
                cb.append({"block": tau, "lam_float": lf, "lam_lo": None, "ok": False})
        out["cholesky_backward_error_certificate"] = cb
        nb = {}
        for k, v in R.get("neighbourhood", {}).items():
            if "parts_at_eta_max" not in v:
                continue
            p = v["parts_at_eta_max"]
            e0 = v["eta_max_lower"]
            if not np.isfinite(e0) or e0 <= 0 or e0 > 1:
                continue
            e3 = float("%.3g" % e0)
            if e3 > e0:
                e3 = float("%.3g" % (e0 * (1 - 5e-3)))
            s = iv.mpf(e3) / iv.mpf(e0)
            # lin scales like eta, rem like eta^2, dP = 2 eps e/(2 g0 - e) is increasing in eta: bounded by its eta_max value
            tot = iv.mpf(p["Z"]) + iv.mpf(p["dP"]) + iv.mpf(p["lin"]) * s + iv.mpf(p["rem"]) * s * s
            m = iv.mpf(R["eps"]) - tot
            nb[k] = {"eta_quoted": e3, "margin_interval_at_quoted": [float(m.a), float(m.b)], "positive": bool(m.a > 0)}
        out["neighbourhood_recheck"] = nb
        fa = "<internal>/i2_projected_v1.npz"
        if os.path.exists(fa):
            A = np.load(fa)
            key = "A_d%d_b+0" % d
            if key in A.files:
                Aa = A[key]; Ab = Z["Ad_b0"]
                sa = np.linalg.svd(Aa, compute_uv=False); sb = np.linalg.svd(Ab, compute_uv=False)
                out["agentA_Ad_crosscheck"] = {"shape_A": list(Aa.shape), "shape_B": list(Ab.shape),
                    "max_abs_diff_same_order": float(np.abs(Aa - Ab).max()) if Aa.shape == Ab.shape else None,
                    "max_abs_B": float(np.abs(Ab).max()),
                    "sv_rel_diff_max": float(np.max(np.abs(sa - sb) / sb.max())) if sa.shape == sb.shape else None,
                    "sigma_min_A": float(sa[-1]), "sigma_min_B": float(sb[-1])}
        try:
            out["agentA_jets_laws_crosscheck"] = jets_crosscheck(Z, d)
        except Exception as ex:                       # report, do not hide
            out["agentA_jets_laws_crosscheck"] = {"error": repr(ex)}
        res["d%d" % d] = out
        print(d, out, flush=True)
    with open(os.path.join(OUT, "s3_check_v1.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
