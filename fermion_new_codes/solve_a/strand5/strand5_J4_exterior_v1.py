#!/usr/bin/env python
"""Strand 5 J4 sec. 10(m)(3) (spec solve/specs/SPEC_strand5_kernel_star.md; builder analyst): the exterior of the stage-1 members, with
strand5_lattice_star_v3's Star, Radial and kernel setup (imported, unchanged).

Per family: the stencil K_n (half-width R), its zero-sum sum_n K_n, its symmetry max |K_n - K_-n|, its second moment k_mom = (1/2) sum n^2 K_n
against the family's k_op, and (packet-built kernels) the sum of the per-packet moments.  Per member (the three most compact by default):
(a) the Gauss decomposition.  With u = r (psi - 1) on the grid (odd at r = 0) and u = A beyond J, sum over all j of j (K * u)_j = -k_mom A
    for a symmetric zero-sum stencil; the grid rows (C(psi - 1))_j = (K * u)_j / r_j = -psi^5 e_b + R1_j hold for j <= J, the rows J < j <=
    J + R are not imposed.  Hence k_op (A_vol - A) = sum_j r_j^2 R1_j + E_ext - Z + (k_mom - k_op) A with E_ext = sum_{J<j<=J+R} j (K * u)_j
    (the exterior rows' residual, from the operator on the grid extended to J + R) and Z = sum_{j<=J+R} j (K * u)_j + k_mom A (the
    zero-sum, symmetry and rounding remainder); each term divided by k_op A, their sum against the record's (A_vol - A)/A.  The same for B
    with u = r (chi - 1), u = -B beyond J and the combined row's matter term.
(b) The exterior clock and combined rows on J < j <= J + R: their largest absolute value, and relative to the largest interior source
    |psi^5 e_b|.
(c) The buffer: if J - r_s < R + 20, the member re-solved at the same z_c (Star.newton) on J_new = ceil(r_s) + R + 20 from the old profile
    continued by the exterior forms; the changes in A, B, A/B - 1, and (a), (b) at the new grid.
One family per call; writes <out>/exterior_<family>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_lattice_star_v3 as S  # noqa: E402


def analyse(star, x, kop, kmom):
    """The Gauss decompositions of (A_vol - A)/A and (B_vol - B)/B and the exterior rows on J < j <= J + R."""
    J, R = star.J, star.op.R
    psi, chi, A, B, mu = star.unpack(x)
    N = chi / psi
    r = star.r
    eb, comb, n, kap = star.matter(psi, N, mu)
    Fx = star.F(x, None, NB_target=float(np.sum(4 * np.pi * r * r * n * psi ** 6)))
    R1, R2 = Fx[:J], Fx[J:2 * J]
    Jx = J + R
    rx = np.arange(1, Jx + 1, dtype=float)
    opx = S.Radial(Jx, star.op.st)
    out = {"J": J, "R": R, "residual_norm": float(np.linalg.norm(Fx[:2 * J]))}
    # A: u = r (psi - 1), u = A beyond J; sum_j j (K*u)_j = -k_mom A + Z; rows j <= J: j (K*u)_j = r^2 (R1 - psi^5 e_b)
    uA = np.concatenate([r * (psi - 1), np.full(R, A)])
    CuA = opx.apply(uA, A)
    jK = rx * rx * CuA
    Eext, Z = float(np.sum(jK[J:])), float(np.sum(jK)) + kmom * A
    Avol = float(np.sum(r * r * psi ** 5 * eb) / kop)
    res = float(np.sum(r * r * R1))
    t = {"residual": res / (kop * A), "exterior": Eext / (kop * A), "zero_sum_symmetry": -Z / (kop * A), "moment": kmom / kop - 1}
    out["A"] = {"vol_minus_rel": Avol / A - 1, "terms_rel": t, "sum_of_terms": float(sum(t.values())), "sum_minus_record": float(sum(t.values()) - (Avol / A - 1)),
                "exterior_rows_max_abs": float(np.max(np.abs(CuA[J:]))) if R > 0 else 0.0,
                "exterior_rows_rel_source": float(np.max(np.abs(CuA[J:])) / max(np.max(np.abs(psi ** 5 * eb)), 1e-300)) if R > 0 else 0.0}
    # B: u = r (chi - 1), u = -B beyond J; sum_j j (K*u)_j = k_mom B + Z; rows j <= J: j (K*u)_j = r^2 (R2 + N psi^5 comb)
    uB = np.concatenate([r * (chi - 1), np.full(R, -B)])
    CuB = opx.apply(uB, -B)
    jK = rx * rx * CuB
    Eext, Z = float(np.sum(jK[J:])), float(np.sum(jK)) - kmom * B
    Bvol = float(np.sum(r * r * N * psi ** 5 * comb) / kop)
    res = float(np.sum(r * r * R2))
    t = {"residual": -res / (kop * B), "exterior": -Eext / (kop * B), "zero_sum_symmetry": Z / (kop * B), "moment": kmom / kop - 1}
    out["B"] = {"vol_minus_rel": Bvol / B - 1, "terms_rel": t, "sum_of_terms": float(sum(t.values())), "sum_minus_record": float(sum(t.values()) - (Bvol / B - 1)),
                "exterior_rows_max_abs": float(np.max(np.abs(CuB[J:]))) if R > 0 else 0.0,
                "exterior_rows_rel_source": float(np.max(np.abs(CuB[J:])) / max(np.max(np.abs(N * psi ** 5 * comb)), 1e-300)) if R > 0 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compact", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Rf = json.load(open(a.family))
    M = Rf["meta"]
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = types.SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, info = S.setup_kernel(args)
    if abs(kop / M["k_op"] - 1) > 1e-12:
        raise SystemExit("k_op mismatch")
    gM, gB = M["couplings"]
    coup = (gM, gB, 1.0, 1.0)
    e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    ns = np.array(sorted(stencil))
    Kv = np.array([stencil[n] for n in ns])
    kmom = 0.5 * float(np.sum(ns * ns * Kv))
    sym = max(abs(stencil[n] - stencil.get(-n, np.nan)) for n in ns)
    res = {"meta": {"family": a.family, "operator": M["operator"], "mlight": M["mlight"], "aH": M["aH"], "couplings": [gM, gB], "k_op": kop,
                    "stencil_half_width": int(max(ns)), "zero_sum": float(np.sum(Kv)), "zero_sum_rel_K0": float(np.sum(Kv) / abs(stencil[0])),
                    "symmetry_max": float(sym), "k_mom": kmom, "k_mom_minus_k_op_rel": kmom / kop - 1,
                    "sum_of_packet_moments": info.get("sum_of_packet_moments"), "kernel_moment_info": info.get("k_moment")}, "members": []}
    mem = [d for d in Rf["members"] if d.get("newton_ok")]
    for d in mem[-a.compact:]:
        zc = d["z_c"]
        key = "z%.6f" % zc
        r = prof[key + "_r"]
        psi = prof[key + "_psi"]
        N = prof[key + "_N"]
        J = len(r)
        star = S.Star(J, stencil, kop, M["mlight"], coup, e1)
        x = np.concatenate([psi, N * psi, [d["A"], d["B"], d["mu_inf"]]])
        rec = {"z_c": zc, "J": J, "r_s": d["r_s"], "buffer": J - d["r_s"], "A": d["A"], "B": d["B"], "A_over_B_minus_1": d["A_over_B_minus_1"],
               "record_A_vol_minus_A_rel": d["A_vol_minus_A"] / d["A"], "record_B_vol_minus_B_rel": d["B_vol_minus_B"] / d["B"]}
        rec["at_J"] = analyse(star, x, kop, kmom)
        R = star.op.R
        need = int(np.ceil(d["r_s"])) + R + 20
        if J < need:
            r2 = np.arange(1, need + 1, dtype=float)
            ps = np.where(r2 <= J, np.interp(r2, r, psi), 1 + d["A"] / r2)
            ch = np.where(r2 <= J, np.interp(r2, r, N * psi), 1 - d["B"] / r2)
            x0 = np.concatenate([ps, ch, [r2[-1] * (ps[-1] - 1), -r2[-1] * (ch[-1] - 1), d["mu_inf"]]])
            st2 = S.Star(need, stencil, kop, M["mlight"], coup, e1)
            x2, inf = st2.newton(x0, float(zc))
            d2 = st2.diagnostics(x2, float(zc))
            rec["extended"] = {"J_new": need, "newton_ok": inf["ok"], "newton_it": inf["it"], "residual": inf["res"], "A": d2["A"], "B": d2["B"],
                               "A_over_B_minus_1": d2["A_over_B_minus_1"], "dA_rel": d2["A"] / d["A"] - 1, "dB_rel": d2["B"] / d["B"] - 1,
                               "d_AoB": d2["A_over_B_minus_1"] - d["A_over_B_minus_1"], "r_s": d2["r_s"], "M": d2["M"],
                               "A_vol_minus_A_rel": d2["A_vol_minus_A"] / d2["A"], "B_vol_minus_B_rel": d2["B_vol_minus_B"] / d2["B"],
                               "analysis": analyse(st2, x2, kop, kmom)}
        else:
            rec["extended"] = None
        res["members"].append(rec)
        e = rec["extended"]
        print("z_c %.4f J %d r_s %.2f buffer %.1f (R %d): A_vol/A-1 %.2e = res %.2e + ext %.2e + zs %.2e + mom %.2e (sum - value %.1e); ext rows %.2e%s (%.0f s)" % (
            zc, J, d["r_s"], J - d["r_s"], R, rec["at_J"]["A"]["vol_minus_rel"], rec["at_J"]["A"]["terms_rel"]["residual"], rec["at_J"]["A"]["terms_rel"]["exterior"],
            rec["at_J"]["A"]["terms_rel"]["zero_sum_symmetry"], rec["at_J"]["A"]["terms_rel"]["moment"], rec["at_J"]["A"]["sum_minus_record"],
            rec["at_J"]["A"]["exterior_rows_max_abs"],
            ("; extended to J %d: dA %.2e dB %.2e d(A/B-1) %.2e, A_vol-A %.2e" % (e["J_new"], e["dA_rel"], e["dB_rel"], e["d_AoB"], e["A_vol_minus_A_rel"])) if e else "; buffer sufficient",
            time.time() - t0), flush=True)
        with open(os.path.join(a.out, "exterior_" + os.path.basename(a.family)), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "exterior_" + os.path.basename(a.family)), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
