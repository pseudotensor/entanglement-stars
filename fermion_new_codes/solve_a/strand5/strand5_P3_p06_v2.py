#!/usr/bin/env python
"""Strand 5 production P3's operator diagnostic (SPEC_strand5_covariant_row.md sec. 9(a); builder analyst).  Per kernel family (the same
--families string as strand5_P3_p06_v1.py, whose functions this script imports: route S's V(q) from the family's stencil, the line kernel
K^V on the ring of 2^18 points, the radial action):
  (i) the DC component: the ring's total sum of K^V (= the route's V at the ring's smallest q by construction), against V(q -> 0) as the
      intercept -k_2/k_0^2 of k_a(q) = k_0 + k_2 q^2 + k_4 q^4 taken (a) from the stencil's moments (exact for route S: k_0 = M_2/2,
      k_2 = (M_2 - M_4)/24, M_p = sum_n n^p K_n, so V(0) = (M_4 - M_2)/(6 M_2^2)) and (b) from a least-squares fit to route S's k_a at the
      five smallest q of the 256-zone (q = 2 pi n/256, n = 1..5; the comparator of the spec's "below the first tabulated q");
  (ii) K^V's partial sums P(N) = K_0 + 2 sum_{n=1..N} K_n against N: |K_n| and P(N) at N = R, 2R, 4R, 1000, 4000, 16000, 65536 (R the
      stencil's half-width), and N_eff, the smallest N with |P(N) - P_total| below 1e-10 |P_total| (and the absolute 1e-10 one);
  (iii) the two checks of P3b, ((V - V(0)) 1)_j and r_j ((V - V(0)) (1/r))_j relative to V(0), redone on a domain J_c = 4 max(N_eff, R) + 400
      over the nodes beyond N_eff and beyond 2 N_eff from both ends (skipped, and stated, when N_eff > 20000: the domain would be too large);
  (iv) if (i) fails by more than 1e-6 relative (the analytic intercept against the ring's smallest-q value), Pi recomputed with K^V's DC
      component corrected to the intercept (K_0 += V0_moment - V0_ring, which adds (V0_moment - V0_ring) sum_j 4 pi r_j^2 f_j g_j to Pi),
      and every P3a ratio's change reported.
Writes the markdown (--md) and the JSON (--out-json).  Numbers emitted, not retyped."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_P3_p06_v1 as P3  # noqa: E402
import strand5_lattice_star_v3 as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    f, table = P3.f, P3.table
    rec, rows_i, rows_ii, rows_iii, rows_iv, summ = {}, [], [], [], [], []
    for item in a.families.split(";"):
        lab, fam_p, s2_p, karg, tab_p, twin_p = item.split("|")
        F = json.load(open(fam_p))
        S2 = json.load(open(s2_p))
        prof = np.load(fam_p.replace(".json", "_profiles.npz"))
        M = F["meta"]
        kk, kv = karg.split(" ", 1)
        args = types.SimpleNamespace(kernel=kv if kk == "--kernel" else None, packets=kv if kk == "--packets" else None, mlight=M["mlight"], aH=M["aH"],
                                     operator=M["operator"])
        stencil, kop, info = S.setup_kernel(args)
        gM, gB = M["couplings"]
        e1t = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
        R = max(stencil)
        ns = np.array(sorted(stencil), dtype=float)
        ks = np.array([stencil[int(n)] for n in ns])
        M2, M4 = float(np.sum(ns ** 2 * ks)), float(np.sum(ns ** 4 * ks))
        V0_mom = (M4 - M2) / (6 * M2 * M2)
        q5 = 2 * np.pi * np.arange(1, 6) / 256
        _, ka5 = P3.V_stencil(stencil, q5)
        X = np.stack([np.ones(5), q5 ** 2, q5 ** 4], 1)
        c = np.linalg.lstsq(X, ka5, rcond=None)[0]
        V0_fit = -c[1] / c[0] ** 2
        nmax = P3.LR // 2
        K, V0S, qg, Vg = P3.line_kernel_V(lambda q: P3.V_stencil(stencil, q)[0], nmax)
        Ptot = float(K[0] + 2 * np.sum(K[1:nmax]) + K[nmax])
        Pn = K[0] + 2 * np.cumsum(np.concatenate([[0.0], K[1:nmax]]))
        rel = np.abs(Pn - Ptot) / abs(Ptot)
        absd = np.abs(Pn - Ptot)
        below = np.nonzero(rel < 1e-10)[0]
        Neff = int(below[0]) if len(below) else None
        below_a = np.nonzero(absd < 1e-10)[0]
        Neff_abs = int(below_a[0]) if len(below_a) else None
        dci = abs(V0_mom / V0S - 1)
        rows_i.append([lab, "%.10e" % V0S, "%.10e" % Ptot, "%.10e" % V0_mom, "%.10e" % V0_fit, f(V0S / V0_mom - 1, 3), f(V0_fit / V0_mom - 1, 3),
                       "fails" if dci > 1e-6 else "passes"])
        Ns = [x for x in (R, 2 * R, 4 * R, 1000, 4000, 16000, 65536) if x < nmax]
        rows_ii.append([lab, str(R), "; ".join("N %d: |K_N| %.3e, P(N)/P_tot - 1 %.3e" % (N_, abs(K[N_]), Pn[N_] / Ptot - 1) for N_ in Ns),
                        str(Neff), str(Neff_abs)])
        # (iii)
        if Neff is not None and Neff <= 20000:
            Jc = 4 * max(Neff, R) + 400
            Kc = K[:2 * Jc + 2]
            rc = np.arange(1, Jc + 1, dtype=float)
            flat = P3.apply_V(Kc, np.ones(Jc), rc) - V0S
            coul = rc * (P3.apply_V(Kc, 1.0 / rc, rc) - V0S / rc)
            out3 = []
            for mult in (1, 2):
                sel = (rc > mult * Neff) & (rc < Jc - mult * Neff)
                out3 += [f(np.max(np.abs(flat[sel])) / abs(V0S), 2) if np.any(sel) else "-", f(np.max(np.abs(coul[sel])) / abs(V0S), 2) if np.any(sel) else "-"]
            rows_iii.append([lab, str(Jc)] + out3)
        else:
            rows_iii.append([lab, "skipped (N_eff %s above 20000)" % Neff, "-", "-", "-", "-"])
        # (iv)
        fam_rec = {"V0_ring": V0S, "P_total": Ptot, "V0_moment": V0_mom, "V0_fit": V0_fit, "R": R, "N_eff": Neff, "N_eff_abs": Neff_abs, "M2": M2, "M4": M4}
        if dci > 1e-6:
            mems = S2["members"]
            five = S2["meta"].get("five_members") or list(range(len(mems)))
            Kfull = K[:2 * max(int(mems[i]["J"]) for i in five) + 4]
            ch = []
            for i in five:
                d = mems[i]
                key = "z%.6f" % d["z_c"]
                r, psi, N, n = prof[key + "_r"], prof[key + "_psi"], prof[key + "_N"], prof[key + "_n"]
                star = S.Star(len(r), stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1t)
                eb, comb, nn, kap = star.matter(psi, N, d["mu_inf"])
                fs, gs = psi ** 5 * eb, N * psi ** 5 * comb
                Pi = float(np.sum(4 * np.pi * r * r * fs * P3.apply_V(Kfull, gs, r)))
                Pic = Pi + (V0_mom - V0S) * float(np.sum(4 * np.pi * r * r * fs * gs))
                meas = d["lock"].get("vac_part_of_AmB_over_A")
                p0, p1 = -Pi / (math.pi * kop * d["A"]), -Pic / (math.pi * kop * d["A"])
                ch.append({"z_c": d["z_c"], "Pi": Pi, "Pi_corrected": Pic, "ratio": p0 / meas if meas else None, "ratio_corrected": p1 / meas if meas else None})
                rows_iv.append([lab, "%.6g" % d["z_c"], "%.6e" % Pi, "%.6e" % Pic, f(p0 / meas, 6) if meas else "-", f(p1 / meas, 6) if meas else "-",
                                f((p1 - p0) / meas, 3) if meas else "-"])
            fam_rec["corrected"] = ch
        rec[lab] = fam_rec
        summ.append("%s: V(0) from the ring's smallest q %s, from the stencil's moments %s (relative %s; %s at 1e-6), from the five-point fit %s; "
                    "K^V's DC partial sums reach the ring total within 1e-10 at N = %s (stencil half-width %d)." % (
                        lab, "%.6e" % V0S, "%.6e" % V0_mom, f(V0S / V0_mom - 1, 3), "fails" if dci > 1e-6 else "passes", "%.6e" % V0_fit, Neff, R))
    out = ["# Strand 5 production P3's operator diagnostic (strand5_P3_p06_v2.py)", "", "## S Summary (emitted)", ""] + ["- " + x for x in summ] + ["",
           "## D1 (i) The DC component: the ring's V(0) (its smallest q), the ring's total sum of K^V, V(0) from the stencil's moments and from the "
           "five-point fit, the relative differences, and (i)'s verdict at 1e-6", "",
           table(["family", "V(0) ring", "sum K^V (ring)", "V(0) moments", "V(0) fit", "ring/moments - 1", "fit/moments - 1", "(i)"], rows_i), "",
           "## D2 (ii) K^V's decay and DC partial sums", "", table(["family", "R", "N: |K_N|, P(N)/P_total - 1", "N_eff (relative 1e-10)", "N_eff (absolute 1e-10)"],
                                                                 rows_ii), "",
           "## D3 (iii) P3b's checks beyond N_eff and 2 N_eff (relative to V(0))", "",
           table(["family", "J_c", "flat beyond N_eff", "Coulomb beyond N_eff", "flat beyond 2 N_eff", "Coulomb beyond 2 N_eff"], rows_iii), "",
           "## D4 (iv) Pi with the DC component corrected to the moments' intercept (only where (i) fails)", "",
           table(["family", "z_c", "Pi", "Pi corrected", "ratio", "ratio corrected", "change"], rows_iv) if rows_iv else "(i) passes in every family: no correction.", ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))


if __name__ == "__main__":
    main()
