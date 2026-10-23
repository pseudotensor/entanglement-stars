#!/usr/bin/env python
"""cvread_i_v1.py (busi lane, inbox #424; ReviewA_cvread_v1.md sec. 4 (i)): Delta N(r), Delta v(r) between X = CV2048 (ISO 128 on 2048
sites, 16 r_s) and P = job (b) (ISO_128_s4 on 4096 sites, 32 r_s) at T/r_s 1.4 (records 1.375 and 1.4375 bracket it), 3.5, 8.0, 11.5,
on matched radii in physical units (r in a and in r_s; N, v dimensionless). Read-only on the stored records.

Fields used, X minus P:
  nodes   N_nodes, v_nodes at r_nodes = 0, 4, 8, ... a (POINT values of the engine's dense solution; the nodes are equal on X's range).
  dense   the dense fields re-solved from each record's stored eps_tot, J_tot by the field module's solve_constraints_ivp (gate_common_v1
          SixTerm.solve, collar flag, support 1e-14), point-sampled at the sites R = 1..2048 a; checked against the stored nodes per side.
  site    the stored site rows N, v (KERNEL-AVERAGED, --site-fields averaged): the rows the comparator's 1.68e-3 was read on; reported.
The reference radius: 2 R_t with R_t the record's outer root (P's; X's printed) where a root exists, else 2 R_birth (the comparator's
gate_profile rule before the root; R_birth from gate_common_v1.birth, equal in X and P to 8.8e-5 a).
Also a sweep over every common clock (both certified): max |Delta| inside 2R and the clock and radius of the site-row 1.68e-3.

  python cvread_i_v1.py X_DIR/X_TAG P_DIR/P_TAG --rs 128 --kits KITS_CODE_DIR --fm-dir FM_DIR --out OUT.npz
"""
import argparse
import os
import sys

import numpy as np

CLOCKS = (1.375, 1.4375, 3.5, 8.0, 11.5)
THR = (1e-3, 1e-4, 1e-5, 1e-6)
BINS_RS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.5, 16.0)
SAMPLE_RS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 15.5, 16.0)


def ragged(F, key, i):
    return np.asarray(F[key][i], float)


def summary(r, d, rref, rs):
    """max |d| on all r, r <= 8 r_s, r <= rref, r > rref, with radii; d at 0 and at rref; innermost r with |d| > thr."""
    a = np.abs(d)
    out = {}
    for nm, sel in (("all", np.ones_like(r, bool)), ("le8", r <= 8 * rs), ("in2R", r <= rref), ("out2R", r > rref)):
        if sel.any():
            j = int(np.argmax(np.where(sel, a, -1.0)))
            out[nm] = (float(a[j]), float(r[j]))
        else:
            out[nm] = (None, None)
    out["at0"] = float(d[int(np.argmin(np.abs(r)))])
    out["at2R"] = float(np.interp(rref, r, d))
    out["inner"] = {}
    for t in THR:
        k = np.nonzero(a > t)[0]
        out["inner"][t] = float(r[k[0]]) if len(k) else None
    return out


def fmtv(x, f="%.2e"):
    return "-" if x is None else (f % x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X"); ap.add_argument("P")
    ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--kits", required=True, help="dir holding gate_common_v1.py")
    ap.add_argument("--fm-dir", dest="fm_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.kits)
    import gate_common_v1 as GC
    rs = a.rs
    xd, xt = GC.split_spec(a.X); pd, pt = GC.split_spec(a.P)
    JX, FX = GC.load_run(xd, xt); JP, FP = GC.load_run(pd, pt)
    nX = int(JX["args"]["nsites"])
    Rx = np.asarray(FX["R"], float); Rp = np.asarray(FP["R"], float)
    assert np.array_equal(Rx, Rp[:nX]), "site grids differ"
    RX, RP = GC.recs_by_T(JX, rs), GC.recs_by_T(JP, rs)
    IX, IP = GC.rows_by_T(FX, rs), GC.rows_by_T(FP, rs)
    bX, bP = GC.birth(JX, rs), GC.birth(JP, rs)
    Rb = bP["R_birth"]
    print("== cvread_i_v1: X %s (nsites %d) - P %s (nsites %s); rs %g a; sites equal on X's range: True" % (xt, nX, pt, JP["args"]["nsites"], rs))
    print("birth: X T %.5f R %.4f a; P T %.5f R %.4f a; 2 R_birth (P) = %.3f a = %.4f r_s; matter R0 = %s r_s" % (
        bX["T_birth"], bX["R_birth"], bP["T_birth"], bP["R_birth"], 2 * Rb, 2 * Rb / rs, JX["args"].get("R0_over_rs")))

    def rref_of(t):
        rx, rp = RX[t].get("R_t"), RP[t].get("R_t")
        if rp:
            return 2.0 * float(rp), "2R_t (P %.4f a, X %s a)" % (float(rp), fmtv(rx, "%.4f"))
        return 2.0 * float(Rb), "2R_birth (no root at this record)"

    six = GC.load_sixterm(a.fm_dir)
    store = {}
    for t in CLOCKS:
        ix, ip = IX[t], IP[t]
        rref, rlab = rref_of(t)
        cx, cp = GC.certified(RX[t]), GC.certified(RP[t])
        print("\n#### T/r_s %.4f (rows X %d, P %d; certified X %s P %s); reference radius %s = %.3f a = %.4f r_s" % (t, ix, ip, cx, cp, rlab, rref, rref / rs))
        # nodes
        rnx = ragged(FX, "r_nodes", ix); rnp = ragged(FP, "r_nodes", ip)
        n = len(rnx); assert np.array_equal(rnx, rnp[:n]), "node radii differ"
        dNn = ragged(FX, "N_nodes", ix) - ragged(FP, "N_nodes", ip)[:n]
        dvn = ragged(FX, "v_nodes", ix) - ragged(FP, "v_nodes", ip)[:n]
        # dense re-solve, point-sampled at the sites and at the nodes
        SX = six.solve(FX, ix); SP = six.solve(FP, ip)
        NxS, vxS = SX[0](Rx), SX[1](Rx); NpS, vpS = SP[0](Rx), SP[1](Rx)
        chk = {"X_N": float(np.max(np.abs(SX[0](rnx) - ragged(FX, "N_nodes", ix)))), "X_v": float(np.max(np.abs(SX[1](rnx) - ragged(FX, "v_nodes", ix)))),
               "P_N": float(np.max(np.abs(SP[0](rnp) - ragged(FP, "N_nodes", ip)))), "P_v": float(np.max(np.abs(SP[1](rnp) - ragged(FP, "v_nodes", ip))))}
        dNd = np.asarray(NxS - NpS, float); dvd = np.asarray(vxS - vpS, float)
        # site rows (kernel-averaged)
        dNs = np.asarray(FX["N"][ix], float) - np.asarray(FP["N"][ip], float)[:nX]
        dvs = np.asarray(FX["v"][ix], float) - np.asarray(FP["v"][ip], float)[:nX]
        print("dense re-solve against the stored nodes (max |dense - nodes| per side): X N %.1e v %.1e; P N %.1e v %.1e" % (chk["X_N"], chk["X_v"], chk["P_N"], chk["P_v"]))
        res = {}
        for nm, r, dN, dv in (("nodes", rnx, dNn, dvn), ("dense", Rx, dNd, dvd), ("site", Rx, dNs, dvs)):
            res[nm] = {"N": summary(r, dN, rref, rs), "v": summary(r, dv, rref, rs)}
            for fl in ("N", "v"):
                s = res[nm][fl]
                print(" %-5s d%s: max all %s at %s r_s | r<=8r_s %s at %s | inside 2R %s at %s r_s | outside 2R %s at %s | d(0) %+.2e d(2R) %+.2e | innermost r_s with |d|> 1e-3 %s 1e-4 %s 1e-5 %s 1e-6 %s" % (
                    nm, fl, fmtv(s["all"][0]), fmtv(s["all"][1] / rs if s["all"][1] is not None else None, "%.3f"), fmtv(s["le8"][0]),
                    fmtv(s["le8"][1] / rs if s["le8"][1] is not None else None, "%.3f"), fmtv(s["in2R"][0]),
                    fmtv(s["in2R"][1] / rs if s["in2R"][1] is not None else None, "%.3f"), fmtv(s["out2R"][0]),
                    fmtv(s["out2R"][1] / rs if s["out2R"][1] is not None else None, "%.3f"), s["at0"], s["at2R"],
                    *[fmtv(s["inner"][q] / rs if s["inner"][q] is not None else None, "%.3f") for q in THR]))
        # binned profile on the nodes (max |d| per bin) and signed samples
        print(" profile (nodes; max |dN|, max |dv| per r/r_s bin; [0, 2R] first):")
        edges = [(0.0, rref)] + [(BINS_RS[k] * rs, BINS_RS[k + 1] * rs) for k in range(len(BINS_RS) - 1)]
        for lo, hi in edges:
            sel = (rnx >= lo) & (rnx <= hi)
            if sel.any():
                print("   r/r_s [%6.3f, %6.3f]  |dN| %.2e  |dv| %.2e" % (lo / rs, hi / rs, float(np.max(np.abs(dNn[sel]))), float(np.max(np.abs(dvn[sel])))))
        print(" signed samples (nodes): " + "; ".join("%g: dN %+.2e dv %+.2e" % (q, float(np.interp(q * rs, rnx, dNn)), float(np.interp(q * rs, rnx, dvn))) for q in SAMPLE_RS))
        store[t] = {"r_nodes": rnx, "dN_nodes": dNn, "dv_nodes": dvn, "dN_dense": dNd, "dv_dense": dvd, "dN_site": dNs, "dv_site": dvs, "rref": rref, "res": res, "chk": chk}

    # sweep over the common certified clocks: where the 1.68e-3 sits, and max inside 2R per clock
    print("\n#### sweep over the common certified clocks (nodes and site rows; inside the reference radius 2R and on r <= 8 r_s)")
    common = sorted(set(RX) & set(RP) & set(IX) & set(IP))
    gated = [t for t in common if GC.certified(RX[t]) and GC.certified(RP[t])]
    sw = []
    for t in gated:
        ix, ip = IX[t], IP[t]
        rref, _ = rref_of(t)
        rnx = ragged(FX, "r_nodes", ix); n = len(rnx)
        dNn = np.abs(ragged(FX, "N_nodes", ix) - ragged(FP, "N_nodes", ip)[:n]); dvn = np.abs(ragged(FX, "v_nodes", ix) - ragged(FP, "v_nodes", ip)[:n])
        dNs = np.abs(np.asarray(FX["N"][ix], float) - np.asarray(FP["N"][ip], float)[:nX]); dvs = np.abs(np.asarray(FX["v"][ix], float) - np.asarray(FP["v"][ip], float)[:nX])
        si = Rx <= rref; s8 = Rx <= 8 * rs; ni = rnx <= rref; n8 = rnx <= 8 * rs
        row = (t, rref, float(dNn[ni].max()), float(dvn[ni].max()), float(dNs[si].max()), float(dvs[si].max()),
               float(dNs[s8].max()), float(Rx[s8][np.argmax(dNs[s8])]), float(dvs[s8].max()), float(Rx[s8][np.argmax(dvs[s8])]),
               float(dNn[n8].max()), float(rnx[n8][np.argmax(dNn[n8])]))
        sw.append(row)
    sw = np.array(sw)
    for row in sw:
        print("  %.4f  2R %.1f a | inside 2R: nodes |dN| %.2e |dv| %.2e, site |dN| %.2e |dv| %.2e | r<=8r_s: site |dN| %.2e at %.3f r_s, |dv| %.2e at %.3f r_s; nodes |dN| %.2e at %.3f r_s" % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7] / rs, row[8], row[9] / rs, row[10], row[11] / rs))
    k = int(np.argmax(sw[:, 6])); kv = int(np.argmax(sw[:, 8])); kn = int(np.argmax(sw[:, 10]))
    ki = int(np.argmax(sw[:, 2])); ks = int(np.argmax(sw[:, 4]))
    print("\nSITE-ROW MAX |dN| on r <= 8 r_s over %d clocks: %.3e at T %.4f, r %.1f a = %.3f r_s (2R there %.1f a)" % (len(sw), sw[k, 6], sw[k, 0], sw[k, 7], sw[k, 7] / rs, sw[k, 1]))
    print("SITE-ROW MAX |dv| on r <= 8 r_s: %.3e at T %.4f, r %.3f r_s" % (sw[kv, 8], sw[kv, 0], sw[kv, 9] / rs))
    print("NODES MAX |dN| on r <= 8 r_s: %.3e at T %.4f, r %.3f r_s" % (sw[kn, 10], sw[kn, 0], sw[kn, 11] / rs))
    print("MAX INSIDE 2R over all clocks: nodes |dN| %.3e at T %.4f (|dv| max %.3e); site |dN| %.3e at T %.4f (|dv| max %.3e)" % (
        sw[ki, 2], sw[ki, 0], sw[:, 3].max(), sw[ks, 4], sw[ks, 0], sw[:, 5].max()))
    tg = [i for i in range(len(sw)) if sw[i, 0] <= 11.5 + 1e-9]
    print("MAX INSIDE 2R through T 11.5: nodes |dN| %.3e |dv| %.3e; site |dN| %.3e |dv| %.3e" % (sw[tg, 2].max(), sw[tg, 3].max(), sw[tg, 4].max(), sw[tg, 5].max()))
    np.savez_compressed(a.out, sweep=sw, clocks=np.array(CLOCKS), R_sites=Rx,
                        **{"%s_%s" % (k_, ("%.4f" % t).replace(".", "p")): v_ for t in CLOCKS for k_, v_ in store[t].items() if isinstance(v_, np.ndarray)})
    print("written %s" % a.out)


if __name__ == "__main__":
    main()
