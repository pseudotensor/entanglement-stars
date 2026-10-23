#!/usr/bin/env python
"""Item 5's figure data (SPEC_strand5_figure.md sec. 3, the lead's contract; builder analyst): reads the strand 5 records, writes one .npz and
the caption numbers; no solve.  Records under --root (box: <internal>/s5_star):
  J1run3/star_v4.json (the continuum reference family), J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json (the covariant row at finite spacing),
  P1run1/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json (the full-node inertia), J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json (the frozen-kernel
  control), P3eff1/P3eff.json (strand5_P3_p06_v4.py's record, the entry '(0.01, 0.1) kernel g0_0_pk' with its kernel and local_twin lists).
Units: M/M_0 and R_*/R_0 with M_0 = m_P^3/(sqrt(8) m^2), R_0 = m_P/(sqrt(8) m^2); in J1's G = m = 1 units M_0 = R_0 = 8^(-1/2); the lattice families
through the TF invariants M~ = G^(3/2) m^2 M, R~ = G^(1/2) m^2 R_s with G = 1/(2 pi k_op), m = m_light.  The lock: (A - B)/(2A), signed.
Checks (each stops the script on failure): the maximum against P2e(c) to 1e-6; 2M/R_s_areal against the compactness to 1e-9 on every reference
member; the lattice 2GM/R_s against its 'compactness' to 1e-9; the lattice M~ at z_c = 0.823379 against P2e(d)'s 0.13584009 to 1e-7; the z_c
matches (stage 2 to 1e-9, P3eff to 1e-6)."""
import argparse
import json
import math
import os
import sys

import numpy as np

S8 = math.sqrt(8.0)
P2EC = {"z_c": 0.8338537642, "M": 0.13585788457, "C": 0.228930742}
P2ED_M = 0.13584009


def stop(msg):
    print("CHECK FAILED: " + msg)
    sys.exit(2)


def rel(a, b):
    return abs(a / b - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="<internal>/s5_star")
    ap.add_argument("--out", required=True, help="the .npz")
    ap.add_argument("--caption", required=True, help="caption_numbers.txt")
    a = ap.parse_args()
    R = lambda p: json.load(open(os.path.join(a.root, p)))
    J1 = R("J1run3/star_v4.json")
    ref = J1["reference"]
    ref_zc = np.array([d["z_c"] for d in ref])
    ref_Mr = np.array([d["M"] for d in ref])
    ref_Rs = np.array([d["R_s_areal"] for d in ref])
    ref_C = np.array([d["compactness_2GM_over_Rs"] for d in ref])
    ref_A = np.array([d["A"] for d in ref])
    ref_B = np.array([d["B"] for d in ref])
    ref_NB = np.array([d["N_B"] for d in ref])
    ref_rs = np.array([d["r_s"] for d in ref])
    c = np.max(np.abs(2 * ref_Mr / ref_Rs / ref_C - 1))
    print("check: 2M/R_s_areal against the compactness, largest relative difference %.3e" % c)
    if c > 1e-9:
        stop("reference compactness")
    mx = J1["controls"]["reference_maximum"]
    for k, v in (("z_c", mx["z_c"]), ("M", mx["M"]), ("C", mx["compactness_2GM_over_Rs"])):
        d = rel(v, P2EC[k])
        print("check: the maximum's %s %.12g against P2e(c) %.12g, relative %.3e" % (k, v, P2EC[k], d))
        if d > 1e-6:
            stop("maximum " + k)
    ref_zmax, ref_Mmax, ref_Rmax, ref_Cmax = mx["z_c"], S8 * mx["M"], S8 * mx["R_s_areal"], mx["compactness_2GM_over_Rs"]
    ref_EbM_max = (mx["M"] - mx["N_B"]) / mx["M"]
    L = R("J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json")
    Kf = R("J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json")
    out_fam = {}
    for lab, F in (("lat", L), ("ker", Kf)):
        kop, ml = F["meta"]["k_op"], F["meta"]["mlight"]
        G = 1.0 / (2 * math.pi * kop)
        mem = F["members"]
        zc = np.array([d["z_c"] for d in mem])
        M = np.array([d["M"] for d in mem])
        Rs = np.array([d["R_s"] for d in mem])
        C = 2 * G * M / Rs
        cc = np.max(np.abs(C / np.array([d["compactness"] for d in mem]) - 1))
        print("check: %s 2GM/R_s against its compactness, largest relative difference %.3e" % (lab, cc))
        if cc > 1e-9:
            stop(lab + " compactness (R_s is not the areal radius?)")
        Mt = G ** 1.5 * ml ** 2 * M
        Rt = G ** 0.5 * ml ** 2 * Rs
        A = np.array([d["A"] for d in mem])
        B = np.array([d["B"] for d in mem])
        out_fam[lab] = {"zc": zc, "M": S8 * Mt, "R": S8 * Rt, "C": C, "lock": (A - B) / (2 * A),
                        "P14": {round(d["z_c"], 6): d["P14_TF_prediction_AmB_over_A"] for d in mem},
                        "meas": {round(d["z_c"], 6): d["measured_AmB_over_A"] for d in mem}, "Mt": Mt}
        if lab == "lat":
            i = int(np.argmin(np.abs(zc - 0.823379)))
            d = rel(Mt[i], P2ED_M)
            print("check: lat M~ at z_c %.6f = %.10f against P2e(d) %.8f, relative %.3e" % (zc[i], Mt[i], P2ED_M, d))
            if d > 1e-7:
                stop("lattice mass against P2e(d)")
    S2 = R("P1run1/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json")
    inert = {}
    for d in S2["members"]:
        inert[d["z_c"]] = int(d["full_node"]["inertia"]) if d.get("full_node") and d["full_node"].get("inertia") is not None else -1
    lat_inertia = []
    for z in out_fam["lat"]["zc"]:
        k = [x for x in inert if abs(x - z) < 1e-9]
        if len(k) != 1:
            stop("stage-2 member for z_c %.9g" % z)
        lat_inertia.append(inert[k[0]])
    P3 = R("P3eff1/P3eff.json")["(0.01, 0.1) kernel g0_0_pk"]

    def preds(lst, fam):
        zz, pp, mm = [], [], []
        for d in lst:
            k = [x for x in fam["P14"] if abs(x - d["z_c"]) < 1e-6]
            if len(k) != 1:
                stop("P3eff member z_c %.9g" % d["z_c"])
            zz.append(d["z_c"])
            pp.append((fam["P14"][k[0]] + d["pred_eff"]) / 2)
            mm.append(fam["meas"][k[0]] / 2)
        return np.array(zz), np.array(pp), np.array(mm)
    ker_pred_zc, ker_pred, ker_meas = preds(P3["kernel"], out_fam["ker"])
    lat_pred_zc, lat_pred, _ = preds(P3["local_twin"], out_fam["lat"])
    units = "M/M_0 with M_0 = m_P^3/(sqrt(8) m^2); R_*/R_0 with R_0 = m_P/(sqrt(8) m^2); lock = (A - B)/(2A)"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, ref_zc=ref_zc, ref_M=S8 * ref_Mr, ref_R=S8 * ref_Rs, ref_C=ref_C, ref_lock=(ref_A - ref_B) / (2 * ref_A), ref_EbM=(ref_Mr - ref_NB) / ref_Mr,
             ref_zmax=ref_zmax, ref_Mmax=ref_Mmax, ref_Rmax=ref_Rmax, ref_Cmax=ref_Cmax, ref_EbM_max=ref_EbM_max,
             lat_zc=out_fam["lat"]["zc"], lat_M=out_fam["lat"]["M"], lat_R=out_fam["lat"]["R"], lat_C=out_fam["lat"]["C"], lat_lock=out_fam["lat"]["lock"],
             lat_inertia=np.array(lat_inertia, dtype=int), lat_pred_zc=lat_pred_zc, lat_pred=lat_pred,
             ker_zc=out_fam["ker"]["zc"], ker_M=out_fam["ker"]["M"], ker_R=out_fam["ker"]["R"], ker_lock=out_fam["ker"]["lock"],
             ker_pred_zc=ker_pred_zc, ker_pred=ker_pred, ker_meas=ker_meas, units=np.array(units))
    iC = int(np.argmax(ref_C))
    rsB = ref_rs / ref_B
    lines = ["ref_zmax %.10f" % ref_zmax, "ref_Mmax (M/M_0) %.10f" % ref_Mmax, "ref_Rmax (R_*/R_0) %.10f" % ref_Rmax, "ref_Cmax (2GM/R_* at the maximum) %.10f" % ref_Cmax,
             "ref_EbM_max %.10f" % ref_EbM_max, "largest ref_C %.10f at z_c %.6g" % (ref_C[iC], ref_zc[iC]),
             "smallest r_s/B %.10f at z_c %.6g" % (rsB.min(), ref_zc[int(np.argmin(rsB))]),
             "ref z_c range %.6g to %.6g (%d members)" % (ref_zc.min(), ref_zc.max(), len(ref_zc)),
             "ref |lock| range %.3e to %.3e" % (np.abs((ref_A - ref_B) / (2 * ref_A)).min(), np.abs((ref_A - ref_B) / (2 * ref_A)).max()),
             "lat |lock| range %.3e to %.3e" % (np.abs(out_fam["lat"]["lock"]).min(), np.abs(out_fam["lat"]["lock"]).max()),
             "ker |lock| range %.3e to %.3e" % (np.abs(out_fam["ker"]["lock"]).min(), np.abs(out_fam["ker"]["lock"]).max()),
             "sign of B - A: ref %s, lat %s, ker %s" % (sorted(set(np.sign(ref_B - ref_A).astype(int).tolist())),
                                                      sorted(set(np.sign(-out_fam["lat"]["lock"]).astype(int).tolist())),
                                                      sorted(set(np.sign(-out_fam["ker"]["lock"]).astype(int).tolist()))),
             "lat inertia per member %s" % lat_inertia,
             "lat largest |M_lat/M_J1(same z_c) - 1| over the lattice members inside J1's z_c range: %.3e" % float(np.max(np.abs(
                 out_fam["lat"]["M"] / np.interp(np.log(out_fam["lat"]["zc"]), np.log(ref_zc), S8 * ref_Mr) - 1)[
                 (out_fam["lat"]["zc"] >= ref_zc.min()) & (out_fam["lat"]["zc"] <= ref_zc.max())]))]
    open(a.caption, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
