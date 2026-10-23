#!/usr/bin/env python
"""Strand 5 production P1, phase 2 (SPEC_strand5_covariant_row.md sec. 3, P1; builder analyst): the spectral certificate on the covariant row
from phase 1's records (strand5_J4_stage2_v10.py --full-node all on J4run4's (0.01, 0.1) local_ka0 couplings (0, 0) family, on the new members
z_max - 0.002, z_max, z_max + 0.002 and 2.8, 2.9 (strand5_J4_members_v1.py), and on the local_klat twin of J4run2), plus one recomputation:
  (1) per member of the family: the full matter-node inertia, omega_1 <= omega_2 <= omega_3 (the fixed-N_B Hessian of M in the volume metric) and
      omega_1/omega_1(dilute) (the dilute member's omega_1, z_c = 0.05);
  (2) the trio at z_max (0.833894, RESULT_strand5_family.md revision 2, I2b): omega_1 and omega_2 at each; the crossing z_x of omega_1 by linear
      interpolation between the two trio members whose omega_1 differ in sign (else by linear extrapolation from the two nearest zero) and by the
      quadratic through the three, against z_max;
  (3) d omega_1/d z_c by the central difference over the trio;
  (4) at z_max: the lowest eigenvector v_1 of the projected full-node Hessian, recomputed here with stage 2 version 10's Exact (the clock row
      re-solved at the member's n_c; the same Householder projection as its full_node()), mapped back to the n_c variation, and its overlap
      |<v_1, t>|/(|v_1| |t|) in the volume metric with the family tangent t = (n_c(z_+) - n_c(z_-))/(z_+ - z_-) on the z_max member's matter nodes
      (the nodes aligned by r), projected onto fixed N_B (orthogonal to the constant in the volume metric);
  (5) the margin table: per family member, omega_1/omega_1(dilute) beside the remainders at that member: r1 the stage-1 member's
      P14_TF_prediction_AmB_over_A (the strain term's lock contribution), r2 = (M_kernel - M_local)/M_local at the same z_c (J4run4's kernel
      (0, 0) family), r4 = (M(0.2, 1) - M(0, 0))/M(0, 0) at the same z_c (J4run4's local_ka0 (0.2, 1) family), r3 the q_0 class from the J2g
      record at the window edge: beta - 1 = q_0/8 (the coefficient channel, gamma^2 = 1 to the digits shown) and the estimate gamma - 1 =
      (3/7) q_0 GM/R_s (the J2g red team's virial form piece/A = (6/7) f(1) GM/R_s with gamma - 1 = piece/(2A); labelled an estimate), r6 "J5
      pending"; for each member before the maximum, whether omega_1/omega_1(dilute) exceeds each |r_i|;
  (6) the second crossing: omega_1, omega_2 and the inertia at z_c = 2.418, 2.8, 2.9, 3.0 (the family and the new members), the location of the
      inertia's change 1 -> 2 by linear interpolation of omega_2, beside the family's mass extrema counted from M(z_c) over the family and the new
      members (the inertia against the count of extrema passed).
Controls: the gradient dM/dn_c against mu_inf 4 pi r^2 (the median |ratio - 1| per member, from the stage-2 records); the Hessian's asymmetry before
symmetrization; the twin's omega's over the family's at each z_c beside the ratio of the families' k_op; the five members of J4s2run9b's version-9
record reproduced (largest relative difference of lowest3 and the inertia).  Numbers emitted, not retyped."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import types

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J4_stage2_v10 as V10  # noqa: E402

V8 = V10.V8
S = V10.S


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def by_z(lst, key="z_c"):
    return {round(float(d[key]), 6): d for d in lst}


def full_node_vec(ex):
    """stage 2 version 10's full_node() with the lowest eigenvector mapped back to the n_c variation on the matter nodes."""
    r, nc = ex.r, ex.nc
    idx = np.nonzero(nc > 0)[0]
    Jm = len(idx)
    Q = np.zeros((ex.J, Jm))
    Q[idx, np.arange(Jm)] = 1.0
    H, asym = ex.hessian(Q)
    del Q
    dvec = np.sqrt(4 * np.pi * r[idx] ** 2)
    Hy = H / dvec[:, None] / dvec[None, :]
    del H
    u = dvec / np.linalg.norm(dvec)
    e1 = np.zeros(Jm)
    e1[0] = 1.0
    v = u - e1
    v /= np.linalg.norm(v)
    Hv = Hy @ v
    Hy -= 2 * np.outer(Hv, v) + 2 * np.outer(v, Hv) - 4 * (v @ Hv) * np.outer(v, v)
    w, Z = sla.eigh(Hy[1:, 1:], subset_by_index=[0, 2], check_finite=False)
    z = np.concatenate([[0.0], Z[:, 0]])
    y = z - 2 * v * (v @ z)
    return w, y, dvec, u, idx, asym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-family", required=True)
    ap.add_argument("--s2-new", required=True)
    ap.add_argument("--s2-twin", required=True)
    ap.add_argument("--s2-old", required=True, help="J4s2run9b's version-9 record of the family (five members)")
    ap.add_argument("--family", required=True, help="the stage-1 family (r1, M)")
    ap.add_argument("--family-new", required=True, help="the new members' stage-1 file (profiles for the tangent and the recomputation)")
    ap.add_argument("--kernel-family", required=True)
    ap.add_argument("--coupling-family", required=True)
    ap.add_argument("--twin-family", required=True)
    ap.add_argument("--j2g", required=True)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--zmax", type=float, default=0.833894)
    ap.add_argument("--dz", type=float, default=0.002)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    S2 = json.load(open(a.s2_family))
    S2n = json.load(open(a.s2_new))
    S2t = json.load(open(a.s2_twin))
    S2o = json.load(open(a.s2_old))
    F1 = json.load(open(a.family))
    FK = json.load(open(a.kernel_family))
    FC = json.load(open(a.coupling_family))
    FT = json.load(open(a.twin_family))
    J2 = json.load(open(a.j2g))
    q0e = float(J2["window edge"]["ka_prime_1_over_ka_0"])
    fam = sorted(S2["members"], key=lambda d: d["z_c"])
    new = sorted(S2n["members"], key=lambda d: d["z_c"])
    summ, out, rec = [], ["# Strand 5 production P1: the spectral certificate on the covariant row (generated by solve/code/strand5/strand5_P1_cert_v1.py)", ""], {}
    # (1) per member
    w1d = fam[0]["full_node"]["lowest3"][0]
    s1 = {round(d["z_c"], 6): d for d in F1["members"]}
    rows1 = []
    for d in fam:
        fn = d["full_node"]
        rows1.append(["%.6g" % d["z_c"], fn["inertia"], "%.6e" % fn["lowest3"][0], "%.6e" % fn["lowest3"][1], "%.6e" % fn["lowest3"][2],
                      "%.6e" % (fn["lowest3"][0] / w1d), fn["matter_nodes"], f(d["exact"]["gradient_over_mu_volume"]["median_abs_minus_1"], 2),
                      f(fn["asymmetry_before_symmetrization"], 2)])
    pre = [d for d in fam if d["z_c"] < a.zmax]
    first_neg = next((d["z_c"] for d in fam if d["full_node"]["inertia"] > 0), None)
    summ.append("P1(1): over the %d members the full-node inertia is %s (z_c ascending); omega_1 > 0 at every member below z_max = %s: %s; the dilute "
                "member's omega_1 = %s; the first member with a negative eigenvalue is z_c = %s." % (
                    len(fam), "".join(str(d["full_node"]["inertia"]) for d in fam), a.zmax,
                    "yes" if all(d["full_node"]["lowest3"][0] > 0 for d in pre) else "NO", "%.6e" % w1d, first_neg))
    # (2), (3) the trio
    trio = [d for d in new if abs(d["z_c"] - a.zmax) <= a.dz * 1.5]
    trio.sort(key=lambda d: d["z_c"])
    zt = np.array([d["z_c"] for d in trio])
    w1t = np.array([d["full_node"]["lowest3"][0] for d in trio])
    w2t = np.array([d["full_node"]["lowest3"][1] for d in trio])
    zx_lin = None
    for i in range(len(zt) - 1):
        if w1t[i] * w1t[i + 1] <= 0:
            zx_lin = float(zt[i] - w1t[i] * (zt[i + 1] - zt[i]) / (w1t[i + 1] - w1t[i]))
    if zx_lin is None and len(zt) >= 2:
        j = int(np.argmin(np.abs(w1t)))
        k = j + 1 if j + 1 < len(zt) else j - 1
        zx_lin = float(zt[j] - w1t[j] * (zt[k] - zt[j]) / (w1t[k] - w1t[j]))
    zx_quad = None
    if len(zt) == 3:
        c = np.polyfit(zt - a.zmax, w1t, 2)
        roots = [float(x.real) + a.zmax for x in np.roots(c) if abs(x.imag) < 1e-14]
        if roots:
            zx_quad = min(roots, key=lambda x: abs(x - a.zmax))
    dw1 = float((w1t[-1] - w1t[0]) / (zt[-1] - zt[0])) if len(zt) >= 2 else None
    rows2 = [["%.6f" % d["z_c"], d["full_node"]["inertia"], "%.6e" % d["full_node"]["lowest3"][0], "%.6e" % d["full_node"]["lowest3"][1],
              "%.6e" % d["full_node"]["lowest3"][2], "%.10e" % d["M"]] for d in trio]
    summ.append("P1(2), (3): at z_c = %s omega_1 = %s and omega_2 = %s; the crossing of omega_1 at z_x = %s (linear) and %s (quadratic), "
                "z_x - z_max = %s and %s; d omega_1/d z_c = %s (central difference over the trio); omega_2 > 0 at the trio: %s." % (
                    ", ".join("%.6f" % z for z in zt), ", ".join("%.5e" % w for w in w1t), ", ".join("%.5e" % w for w in w2t), f(zx_lin, 8), f(zx_quad, 8),
                    f(zx_lin - a.zmax if zx_lin is not None else None, 3), f(zx_quad - a.zmax if zx_quad is not None else None, 3), f(dw1, 5),
                    "yes" if np.all(w2t > 0) else "NO"))
    # (4) the eigenvector and the tangent at z_max
    FN = json.load(open(a.family_new))
    prof = np.load(a.family_new.replace(".json", "_profiles.npz"))
    M = FN["meta"]
    args = types.SimpleNamespace(kernel=None, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, info = S.setup_kernel(args)
    gM, gB = M["couplings"]
    e1t = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    memn = by_z(FN["members"])

    def nc_of(zc):
        key = "z%.6f" % zc
        return prof[key + "_r"], prof[key + "_n"] * prof[key + "_psi"] ** 6

    zc0 = min(memn, key=lambda z: abs(z - a.zmax))
    d0 = memn[zc0]
    key = "z%.6f" % zc0
    r, psi, N, n, qF = (prof[key + s] for s in ("_r", "_psi", "_N", "_n", "_qF"))
    J = len(r)
    star = S.Star(J, stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1t)
    x = np.concatenate([psi, N * psi, [d0["A"], d0["B"], d0["mu_inf"]]])
    Fm = V8.Fam(star, M["mlight"], e1t)
    ex = V10.Exact(Fm, star, x, n, kop, psi * psi * qF)
    w, y1, dvec, u, idx, asym = full_node_vec(ex)
    zm = min(memn, key=lambda z: abs(z - (a.zmax - a.dz)))
    zp = min(memn, key=lambda z: abs(z - (a.zmax + a.dz)))
    rm_, ncm = nc_of(zm)
    rp_, ncp = nc_of(zp)
    rr = r[idx]
    gm = dict(zip(np.round(rm_, 6), ncm))
    gp = dict(zip(np.round(rp_, 6), ncp))
    t = np.array([(gp.get(round(x_, 6), 0.0) - gm.get(round(x_, 6), 0.0)) for x_ in rr]) / (zp - zm)
    ty = dvec * t
    ty_perp = ty - (u @ ty) * u
    ov = float(abs(y1 @ ty_perp) / (np.linalg.norm(y1) * np.linalg.norm(ty_perp)))
    frac_removed = float(abs(u @ ty) / np.linalg.norm(ty))
    rec["overlap"] = {"z_c": zc0, "z_minus": zm, "z_plus": zp, "overlap": ov, "omega_recomputed": w.tolist(), "asymmetry": asym,
                      "tangent_constraint_component": frac_removed, "matter_nodes": int(len(idx))}
    summ.append("P1(4): at z_c = %s the recomputed lowest three eigenvalues %s (stage 2's record: %s); the overlap of v_1 with the family tangent "
                "(the z_c = %s and %s members, projected onto fixed N_B; the tangent's component along the constraint before projection %s of its norm) "
                "is %s." % ("%.6f" % zc0, ", ".join("%.6e" % x_ for x_ in w),
                            ", ".join("%.6e" % x_ for x_ in by_z(new)[round(zc0, 6)]["full_node"]["lowest3"]) if round(zc0, 6) in by_z(new) else "-",
                            "%.6f" % zm, "%.6f" % zp, f(frac_removed, 3), "%.6f" % ov))
    # (5) the margin table
    fk, fc = by_z(FK["members"]), by_z(FC["members"])
    rows5 = []
    r3b = q0e / 8.0
    for d in fam:
        z = round(d["z_c"], 6)
        m1 = s1.get(z, {})
        ratio = d["full_node"]["lowest3"][0] / w1d
        r1 = m1.get("P14_TF_prediction_AmB_over_A")
        r2 = (fk[z]["M"] - d["M"]) / d["M"] if z in fk else None
        r4 = (fc[z]["M"] - d["M"]) / d["M"] if z in fc else None
        gmr = m1.get("compactness", 0.0) / 2.0
        r3g = (3.0 / 7.0) * q0e * gmr
        marks = "-"
        if d["z_c"] < a.zmax:
            marks = "".join("y" if (rv is not None and ratio > abs(rv)) else ("n" if rv is not None else "-") for rv in (r1, r2, r3b, r3g, r4))
        rows5.append(["%.6g" % d["z_c"], "%.6e" % ratio, f(r1, 3), f(r2, 3), "%.4e" % r3b, "%.4e" % r3g, f(r4, 3), "J5 pending", marks])
    bad = [r_[0] for r_ in rows5 if "n" in r_[8]]
    summ.append("P1(5): the margin omega_1/omega_1(dilute) exceeds every listed remainder (r1, r2, r3 beta, r3 gamma estimate, r4) at %s of the %d "
                "members below z_max%s; r3 from the J2g record's window-edge q_0 = %s (beta - 1 = q_0/8 = %s); r6 pending J5." % (
                    len(pre) - len(bad), len(pre), (" (not at z_c = %s)" % ", ".join(bad)) if bad else "", "%.6e" % q0e, "%.4e" % r3b))
    # (6) the second crossing
    sec = sorted([d for d in fam if d["z_c"] > 2.0] + [d for d in new if d["z_c"] > 2.0], key=lambda d: d["z_c"])
    rows6 = [["%.6g" % d["z_c"], d["full_node"]["inertia"], "%.6e" % d["full_node"]["lowest3"][0], "%.6e" % d["full_node"]["lowest3"][1], "%.10e" % d["M"]]
             for d in sec]
    z6 = [d["z_c"] for d in sec]
    w26 = [d["full_node"]["lowest3"][1] for d in sec]
    zc2 = None
    for i in range(len(z6) - 1):
        if w26[i] * w26[i + 1] <= 0:
            zc2 = z6[i] - w26[i] * (z6[i + 1] - z6[i]) / (w26[i + 1] - w26[i])
    allm = sorted(fam + new, key=lambda d: d["z_c"])
    Ms = [d["M"] for d in allm]
    ext = [allm[i]["z_c"] for i in range(1, len(allm) - 1) if (Ms[i] - Ms[i - 1]) * (Ms[i + 1] - Ms[i]) < 0]
    summ.append("P1(6): beyond z_c = 2 the inertia is %s at z_c = %s; omega_2 changes sign at z_c = %s (linear); the mass extrema over the family and "
                "the new members lie at the members z_c = %s (count %d)." % (
                    ", ".join(str(d["full_node"]["inertia"]) for d in sec), ", ".join("%.4g" % z for z in z6), f(zc2, 6), ", ".join("%.6g" % z for z in ext), len(ext)))
    # controls
    tw = by_z(S2t["members"])
    rowsT = []
    kr = FT["meta"]["k_op"] / F1["meta"]["k_op"]
    for d in fam:
        z = round(d["z_c"], 6)
        if z in tw and "full_node" in tw[z]:
            a1 = np.array(d["full_node"]["lowest3"])
            b1 = np.array(tw[z]["full_node"]["lowest3"])
            rowsT.append(["%.6g" % d["z_c"], " ".join("%.6e" % x_ for x_ in b1 / a1 - 1), "%d/%d" % (tw[z]["full_node"]["inertia"], d["full_node"]["inertia"])])
    old = by_z(S2o["members"])
    od = []
    for d in fam:
        z = round(d["z_c"], 6)
        if z in old and "full_node" in old[z]:
            a1 = np.array(d["full_node"]["lowest3"])
            b1 = np.array(old[z]["full_node"]["lowest3"])
            od.append((d["z_c"], float(np.max(np.abs(a1 / b1 - 1))), old[z]["full_node"]["inertia"] == d["full_node"]["inertia"]))
    grad = [d["exact"]["gradient_over_mu_volume"]["median_abs_minus_1"] for d in fam + new]
    asy = [d["full_node"]["asymmetry_before_symmetrization"] for d in fam + new]
    summ.append("Controls: the gradient's median |dM/dn_c/(mu_inf 4 pi r^2) - 1| at most %s over the family and the new members; the Hessian's "
                "asymmetry before symmetrization at most %s; the twin's k_op over the family's %s - 1 = %s, its omega's over the family's - 1 between %s "
                "and %s; the version-9 five members reproduced to %s (inertia equal: %s)." % (
                    f(max(grad), 2), f(max(asy), 2), "%.10f" % kr, "%.3e" % (kr - 1),
                    f(min(float(x_) for r_ in rowsT for x_ in r_[1].split()), 3) if rowsT else "-",
                    f(max(float(x_) for r_ in rowsT for x_ in r_[1].split()), 3) if rowsT else "-",
                    f(max(x_[1] for x_ in od), 2) if od else "-", all(x_[2] for x_ in od) if od else "-"))
    out += ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""]
    out += ["## P1a The family (J4run4 (0.01, 0.1) local_ka0, couplings (0, 0)): the full matter-node inertia and the lowest three eigenvalues of the "
            "fixed-N_B Hessian of M (volume metric), omega_1/omega_1(dilute), the matter nodes, the gradient check and the asymmetry", "",
            table(["z_c", "inertia", "omega_1", "omega_2", "omega_3", "omega_1/omega_1(dilute)", "matter nodes", "gradient median |ratio - 1|", "asymmetry"], rows1), "",
            "## P1b The trio at z_max", "", table(["z_c", "inertia", "omega_1", "omega_2", "omega_3", "M"], rows2), "",
            "## P1d The margin table (marks for members below z_max: y where omega_1/omega_1(dilute) exceeds |r_i| for r1, r2, r3 beta, r3 gamma, r4 in order)", "",
            table(["z_c", "omega_1/omega_1(dilute)", "r1 (P14 TF strain lock)", "r2 kernel mass shift", "r3 beta - 1 (edge)", "r3 gamma - 1 (estimate)",
                   "r4 coupling mass shift", "r6", "marks"], rows5), "",
            "## P1e The second crossing", "", table(["z_c", "inertia", "omega_1", "omega_2", "M"], rows6), "",
            "## P1f The twin (J4run2 local_klat (0.01, 0.1), couplings (0, 0)): its lowest three over the family's - 1, and the inertias (twin/family)", "",
            table(["z_c", "omega ratios - 1", "inertia"], rowsT), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    rec.update({"zx_lin": zx_lin, "zx_quad": zx_quad, "dw1_dz": dw1, "w1_dilute": w1d, "q0_edge": q0e, "second_crossing": zc2, "mass_extrema": ext})
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
