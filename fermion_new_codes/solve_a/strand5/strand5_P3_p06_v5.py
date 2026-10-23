#!/usr/bin/env python
"""Strand 5 production P3 revision 4 (JUDGMENT_strand5_production.md sec. 7.2, A2, the lead's BUILDER item; builder analyst): the operator checks
completed, for both multipliers (V_code of P3 versions 1 and 3, the zero-sum form; V_eff of version 4, the complete symbol), on the four kernel
stencils and their local twins, every inventory included ((0.01, 0.1) too):
  R4a  the zero-wavevector reference from the stencil's moments: V_code(0) = (M_4 - M_2)/(6 M_2^2), V_eff(0) = V_code(0) + 1/(6 M_2)
       (M_k = sum_n n^k K_n; 1/(12 k_a(0)) with k_a(0) = M_2/2); the ring's value at its smallest q against it (ring lengths 2^17, 2^18, 2^19);
  R4b  the action-convergence criterion in place of N_eff: Pi with the line kernel truncated at N (K_n = 0 beyond N) for N = R, 2R, 4R, ... up to
       2 J + 2 (the largest index the radial action uses), per member; the smallest N with |Pi(N)/Pi(2J + 2) - 1| below 1e-8 (N_act) and the
       kernel's largest |K_n| beyond it;
  R4c  P3b's two checks, ((V - V(0)) 1)_j and r_j ((V - V(0)) (1/r))_j relative to V(0) (the moments' value), beyond L_0 = max(2R, N_act) on
       a domain J_c: the ring length varied at fixed J_c = 4 L_0 + 400 (2^17, 2^18, 2^19) and J_c varied at the ring 2^19 (2 and 4 times that);
  R4d  an independent spectral evaluation of Pi: (V g) from the odd extension of u = r g on a ring of L_S = the power of 2 at or above 8 J points,
       V(q) evaluated on that ring directly (no line kernel), beside the line-kernel Pi of version 4 (ring 2^18), for both multipliers; the
       prediction/measured ratios of revision 6's table F2 recomputed with it, and their changes (the lead's flag: any near-maximum ratio moving by
       more than 1e-4);
  R4e  the same-solve assertion: the stage-2 record's members (z_c, A, J, M) against the stage-1 family record's at the five members, and the
       profiles' lengths against J.
The radial action (V g)_j = (1/r_j) sum_i (K_|j-i| - K_(j+i)) u_i by FFT convolutions (Toeplitz and Hankel parts; checked against the dense
action of version 1 on a test vector).  Writes --md and --out-json.  Labelled: for the record, not a precondition of item 5's delivery."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import types

import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_P3_p06_v1 as P3  # noqa: E402
import strand5_P3_p06_v3 as P3v3  # noqa: E402
import strand5_P3_p06_v4 as P4  # noqa: E402
import strand5_lattice_star_v3 as S  # noqa: E402

f, table = P3.f, P3.table


def kernel_ring(Vfun, V0, nmax, LR):
    """K^V_n, n = 0..nmax, from the ring of LR points (V even; V at q = 0 from the moments)."""
    m = np.arange(LR // 2 + 1)
    q = 2 * np.pi * m / LR
    V = np.empty(len(q))
    V[1:] = Vfun(q[1:])
    V[0] = V0
    full = np.concatenate([V, V[-2:0:-1]])
    K = np.real(np.fft.ifft(full))
    return K[:nmax + 1], float(V[1])


def apply_V_fft(K, g, r):
    """(V g)_j = (1/r_j) sum_{i=1..J} (K_|j-i| - K_(j+i)) u_i, u = r g, by FFT convolutions; K needs indices 0..2J."""
    J = len(r)
    u = r * g
    Ksym = np.concatenate([K[1:J][::-1], K[:J]])
    T = fftconvolve(Ksym, u)[J - 1:2 * J - 1]
    Kp = K[:2 * J + 2]
    H = fftconvolve(Kp, u[::-1])[J + 1:2 * J + 1]
    return (T - H) / r


def apply_V_spectral(Vfun, V0, g, r, LS):
    """(V g) from the odd extension of u = r g on a ring of LS points and V(q) on that ring directly."""
    J = len(r)
    u = r * g
    w = np.zeros(LS)
    w[1:J + 1] = u
    w[LS - J:] = -u[::-1]
    W = np.fft.rfft(w)
    q = 2 * np.pi * np.arange(len(W)) / LS
    V = np.empty(len(q))
    V[1:] = Vfun(q[1:])
    V[0] = V0
    vw = np.fft.irfft(W * V, n=LS)
    return vw[1:J + 1] / r


def moments(stencil):
    ns = np.array(sorted(stencil), dtype=float)
    ks = np.array([stencil[int(n)] for n in ns])
    M2, M4 = float(np.sum(ns ** 2 * ks)), float(np.sum(ns ** 4 * ks))
    vc = (M4 - M2) / (6 * M2 * M2)
    return {"M2": M2, "M4": M4, "code": vc, "eff": vc + 1 / (6 * M2), "ka0": M2 / 2}


def member_data(F, S2, prof, stencil, kop, e1t, M):
    gM, gB = M["couplings"]
    mems = S2["members"]
    five = S2["meta"].get("five_members") or list(range(len(mems)))
    out = []
    for i in five:
        d = mems[i]
        key = "z%.6f" % d["z_c"]
        r, psi, N = prof[key + "_r"], prof[key + "_psi"], prof[key + "_N"]
        star = S.Star(len(r), stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1t)
        eb, comb, nn, kap = star.matter(psi, N, d["mu_inf"])
        fm = [m for m in F["members"] if abs(m["z_c"] - d["z_c"]) < 1e-12]
        out.append({"z_c": d["z_c"], "r": r, "f": psi ** 5 * eb, "g": N * psi ** 5 * comb, "A": d["A"], "J": int(d["J"]),
                    "measured": d["lock"].get("vac_part_of_AmB_over_A"), "stage1": fm[0] if fm else None, "stage2": d, "profile_len": len(r)})
    return out


def Pi(mem, Vg):
    r = mem["r"]
    return float(np.sum(4 * np.pi * r * r * mem["f"] * Vg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True, help="the P3eff launcher's list: label|family|stage2|karg|table|twin-stage2; ...")
    ap.add_argument("--rev6-json", required=True, help="revision 6's P3eff.json (the F2 ratios to compare)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    R6 = json.load(open(a.rev6_json))
    rec, A_rows, B_rows, C_rows, D_rows, E_rows, summ, flags = {}, [], [], [], [], [], [], []
    # the FFT action against the dense one
    rt = np.arange(1, 1201, dtype=float)
    Kt = np.exp(-0.01 * np.arange(0, 2500)) * np.cos(0.3 * np.arange(0, 2500))
    gt = np.exp(-rt / 300.0)
    chk = float(np.max(np.abs(apply_V_fft(Kt, gt, rt) - P3.apply_V(Kt, gt, rt))) / np.max(np.abs(P3.apply_V(Kt, gt, rt))))
    summ.append("The FFT radial action against version 1's dense action on a test vector: largest relative difference %.2e." % chk)
    done_stencils = {}
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
        R = int(max(stencil))
        mo = moments(stencil)
        Vfun = {"code": lambda q, st=stencil: P3v3.V_stencil3(st, q)[0], "eff": lambda q, st=stencil: P4.V_eff(st, q)}
        mem = member_data(F, S2, prof, stencil, kop, e1t, M)
        Jmax = max(m_["J"] for m_ in mem)
        frec = {"R": R, "moments": mo, "members": []}
        # R4e: the same-solve assertion
        for m_ in mem:
            s1, s2 = m_["stage1"], m_["stage2"]
            ok = s1 is not None and abs(s1["A"] / s2["A"] - 1) < 1e-12 and int(s1["J"]) == int(s2["J"]) and abs(s1["M"] / s2["M"] - 1) < 1e-12 \
                and m_["profile_len"] == int(s2["J"])
            E_rows.append([lab, "%.6g" % m_["z_c"], str(ok), "%.3e" % (abs(s1["A"] / s2["A"] - 1) if s1 else float("nan")), "%d/%d/%d" % (
                int(s1["J"]) if s1 else -1, int(s2["J"]), m_["profile_len"])])
            if not ok:
                flags.append("%s z_c %.6g: the same-solve assertion fails" % (lab, m_["z_c"]))
        stkey = (M["mlight"], M["aH"], kv)
        for mult in ("code", "eff"):
            V0 = mo[mult]
            # R4a: the ring's smallest q against the moments (computed once per stencil)
            if (stkey, mult) not in done_stencils:
                ra = []
                for LR in (2 ** 17, 2 ** 18, 2 ** 19):
                    q1 = 2 * np.pi / LR
                    ra.append(float(Vfun[mult](np.array([q1]))[0] / V0 - 1))
                A_rows.append([lab.split(" g")[0], mult, "%.10e" % V0] + ["%.3e" % x for x in ra])
                done_stencils[(stkey, mult)] = True
            nK = 2 * Jmax + 2
            K18, _ = kernel_ring(Vfun[mult], V0, max(nK, 2 * (16 * max(2 * R, 1) + 400) + 2), 2 ** 18)
            # R4b: action convergence per member
            Ns = sorted(set([R * 2 ** k for k in range(0, 40) if R * 2 ** k < nK] + [nK]))
            nacts = []
            for m_ in mem:
                J = m_["J"]
                full = Pi(m_, apply_V_fft(K18[:2 * J + 2], m_["g"], m_["r"]))
                nact = None
                for N in Ns:
                    Kt_ = K18[:2 * J + 2].copy()
                    if N < len(Kt_) - 1:
                        Kt_[N + 1:] = 0
                    p = Pi(m_, apply_V_fft(Kt_, m_["g"], m_["r"]))
                    if abs(p / full - 1) < 1e-8:
                        nact = N
                        break
                tail = float(np.max(np.abs(K18[(nact or nK) + 1:2 * J + 2]))) if (nact or nK) + 1 < 2 * J + 2 else 0.0
                nacts.append(nact or nK)
                B_rows.append([lab, mult, "%.6g" % m_["z_c"], str(J), str(nact), "%.3e" % tail])
            L0 = max(2 * R, max(nacts))
            # R4c: the checks, ring length and domain varied separately
            Jc0 = 4 * L0 + 400
            for tag, LR, Jc in [("ring 2^17", 2 ** 17, Jc0), ("ring 2^18", 2 ** 18, Jc0), ("ring 2^19", 2 ** 19, Jc0), ("domain x2 (ring 2^19)", 2 ** 19, 2 * Jc0),
                                ("domain x4 (ring 2^19)", 2 ** 19, 4 * Jc0)]:
                if LR // 2 < 2 * Jc + 2:
                    C_rows.append([lab.split(" g")[0], mult, tag, str(L0), str(Jc), "ring too short", "-"])
                    continue
                Kc, _ = kernel_ring(Vfun[mult], V0, 2 * Jc + 2, LR)
                rc = np.arange(1, Jc + 1, dtype=float)
                flat = apply_V_fft(Kc, np.ones(Jc), rc) - V0
                coul = rc * (apply_V_fft(Kc, 1.0 / rc, rc) - V0 / rc)
                sel = (rc > L0) & (rc < Jc - L0)
                C_rows.append([lab.split(" g")[0], mult, tag, str(L0), str(Jc), "%.3e" % (np.max(np.abs(flat[sel])) / abs(V0)),
                               "%.3e" % (np.max(np.abs(coul[sel])) / abs(V0))])
            # R4d: the spectral Pi and the F2 ratios
            for m_ in mem:
                J = m_["J"]
                LS = 1 << int(math.ceil(math.log2(8 * J)))
                p_k = Pi(m_, apply_V_fft(K18[:2 * J + 2], m_["g"], m_["r"]))
                p_s = Pi(m_, apply_V_spectral(Vfun[mult], V0, m_["g"], m_["r"], LS))
                pred_k = -p_k / (math.pi * kop * m_["A"])
                pred_s = -p_s / (math.pi * kop * m_["A"])
                ms = m_["measured"]
                old = [x for x in R6.get(lab, {}).get("kernel", []) if abs(x["z_c"] - m_["z_c"]) < 1e-9]
                old_ratio = (old[0]["pred_" + mult] / old[0]["measured"]) if (old and old[0]["measured"]) else None
                rk = pred_k / ms if ms else None
                rs = pred_s / ms if ms else None
                d6 = (rs - old_ratio) if (rs is not None and old_ratio is not None) else None
                D_rows.append([lab, mult, "%.6g" % m_["z_c"], "%.6e" % p_k, "%.6e" % p_s, "%.2e" % (p_s / p_k - 1), f(old_ratio, 7), f(rk, 7), f(rs, 7),
                               "%.2e" % d6 if d6 is not None else "-", str(LS)])
                near = 0.8 <= m_["z_c"] <= 1.1
                if near and d6 is not None and abs(d6) > 1e-4:
                    flags.append("%s %s z_c %.6g: the ratio moves by %.2e (above 1e-4)" % (lab, mult, m_["z_c"], d6))
                frec["members"].append({"mult": mult, "z_c": m_["z_c"], "Pi_kernel18": p_k, "Pi_spectral": p_s, "LS": LS, "ratio_rev6": old_ratio,
                                        "ratio_kernel18": rk, "ratio_spectral": rs, "change_vs_rev6": d6})
            frec.setdefault("L0", {})[mult] = L0
        rec[lab] = frec
    summ.append("Flags (the same-solve assertion; near-maximum F2 ratios moving by more than 1e-4): " + ("; ".join(flags) if flags else "none") + ".")
    md = ["# Strand 5 production P3 revision 4 (strand5_P3_p06_v5.py): the operator checks completed for both multipliers", "", "## Emitted summary", ""] + [
        "- " + s for s in summ] + ["",
        "## R4a The zero-wavevector reference: the moments' V(0) and the ring's value at its smallest q relative to it (rings 2^17, 2^18, 2^19)", "",
        table(["inventory", "multiplier", "V(0) from the moments", "ring 2^17", "ring 2^18", "ring 2^19"], A_rows), "",
        "## R4b The action-convergence radius: the smallest truncation N with |Pi(N)/Pi(2J + 2) - 1| < 1e-8, and the kernel's largest |K_n| beyond it", "",
        table(["family", "multiplier", "z_c", "J", "N_act", "max |K_n| beyond N_act"], B_rows), "",
        "## R4c P3b's checks relative to the moments' V(0), beyond L_0 = max(2R, N_act): the ring length and the domain varied separately", "",
        table(["inventory", "multiplier", "variation", "L_0", "J_c", "flat check", "Coulomb check"], C_rows), "",
        "## R4d Pi by the line kernel (ring 2^18) and by the independent spectral route (ring L_S), and table F2's prediction/measured", "",
        table(["family", "multiplier", "z_c", "Pi (kernel 2^18)", "Pi (spectral)", "spectral/kernel - 1", "ratio (revision 6)", "ratio (kernel, here)",
               "ratio (spectral)", "spectral - revision 6", "L_S"], D_rows), "",
        "## R4e The same-solve assertion (stage-2 members against the stage-1 family record and the profiles)", "",
        table(["family", "z_c", "same solve", "|A_1/A_2 - 1|", "J stage 1 / stage 2 / profile"], E_rows), ""]
    open(a.md, "w").write("\n".join(md) + "\n")
    json.dump({"families": rec, "summary": summ, "flags": flags}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join("- " + s for s in summ))


if __name__ == "__main__":
    main()
