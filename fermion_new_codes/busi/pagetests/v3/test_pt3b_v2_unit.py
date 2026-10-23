#!/usr/bin/env python
"""test_pt3b_v2_unit.py (busi; B' v2 unit tests, the additions the two checks of design v1 ask for; box CPU). test_pt3b_unit.py (B1 to B10)
is unchanged and re-run beside this file.
  C1 the production kernel lifted: pt3b_v2lib.engine_kernels = engine_fields_v4.kernels; engine_avg_radial = engine_fields_v12
     .site_fields_kernel_average; the engine's back-average of the re-solved dense F1 fields = the record's stored site rows (reviewer A 'wrong' 4)
  C2 the averaging's discrete adjoint identity <W s, u> = <s, W^dag u> (reviewer B's B8 addition)
  C3 the cube's 3D Gaussian on the axis = the engine's radial formula at the same sites
  C4 the root-relative clock ramp: f = 0 at and outside the baseline root, w = v/N kept (reviewer A 'wrong' 2, reviewer B)
  C5 exterior geometry: the moved splice runs with onset 2.5 L, every control with half-side >= onset + 0.5 L (reviewer A 'wrong' 1)
  C6 deterministic windows (G11): two shift-invert solves of the same block give the same count and energies
  C7 pole selection (G10): on a two-pole signal the nearest-energy pole is chosen, not the dominant one
  C8 censoring: a threshold whose clock was not reached is 'censored' with a lower bound, never a zero count
  B12 (reviewer B bpchk sec. 2, the multimode coherent-channel reference): an independently assembled interior (five levels: three overlapping, one
     far, one dark) on a tight-binding lead with a partially reflecting barrier (a coherent return) and an absorber, inputs including a
     retained-exterior split mode; the REFERENCE builds the explicit Markovian bath (orthonormal time-bin modes (x, k) with amplitudes
     (2 eta_x dt)^(1/2) psi_x at the step midpoints) and computes every quantity from those amplitudes by SVD; the PRODUCTION path propagates
     by expm_multiply in two column batches, checkpoints the states, assembles G and R with gram_assemble, serializes them to npz, reloads,
     and reads them with bins_reader. Compared: the full K^dag K (off-diagonal included) and the threshold counts; the return amplitudes
     (against an eigendecomposition); the bin entropies, S(b | R) (one negative), the early-late mutual information, the retained-output
     mutual information, the outgoing number, a prepared hole's depletion; all again under a random unitary rotation of the input basis.
     Tolerance 1e-4 absolute (reviewer B's target). A second reference, the same system on a long unitary lead with the output the region
     beyond the absorber's onset, is printed beside (the dilation against a unitary exterior; not a pass criterion here, B11 is)."""
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_kernels as KR  # noqa: E402
import pt3b_v2lib as V2  # noqa: E402


def chk(name, ok, **kw):
    r = {"check": name, "verdict": "PASS" if ok else "FAIL"}
    r.update(kw)
    K3.log("  %s  %s" % (r["verdict"], name))
    return r


def C1():
    out = {"checks": []}
    sys.path.insert(0, V2.FM_DIR)
    import engine_fields_v4 as F4
    r = np.linspace(0.05, 40, 300); R = np.array([0.5, 3.0, 12.0, 30.0])
    a = F4.kernels(r, R, 4.0); b = V2.engine_kernels(r, R, 4.0)
    d = max(float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))
    out["checks"].append(chk("C1 engine_kernels = engine_fields_v4.kernels", d == 0.0, max_diff=d))
    p = V2.dense_record_profile_diag(*CU.REC_F1)
    kl = p["kernel_lift"]
    out["lift"] = kl; out["resolve_info"] = p["resolve_info"]
    out["checks"].append(chk("C1 my radial back-average = the engine's site_fields_kernel_average (1e-13)",
                             kl["mine_vs_engine_N_max"] < 1e-13 and kl["mine_vs_engine_v_max"] < 1e-13, **kl))
    out["checks"].append(chk("C1 the engine's back-average of the re-solved dense F1 fields = the record's stored site rows (1e-10)",
                             kl["engine_vs_site_N_max"] < 1e-10 and kl["engine_vs_site_v_max"] < 1e-10, **kl))
    return out


def C2():
    d = V2.adjoint_defect(H=10, ell=2.0)
    return {"checks": [chk("C2 averaging adjoint <W s, u> = <s, W^dag u> (1e-12)", d < 1e-12, rel_defect=d)]}


def C3():
    """v1.1: on the smooth F1 profile (the first version used F2, whose v has a kink at R* = L/2: there the cube's lattice sum and the engine's
    continuum quadrature differ by 2.8e-4 in v, a lattice-quadrature difference on a kink, printed here as the F2 line)."""
    out = {"checks": []}
    L = 16.0; H = 40
    for nm, prof in (("F1", CU.dense_record_profile(*CU.REC_F1)), ("F2", CU.f2_profile())):
        N, v, r, info = V2.gauss_cube_fields(prof, L, H, 0.0, ell=2.0)
        Ne, ve, _ = V2.engine_avg_radial(lambda x: prof["N"](x / L), lambda x: prof["v"](x / L), np.arange(1, H + 1, dtype=float), 2.0)
        dN = float(np.abs(N[H + 1:, H, H] - Ne).max()); dv = float(np.abs(v[0][H + 1:, H, H] - ve).max())
        if nm == "F1":
            out["checks"].append(chk("C3 [v1.1] cube 3D Gaussian on the axis = the engine radial formula, smooth F1 (1e-9)", dN < 1e-9 and dv < 1e-9, dN=dN, dv=dv, **info))
        else:
            out["F2_kink_lattice_quadrature"] = {"dN": dN, "dv": dv}
    return out


def C4():
    prof = CU.f2_profile()
    L = 16.0; H = 40
    N0, v0, r, _ = V2.gauss_cube_fields(prof, L, H, 0.0, ell=0.0)
    Rb = 0.695 * L
    N2, v2, _, _ = V2.gauss_cube_fields(prof, L, H, 2.0, ell=0.0, ramp="root", R_ramp=Rb)
    outside = r >= Rb
    d_out = float(max(np.abs(N2 - N0)[outside].max(), np.abs(v2[0] - v0[0])[outside].max()))
    inner = (r < 0.85 * Rb) & (r > 0)
    ratio = float(np.abs(N2[inner] / N0[inner] - math.exp(-2.0)).max())
    w0 = v0[0][inner] / N0[inner]; w2 = v2[0][inner] / N2[inner]
    return {"checks": [chk("C4 root-relative ramp: fields unchanged at and outside R_b, N x e^-d inside 0.85 R_b, w kept",
                           d_out == 0.0 and ratio < 1e-14 and float(np.abs(w2 - w0).max()) < 1e-14, d_out=d_out, ratio_dev=ratio)]}


def C5():
    a = V2.exterior_geometry("reviewb_out", 2.0, 2.5); b = V2.exterior_geometry("none", 2.25, 2.5); c = V2.exterior_geometry("none", 2.0, 2.5)
    ok = a == (2.5, 3.0) and b == (2.25, 2.75) and c == (2.0, 2.5)
    return {"checks": [chk("C5 exterior geometry: moved splice onset 2.5 L box 3 L; onset 2.25 L box 2.75 L; default unchanged", ok, a=a, b=b, c=c)]}


def C6():
    prof = CU.f2_profile()
    L = 8.0; H = 20
    N, v, r, _ = V2.gauss_cube_fields(prof, L, H, 0.0, ell=1.0)
    h = CU.cubic_csr(N, v)
    im = CU.masks(r, 0.6 * L)
    i1, i2 = {}, {}
    e1, _, d1 = V2.interior_window_det(h, im, 0.0, 0.2, dense_max=100, info=i1)
    e2, _, _ = V2.interior_window_det(h, im, 0.0, 0.2, dense_max=100, info=i2)
    ok = len(e1) == len(e2) and i1["n_zero"] == i2["n_zero"] and (len(e1) == 0 or float(np.abs(e1 - e2).max()) < 1e-12)
    return {"checks": [chk("C6 deterministic shift-invert window: same count, same zero modes, same energies", ok, n=len(e1), dim=d1,
                           zeros=[i1["n_zero"], i2["n_zero"]], cluster=i1.get("zero_cluster"))]}


def C7():
    dt = 1.0; t = np.arange(200) * dt
    sig = 0.9 * np.exp(-1j * 0.8 * t - 0.5 * 0.05 * t) + 0.1 * np.exp(-1j * 0.1 * t - 0.5 * 0.002 * t)
    w, c = V2.select_pole(sig, dt, 0.1, n_poles=4)
    ok = abs(w.real - 0.1) < 1e-6 and abs(-2 * w.imag - 0.002) < 1e-6
    return {"checks": [chk("C7 pole selection: the pole nearest e_ref (0.1, Gamma 0.002) beside a dominant one (0.8, 0.05)", ok, Re=float(w.real), Gamma=float(-2 * w.imag))]}


def C8():
    L = 10.0
    G = {100.0: np.diag([0.9, 0.3, 0.01])}
    cc = V2.censored_counts(G, L, (0.1, 1e-3), 100.0)
    ok = cc["0.001"]["time"]["status"] == "censored" and cc["0.001"]["time"]["lower_bound"] == 1 and "count" not in cc["0.001"]["time"]
    return {"checks": [chk("C8 censored threshold: status censored with a lower bound, no zero count", ok, counts=cc)]}


# ------------------------------------------------------------------------------------------------------------------- B12
def b12_system(M=200, J=0.5, barrier_site=15, Vb=0.3, cap_from=140, eta0=0.1):
    lev = np.array([0.10, 0.12, 0.14, 0.30, 0.20]); gc = np.array([0.05, 0.05, 0.04, 0.06, 0.0])
    nI = len(lev); d = nI + M
    Hm = np.zeros((d, d))
    Hm[np.arange(nI), np.arange(nI)] = lev
    for j in range(M - 1):
        Hm[nI + j, nI + j + 1] = Hm[nI + j + 1, nI + j] = -J
    Hm[nI + barrier_site, nI + barrier_site] = Vb
    Hm[:nI, nI] = gc; Hm[nI, :nI] = gc
    eta = np.zeros(d)
    if cap_from is not None and cap_from < M:
        x = np.arange(M - cap_from) / max(M - cap_from - 1, 1)
        eta[nI + cap_from:] = eta0 * x ** 2
    imask = np.zeros(d, bool); imask[:nI] = True
    pk = np.zeros(d); js = np.arange(M); pk[nI:] = np.exp(-0.5 * ((js - (cap_from or 140) + 6) / 3.0) ** 2) * np.cos(1.2 * js)
    pk /= np.linalg.norm(pk)
    F = np.zeros((d, 6), complex)
    F[0, 0] = F[1, 1] = F[2, 2] = 1.0; F[4, 3] = 1.0                       # levels 0.10, 0.12, 0.14 and the dark one
    l3 = np.zeros(d); l3[3] = 1.0
    F[:, 4] = (l3 + pk) / math.sqrt(2); F[:, 5] = (l3 - pk) / math.sqrt(2)  # the retained-exterior split pair
    C0 = np.zeros((6, 6), complex)
    C0[0, 0] = 1.0; C0[1, 1] = 0.5; C0[2, 2] = 0.5; C0[1, 2] = C0[2, 1] = 0.3; C0[4, 4] = 1.0
    Ch = np.zeros((6, 6), complex); Ch[0, 0] = Ch[2, 2] = Ch[4, 4] = 1.0     # the prepared hole: level 0.12 empty
    return Hm, eta, imask, F, C0, Ch


def B12():
    out = {"checks": []}
    Hm, eta, im, F, C0, Ch = b12_system()
    T = 300.0; nb = 10; tb = np.linspace(0, T, nb + 1)[1:]
    Hn = Hm - 1j * np.diag(eta)
    # ---------------- the reference: explicit bath amplitudes at the step midpoints, fine dt, exact step propagator
    dt = 0.02; nst = int(round(T / dt))
    Uh = expm(-1j * Hn * (dt / 2)); Uf = Uh @ Uh
    cap = np.where(eta > 0)[0]; sq = np.sqrt(2 * eta[cap] * dt)
    psi = F.copy(); A_bins = [[] for _ in range(nb)]; S_ref_t = []; R_ref = []
    k_bin = 0
    for k in range(nst):
        mid = Uh @ psi
        A_bins[min(int((k + 0.5) * dt / (T / nb)), nb - 1)].append(sq[:, None] * mid[cap])
        psi = Uf @ psi
        if abs((k + 1) * dt - tb[k_bin]) < 1e-9:
            R_ref.append(psi[im].conj().T @ psi[im]); k_bin += 1
            if k_bin == nb:
                break
    A_bins = [np.concatenate(b, axis=0) for b in A_bins]
    wv, Vr = np.linalg.eig(Hn); Vi = np.linalg.inv(Vr)
    S_eig = lambda t: F.conj().T @ (Vr * np.exp(-1j * wv * t)) @ (Vi @ F)

    def S_svd(Amat, C):
        if Amat.shape[0] == 0:
            return 0.0
        w, Vc = np.linalg.eigh(0.5 * (C + C.conj().T)); Chh = (Vc * np.sqrt(np.clip(w, 0, None))) @ Vc.conj().T
        nu = np.linalg.svd(Amat @ Chh, compute_uv=False) ** 2
        return float(K3.s_bin(np.clip(nu, 0, 1)).sum())

    def ref_quantities(C, Q=None):
        Ab = [A @ Q for A in A_bins] if Q is not None else A_bins
        Cq = Q.conj().T @ C @ Q if Q is not None else C
        cum = [np.concatenate(Ab[: n + 1], axis=0) for n in range(nb)]
        KK = cum[-1].conj().T @ cum[-1]
        S_R = [S_svd(cum[n], Cq) for n in range(nb)]
        S_b = [S_svd(Ab[n], Cq) for n in range(nb)]
        mi = [S_svd(cum[n], Cq) + S_svd(np.concatenate(Ab[n + 1:], axis=0), Cq) - S_R[-1] for n in range(nb - 1)]
        ro = []
        for n in range(nb):
            Rm = R_ref[n] if Q is None else Q.conj().T @ R_ref[n] @ Q
            wr, Vrr = np.linalg.eigh(0.5 * (Rm + Rm.conj().T)); Rh = (Vrr * np.sqrt(np.clip(wr, 0, None))) @ Vrr.conj().T
            joint = np.concatenate([Rh, cum[n]], axis=0)                    # an amplitude map with Gram R + A^dag A
            ro.append(S_svd(Rh, Cq) + S_R[n] - S_svd(joint, Cq))
        number = float(np.real(np.trace(Cq @ KK)))
        return {"KK": KK, "S_R": np.array(S_R), "S_b": np.array(S_b), "MI": np.array(mi), "RO": np.array(ro), "number": number}

    # ---------------- the production path: two column batches by expm_multiply, checkpoints, gram_assemble, npz round trip, bins_reader
    def production(Fin, C, tag):
        Asp = sp.csc_matrix(-1j * Hn)
        batches = [Fin[:, :3], Fin[:, 3:]]
        states = [expm_multiply(Asp, b, start=0.0, stop=T, num=nb + 1, endpoint=True)[1:] for b in batches]
        GL, RL = [], []
        for n in range(nb):
            G, R = V2.gram_assemble([states[0][n], states[1][n]], im)
            GL.append(G); RL.append(R)
        Sdiag = None
        pth = os.path.join(OUT, "b12_%s.npz" % tag)
        np.savez_compressed(pth, t=tb, G=np.array(GL), R=np.array(RL))
        Z = np.load(pth)
        Gl = [(float(t), g) for t, g in zip(Z["t"], Z["G"])]
        rd = V2.bins_reader(Gl, C, R_list=list(Z["R"]))
        return {"KK": Z["G"][-1], "S_R": np.array([b["S_R"] for b in rd["bins"]]), "S_b": np.array([b["S_b"] for b in rd["bins"]]),
                "MI": np.array([m["I_early_late"] for m in rd["early_late"]]), "RO": np.array([x["I_retained_output"] for x in rd["retained_output"]]),
                "number": rd["bins"][-1]["number"], "S_given": [b.get("S_b_given_R") for b in rd["bins"]], "states": states}

    ref = ref_quantities(C0); pro = production(F, C0, "base")
    tol = 1e-4
    dKK = float(np.abs(ref["KK"] - pro["KK"]).max())
    pr = np.linalg.eigvalsh(ref["KK"]); pp = np.linalg.eigvalsh(pro["KK"])
    thr = {th: (int(np.sum(pr > th)), int(np.sum(pp > th))) for th in (1e-1, 1e-2, 1e-3, 0.5)}
    out["checks"].append(chk("B12 full K^dag K incl. off-diagonal (1e-4) and threshold counts equal", dKK < tol and all(a == b for a, b in thr.values()),
                             max_abs=dKK, thresholds={"%g" % k: v for k, v in thr.items()}, p_ref=[float(x) for x in pr]))
    # returns: production S(t) from the checkpointed states vs the eigendecomposition
    S_pro = [np.concatenate([F.conj().T @ pro["states"][0][n], F.conj().T @ pro["states"][1][n]], axis=1) for n in range(nb)]
    dS = float(max(np.abs(S_pro[n] - S_eig(tb[n])).max() for n in range(nb)))
    out["checks"].append(chk("B12 return amplitudes S(t) = F^dag U(t) F against the eigendecomposition (1e-8)", dS < 1e-8, max_abs=dS))
    dsr = float(np.abs(ref["S_R"] - pro["S_R"]).max()); dsb = float(np.abs(ref["S_b"] - pro["S_b"]).max())
    out["checks"].append(chk("B12 bin entropies S(R_n), S(b_n) of the mixed multimode covariance (1e-4)", dsr < tol and dsb < tol,
                             dS_R=dsr, dS_b=dsb, S_b_given_R=pro["S_given"], S_R_ref=ref["S_R"].tolist()))
    # v1.1: the purification case on its own (the multimode covariance's mixed inputs dominate its increments): only the split mode occupied
    Cp = np.zeros((6, 6), complex); Cp[4, 4] = 1.0
    rp = ref_quantities(Cp); pp_ = production(F, Cp, "pair")
    neg = [x for x in pp_["S_given"] if x is not None and x < -1e-3]
    rneg = [rp["S_R"][n] - rp["S_R"][n - 1] for n in range(1, nb)]
    out["checks"].append(chk("B12 the retained-exterior split mode alone: S(b | R) < 0 in a bin, reference = production (1e-4)",
                             len(neg) > 0 and float(np.abs(rp["S_R"] - pp_["S_R"]).max()) < tol, S_b_given_R=pp_["S_given"], ref_increments=rneg))
    dmi = float(np.abs(ref["MI"] - pro["MI"]).max()); dro = float(np.abs(ref["RO"] - pro["RO"]).max())
    out["checks"].append(chk("B12 early-late and retained-output mutual informations (1e-4)", dmi < tol and dro < tol, dMI=dmi, dRO=dro,
                             MI_ref=ref["MI"].tolist(), RO_ref=ref["RO"].tolist()))
    dn = abs(ref["number"] - pro["number"])
    rh = ref_quantities(Ch); ph = production(F, Ch, "hole")
    dh = abs(rh["number"] - ph["number"])
    out["checks"].append(chk("B12 outgoing number and the prepared hole's depletion (1e-4)", dn < tol and dh < tol, number=[ref["number"], pro["number"]],
                             hole_number=[rh["number"], ph["number"]], depletion=float(ref["number"] - rh["number"])))
    # rotation of the input basis
    rng = np.random.default_rng(11)
    Qm, _ = np.linalg.qr(rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6)))
    pq = production(F @ Qm, Qm.conj().T @ C0 @ Qm, "rot")
    drot = max(float(np.abs(pq["S_R"] - pro["S_R"]).max()), float(np.abs(pq["MI"] - pro["MI"]).max()), float(np.abs(pq["RO"] - pro["RO"]).max()),
               abs(pq["number"] - pro["number"]), float(np.abs(np.sort(np.linalg.eigvalsh(pq["KK"])) - np.sort(pp)).max()))
    out["checks"].append(chk("B12 a random unitary rotation of the inputs (and new batch contents) leaves every physical readout unchanged (1e-10)",
                             drot < 1e-10, max_dev=drot))
    # the unitary-lead reference (printed)
    Hl, etal, iml, Fl, _, _ = b12_system(M=1200, cap_from=None)
    Fl[:, 4:] = 0.0
    js = np.arange(1200); pk = np.zeros(Hl.shape[0]); pk[5:] = np.exp(-0.5 * ((js - 134) / 3.0) ** 2) * np.cos(1.2 * js); pk /= np.linalg.norm(pk)
    l3 = np.zeros(Hl.shape[0]); l3[3] = 1.0
    Fl[:, 4] = (l3 + pk) / math.sqrt(2); Fl[:, 5] = (l3 - pk) / math.sqrt(2)
    Pl = expm_multiply(sp.csc_matrix(-1j * Hl), Fl, start=0.0, stop=T, num=nb + 1, endpoint=True)[1:]
    reg = np.zeros(Hl.shape[0], bool); reg[5 + 140:] = True
    w, Vc = np.linalg.eigh(C0); Chh = (Vc * np.sqrt(np.clip(w, 0, None))) @ Vc.conj().T
    S_lead = [float(K3.cov_entropy(Chh @ (P[reg].conj().T @ P[reg]) @ Chh)["S"]) for P in Pl]
    out["unitary_lead_S_R"] = S_lead; out["dilation_S_R"] = pro["S_R"].tolist()
    out["unitary_lead_vs_dilation_max"] = float(np.max(np.abs(np.array(S_lead) - pro["S_R"])))
    K3.log("  B12 printed: unitary lead S(R_n) vs dilation, max |diff| %.3g" % out["unitary_lead_vs_dilation_max"])
    return out


TESTS = {"C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "C6": C6, "C7": C7, "C8": C8, "B12": B12}
OUT = "."


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(a.out, exist_ok=True)
    names = [x for x in a.only.split(",") if x] or list(TESTS)
    res = {"script": "test_pt3b_v2_unit.py", "own_md5": K3.own_md5s(), "tests": {}}
    for nm in names:
        K3.log("start %s" % nm)
        try:
            r = TESTS[nm]()
        except Exception as ex:
            r = {"checks": [chk("%s crashed" % nm, False, error=repr(ex), tb=traceback.format_exc())]}
        res["tests"][nm] = r
        K3.dump_json(res, os.path.join(a.out, "bprime_v2_unit%s.json" % ("" if not a.only else "_" + "_".join(names))))
    allc = [c for t in res["tests"].values() for c in t["checks"]]
    K3.log("%d checks: %d PASS, %d FAIL" % (len(allc), sum(c["verdict"] == "PASS" for c in allc), sum(c["verdict"] == "FAIL" for c in allc)))


if __name__ == "__main__":
    main()
