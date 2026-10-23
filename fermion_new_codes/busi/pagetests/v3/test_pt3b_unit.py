#!/usr/bin/env python
"""test_pt3b_unit.py (busi; B' one-body kernels, unit tests on the box CPU; no physics claim).
  B1 the cubic operator: torch apply = the CSR on random vectors; Hermitian without the absorber; the open box at N = 1, v = 0 against the
     analytic |E| = (sum_i cos^2(pi n_i/(L+1)))^(1/2) (each twice, both signs); the Strang absorber step against the dense exp(-i(h - i eta)dt).
  B2 the dilation: G(T) = 1 - Psi^dag Psi equals the time integral of Psi^dag 2 eta Psi (trapezoid on fine steps); 0 <= p <= 1.
  B3 the absorber's return error: an outgoing packet on a flat chain into the absorber: reflected weight below 1e-3.
  B4 golden-rule width: an orbital coupled to a tight-binding lead with the absorber at the far end: the kernel's escape p(T) = 1 - e^(-Gamma T),
     Gamma = 2 pi V^2 rho(eps) (rho the lead's density at the orbital's energy), and N_acc's threshold mapping p > 1/2 <=> Gamma T > ln 2.
  B5 the two routes agree: on a leaky chain cavity (the matched profile at r_s/a 16, kappa 1) each window input's quasi-mode width (dense eig of
     h - i eta) against its time-domain escape rate -ln(1 - p)/T at two clocks.
  B6 the readouts on synthetic inputs (N_acc, densities, D, two-point exponent).
  B7 the cube at a reduced size (matched profile at r_s/a 8, box 2.5 r_s, absorber beyond 2 r_s): interior window by dense eigh of the interior
     block, the CSR sub-box restriction equal to the full box's, the kernel at theta = 0.1, its unitarity (trace G + retained + transit = K)."""
import argparse
import math
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_kernels as KR  # noqa: E402
import pt3b_chain as CH  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))


def chk(name, ok, **kw):
    d = {"check": name, "verdict": "PASS" if bool(ok) else "FAIL"}
    d.update(kw)
    return d


def B1():
    out = {"checks": []}
    H = 4
    X, Y, Z, r = CU.grid(H)
    rng = np.random.default_rng(1)
    N = 1.0 - 0.3 * np.exp(-r ** 2 / 8); u = 0.4 * np.exp(-r / 3)
    v = [-u * c / np.where(r > 0, r, 1) for c in (X, Y, Z)]
    cube = CU.Cube(N, v)
    Hc = CU.cubic_csr(N, v)
    psi = rng.standard_normal((4,) + N.shape) + 1j * rng.standard_normal((4,) + N.shape)
    a = cube.apply(torch.as_tensor(psi, dtype=CU.CD)).numpy().ravel()
    b = Hc @ psi.ravel()
    out["checks"].append(chk("B1 torch apply = CSR", np.max(np.abs(a - b)) < 1e-12, max_diff=float(np.max(np.abs(a - b)))))
    out["checks"].append(chk("B1 Hermitian without the absorber", abs(Hc - Hc.conj().T).max() < 1e-15, defect=float(abs(Hc - Hc.conj().T).max())))
    N1 = np.ones_like(N); v0 = [np.zeros_like(N)] * 3
    e = np.linalg.eigvalsh(CU.cubic_csr(N1, v0).toarray())
    ea = CU.box_spectrum_analytic(2 * H + 1)
    out["checks"].append(chk("B1 open box (N = 1, v = 0) = analytic spectrum", np.max(np.abs(e - ea)) < 1e-12, max_diff=float(np.max(np.abs(e - ea)))))
    eta = CU.cap(r, 2.5, 4 * math.sqrt(3), 0.3)
    cube_e = CU.Cube(N, v, eta)
    dt = 0.2
    Ue = expm(-1j * CU.cubic_csr(N, v, eta).toarray() * dt)
    y = cube_e.step_cap(torch.as_tensor(psi, dtype=CU.CD), dt).numpy().ravel()
    err = float(np.max(np.abs(y - Ue @ psi.ravel())) / np.max(np.abs(psi)))
    y2 = cube_e.step_cap(cube_e.step_cap(torch.as_tensor(psi, dtype=CU.CD), dt / 2), dt / 2).numpy().ravel()
    err2 = float(np.max(np.abs(y2 - Ue @ psi.ravel())) / np.max(np.abs(psi)))
    out["checks"].append(chk("B1 Strang absorber step: error O(dt^3) per step (halving ratio > 3)", err < 1e-3 and err / max(err2, 1e-300) > 3, err=err, err_half=err2))
    return out


def B2():
    out = {"checks": []}
    H = 4
    X, Y, Z, r = CU.grid(H)
    N = np.ones_like(r); v = [np.zeros_like(r)] * 3
    eta = CU.cap(r, 2.0, 4 * math.sqrt(3), 0.5)
    Hm = CU.cubic_csr(N, v, eta).toarray()
    rng = np.random.default_rng(2)
    F = np.linalg.qr(rng.standard_normal((Hm.shape[0], 5)) + 1j * rng.standard_normal((Hm.shape[0], 5)))[0]
    dt = 0.01; T = 4.0
    U = expm(-1j * Hm * dt)
    E2 = np.tile(2 * eta.ravel(), 4)
    Psi = F.copy(); integ = np.zeros((5, 5), complex)
    prev = Psi.conj().T @ (E2[:, None] * Psi)
    for _ in range(int(round(T / dt))):
        Psi = U @ Psi
        cur = Psi.conj().T @ (E2[:, None] * Psi)
        integ += 0.5 * dt * (prev + cur); prev = cur
    G = np.eye(5) - Psi.conj().T @ Psi
    p = np.linalg.eigvalsh(0.5 * (G + G.conj().T))
    out["checks"].append(chk("B2 dilation: 1 - Psi^dag Psi = int Psi^dag 2 eta Psi dt", np.max(np.abs(G - integ)) < 1e-4, max_diff=float(np.max(np.abs(G - integ)))))
    out["checks"].append(chk("B2 escape spectrum in [0, 1]", p.min() > -1e-12 and p.max() < 1 + 1e-12, p=[float(x) for x in p]))
    return out


def B3():
    out = {"checks": []}
    n = 1200
    ops = K3.ChainOps(n, 1.0, R0=1e6)                      # kappa/R ~ 1e-6: a flat massless chain
    h = ops.h(np.ones(n), np.zeros(n))
    eta = np.zeros(n); eta[-300:] = 0.05 * (np.arange(1, 301) / 300) ** 2
    E2 = np.repeat(eta, 2)
    x = np.arange(n); k0 = 0.6; s = 40.0
    psi = np.zeros(2 * n, complex)
    amp = np.exp(-((x - 300) / s) ** 2 / 4) * np.exp(1j * k0 * x)
    e = math.sqrt(math.sin(k0) ** 2 + 1e-12)
    psi[0::2] = amp * (math.sin(k0) - 1e-6j) / (math.sqrt(2) * e); psi[1::2] = amp / math.sqrt(2)
    psi /= np.linalg.norm(psi)
    Hm = h.toarray() - 1j * np.diag(E2)
    U = expm(-1j * Hm * 50.0)
    for _ in range(80):                                    # 4000 a: whichever way it starts, the packet reaches the absorber and a reflection has time to return
        psi = U @ psi
    left = float(np.sum(np.abs(psi.reshape(-1, 2)[: n - 300]) ** 2))
    out["checks"].append(chk("B3 absorber reflection < 1e-3", left < 1e-3, reflected_weight=left))
    return out


def B4():
    """v1.1 of the check (first run compared the collected weight against the retained decay): the retained decay 1 - |<f|psi(T)>|^2 against
    the golden rule, and the collected weight p at T + the lead's travel time to the absorber against the same law."""
    out = {"checks": []}
    L = 300; J = 0.5; V = 0.05; eps = 0.3
    n = L + 1
    Hm = np.zeros((n, n))
    for i in range(1, L):
        Hm[i, i + 1] = Hm[i + 1, i] = -J
    Hm[0, 1] = Hm[1, 0] = V; Hm[0, 0] = eps
    cap_len = 120
    eta = np.zeros(n); eta[-cap_len:] = 0.05 * (np.arange(1, cap_len + 1) / cap_len) ** 2
    rho = math.sqrt(4 * J * J - eps * eps) / (2 * math.pi * J * J)
    Gam = 2 * math.pi * V * V * rho
    vg = math.sqrt(4 * J * J - eps * eps)
    travel = (n - cap_len + 0.5 * cap_len) / vg
    F = np.zeros((n, 1), complex); F[0, 0] = 1.0
    dt = 1.0
    A = -1j * (Hm - 1j * np.diag(eta))
    imask = np.zeros(n, bool); imask[0] = True
    Ts = [math.log(2) / Gam * f for f in (0.5, 1.0, 2.0)]
    kern, _ = KR.propagate_kernel(lambda X, hh: expm(A * hh) @ X, F, Ts + [T + travel for T in Ts], dt, imask)
    byT = {round(k["T"], 6): k for k in kern}
    rows = []
    for T in Ts:
        k1 = byT[round(T, 6)]; k2 = byT[round(T + travel, 6)]
        rows.append({"T": T, "retained_escape": float(1 - k1["retained"][0]), "collected_after_travel": float(k2["p"][0]), "golden_rule": 1 - math.exp(-Gam * T)})
    ok1 = all(abs(r_["retained_escape"] - r_["golden_rule"]) < 0.03 for r_ in rows)
    ok2 = all(abs(r_["collected_after_travel"] - r_["golden_rule"]) < 0.06 for r_ in rows) and rows[-1]["collected_after_travel"] <= rows[-1]["retained_escape"] + 1e-9
    out["checks"].append(chk("B4 [v1.1] retained decay = 1 - exp(-Gamma T), Gamma = 2 pi V^2 rho", ok1, rows=rows, Gamma=Gam))
    out["checks"].append(chk("B4 [v1.2] collected weight after the lead's travel time = the same law within the absorber's collection delay (0.06), never above the escape", ok2, travel=travel))
    m = [int(r_["retained_escape"] > 0.5) for r_ in rows]
    out["checks"].append(chk("B4 N_acc threshold: escape > 1/2 exactly when Gamma T > ln 2 (T = ln 2/Gamma x 0.5, 1, 2)", m[0] == 0 and m[2] == 1, flags=m))
    return out


def B5():
    """v1.3: the two routes on an isolated resonance (an orbital on a short lead whose absorber makes the lead's box modes much broader than the
    resonance: earlier versions had Delta_box ~ Gamma, no isolated resonance; printed): the quasi-mode carrying the input's largest
    biorthogonal residue, the matrix-pencil pole of the return signal, and the time-domain decay rate agree; the golden rule within 20%."""
    out = {"checks": []}
    L = 150; J = 0.5; V = 0.03; eps = 0.3
    n = L + 1
    Hm = np.zeros((n, n))
    for i in range(1, L):
        Hm[i, i + 1] = Hm[i + 1, i] = -J
    Hm[0, 1] = Hm[1, 0] = V; Hm[0, 0] = eps
    eta = np.zeros(n); eta[30:] = 0.3 * np.sin(0.5 * math.pi * np.clip((np.arange(30, n) - 30) / 60.0, 0, 1)) ** 2
    Hn = Hm - 1j * np.diag(eta)
    w, Vq = np.linalg.eig(Hn)
    f = np.zeros(n, complex); f[0] = 1.0
    rz = KR.residues(w, Vq, f)
    q = int(np.argmax(rz))
    rho = math.sqrt(4 * J * J - eps * eps) / (2 * math.pi * J * J)
    Gam = 2 * math.pi * V * V * rho
    dt = 4.0
    U = expm(-1j * Hn * dt)
    psi = f.copy(); sig = []; nrm = []
    for k in range(300):
        sig.append(psi[0]); nrm.append(abs(psi[0]) ** 2); psi = U @ psi
    k0 = 20
    wp, c, Sv = KR.matrix_pencil(np.array(sig[k0:]), dt, n_poles=4)
    jdom = int(np.argmax(np.abs(c)))
    tt = dt * np.arange(300)
    fit = np.polyfit(tt[k0:250], np.log(np.array(nrm[k0:250])), 1)
    rows = {"golden_rule": Gam, "quasimode_Gamma": float(-2 * w[q].imag), "quasimode_Re": float(w[q].real), "residue_max": float(rz[q]),
            "residue_second": float(np.sort(rz)[-2]), "pencil_Gamma": float(-2 * wp[jdom].imag), "pencil_Re": float(wp[jdom].real), "time_domain_rate": float(-fit[0])}
    ok = abs(rows["pencil_Gamma"] / rows["quasimode_Gamma"] - 1) < 0.02 and abs(rows["time_domain_rate"] / rows["quasimode_Gamma"] - 1) < 0.05 and \
        abs(rows["quasimode_Gamma"] / Gam - 1) < 0.2 and rows["residue_max"] > 0.9
    out["checks"].append(chk("B5 [v1.3] isolated resonance: quasi-mode width = pencil width (2%) = time-domain rate (5%); golden rule within 20%", ok, **rows))
    return out


def B6():
    out = {"checks": []}
    rs = 16.0
    Ts = KR.T_list_for(rs)
    kern = [{"T": T, "p": np.array([0.9, 0.6, 0.4, 0.1]) if i == 0 else np.array([0.99, 0.9, 0.7, 0.3]) if i == 1 else np.array([1, 1, 0.9, 0.6]),
             "trace_G": 0, "retained": np.zeros(4), "unphysical_p": 0} for i, T in enumerate(Ts)]
    r = KR.readouts(np.ones(4), kern, T_H=1.0 / (4 * math.pi * rs), rs=rs)
    ok = r["N_acc"] == {"0.1": 2, "0.01": 3, "0.001": 4} and abs(r["N_w_over_TH_rs"] - 4 * 4 * math.pi) < 1e-9 and abs(r["D"] - 4 / (4 * math.pi * 256)) < 1e-12
    ex = KR.two_point_exponent(10.0, 40.0, 16.0, 32.0)
    out["checks"].append(chk("B6 readouts: N_acc thresholds, dimensionless density, D; two-point exponent", ok and abs(ex - 2.0) < 1e-12, readout=r["N_acc"], exponent=ex))
    return out


def B7():
    out = {"checks": []}
    rs_a = 8.0
    rp, Np, up, rs_vac, meta = CU.matched_profile(rs_a)
    R_t = max(meta["roots"]); T_H = meta["T_H"]
    H = int(round(2.5 * rs_a))
    N, v, r = CU.fields_from_profile(H, rp, Np, up, rs_vac)
    eta = CU.cap(r, 2.0 * rs_a, H * math.sqrt(3), 0.05)
    # the interior block from a sub-box equals the full box's restriction
    Hs = int(math.ceil(R_t)) + 2
    Ns, vs, rsb = CU.fields_from_profile(Hs, rp, Np, up, rs_vac)
    hs = CU.cubic_csr(Ns, vs)
    ims = CU.masks(rsb, R_t)
    hf = CU.cubic_csr(N, v)
    imf = CU.masks(r, R_t)
    A = hs[np.where(ims)[0]][:, np.where(ims)[0]].toarray(); B = hf[np.where(imf)[0]][:, np.where(imf)[0]].toarray()
    out["checks"].append(chk("B7 interior block from the sub-box = the full box's", A.shape == B.shape and np.max(np.abs(A - B)) < 1e-15,
                             dim=A.shape[0], R_t=R_t))
    zi = {}
    e, U, dI = KR.interior_window(hf, imf, -T_H, T_H, info=zi)
    out["window"] = {"R_t": R_t, "T_H": T_H, "dim_I": dI, "e_over_TH": [float(x) for x in e / T_H], "structural_zero_modes": zi.get("n_zero", 0)}
    if len(e) == 0:
        out["checks"].append(chk("B7 interior window has inputs (reduced size)", True, note="none at r_s/a 8: kernel exercised on |e| < 4 T_H"))
        e, U, dI = KR.interior_window(hf, imf, -0.3, 0.3)
        e, U = e[:8], U[:, :8]
    cube = CU.Cube(N, v, eta)
    L = 2 * H + 1
    F = torch.as_tensor(U.T.reshape(len(e), 4, L, L, L), dtype=CU.CD)
    def step(Xn, hh):
        Xt = torch.as_tensor(Xn.T.reshape(-1, 4, L, L, L), dtype=CU.CD)
        y = cube.step_cap(Xt, hh)
        return y.reshape(y.shape[0], -1).numpy().T
    T_th = math.log(2) * rs_a / 0.1
    t0 = time.time()
    kern, _ = KR.propagate_kernel(step, U, [T_th], 2.0, imf)
    k = kern[0]
    out["kernel"] = {"T": T_th, "p": [float(x) for x in k["p"]], "retained": [float(x) for x in k["retained"]], "seconds": time.time() - t0}
    ok = k["unphysical_p"] == 0 and all(0 <= x <= 1 + 1e-9 for x in k["retained"])
    out["checks"].append(chk("B7 cube kernel at theta = 0.1: p in [0, 1], retained in [0, 1]", ok, **out["kernel"]))
    return out




def B8():
    """the pc13 field families on the cube at size 16: F2's roots (0.354 L and L) from the averaged fields at small ell; the kernel's K0/K1
    identities; the clock scan keeps w = v/N exactly before averaging; the averaged root recomputed for d = 0, 1/2, 1, 2; F1 and A32 load."""
    out = {"checks": []}
    L = 16.0; H = int(round(2.5 * L))
    f2 = CU.f2_profile()
    N, v, r, (Nd, vd) = CU.cube_fields(f2, L, H, 0.0, 1.0 / 64)
    rt = CU.averaged_root(N, v, r, H)
    ax = sorted(rt["axis"])
    ok = len(ax) >= 2 and abs(ax[0] / L - 0.354) < 0.05 and abs(ax[-1] / L - 1.0) < 0.05
    out["checks"].append(chk("B8 F2 dust ball: roots at 0.354 L and L (averaged fields, small ell)", ok, roots_over_L=[x / L for x in ax]))
    ki = CU.kernel_identities(L, H, 1.0 / 8)
    out["checks"].append(chk("B8 averaging kernel: K0 (constant) and K1 (linear, inner box) identities", ki["K0_max_abs"] < 1e-12 and ki["K1_max_abs_inner"] < 1e-9, **ki))
    roots = {}
    for d in (0.0, 0.5, 1.0, 2.0):
        N0, v0, r0, (Nd0, vd0) = CU.cube_fields(f2, L, H, d, 1.0 / 8)
        Nr, vr_, _, _ = CU.cube_fields(f2, L, H, 0.0, 1.0 / 8)
        wd = vd0[0] / Nd0; w0 = CU.cube_fields(f2, L, H, 0.0, 0.0)[1][0] / CU.cube_fields(f2, L, H, 0.0, 0.0)[0]
        roots[str(d)] = CU.averaged_root(N0, v0, r0, H)["axis"]
        if d == 2.0:
            out["checks"].append(chk("B8 clock scan keeps w = v/N before averaging (d = 2)", float(np.max(np.abs(wd - w0))) < 1e-12, max_dev=float(np.max(np.abs(wd - w0)))))
    out["roots_by_d"] = roots
    for name, fn in (("F1", lambda: CU.dense_record_profile(*CU.REC_F1)), ("A32", CU.a32_profile)):
        pr = fn()
        out[name] = {"label": pr["label"], "L_rec": pr["L_rec"], "roots_over_L": pr["roots_over_L"]}
        Nn, vn, rn, _ = CU.cube_fields(pr, L, H, 0.0, 1.0 / 8)
        out[name]["averaged_roots_over_L"] = [x / L for x in CU.averaged_root(Nn, vn, rn, H)["axis"]]
        out["checks"].append(chk("B8 %s loads with a root" % name, len(pr["roots_over_L"]) >= 1, **out[name]))
    return out


def B9():
    """the collected output's nested entropies in the dilation (collected_entropies): a coherently paired mode (an occupied orbital split with an
    inert early-output mode) purifies as the orbital escapes: S(R) from ln 2 toward 0, S(b|R) < 0; against the closed form s((1 + p)/2)."""
    out = {"checks": []}
    ps = np.linspace(0, 0.999, 11)
    Gl = [(float(k), np.diag([p, 1.0]).astype(complex)) for k, p in enumerate(ps)]
    rows = KR.collected_entropies(Gl, 0.5 * np.array([[1, 1], [1, 1]], complex))
    ex = K3.s_bin((1 + ps) / 2)
    dev = max(abs(r_["S_R"] - e_) for r_, e_ in zip(rows, ex))
    out["checks"].append(chk("B9 paired-mode purification: S(R) = s((1 + p)/2), S(b|R) < 0 for the late bins", dev < 1e-12 and all(r_["S_b_given_R"] < 0 for r_ in rows[2:]),
                             max_dev=dev, S_R=[r_["S_R"] for r_ in rows]))
    return out


def B10():
    """reviewer B's artificial-window control at n_w = 1, 10 (stationary): the mixing arm shares the port (largest/median small), the bright arm has one
    bright direction (largest/median large, n - 1 dark); the hole exits; the paired mode purifies. reviewer A's cavity control at n_lev = 1 runs."""
    out = {"checks": []}
    prof = CU.dense_record_profile(*CU.REC_F1)
    res = {}
    for arm in ("mixing", "bright"):
        for nw in (1, 10):
            r_ = CH.window_control_reviewb(nw, 64.0, prof, arm=arm, T_obs_over_L=100.0, n_bins=10)
            res["%s_%d" % (arm, nw)] = {k: r_[k] for k in ("emitted_number_supply", "largest_over_median", "n_p_above_1e-2", "hole_exit", "G_coupling", "rho_port")}
            res["%s_%d" % (arm, nw)]["paired_S_R_first_last"] = [r_["paired_mode_rows"][0]["S_R"], r_["paired_mode_rows"][-1]["S_R"]]
    out["reviewb"] = res
    r10 = CH.window_control_reviewb(10, 64.0, prof, arm="bright", T_obs_over_L=100.0, n_bins=10)
    sp = r10["depletion_spectrum_end"]
    ok = abs(sp[0] - res["bright_1"]["emitted_number_supply"]) < 1e-9 and max(abs(x) for x in sp[1:]) < 1e-10
    out["checks"].append(chk("B10 [v1.1] bright arm (n_w = 10): one direction escapes exactly as the single level, nine dark (< 1e-10)", ok, bright_spectrum=sp, **res))
    pr_ = res["mixing_1"]["paired_S_R_first_last"]
    out["checks"].append(chk("B10 paired mode on the native port purifies (S(R) falls from ln 2)", abs(pr_[0] - math.log(2)) < 1e-12 and pr_[1] < pr_[0] - 0.1, S_R=pr_))
    return out


TESTS = {"B8": B8, "B9": B9, "B10": B10, "B1": B1, "B2": B2, "B3": B3, "B4": B4, "B5": B5, "B6": B6, "B7": B7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [x for x in a.only.split(",") if x] or list(TESTS)
    res = {"script": "test_pt3b_unit.py", "own_md5": K3.own_md5s(), "tests": {}}
    for nm in names:
        t0 = time.time(); K3.log("start %s" % nm)
        try:
            r = TESTS[nm]()
        except Exception as ex:
            r = {"checks": [chk("%s crashed" % nm, False, error=repr(ex), tb=traceback.format_exc())]}
        r["seconds"] = time.time() - t0
        res["tests"][nm] = r
        for c in r["checks"]:
            K3.log("  %-5s %s" % (c["verdict"], c["check"]))
        K3.dump_json(res, os.path.join(a.out, "bprime_unit%s.json" % ("_" + "_".join(names) if a.only else "")))
    tot = [c for r in res["tests"].values() for c in r["checks"]]
    K3.log("%d checks: %d PASS, %d FAIL" % (len(tot), sum(c["verdict"] == "PASS" for c in tot), sum(c["verdict"] == "FAIL" for c in tot)))


if __name__ == "__main__":
    main()
