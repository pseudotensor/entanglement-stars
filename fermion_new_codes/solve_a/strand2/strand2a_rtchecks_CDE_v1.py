#!/usr/bin/env python
"""Strand 2a red-team checks, parts C, D and E (spec solve/specs/SPEC_strand2a_redteam_checks.md; builder analyst).  One-body rings
of L sites with the 1-component Weyl D of strand 0 (D[x, x+1] = -i/2, D[x, x-1] = +i/2, so D|p> = sin p |p>), as in the addendum
(strand2a_addendum_transport.py, _v3.py): A = m psi^-1 + i psi^-3 D, E = (A^dag A)^(1/2), U = A E^-1, B = E^(1/2) (N psi) E^(1/2);
the law's positive spectrum is eig(B); the flat positive basis v_p^+ = P (|p>, U|p>)/sqrt2; the native h_nat(s) = m (1 x sigma_3) +
s^-1 (D x sigma_1) s^-1, rotated by W = 1 x (1/2)(1 - i(sigma_x + sigma_y + sigma_z)) as in the addendum.  One part per call.

--part C (A.1; m in {0.1, 0.2}; psi = 1 + eps cos(2 pi x/L), N = 1):
  C.1 at L in {64, 256}: B^(1) = dB/deps and B^(2) = (1/2) d^2B/deps^2 at eps = 0 by central differences of the matrix (steps 1e-3 and
      5e-4, both reported); in the plane-wave basis (the exact flat eigenbasis, E_p = (m^2 + sin^2 p)^(1/2)) the band-bottom doublet
      {p = 0, p = pi}: the first-order matrix B^(1) on it, the second-order effective matrix H2_ab = B^(2)_ab + sum_{p not in the doublet}
      B^(1)_ap B^(1)_pb/(m - E_p) and its eigenvalues, its two parts separately (the direct second derivative and the sum over
      intermediate levels); against the finite-eps coefficient (E_low(eps) - m)/eps^2 at eps in {1e-3, 1e-2} and -4 m.
  C.2 at m = 0.1, L in {64, 128, 256, 512, 1024}: (a) eps = 0.05 x 64/L and (b) eps = 0.05: the shift m - E_low, shift/(m eps^2), the
      eps power from the bracketing pair (eps/2, eps), the inverse participation ratio sum |v_x|^4 of the lowest eigenvector and L times it.
--part D (A.1c; L = 2048, m = 0.1, psi = N = 1): Delta_x = M1_x - X1_x (M1_x = (1/2){P_x, h_*}, X1_x the addendum's ruler vertex
  at site x) in: the single-valley packets at 0 and at pi (Gaussian in p with widths sigma in {0.01, 0.03, 0.1}, the addendum's
  construction), their incoherent equal mixture (the mean), the coherent superpositions (|0> + e^(i alpha)|pi>)/sqrt2 with alpha in
  {0, pi/2, pi}; at the site x = 0, at the neighbouring site x = 1, and their average (the two-site unit cell).
--part E (A.4; L in {1024, 4096}, m in {0.1, 0.2}; psi = 1 + u, u_x = eps cos(2 pi j x/L), j in {1, 2, 3, 4} at L = 1024 and {1, 2} at
  L = 4096): the first-order derivatives V_law(p, p') = d<v_p^+|h_*|v_p'^+>/deps and V_nat likewise (central differences at eps = 1e-5
  and 1e-6; their agreement), u_pp' = <p|u|p'>/eps, Delta/u_pp' = (V_law - V_nat)/u_pp' between band-bottom states of each valley with
  max(|p|, |p'|) <= w m (w in {1/8, 1/4, 1/2}; p measured from the valley); the fit Delta/u = b (p' - p) + c (p' - p)^2 per (L, m,
  valley, window) over all j, and over j in {1, 2}; the red team's expression dB_pp' + (E_p' - E_p)(U^dag dU)_pp'/2 (dB = dE +
  (E_p E_p')^(1/2) u_pp', dE and dU by the same central differences) against the direct derivative of the law element.
Writes <out>/rtchecks_<part>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import socket
import time

import numpy as np
import scipy.sparse as sp

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def herm(M):
    return 0.5 * (M + M.conj().T)


def ring_D(L):
    D = np.zeros((L, L), dtype=complex)
    for x in range(L):
        D[x, (x + 1) % L] += -0.5j
        D[x, (x - 1) % L] += 0.5j
    return D


def law_parts(m, psi, D):
    """E, E^(1/2), U and B = E^(1/2) psi E^(1/2) (N = 1) for the ring."""
    p = np.asarray(psi, float)
    A = m * np.diag(1 / p) + 1j * (p ** -3)[:, None] * D
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    E = (V * np.sqrt(w)) @ V.conj().T
    Eh = (V * w ** 0.25) @ V.conj().T
    U = A @ ((V / np.sqrt(w)) @ V.conj().T)
    B = herm(Eh @ (p[:, None] * Eh))
    return E, Eh, U, B


def dft(L):
    x = np.arange(L)
    return np.exp(2j * np.pi * np.outer(x, x) / L) / np.sqrt(L)       # column n = plane wave p = 2 pi n/L


# ------------------------------------------------------------------ part C
def part_C():
    out = {"C1": [], "C2": []}
    for L in (64, 256):
        D = ring_D(L)
        F = dft(L)
        xs = np.arange(L)
        c = np.cos(2 * np.pi * xs / L)
        pk = 2 * np.pi * np.arange(L) / L
        for m in (0.1, 0.2):
            Ep = np.sqrt(m * m + np.sin(pk) ** 2)
            B0 = law_parts(m, np.ones(L), D)[3]
            for hstep in (1e-3, 5e-4):
                Bp = law_parts(m, 1 + hstep * c, D)[3]
                Bm = law_parts(m, 1 - hstep * c, D)[3]
                B1 = (Bp - Bm) / (2 * hstep)
                B2 = (Bp - 2 * B0 + Bm) / (2 * hstep * hstep)
                B1k = F.conj().T @ B1 @ F
                B2k = F.conj().T @ B2 @ F
                dbl = [0, L // 2]
                others = [k for k in range(L) if k not in dbl]
                H1 = B1k[np.ix_(dbl, dbl)]
                Hd = B2k[np.ix_(dbl, dbl)]
                Hs = np.zeros((2, 2), dtype=complex)
                for k in others:
                    Hs += np.outer(B1k[dbl, k], B1k[k, dbl]) / (m - Ep[k])
                H2 = Hd + Hs
                ev2 = np.linalg.eigvalsh(herm(H2))
                row = {"L": L, "m": m, "fd_step": hstep, "B0_vs_plane_waves": float(np.max(np.abs(F.conj().T @ B0 @ F - np.diag(Ep)))),
                       "H1_doublet_max_abs": float(np.max(np.abs(H1))), "H2": [[[float(z.real), float(z.imag)] for z in r_] for r_ in H2],
                       "H2_eigs": ev2.tolist(), "direct_second_derivative_diag": [float(Hd[0, 0].real), float(Hd[1, 1].real)],
                       "sum_over_levels_diag": [float(Hs[0, 0].real), float(Hs[1, 1].real)], "minus_4m": -4 * m}
                for eps in (1e-3, 1e-2):
                    w = np.linalg.eigvalsh(law_parts(m, 1 + eps * c, D)[3])
                    row["finite_eps_%g" % eps] = float((w[0] - m) / eps ** 2)
                out["C1"].append(row)
                print("C1 L %d m %g step %g: H2 eigs %s, finite eps 1e-3 %.6g" % (L, m, hstep, ev2, row["finite_eps_0.001"]), flush=True)
    m = 0.1
    for seq in ("a", "b"):
        for L in (64, 128, 256, 512, 1024):
            D = ring_D(L)
            xs = np.arange(L)
            c = np.cos(2 * np.pi * xs / L)
            eps = 0.05 * 64 / L if seq == "a" else 0.05
            res = {}
            for e in (eps, eps / 2):
                w, V = np.linalg.eigh(law_parts(m, 1 + e * c, D)[3])
                v = V[:, 0] / np.linalg.norm(V[:, 0])
                res[e] = (m - w[0], float(np.sum(np.abs(v) ** 4)))
            sh, ipr = res[eps]
            out["C2"].append({"sequence": seq, "L": L, "m": m, "eps": eps, "eps_m_over_q": eps * m / (2 * np.pi / L), "shift": float(sh),
                              "shift_over_m_eps2": float(sh / (m * eps * eps)), "eps_power_pair": float(np.log(res[eps][0] / res[eps / 2][0]) / np.log(2.0)),
                              "ipr": ipr, "L_ipr": L * ipr, "shift_half_eps": float(res[eps / 2][0]), "ipr_half_eps": res[eps / 2][1]})
            print("C2 %s L %d eps %.5g: shift %.6g power %.4f L*ipr %.4g" % (seq, L, eps, sh, out["C2"][-1]["eps_power_pair"], L * ipr), flush=True)
    return out


# ------------------------------------------------------------------ part D
def part_D(L=2048, m=0.1):
    D = ring_D(L)
    E, Eh, U, B0 = law_parts(m, np.ones(L), D)
    xs = np.arange(L)
    Zl = np.zeros((L, L), dtype=complex)
    # h_* in the block layout (u-block, then the U-block); mode layout j = 2x + chiral is a permutation, applied to vectors below
    hblk = np.block([[Zl, B0 @ U.conj().T], [U @ B0, Zl]])

    def to_mode(vb):
        v = np.zeros(2 * L, dtype=complex)
        v[0::2] = vb[:L]
        v[1::2] = vb[L:]
        return v

    def site_delta(x):
        Px = np.zeros(2 * L)
        Px[2 * x] = Px[2 * x + 1] = 1.0
        hm = np.zeros((2 * L, 2 * L), dtype=complex)
        hm[0::2, 0::2] = hblk[:L, :L]
        hm[0::2, 1::2] = hblk[:L, L:]
        hm[1::2, 0::2] = hblk[L:, :L]
        hm[1::2, 1::2] = hblk[L:, L:]
        M1 = 0.5 * (Px[:, None] * hm + hm * Px[None, :])
        X0 = np.outer(Eh[:, x], Eh[x, :])
        Xb = np.block([[Zl, X0 @ U.conj().T], [U @ X0, Zl]])
        X1 = np.zeros((2 * L, 2 * L), dtype=complex)
        X1[0::2, 0::2] = Xb[:L, :L]
        X1[0::2, 1::2] = Xb[:L, L:]
        X1[1::2, 0::2] = Xb[L:, :L]
        X1[1::2, 1::2] = Xb[L:, L:]
        return M1 - X1
    Dl = {0: site_delta(0), 1: site_delta(1)}

    def vp(n):
        k = np.exp(2j * np.pi * n * xs / L) / np.sqrt(L)
        return to_mode(np.concatenate([k, U @ k]) / np.sqrt(2))

    def packet(center, sig):
        c = np.zeros(2 * L, dtype=complex)
        for n in range(-L // 4, L // 4):
            nn = (n + (L // 2 if center == "pi" else 0)) % L
            p = 2 * np.pi * n / L
            c += np.exp(-p * p / (2 * sig * sig)) * vp(nn)
        return c / np.linalg.norm(c)
    out = {"L": L, "m": m, "rows": []}
    for sig in (0.01, 0.03, 0.1):
        p0, ppi = packet("0", sig), packet("pi", sig)
        for x in (0, 1, "cell"):
            def ev(st):
                if x == "cell":
                    return 0.5 * (float(np.real(st.conj() @ Dl[0] @ st)) + float(np.real(st.conj() @ Dl[1] @ st)))
                return float(np.real(st.conj() @ Dl[x] @ st))
            r = {"sigma": sig, "site": x, "valley_0": ev(p0), "valley_pi": ev(ppi)}
            r["incoherent_mixture"] = 0.5 * (r["valley_0"] + r["valley_pi"])
            for alpha in (0.0, np.pi / 2, np.pi):
                r["coherent_alpha_%.4f" % alpha] = ev((p0 + np.exp(1j * alpha) * ppi) / np.sqrt(2))
            r["overlap_0_pi"] = float(abs(p0.conj() @ ppi))
            out["rows"].append(r)
            print("D sigma %g site %s: %s" % (sig, x, json.dumps({k: v for k, v in r.items() if k not in ("sigma", "site")})), flush=True)
    return out


# ------------------------------------------------------------------ part E
def native_sparse(m, s, L):
    rows, cols, vals = [], [], []
    for x in range(L):
        for (dx, a) in ((1, -0.5j), (-1, 0.5j)):
            y = (x + dx) % L
            rows.append(x)
            cols.append(y)
            vals.append(a / (s[x] * s[y]))
    Ds = sp.csr_matrix((vals, (rows, cols)), shape=(L, L))
    return sp.kron(sp.identity(L), sp.csr_matrix(m * SZ)) + sp.kron(Ds, sp.csr_matrix(SX))


def part_E():
    out = {"rows": [], "fits": []}
    W2 = 0.5 * (I2 - 1j * (SX + SY + SZ))
    for L in (1024, 4096):
        D = ring_D(L)
        xs = np.arange(L)
        Wf = sp.kron(sp.identity(L), sp.csr_matrix(W2)).tocsr()
        js = (1, 2, 3, 4) if L == 1024 else (1, 2)
        for m in (0.1, 0.2):
            E1, Eh1, U1, B1 = law_parts(m, np.ones(L), D)
            pk = 2 * np.pi * np.arange(L) / L
            Ep = np.sqrt(m * m + np.sin(pk) ** 2)

            def kvec(n):
                return np.exp(2j * np.pi * n * xs / L) / np.sqrt(L)

            def vp_mode(n):
                k = kvec(n)
                vb = np.concatenate([k, U1 @ k]) / np.sqrt(2)
                v = np.zeros(2 * L, dtype=complex)
                v[0::2], v[1::2] = vb[:L], vb[L:]
                return v
            wmax = 0.5 * m
            nmax = int(np.floor(wmax / (2 * np.pi / L)))
            for j in js:
                q = 2 * np.pi * j / L
                pairs = []
                for valley, n0, nu in (("0", 0, +1), ("pi", L // 2, -1)):
                    for a in range(-nmax, nmax + 1):
                        for sgn in (+1, -1):
                            b = a + sgn * j
                            if abs(b) <= nmax:
                                pairs.append((valley, nu, a, b, (n0 + a) % L, (n0 + b) % L))
                K = {n: kvec(n) for pr in pairs for n in (pr[4], pr[5])}
                VPm = {n: vp_mode(n) for n in K}
                c = np.cos(q * xs)
                der = {}
                for step in (1e-5, 1e-6):
                    vals = {}
                    for sgn in (+1, -1):
                        psi = 1 + sgn * step * c
                        E, Eh, U, B = law_parts(m, psi, D)
                        BUd, UB, U1dU = B @ U.conj().T, U @ B, U1.conj().T @ U
                        hn = Wf @ native_sparse(m, psi, L) @ Wf.conj().T
                        el = {}
                        for (valley, nu, a, b, na, nb) in pairs:
                            ka, kb = K[na], K[nb]
                            law = 0.5 * (ka.conj() @ (BUd @ (U1 @ kb)) + (U1 @ ka).conj() @ (UB @ kb))
                            nat = VPm[na].conj() @ (hn @ VPm[nb])
                            el[(na, nb)] = (law, nat, ka.conj() @ (E @ kb), ka.conj() @ (U1dU @ kb), ka.conj() @ (B @ kb))
                        vals[sgn] = el
                    der[step] = {key: tuple((vals[1][key][i] - vals[-1][key][i]) / (2 * step) for i in range(5)) for key in vals[1]}
                for (valley, nu, a, b, na, nb) in pairs:
                    d5, d6 = der[1e-5][(na, nb)], der[1e-6][(na, nb)]
                    upp = complex(np.sum(np.conj(K[na]) * c * K[nb]))
                    Vl, Vn = d5[0], d5[1]
                    dE, dUU, dB = d5[2], d5[3], d5[4]
                    p, pp = 2 * np.pi * a / L, 2 * np.pi * b / L
                    Ea, Eb = Ep[na], Ep[nb]
                    rt = (dE + np.sqrt(Ea * Eb) * upp) + 0.5 * (Eb - Ea) * dUU
                    out["rows"].append({"L": L, "m": m, "j": j, "valley": valley, "nu": nu, "p": p, "pp": pp, "u_pp": [upp.real, upp.imag],
                                        "V_law": [Vl.real, Vl.imag], "V_nat": [Vn.real, Vn.imag],
                                        "Delta_over_u": [((Vl - Vn) / upp).real, ((Vl - Vn) / upp).imag],
                                        "fd_agreement_law": float(abs(d5[0] - d6[0]) / max(abs(d5[0]), 1e-300)),
                                        "fd_agreement_nat": float(abs(d5[1] - d6[1]) / max(abs(d5[1]), 1e-300)),
                                        "redteam_expr": [rt.real, rt.imag], "redteam_minus_direct": float(abs(rt - Vl)),
                                        "dB_minus_dE_minus_sqrtEE_u": float(abs(dB - dE - np.sqrt(Ea * Eb) * upp)),
                                        "maxabs_p_over_m": float(max(abs(p), abs(pp)) / m)})
                print("E L %d m %g j %d: %d elements (%.0f s)" % (L, m, j, len(pairs), time.time() - T0), flush=True)
            for valley in ("0", "pi"):
                for wfrac in (0.125, 0.25, 0.5):
                    for jset in ("all", "1,2"):
                        sel = [r for r in out["rows"] if r["L"] == L and r["m"] == m and r["valley"] == valley and r["maxabs_p_over_m"] <= wfrac + 1e-12
                               and (jset == "all" or r["j"] in (1, 2))]
                        if len(sel) < 3:
                            continue
                        X = np.array([r["pp"] - r["p"] for r in sel])
                        Y = np.array([complex(*r["Delta_over_u"]) for r in sel])
                        Am = np.column_stack([X, X ** 2]).astype(complex)
                        coef = np.linalg.lstsq(Am, Y, rcond=None)[0]
                        nu = +1 if valley == "0" else -1
                        out["fits"].append({"L": L, "m": m, "valley": valley, "window_over_m": wfrac, "j_set": jset, "n_points": len(sel),
                                            "n_transfers": len(set(np.round(X, 12))), "b": [coef[0].real, coef[0].imag], "c": [coef[1].real, coef[1].imag],
                                            "prediction_b": [0.0, -2.0 * nu], "prediction_c": [1.0 / m, 0.0],
                                            "rel_residual": float(np.linalg.norm(Am @ coef - Y) / np.linalg.norm(Y))})
    return out


T0 = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", required=True, choices=["C", "D", "E"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"meta": {"host": socket.gethostname(), "part": a.part}}
    res[a.part] = {"C": part_C, "D": part_D, "E": part_E}[a.part]()
    res["meta"]["seconds"] = time.time() - T0
    with open(os.path.join(a.out, "rtchecks_%s.json" % a.part), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("part %s done (%.0f s)" % (a.part, time.time() - T0), flush=True)


if __name__ == "__main__":
    main()
