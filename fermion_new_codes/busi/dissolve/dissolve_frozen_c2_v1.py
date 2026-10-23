#!/usr/bin/env python
"""dissolve_frozen_c2_v1.py (busi f52b1152, 2026-10-20, inbox #323): C6-2 of REVIEWA_write_DISSOLVE_v1 (MERGED SPECIFICATION), the frozen-field
control. The ball's columns of channel kappa = 1 at a stored complex128 slice of the (32, 4) massless history are propagated exactly (eigenbasis)
on that record's fields held fixed, by the chain operator of Sec. IV A at m = 0 (the engine's assembly 0.5{N, h0} + 0.5{v, Q}; h0 = sigma_1 p_r +
sigma_2 kappa/r_n, p_r = (T - T^dag)/2i, Q = (T^2 - T^dag2)/4i, site-major interleaved spinor, no hop across the origin, the last site open),
read every 2a to --span r_s. The operator is checked against the slice's stored eigenvalues e_1 before use (applied fields only).
Busi's parameterized version of the write desk's dissolve_frozen_c1_v1.py (ea83ffae; T0 = 2.0, R_t = 25.725, R* = 27.235 hard-coded there):
here T0 is the slice's clock, R_t is the dense outer root of the run json's record at T0, R* the dense outer root of the record at --rstar-T
(default 1.8125, the deepest record), and --fieldset applied|dense picks N_applied, v_applied (the matter's frozen control) or the run's
kernel-averaged N, v at T0 from the fields npz (the dense comparator).
Readouts per 2a: the ball's number and bare (h0) energy inside R_t and R*, the Killing (h) energy inside R*, the momentum content inside R_t
(the restricted columns' plane-wave weight: the share at |ak| >= pi/4, at |ak| >= z_c = arccos(N/|v|) at R_t - 4a, and at k > 0). Against the
records: f_C over 0.25, 0.5, 0.75 and 1.0 r_s = (frozen fall of the weighted bare energy inside R_t) / (the records' fall of the ball's bare
shell energy eps_tot - eps_pairs inside R_t), from --records-fields (default: the stored run's fields npz in the kit, which the regenerated
history reproduces within the floor and which runs to T/r_s = 40).
Usage: dissolve_frozen_c2_v1.py --slice S.npz --run-json J.json --records-fields F.npz [--fieldset applied|dense] [--fields F2.npz]
       [--rstar-T 1.8125] [--span 1.0] --out DIR. numpy only; one dense 4096 x 4096 eigendecomposition (about 2 min at 3 to 8 threads)."""
import argparse, json, os, time
import numpy as np


def chain_ops(R, kappa):
    n = len(R)
    s1 = np.array([[0, 1], [1, 0]], complex); s2 = np.array([[0, -1j], [1j, 0]], complex)
    S = np.eye(n, k=1)                                   # (S f)_i = f_{i+1}: no hop across the origin, the last site open
    p = (S - S.T) / 2j
    Q = (S @ S - S.T @ S.T) / 4j
    h0 = np.kron(p, s1) + np.kron(np.diag(kappa / np.asarray(R, float)), s2)
    return h0, np.kron(Q, np.eye(2))


def clocked(h0, Q2, N, v):
    Nm = np.repeat(np.asarray(N, float), 2)[:, None]; vm = np.repeat(np.asarray(v, float), 2)[:, None]
    return 0.5 * (Nm * h0 + h0 * Nm.T) + 0.5 * (vm * Q2 + Q2 * vm.T)


def site_sum(x):
    return x[0::2] + x[1::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True); ap.add_argument("--run-json", required=True); ap.add_argument("--records-fields", required=True)
    ap.add_argument("--fieldset", default="applied", choices=["applied", "dense"]); ap.add_argument("--fields", default=None, help="the run's fields npz (dense fieldset)")
    ap.add_argument("--rstar-T", type=float, default=1.8125); ap.add_argument("--span", type=float, default=1.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Z = np.load(a.slice, allow_pickle=True)
    R = np.asarray(Z["R"], float); T0 = float(Z["T"]); rs = float(Z["rs"]) if "rs" in Z.files else 32.0
    Phi0 = np.asarray(Z["dust_1"], complex); w = float(np.asarray(Z["weights"]).ravel()[0]) if "weights" in Z.files else 16.0
    J = json.load(open(a.run_json))
    recs = {round(r["T_over_rs"], 6): r for r in J["records"] if r.get("E") is not None}
    r0 = recs[round(T0 / rs, 6)]; rD = recs[round(a.rstar_T, 6)]
    Rt = float(r0["trapping_zeros"][-1]); Rin = float(r0["trapping_zeros"][0]); Rstar = float(rD["trapping_zeros"][-1])
    if a.fieldset == "applied":
        N, v = np.asarray(Z["N_applied"], float), np.asarray(Z["v_applied"], float)
    else:
        F = np.load(a.fields, allow_pickle=True); i = int(np.argmin(np.abs(F["T"] - T0))); assert abs(F["T"][i] - T0) < 1e-9
        N, v = np.asarray(F["N"][i], float), np.asarray(F["v"][i], float)
    tag = os.path.basename(a.slice).replace(".npz", "") + "_" + a.fieldset
    L = open(os.path.join(a.out, "frozen_c2_%s.log" % tag), "w")
    pr = lambda s: (L.write(s + "\n"), print(s, flush=True))  # noqa: E731
    pr("dissolve_frozen_c2_v1: slice %s (T = %.4f a, T/r_s = %.4f), fieldset %s, sites %d, ball columns %s, weight %.1f; R_in %.4f R_t %.4f (record at T0), R* %.4f (record at %.4f)" % (
        a.slice, T0, T0 / rs, a.fieldset, len(R), Phi0.shape, w, Rin, Rt, Rstar, a.rstar_T))
    h0, Q2 = chain_ops(R, 1.0)
    h = clocked(h0, Q2, N, v)
    e, U = np.linalg.eigh(h)
    if a.fieldset == "applied" and "e_1" in Z.files:
        pr("operator check: max |e - e_1 (stored)| = %.3e (sorted); e range %.6f to %.6f" % (np.max(np.abs(np.sort(e) - np.sort(np.asarray(Z["e_1"], float)))), e.min(), e.max()))
    u = lambda r: float(np.interp(r, R, np.abs(v) / N))  # noqa: E731
    for r in (Rt - 8, Rt - 4, Rt - 2, Rt):
        pr("  u = |v|/N at r = %.3f: %.4f, z_c = arccos(1/u) = %.4f" % (r, u(r), np.arccos(1 / u(r)) if u(r) > 1 else 0.0))
    zc4 = np.arccos(1 / u(Rt - 4)) if u(Rt - 4) > 1 else 0.0
    c0 = U.conj().T @ Phi0
    inside = lambda Phi, Rc: float(np.sum(np.abs(Phi[np.repeat(R < Rc, 2)]) ** 2))  # noqa: E731

    def energy(Phi, Rc, op):
        d = site_sum(np.real(np.sum(np.conj(Phi) * (op @ Phi), axis=1)))
        return float(d[R < Rc].sum())

    def momentum(Phi):
        m = R < Rt; k = 2 * np.pi * np.fft.fftfreq(int(m.sum())); wk = np.zeros_like(k)
        for j in range(Phi.shape[1]):
            for c in (0, 1):
                wk += np.abs(np.fft.fft(Phi[c::2, j][m])) ** 2
        s = wk.sum()
        return float(wk[np.abs(k) >= np.pi / 4].sum() / s), float(wk[np.abs(k) >= zc4].sum() / s), float(wk[k > 0].sum() / s)
    n_tot = inside(Phi0, 1e18)
    nstep = int(round(a.span * rs / 2.0)); rows = []
    for s in range(nstep + 1):
        t = 2.0 * s
        Phi = U @ (np.exp(-1j * e * t)[:, None] * c0)
        q = momentum(Phi)
        rows.append([t / rs, inside(Phi, Rt), inside(Phi, Rstar), energy(Phi, Rt, h0), energy(Phi, Rstar, h0), energy(Phi, Rstar, h), q[0], q[1], q[2]])
        x = rows[-1]
        pr("T/r_s %.4f (+%.4f): N<R_t %.4f (%.3f) N<R* %.4f | E0<R_t %.4f E0<R* %.4f K<R* %.4f | |ak|>=pi/4 %.3f, >=z_c %.3f, k>0 %.3f" % (
            (T0 + t) / rs, t / rs, x[1], x[1] / n_tot, x[2], x[3], x[4], x[5], x[6], x[7], x[8]))
    rows = np.array(rows)
    F = np.load(a.records_fields, allow_pickle=True)
    Tr = np.asarray(F["T"], float) / rs; Rf = np.asarray(F["R"], float); eb = F["eps_tot"] - F["eps_pairs"]
    rec = lambda T, Rc: float(eb[int(np.argmin(np.abs(Tr - T)))][Rf < Rc].sum())  # noqa: E731
    pr("weight check: %.1f x frozen E0<R* at T0 = %.4f against the records' %.4f" % (w, w * rows[0, 4], rec(T0 / rs, Rstar)))
    out = {}
    for dt in (0.25, 0.5, 0.75, 1.0):
        j = int(round(dt * rs / 2.0))
        if j > nstep or (T0 / rs + dt) > Tr.max() + 1e-9:
            continue
        fr_t = w * (rows[0, 3] - rows[j, 3]); fr_s = w * (rows[0, 4] - rows[j, 4])
        rc_t = rec(T0 / rs, Rt) - rec(T0 / rs + dt, Rt); rc_s = rec(T0 / rs, Rstar) - rec(T0 / rs + dt, Rstar)
        out[dt] = (fr_t / rc_t, fr_s / rc_s)
        pr("f_C over %.2f r_s: frozen fall inside R_t %.4f against the records' %.4f: f_C %.2f; inside R* %.4f against %.4f: %.2f; number inside R_t %.3f -> %.3f of the ball" % (
            dt, fr_t, rc_t, fr_t / rc_t, fr_s, rc_s, fr_s / rc_s, rows[0, 1] / n_tot, rows[j, 1] / n_tot))
    half = next((rows[i, 0] for i in range(len(rows)) if rows[i, 1] <= 0.5 * rows[0, 1]), None)
    pr("the exit's half-clock (number inside R_t at half its T0 value): %s r_s after T0" % ("%.4f" % half if half is not None else "beyond the span"))
    pr("bands (row-18): f_C >= 0.5 the carrier's exit accounts for the dissolution; 0.25 to 0.5 shared; below 0.25 not the carrier's")
    pr("seconds %.0f" % (time.time() - t0)); L.close()
    np.savez_compressed(os.path.join(a.out, "frozen_c2_%s.npz" % tag), rows=rows, cols=np.array(["dT_over_rs", "N_in_Rt", "N_in_Rstar", "E0_in_Rt", "E0_in_Rstar", "K_in_Rstar", "q_quarter", "q_zc", "q_kpos"]),
                        T0=T0, Rt=Rt, Rin=Rin, Rstar=Rstar, fieldset=a.fieldset, n_tot=n_tot, weight=w, fC=np.array([[k, *v_] for k, v_ in out.items()]))


if __name__ == "__main__":
    main()
