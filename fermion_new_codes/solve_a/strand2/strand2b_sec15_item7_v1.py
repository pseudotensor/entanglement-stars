#!/usr/bin/env python
"""Strand 2b item 7 (spec solve/specs/SPEC_strand2b_row.md sec. 15, after the second reviewer B red team; builder analyst): the cube's two
non-uniform classes between k = 5 and 10, with strand2b_c8_lib_v6 (Newton never executes a rejected step).  The enclosure case: (2,0),
m 0.1, (g_M, g_B, Lambda) = (0.2, 1, 1), d_0 fixed at the record's value.  One part per call (--part):
class1, class2: branch-preserving natural-parameter continuation in k from item 5's k = 5 representative (re-converged at k = 5), steps
  dk at most 0.25 (halved on rejection down to 1/512), a secant predictor, the corrector the FD Newton (40 iterations; acceptance at the
  noise floor: best residual at most max(1e-9, 10 times the largest change of |F| under ten random 1e-12 displacements)); a step is
  accepted only if the corrected solution is positive, within half the current amplitude of the predictor and at a nonzero amplitude
  (the amplitude a = the distance to the uniform solution at the same k minimized over the cube's 48 symmetries); per accepted step: k,
  a, E, E_sel, the block and cluster gaps, the three smallest singular values of the FD Jacobian, the eigenvalue of its symmetric part of
  smallest modulus with its sign, the eigenvalue of J of smallest modulus, the residual and the Newton count; the outcome: the amplitude
  reaching zero (the steps stall as a -> 0), a fold (the steps stall at a finite amplitude with sigma_min -> 0), or persistence to k = 10.
uniform: the uniform branch from k = 2 to 10 in steps of 0.25 (secant predictor, the same corrector), per k the three smallest singular
  values of the FD Jacobian, the symmetric part's eigenvalue of smallest modulus with its sign, the Rayleigh quotient v^T J v along the
  right singular vector of sigma_min, the eigenvalue of J of smallest modulus (real and imaginary parts); then 8 further k in the step
  around the smallest sigma_min (spacing 1/32), and the two-step Jacobian check there (h = 1e-6 against 1e-7, the largest relative entry
  difference and the two sigma_min).  The exact k-derivative of F (library v6, F affine in k) is checked against a central difference
  in k at k = 5.
doublet: (2,0) at (g_M, g_B) = (0.05, 0), Lambda = 1, k = 1, d_0 fixed at the flat start's value (2): the uniform continuation solution,
  the doublet's two levels, their splitting, the gap above them, Tr rho^2 (rho = P_low/d_0), the 2 x 2 clock-source matrices
  S_x = C^dag M_x C on the doublet (C its two columns) and the 2 x 2 adjoint-source matrices T_yx = d/dpsi_y [(O C)^dag M_x(psi') (O C)]
  at fixed Fock vector (central differences, h = 1e-4, the overlap projection of sec. 10), each with its non-scalar part
  ||A - (tr A/2) 1||/||A|| and its two eigenvalues.
Writes <out>/item7_<part>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib_v6 as C  # noqa: E402
import native_core as nc  # noqa: E402
from strand2b_c8_unique_post_v1 import symmetries, act  # noqa: E402

LAT = S2.lattice("C8")
K, _ = nc.K_matrix(LAT["positions"].tolist())
K = np.asarray(K, float)
PERMS = symmetries(LAT["positions"])


def make(m, gM, gB, Lam, k, d0=None):
    bm = C.BlockModel(LAT, m, gM, gB, Lam, (2, 0))
    s = C.SystemC8v6(bm, K, k)
    s.d0_fixed = d0
    return s


def noise(sys_, z):
    F0 = sys_.F(z)[0]
    rng = np.random.default_rng(5)
    return float(max(np.linalg.norm(sys_.F(z + 1e-12 * rng.normal(size=z.size))[0] - F0) for _ in range(10)))


def solve(sys_, z0):
    zz, inf = C.newton_fd(sys_, np.array(z0), maxit=40)
    nf = noise(sys_, zz) if np.all(zz > 0) else None
    best = inf.get("best")
    ok = bool(nf is not None and best is not None and best <= max(1e-9, 10 * nf) and np.all(zz > 0))
    return zz, ok, best, nf, inf


def amplitude(z, zu, n):
    return float(min(np.linalg.norm(act(zu, p, n) - z) for p in PERMS))


def spectral(sys_, z, h=1e-6):
    J = C.jacobian_fd(sys_, z, h=h)
    U, sv, Vt = np.linalg.svd(J)
    v = Vt[-1]
    Ssym = 0.5 * (J + J.T)
    es = np.linalg.eigvalsh(Ssym)
    i = int(np.argmin(np.abs(es)))
    ej = np.linalg.eigvals(J)
    jj = int(np.argmin(np.abs(ej)))
    return J, {"sigma_smallest3": [float(x) for x in sv[-3:][::-1]], "sym_eig_smallest_modulus": float(es[i]),
               "sym_eig_lowest": float(es[0]), "rayleigh_vJv": float(v @ J @ v), "J_eig_smallest_modulus": [float(ej[jj].real), float(ej[jj].imag)]}


def describe(sys_, z, zu=None, h=1e-6):
    n, k = sys_.n, sys_.k
    Fz, st, src, t = sys_.F(z)
    psi, N = z[:n], z[n:]
    _, sp = spectral(sys_, z, h)
    out = {"residual": float(np.linalg.norm(Fz)), "E": st["E"], "gap_block": st["gap_block"], "gap_cluster": st["gap_cluster"],
           "E_sel": float(-k * (psi - 1) @ K @ (N * psi - 1) + np.dot(N, src)), "min_component": float(z.min()), "psi": psi.tolist(), "N": N.tolist()}
    out.update(sp)
    if zu is not None:
        out["amplitude"] = amplitude(z, zu, n)
    return out


def part_class(rec, cls, zstart, out, t0, kmax=10.0, dk_max=0.25, dk_min=1.0 / 512):
    m, gM, gB, Lam, d0 = rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), rec["d0"]
    k = 5.0
    s = make(m, gM, gB, Lam, k, d0)
    zu0 = np.concatenate([rec["uniform_k5"]["psi"], rec["uniform_k5"]["N"]])
    zu, oku, _, _, _ = solve(s, zu0)
    z, ok, best, nf, inf = solve(s, zstart)
    res = {"class": cls, "start_k5": {"converged": ok, "best": best, "noise": nf, "stop": inf.get("why")}, "steps": [], "rejections": []}
    if not ok:
        res["outcome"] = "the k = 5 representative did not re-converge"
        return res
    d = describe(s, z, zu)
    d.update({"k": k, "dk": 0.0, "best": best, "noise_floor": nf, "newton": len(inf["residuals"])})
    res["steps"].append(d)
    hist = [(k, z, zu)]
    dk = dk_max
    print("%s k %.5f a %.6g sigma %s (%.0f s)" % (cls, k, d["amplitude"], d["sigma_smallest3"][0], time.time() - t0), flush=True)
    while k < kmax - 1e-12:
        kn = min(kmax, k + dk)
        s.k = kn
        if len(hist) >= 2:
            (k1, z1, u1), (k2, z2, u2) = hist[-2], hist[-1]
            zp = z2 + (z2 - z1) * (kn - k2) / (k2 - k1)
            up = u2 + (u2 - u1) * (kn - k2) / (k2 - k1)
        else:
            zp, up = hist[-1][1], hist[-1][2]
        a_last = res["steps"][-1]["amplitude"]
        zun, oku, _, _, _ = solve(s, up)
        zn, ok, best, nf, inf = solve(s, zp)
        reason = None
        if not oku:
            reason = "uniform branch not converged"
        elif not ok:
            reason = "class corrector not converged (%s)" % inf.get("why")
        else:
            an = amplitude(zn, zun, s.n)
            if an < 1e-6:
                reason = "corrector reached the uniform orbit (a = %.2e)" % an
            elif np.linalg.norm(zn - zp) > 0.5 * a_last:
                reason = "corrector moved %.3e from the predictor (half the amplitude %.3e)" % (np.linalg.norm(zn - zp), 0.5 * a_last)
        if reason is not None:
            res["rejections"].append({"k_from": k, "k_to": kn, "dk": dk, "reason": reason})
            dk /= 2
            if dk < dk_min:
                res["outcome"] = "stalled at k = %.6f with amplitude %.6g (last dk %.2e): %s" % (k, a_last, dk * 2, reason)
                break
            continue
        k = kn
        d = describe(s, zn, zun)
        d.update({"k": k, "dk": dk, "best": best, "noise_floor": nf, "newton": len(inf["residuals"])})
        res["steps"].append(d)
        hist.append((k, zn, zun))
        print("%s k %.5f dk %.4g a %.6g E %.9f sigma %.4g sym %.4g (%.0f s)" % (cls, k, dk, d["amplitude"], d["E"], d["sigma_smallest3"][0],
                                                                            d["sym_eig_smallest_modulus"], time.time() - t0), flush=True)
        dk = min(dk_max, 2 * dk)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    else:
        res["outcome"] = "persists to k = %g (amplitude %.6g)" % (kmax, res["steps"][-1]["amplitude"])
    if "outcome" not in res:
        res["outcome"] = "persists to k = %g (amplitude %.6g)" % (kmax, res["steps"][-1]["amplitude"])
    return res


def part_uniform(rec, out, t0):
    m, gM, gB, Lam, d0 = rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), rec["d0"]
    s = make(m, gM, gB, Lam, 2.0, d0)
    z = np.concatenate([rec["uniform_k2"]["psi"], rec["uniform_k2"]["N"]])
    res = {"scan": [], "refine": []}
    # the exact k-derivative against a central difference in k at k = 5 (at the k = 5 uniform record)
    s5 = make(m, gM, gB, Lam, 5.0, d0)
    z5 = np.concatenate([rec["uniform_k5"]["psi"], rec["uniform_k5"]["N"]])
    dfa = s5.dF_dk(z5)
    s5.k = 5.0 + 1e-3
    fp = s5.F(z5)[0]
    s5.k = 5.0 - 1e-3
    fm = s5.F(z5)[0]
    res["dF_dk_check"] = float(np.max(np.abs(dfa - (fp - fm) / 2e-3)) / np.max(np.abs(dfa)))
    hist = []

    def at(k, zg):
        s.k = k
        zz, ok, best, nf, inf = solve(s, zg)
        d = describe(s, zz)
        d.update({"k": k, "converged": ok, "best": best, "noise_floor": nf})
        return zz, d
    for k in np.arange(2.0, 10.0 + 1e-9, 0.25):
        if len(hist) >= 2:
            (k1, z1), (k2, z2) = hist[-2], hist[-1]
            zg = z2 + (z2 - z1) * (k - k2) / (k2 - k1)
        else:
            zg = z if not hist else hist[-1][1]
        zz, d = at(float(k), zg)
        res["scan"].append(d)
        if d["converged"]:
            hist.append((float(k), zz))
        print("uniform k %.2f: conv %s sigma %s sym %.4g rq %.4g eigJ %s (%.0f s)" % (k, d["converged"], np.round(d["sigma_smallest3"], 5),
                                                                                   d["sym_eig_smallest_modulus"], d["rayleigh_vJv"],
                                                                                   np.round(d["J_eig_smallest_modulus"], 5), time.time() - t0), flush=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    sig = [d["sigma_smallest3"][0] if d["converged"] else np.inf for d in res["scan"]]
    i = int(np.argmin(sig))
    k0 = res["scan"][i]["k"]
    zc = dict(hist)[k0] if k0 in dict(hist) else hist[-1][1]
    for k in sorted(set(np.round(np.arange(k0 - 0.125, k0 + 0.125 + 1e-9, 1.0 / 32), 10)) - {k0}):
        if k < 2.0 or k > 10.0:
            continue
        zz, d = at(float(k), zc)
        res["refine"].append(d)
        print("refine k %.5f: sigma %s sym %.4g eigJ %s (%.0f s)" % (k, np.round(d["sigma_smallest3"], 6), d["sym_eig_smallest_modulus"],
                                                                     np.round(d["J_eig_smallest_modulus"], 6), time.time() - t0), flush=True)
    allp = [d for d in res["scan"] + res["refine"] if d["converged"]]
    jbest = min(allp, key=lambda d: d["sigma_smallest3"][0])
    s.k = jbest["k"]
    zb = np.concatenate([jbest["psi"], jbest["N"]])
    J6 = C.jacobian_fd(s, zb, h=1e-6)
    J7 = C.jacobian_fd(s, zb, h=1e-7)
    res["two_step_check"] = {"k": jbest["k"], "max_rel_entry_diff": float(np.max(np.abs(J6 - J7)) / np.max(np.abs(J6))),
                             "sigma_min_h1e-6": float(np.linalg.svd(J6, compute_uv=False)[-1]),
                             "sigma_min_h1e-7": float(np.linalg.svd(J7, compute_uv=False)[-1])}
    print("two-step check at k %.5f: %s" % (jbest["k"], res["two_step_check"]), flush=True)
    return res


def part_doublet(rec, t0):
    m, Lam = rec["m"], float(rec["Lambda"])
    s = make(m, 0.05, 0.0, Lam, 1.0)
    z, inf = C.continuation_fd(s)
    n = s.n
    psi, N = z[:n], z[n:]
    Fz, st, src, t = s.F(z)
    e = st["e"]
    Cd = st["C"]
    d0 = st["d0"]
    rho = Cd @ Cd.conj().T / d0
    out = {"d0": d0, "continuation_ok": bool(inf.get("ok")), "residual": float(np.linalg.norm(Fz)), "levels": [float(x) for x in e[:4]],
           "splitting": float(e[d0 - 1] - e[0]), "gap_above": float(e[d0] - e[d0 - 1]), "Tr_rho2": float(np.real(np.trace(rho @ rho))),
           "psi": psi.tolist(), "N": N.tolist(), "source": src.tolist()}

    def ns(A):
        A = 0.5 * (A + A.conj().T)
        return float(np.linalg.norm(A - np.trace(A) / 2 * np.eye(2)) / max(np.linalg.norm(A), 1e-300)), [float(x) for x in np.linalg.eigvalsh(A)]
    clock = []
    for x in range(n):
        A = Cd.conj().T @ st["b"]["M"][x] @ Cd
        f, ev = ns(A)
        clock.append({"x": x, "nonscalar": f, "eigenvalues": ev, "trace_over_2": float(np.real(np.trace(A)) / 2)})
    out["clock_sources"] = clock
    h = 1e-4
    adj = []
    for y in range(n):
        mats = []
        for sgn in (+1, -1):
            ps = np.array(psi, float)
            ps[y] += sgn * h
            bn = s.bm.build(ps)
            O = s.bm.overlap(bn, st["b"])
            Cp = O @ Cd
            mats.append(np.array([Cp.conj().T @ bn["M"][x] @ Cp for x in range(n)]))
        T = (mats[0] - mats[1]) / (2 * h)
        for x in range(n):
            f, ev = ns(T[x])
            adj.append({"y": y, "x": x, "nonscalar": f, "eigenvalues": ev, "trace_over_2": float(np.real(np.trace(T[x])) / 2)})
    out["adjoint_sources"] = adj
    out["clock_nonscalar_max"] = max(c["nonscalar"] for c in clock)
    out["adjoint_nonscalar_max"] = max(c["nonscalar"] for c in adj)
    big = max(abs(c["trace_over_2"]) for c in adj)
    out["adjoint_nonscalar_max_significant"] = max((c["nonscalar"] for c in adj if abs(c["trace_over_2"]) > 1e-3 * big), default=None)
    print("doublet: d0 %d levels %s splitting %.3e gap above %.3e Tr rho^2 %.6f; clock non-scalar max %.3e; adjoint %.3e (%.0f s)" % (
        d0, np.round(out["levels"], 8), out["splitting"], out["gap_above"], out["Tr_rho2"], out["clock_nonscalar_max"], out["adjoint_nonscalar_max"],
        time.time() - t0), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True, help="the enclosure case record (m, couplings, Lambda, d0)")
    ap.add_argument("--item5", required=True, help="item 5's item5.json (the k = 2 and k = 5 states)")
    ap.add_argument("--part", required=True, choices=["class1", "class2", "uniform", "doublet"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = json.load(open(a.record))
    I5 = json.load(open(a.item5))
    rec = {"m": R["m"], "gM": R["gM"], "gB": R["gB"], "Lambda": R["Lambda"], "d0": R["d0"]}
    uni = {r["k"]: r for r in I5["part_a"]["uniform"]}
    rec["uniform_k2"], rec["uniform_k5"] = uni[2.0], uni[5.0]
    out = os.path.join(a.out, "item7_%s.json" % a.part)
    if a.part in ("class1", "class2"):
        name = "class %s" % a.part[-1]
        r5 = [r for r in I5["part_a"][name] if r["k"] == 5.0][0]
        res = part_class(rec, name, np.concatenate([r5["psi"], r5["N"]]), out, t0)
    elif a.part == "uniform":
        res = part_uniform(rec, out, t0)
    else:
        res = part_doublet(rec, t0)
    res["meta"] = {"record": a.record, "item5": a.item5, "seconds": time.time() - t0}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("done", a.part, res.get("outcome", ""), "%.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
