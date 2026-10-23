#!/usr/bin/env python
"""Strand 6 (j) on ring4 (SPEC_strand6_validation.md, item (j), with the lead's amendment of JUDGMENT_strand6_validation.md sec. 11; builder
analyst).  Per 2b case record (solve/results/strand2b/pass2v13/case_R4_*.json: the block, m, the couplings, k, the solved psi and N, d0), on
the full c-basis Fock sector N_c of ring4 (8 c-modes) with strand2b_lib_v6's sources e_x = rho^P_x[psi] (the law's clock derivative, the
positive form, the seed at the case's couplings) and the isolated-exterior Schur stiffness K of native_core (G = K^-1 its exterior Green
function on the sites):
  J1 (1): ||[e_z, e_w]|| / (||e_z|| ||e_w||) (operator 2-norms) for the six pairs, at psi = 1 and at the case's psi, on the full sector and
      compressed to the case's own (n_p, n_h) block (the eigenspace of n_exc with eigenvalue n_p + n_h: the block holding the lowest
      multiplet), the compressed commutator taken between the compressed operators;
  J2 (2): s1_x = (t/k) sum_z G_xz e_z at t = 1 and [s_x, s_y]^(2) = (t^2/k^2) sum_{z,w} G_xz G_yw [e_z, e_w] (= [s1_x, s1_y]), at the case's psi:
      ||[s_x, s_y]^(2)|| / (||s1_x|| ||s1_y||) for the pairs (0, 1), (0, 2) and the largest over the six; in the stellar-type state rho (the
      projector on the case's lowest d0 levels of H = sum_y N_y e_y on the block, over d0, as strand2b_lib_v6.System.state): |<[s_x, s_y]^(2)>|
      and its standard deviation sqrt(<C^dag C> - |<C>|^2), each over |<s1_x>| |<s1_y>|;
  J3 (3): Var(e_x) = <e_x^2> - <e_x>^2 in rho (equal to Var(C_x): the constraint's second term is a c-number at the scalar solution), sigma/<e_x>
      per site; for the windows f = all sites and the four adjacent two-site windows {x, x+1}: sigma(e[f])/<e[f]>, N_f = <sum_x f_x n_x^A2> with
      n_x^A2 = d_x^dag d_x + f_x^dag f_x the particle and hole site densities in the A2 frame (the polar-transported fields (c_u, U^dag c_d):
      d_x = sum_r R_xr d_r and f_x = sum_r conj(R_xr) f_r with R the eigenvectors of B_0 at N = 1, so that the A2 particle vectors are
      P [1; U]/sqrt 2 site by site), eps_f = <e[f]>/N_f, F_E,f = Var(e[f])/(eps_f^2 N_f) and sqrt(F_E,f/N_f); beside N_f the bare
      normal-ordered charge <sum_x f_x :c_x^dag c_x:> in the site frame (c-mode occupations minus the light vacuum's at psi); the same in the
      free state (g_M = g_B = 0) of the same block at the case's own psi and N (the builder's choice of the free control's fields; d0 at
      tolerance 1e-9).
Controls: C-j1 the state's <e_x> against the record's 'source'; C-j2 sum_x n_x^A2 = n_exc and sum_x :n_x: = (N_c - n_W) 1 on the sector; C-j3
the Hermiticity of e_x.  Writes --out-json and --md (tables J1 to J3; the cube's J4 is a separate script).  Single-threaded, seconds per case."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "strand2"))
sys.path.insert(0, os.path.join(HERE, "..", "strand2", "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_lib_v6 as B  # noqa: E402
import native_core as nc  # noqa: E402

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def herm(A):
    return 0.5 * (A + A.conj().T)


def nrm(M):
    return float(np.linalg.norm(M, 2))


def comm(X, Y):
    return X @ Y - Y @ X


def ev(rho, O):
    return complex(np.trace(rho @ O))


def f(x, d=4):
    if x is None:
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return ("%." + str(d) + "g") % x
    return "%.3e" % x


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def block_basis(nexc, kexc):
    w, V = np.linalg.eigh(nexc)
    return V[:, np.abs(w - kexc) < 1e-6]


def site_densities(rp, b, fops):
    """n_x^A2 (particle plus hole site densities in the A2 frame) and :n_x: (the site frame's normal-ordered c-charge) on sector N_c."""
    light = b["light"]
    Nc, nW, ncp = rp.Nc, rp.nW, rp.lat["ncomp"]
    P, Hh = rp.qp_ops(light, Nc)
    R = light.R
    D = fops.dim(Nc)
    Ps = np.tensordot(R, P, axes=([1], [0])) if P is not None else None
    Hs = np.tensordot(R.conj(), Hh, axes=([1], [0])) if Hh is not None else None
    gam = np.real(np.diag(light.vm @ light.vm.conj().T))
    a = fops.ann[Nc] if Nc >= 1 else None
    nA2, nc_ = [], []
    for x in range(rp.n):
        M = np.zeros((D, D), dtype=complex)
        Q = np.zeros((D, D), dtype=complex)
        for c in range(ncp):
            w = x * ncp + c
            if Ps is not None:
                M += Ps[w].conj().T @ Ps[w]
            if Hs is not None:
                M += Hs[w].conj().T @ Hs[w]
            for s in range(2):
                j = 2 * w + s
                if a is not None:
                    Q += a[j].conj().T @ a[j]
                Q -= gam[j] * np.eye(D)
        nA2.append(herm(M))
        nc_.append(herm(Q))
    return np.array(nA2), np.array(nc_)


def variances(rho, e, nA2, nch, windows):
    n = len(e)
    mean = np.array([ev(rho, e[x]).real for x in range(n)])
    var = np.array([ev(rho, e[x] @ e[x]).real for x in range(n)]) - mean ** 2
    out = {"site_mean": mean.tolist(), "site_var": var.tolist(), "site_sigma_over_mean": (np.sqrt(np.maximum(var, 0)) / mean).tolist(), "windows": {}}
    for lab, fw in windows.items():
        fw = np.asarray(fw, float)
        ef = np.tensordot(fw, e, axes=(0, 0))
        m_ = ev(rho, ef).real
        v_ = ev(rho, ef @ ef).real - m_ ** 2
        Nf = ev(rho, np.tensordot(fw, nA2, axes=(0, 0))).real
        Qf = ev(rho, np.tensordot(fw, nch, axes=(0, 0))).real
        eps = m_ / Nf if Nf != 0 else None
        F = v_ / (eps * eps * Nf) if (eps and Nf) else None
        out["windows"][lab] = {"mean": m_, "var": v_, "sigma_over_mean": float(np.sqrt(max(v_, 0)) / m_) if m_ else None, "N_f": Nf,
                               "bare_charge": Qf, "eps_f": eps, "F_E": F, "sqrt_F_over_N": float(np.sqrt(F / Nf)) if (F is not None and F >= 0 and Nf > 0) else None}
    return out


def do_case(p, lat, K, G, fops):
    rec = json.load(open(p))
    npp, nh = (int(x) for x in rec["block"].split(","))
    Lam = np.inf if rec["Lambda"] == "inf" else float(rec["Lambda"])
    m, gM, gB, k = float(rec["m"]), float(rec["gM"]), float(rec["gB"]), float(rec["k"])
    nW = lat["nsite"] * lat["ncomp"]
    Nc = nW + npp - nh
    kexc = npp + nh
    psi, N = np.array(rec["psi"]), np.array(rec["N"])
    rp = B.RhoP(lat, m, gM, gB, Lam, Nc, fops)
    sy = B.System(rp, K, k, kexc)
    sy.d0_fixed = int(rec["d0"])
    out = {"case": os.path.basename(p), "block": rec["block"], "m": m, "gM": gM, "gB": gB, "Lambda": rec["Lambda"], "k": k, "Nc": Nc, "d0": rec["d0"]}
    J1 = {}
    for lab, ps in (("psi1", np.ones(lat["nsite"])), ("case", psi)):
        b = rp.build(ps)
        e = b["rho"]
        W = block_basis(b["nexc"], kexc)
        ec = np.array([herm(W.conj().T @ e[x] @ W) for x in range(len(e))])
        row = {}
        for (z, w) in PAIRS:
            row["%d%d" % (z, w)] = {"full": nrm(comm(e[z], e[w])) / (nrm(e[z]) * nrm(e[w])),
                                    "compressed": nrm(comm(ec[z], ec[w])) / (nrm(ec[z]) * nrm(ec[w]))}
        J1[lab] = {"pairs": row, "block_dim": int(W.shape[1]), "sector_dim": int(e.shape[1]),
                   "hermiticity": max(float(np.max(np.abs(e[x] - e[x].conj().T))) for x in range(len(e)))}
    out["J1"] = J1
    st = sy.state(psi, N)
    rho = st["rho"]
    b = st["b"]
    e = b["rho"]
    src = np.array([ev(rho, e[x]).real for x in range(len(e))])
    out["C_j1_source_max_rel"] = float(np.max(np.abs(src / np.array(rec["source"]) - 1)))
    s1 = np.array([(1.0 / k) * np.tensordot(G[x], e, axes=(0, 0)) for x in range(len(e))])
    J2 = {}
    for (x, y) in PAIRS:
        C2 = (1.0 / k ** 2) * sum(G[x, z] * G[y, w] * comm(e[z], e[w]) for z in range(len(e)) for w in range(len(e)))
        mC = ev(rho, C2)
        sd = np.sqrt(max(ev(rho, C2.conj().T @ C2).real - abs(mC) ** 2, 0.0))
        sx, sy_ = ev(rho, s1[x]).real, ev(rho, s1[y]).real
        J2["%d%d" % (x, y)] = {"norm_ratio": nrm(C2) / (nrm(s1[x]) * nrm(s1[y])), "abs_mean_over_s1s1": abs(mC) / abs(sx * sy_),
                               "sd_over_s1s1": sd / abs(sx * sy_), "check_equals_s1_commutator": nrm(C2 - comm(s1[x], s1[y])) / max(nrm(C2), 1e-300)}
    out["J2"] = J2
    nA2, nch = site_densities(rp, b, fops)
    out["C_j2_nA2_sum_minus_nexc"] = float(np.max(np.abs(nA2.sum(0) - b["nexc"])))
    out["C_j2_charge_sum_minus_NcmnW"] = float(np.max(np.abs(nch.sum(0) - (Nc - nW) * np.eye(nch.shape[1]))))
    n = lat["nsite"]
    windows = {"all": np.ones(n)}
    for x in range(n):
        fw = np.zeros(n)
        fw[x] = fw[(x + 1) % n] = 1
        windows["{%d,%d}" % (x, (x + 1) % n)] = fw
    out["J3"] = variances(rho, e, nA2, nch, windows)
    out["J3"]["d0"] = st["d0"]
    rp0 = B.RhoP(lat, m, 0.0, 0.0, Lam, Nc, fops)
    sy0 = B.System(rp0, K, k, kexc)
    st0 = sy0.state(psi, N)
    nA20, nch0 = site_densities(rp0, st0["b"], fops)
    out["J3_free"] = variances(st0["rho"], st0["b"]["rho"], nA20, nch0, windows)
    out["J3_free"]["d0"] = st0["d0"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="a glob of the 2b case records")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    lat = S2.lattice("R4")
    K, G = nc.K_matrix(nc.lattice("ring4")["positions"])
    fops = B.FockOps(8)
    t0 = time.time()
    res = []
    for p in sorted(glob.glob(a.cases)):
        r = do_case(p, lat, K, G, fops)
        res.append(r)
        print("%s: J2 (0,1) %.3e (0,2) %.3e; F_E(all) %s; C-j1 %.1e (%.0f s)" % (
            r["case"], r["J2"]["01"]["norm_ratio"], r["J2"]["02"]["norm_ratio"], f(r["J3"]["windows"]["all"]["F_E"]), r["C_j1_source_max_rel"],
            time.time() - t0), flush=True)
    json.dump({"meta": {"script": "strand6j_constraint_var_v1.py", "builder": "analyst", "cases": a.cases, "seconds": time.time() - t0},
               "cases": res}, open(a.out_json, "w"), indent=1, default=float)
    L = ["# Strand 6 (j) on ring4: the source commutators, the second-order ruler commutator and the constraint's variance (strand6j_constraint_var_v1.py)", ""]
    r1 = []
    for r in res:
        for lab in ("psi1", "case"):
            j = r["J1"][lab]
            r1.append([r["case"][8:-5], lab, "%d/%d" % (j["block_dim"], j["sector_dim"])] +
                      ["%s / %s" % (f(j["pairs"][q]["full"]), f(j["pairs"][q]["compressed"])) for q in ("01", "02")] +
                      ["%s / %s" % (f(max(j["pairs"][q]["full"] for q in j["pairs"])), f(max(j["pairs"][q]["compressed"] for q in j["pairs"])))])
    L += ["## J1 ||[e_z, e_w]|| / (||e_z|| ||e_w||): full sector / compressed to the case's own block", "",
          table(["case", "psi", "block/sector dim", "(0,1)", "(0,2)", "largest of six"], r1), ""]
    r2 = []
    for r in res:
        j = r["J2"]
        mx = max(j, key=lambda q: j[q]["norm_ratio"])
        r2.append([r["case"][8:-5]] + ["%s; %s; %s" % (f(j[q]["norm_ratio"]), f(j[q]["abs_mean_over_s1s1"]), f(j[q]["sd_over_s1s1"])) for q in ("01", "02")] +
                  ["%s (%s)" % (f(j[mx]["norm_ratio"]), mx), f(max(j[q]["check_equals_s1_commutator"] for q in j), 2)])
    L += ["## J2 [s_x, s_y]^(2) at t = 1: norm ratio; |<.>|/(|<s1_x>| |<s1_y>|); sd/(|<s1_x>| |<s1_y>|)", "",
          table(["case", "(0,1)", "(0,2)", "largest norm ratio (pair)", "identity check"], r2), ""]
    r3 = []
    for r in res:
        for lab, J in (("state", r["J3"]), ("free", r["J3_free"])):
            w = J["windows"]
            r3.append([r["case"][8:-5], lab, str(J["d0"]), ", ".join(f(x) for x in J["site_sigma_over_mean"])] +
                      ["%s; N_f %s; Q %s; F_E %s; sqrt(F/N) %s" % (f(w[q]["sigma_over_mean"]), f(w[q]["N_f"]), f(w[q]["bare_charge"]), f(w[q]["F_E"]),
                                                                   f(w[q]["sqrt_F_over_N"])) for q in ("all", "{0,1}")])
    L += ["## J3 The constraint's variance: sigma/<e_x> per site, and per window sigma(e[f])/<e[f]>, N_f (A2), the bare charge Q, F_E, sqrt(F_E/N_f)", "",
          table(["case", "state", "d0", "sigma/<e_x> per site", "f = all", "f = {0,1}"], r3), ""]
    L += ["## Controls", "", table(["case", "C-j1 max |<e_x>/source - 1|", "C-j2 sum n^A2 - n_exc", "C-j2 sum :n: - (N_c - n_W)", "C-j3 hermiticity"],
                                   [[r["case"][8:-5], f(r["C_j1_source_max_rel"], 2), f(r["C_j2_nA2_sum_minus_nexc"], 2), f(r["C_j2_charge_sum_minus_NcmnW"], 2),
                                     f(max(r["J1"][q]["hermiticity"] for q in r["J1"]), 2)] for r in res]), ""]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("done %d cases %.0f s" % (len(res), time.time() - t0))


if __name__ == "__main__":
    main()
