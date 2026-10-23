#!/usr/bin/env python
"""engine_J0_v1.py (builder analyst; SPEC_final4b_engine.md sec. 7, job J0): the engine's checks before any run.
(1) Hermiticity of h_m on the chain and on the 3D box; (2) the chain at N = 1 against flow_chart_v4.chain_hamiltonian (v = 0 and a
nonzero v: identical to rounding); (3) the periodic 3D operator at uniform N, v against the symbol N sqrt(m^2 + sum sin^2 p) + v . sin(2p)/2
(both bands); (4) the vacuum's normal-ordered sources (form (ii) with the sea only) zero; (5) form (i) = form (ii) at a bound configuration;
(6) the constraint solver: J = 0 gives N = 1 everywhere and r v^2 = 2gE beyond the source exactly; the charge independent of the collar and
equal to the Gauss sum; (7) the sign check of spec 3.3: with the static-frame momentum J_n = |v_n| eps_n/N_n iterated to self-consistency,
N < 1 inside and N = 1 beyond; (8) the chain's two valleys: the spectrum of the chain at +kappa equals that at -kappa (sigma_z conjugation
of the pi valley, spec 2.1).  Prints one JSON.  Args: --out."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("FLOW_CODE", os.path.join(HERE, "..", "flow")))
import engine_fields_v1 as FLD  # noqa: E402
import engine_ops_v1 as OPS  # noqa: E402
import flow_chart_v4 as FC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"script": "engine_J0_v1.py"}
    rng = np.random.default_rng(5)
    # (1), (2) the chain
    n, m = 96, 0.2
    h, Q, R = OPS.chain_parts(n, m, 3.0)
    N = 1 - 0.3 * np.exp(-R ** 2 / 200.0)
    v = -0.5 * (R / 10) * np.exp(-R ** 2 / 200.0)
    hm = OPS.clocked(h, Q, N, v)
    res["chain_hermiticity"] = float(np.max(np.abs(hm - hm.conj().T)))
    mz, yy, Rf = FC.radial_fields(n, m, 3.0)
    res["chain_vs_flow_chart_v4_v0"] = float(np.max(np.abs(OPS.clocked(h, Q, np.ones(n), np.zeros(n)) - FC.chain_hamiltonian(mz, yy, np.zeros(n)))))
    res["chain_vs_flow_chart_v4_v"] = float(np.max(np.abs(OPS.clocked(h, Q, np.ones(n), v) - FC.chain_hamiltonian(mz, yy, v))))
    # (8) +kappa against -kappa
    hp, Qp, _ = OPS.chain_parts(n, m, 3.0)
    hn, Qn, _ = OPS.chain_parts(n, m, -3.0)
    ep = np.sort(np.linalg.eigvalsh(OPS.clocked(hp, Qp, N, v)))
    en = np.sort(np.linalg.eigvalsh(OPS.clocked(hn, Qn, N, v)))
    res["chain_plus_minus_kappa_spectra"] = float(np.max(np.abs(ep - en)))
    # (3) periodic 3D dispersion at uniform N, v
    L = 6
    h3, Q3 = OPS.lattice3d_parts(L, m, periodic=True)
    N0, v0 = 0.8, np.array([0.1, -0.2, 0.05])
    hm3 = OPS.clocked3d(h3, Q3, np.full(L ** 3, N0), np.tile(v0, (L ** 3, 1)))
    res["3d_hermiticity"] = float(np.max(np.abs(hm3 - hm3.conj().T)))
    e3 = np.sort(np.linalg.eigvalsh(hm3))
    k = 2 * np.pi * np.arange(L) / L
    K = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)
    E = N0 * np.sqrt(m * m + np.sum(np.sin(K) ** 2, 1))
    dop = np.sum(v0[None, :] * np.sin(2 * K) / 2, 1)
    sym = np.sort(np.concatenate([E + dop, E + dop, -E + dop, -E + dop]))
    res["3d_dispersion_vs_symbol"] = float(np.max(np.abs(e3 - sym)))
    # (4), (5) the sources
    e, U = np.linalg.eigh(hm)
    pos = e > 0
    EF, Ts = 0.19, 0.002
    f = OPS.fermi(e[pos], EF, Ts)
    e1, J1 = OPS.sources_static(h, Q, U[:, pos], f)
    Phi = np.hstack([U[:, ~pos], U[:, pos]])
    ff = np.concatenate([np.ones((~pos).sum()), f])
    e2, J2 = OPS.sources_dynamic(h, Q, Phi, ff, U[:, ~pos])
    ev, Jv = OPS.sources_dynamic(h, Q, U[:, ~pos], np.ones((~pos).sum()), U[:, ~pos])
    res["form_i_minus_form_ii"] = [float(np.max(np.abs(e1 - e2))), float(np.max(np.abs(J1 - J2)))]
    res["vacuum_sources_form_ii"] = [float(np.max(np.abs(ev))), float(np.max(np.abs(Jv)))]
    res["occupied_bound"] = float(f.sum())
    # (6), (7) the constraint solver
    R2 = np.arange(1, 401, dtype=float)
    eps = 0.5 * np.exp(-((R2 - 20) / 6) ** 2)
    g = 0.25
    Nc, vc, Wf, info = FLD.solve_constraints(eps, 0 * eps, g, R2)
    Eg = float(np.sum(Nc ** 2 * eps))
    res["J0_solver_info"] = info
    res["J0_N_minus_1_max"] = float(np.max(np.abs(Nc - 1)))
    res["J0_exterior"] = FLD.exterior_check(Nc, vc, R2, Wf, g, Eg, 40.0)
    res["J0_charge_collars"] = [FLD.charge(Wf, g, c) for c in (60, 150, 300, 399)]
    res["J0_charge_gauss"] = Eg
    for it in range(80):
        J = np.abs(vc) * eps / Nc
        Nc, vc, Wf, info = FLD.solve_constraints(eps, J, g, R2, N0=Nc)
    Eg = float(np.sum(Nc ** 2 * eps))
    r1, r2 = FLD.residuals(eps, J, g, R2, Nc, vc, Wf)
    res["sign_check"] = {"N_min": float(Nc.min()), "R_at_N_min": float(R2[np.argmin(Nc)]), "N_max": float(Nc.max()),
                         "N_inside_below_1": bool(Nc[R2 < 20].max() < 1), "exterior": FLD.exterior_check(Nc, vc, R2, Wf, g, Eg, 40.0),
                         "charge_collars": [FLD.charge(Wf, g, c) for c in (60, 150, 300, 399)], "charge_gauss": Eg,
                         "trapping": FLD.trapping_radius(Nc, vc, R2), "residual_H": float(np.max(np.abs(r1))), "residual_M": float(np.max(np.abs(r2))),
                         "min_N_minus_absv": float(np.min(Nc - np.abs(vc)))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
