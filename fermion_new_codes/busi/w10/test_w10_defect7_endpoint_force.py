#!/usr/bin/env python
"""test_w10_defect7_endpoint_force.py (busi W10): defect 7 of REVIEWA_REVIEW_web40i section 5. The archived momentum-flux reader omits
the open-chain endpoint force: F_cut = F_archived + (1/2) Tr D {P, i[h_b, Q]}. Gate: the cut-locality identity on a fixture.
The identity: the enclosed radial momentum Tr D (1/2){P, Q} (P the sites below the cut, Q the two-step current) changes at the rate
Tr D i[h_m, (1/2){P, Q}] = (1/2) Tr D {i[h_m, P], Q} + (1/2) Tr D {P, i[h_m, Q]} (the face flux plus the forces); the forces split by the
parts of h_m (on-site: centrifugal; flow: the flow gradient; bond: zero in the bulk), and at unit clock the bond part is exactly the two
end bonds, i[h_bond, Q] = (i/8) sigma_x (|N-2><N-1| - |N-1><N-2| - |0><1| + |1><0|). Fixture: 64 sites, kappa 1, m 0.2, a nonuniform
flow, the clock unit or nonuniform; the states: the ground state of h_m (held, touches the origin) and a filling of 6 packets localized
at R 30 (width 3). Checks: the identity closes to roundoff at every cut; the archived form (every term but the bond force) misses the
origin traction, the same for every interior cut (2 to N - 2) at unit clock; P = 1 gives the sum of the two end tractions; the
commutator's explicit end-bond formula; the localized filling has no traction. Optionally page_ops_v6.chain_sparse (W10_PAGE_DIR) and
engine_vacflow_v1.chain_sparse (W10_ENGINE_DIR) are checked equal to the lib's chain.
Run: qi2 python test_w10_defect7_endpoint_force.py"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def ground_state(hfull):
    e, U = np.linalg.eigh(hfull.toarray())
    occ = e < 0
    return U[:, occ], np.ones(int(occ.sum()))


def localized_filling(n, center, width, nmodes, seed=0):
    rng = np.random.default_rng(seed)
    sites = np.arange(n)
    cols = []
    for _ in range(nmodes):
        k0 = rng.uniform(-math.pi, math.pi)
        env = np.exp(-((sites - center) / (2.0 * width)) ** 2) * np.exp(1j * k0 * sites)
        spin = rng.normal(size=2) + 1j * rng.normal(size=2)
        col = np.zeros(2 * n, complex); col[0::2] = env * spin[0]; col[1::2] = env * spin[1]
        cols.append(col)
    X = np.array(cols).T
    Qm, _ = np.linalg.qr(X)
    return Qm, np.ones(nmodes)


def explicit_end_commutator(n):
    """(i/8) sigma_x (|N-2><N-1| - |N-1><N-2| - |0><1| + |1><0|) in the flattened basis."""
    C = np.zeros((2 * n, 2 * n), complex)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    for (a_, b_, sgn) in ((n - 2, n - 1, 1.0), (n - 1, n - 2, -1.0), (0, 1, -1.0), (1, 0, 1.0)):
        C[2 * a_: 2 * a_ + 2, 2 * b_: 2 * b_ + 2] = (1j / 8.0) * sgn * sx
    return C


def main():
    n, m, kappa = 64, 0.2, 1.0
    R = np.arange(n, dtype=float) + 1.0
    hs, Qs = W10.chain_sparse_ref(R, m, kappa)
    for d in (os.environ.get("W10_PAGE_DIR"), os.environ.get("W10_ENGINE_DIR")):
        if d and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import page_ops_v6 as P6
        h6, Q6 = P6.chain_sparse(R, m, kappa)
        check(abs(h6 - hs).max() < 1e-15 and abs(Q6 - Qs).max() < 1e-15, "page_ops_v6.chain_sparse equals the lib's chain (the same conventions)")
    except ImportError:
        print("INFO page_ops_v6 not importable (W10_PAGE_DIR): the lib's verbatim copy stands for it")
    try:
        import engine_vacflow_v1 as VF
        hv, Qv = VF.chain_sparse(R, m, kappa)
        check(abs(hv - hs).max() < 1e-15 and abs(Qv - Qs).max() < 1e-15, "engine_vacflow_v1.chain_sparse (the engine's Chan.hs, Chan.Qs) equals the lib's chain")
    except Exception:  # noqa: BLE001
        print("INFO engine_vacflow_v1 not importable (W10_ENGINE_DIR or its own imports): not compared")
    v = -0.4 * np.exp(-((R - 30.0) / 8.0) ** 2)
    for clock_label, N in (("unit clock", np.ones(n)), ("nonuniform clock N = 1 + 0.15 exp(-((R - 30)/12)^2)", 1.0 + 0.15 * np.exp(-((R - 30.0) / 12.0) ** 2))):
        hfull = W10.clocked_full(hs, Qs, N, v)
        op = W10.bond_traction_operator(hs, Qs, N, W10.END_WINDOW)
        if clock_label == "unit clock":
            Cx = explicit_end_commutator(n)
            check(np.max(np.abs(op["C"].toarray() - Cx)) < 1e-15, "unit clock: i[h_bond, Q] equals (i/8) sigma_x (|N-2><N-1| - |N-1><N-2| - |0><1| + |1><0|) to %.1e: the two end bonds and nothing else" % np.max(np.abs(op["C"].toarray() - Cx)))
            check(op["max_outside"] < 1e-15, "unit clock: no entry of the commutator outside the two end windows (%.1e)" % op["max_outside"])
        else:
            print("INFO %s: the commutator's largest entry outside the end windows is %.3e (the clock-gradient force of the rows, not a traction)" % (clock_label, op["max_outside"]))
        states = {"the ground state of h_m (held, touches the origin)": ground_state(hfull),
                  "a filling of 6 packets localized at R 30 (width 3)": localized_filling(n, 29.0, 3.0, 6)}
        for sname, (cols, f) in states.items():
            origin, outer = W10.end_tractions(op, cols[:op["n_rows"]], cols[-op["n_rows"]:], f)
            terms = {cut: W10.momentum_identity_terms(hs, Qs, N, v, cols, f, cut) for cut in range(1, n + 1)}
            scale = max(1.0, max(abs(t["rate_exact"]) for t in terms.values()), max(abs(t["face"]["bond"]) for t in terms.values()))
            closure = max(abs(t["rate_exact"] - t["sum_terms"]) for t in terms.values())
            check(closure < 1e-12 * scale, "%s, %s: the identity rate = faces + forces closes at every cut 1..%d (largest defect %.1e; scale %.2e)" % (clock_label, sname, n, closure, scale))
            defect = np.array([terms[c]["defect_without_bond_force"] for c in range(2, n - 1)])
            if clock_label == "unit clock":
                check(np.max(np.abs(defect - origin)) < 1e-13 * scale,
                      "gate (the cut-locality identity on a fixture): %s: the archived form's defect equals the origin traction %.6e for every interior cut 2..%d (spread %.1e)" % (sname, origin, n - 2, np.max(np.abs(defect - origin))))
                check(abs(terms[n]["force"]["bond"] - (origin + outer)) < 1e-13 * scale, "%s: with P = 1 the bond force is the sum of the two end tractions (origin %.3e + outer %.3e)" % (sname, origin, outer))
            else:
                print("INFO %s, %s: the archived form's defect over the interior cuts ranges %.3e to %.3e (the origin traction %.3e plus the clock-gradient force inside the cut)" % (clock_label, sname, defect.min(), defect.max(), origin))
            if "localized" in sname:
                check(abs(origin) < 1e-12 and abs(outer) < 1e-12, "%s: both end tractions vanish for the localized filling (%.1e, %.1e)" % (sname, origin, outer))
            else:
                check(abs(origin) > 1e-6, "%s: the origin traction is not zero (%.6e): the held state's balance closes only with it (its exact rate is %.1e)" % (sname, origin, terms[10]["rate_exact"]))
    print("test_w10_defect7_endpoint_force: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
