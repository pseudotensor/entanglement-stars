#!/usr/bin/env python
"""stageA_declare_v1.py (busi; B' s = 16 stage A, the declarations ReviewA_bpchk2_v1 and ReviewB_bpchk2_v1 require BEFORE stage A is read).
Box CPU. Writes into --out:
  stageA_manifest_v1.json: the literal port tag 'port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)' (with the eta and
    onset variants of runs 2a and 2b), the hypothesis ladder (killing_null, exit_hole, supply_filled, file:thermal_T0) with the bath and hole
    conventions, the vocabulary rule, the readout meanings (reviewer A bpchk2 sec. 1.3 table; reviewer B bpchk2 sec. 1 table), the prep md5s and the
    input-basis identity check across the three preps (reviewer A sec. 6: a different basis is a different experiment);
  hole_packet_v1.npz: the declared hole packet (reviewer A sec. 2 item 2): the combination of exit-window columns of minimum radial spread about
    the mid-shell radius r_mid = (inner + outer averaged root)/2, i.e. the lowest eigenvector of <u_i|(r - r_mid)^2|u_j> on the exit span;
  cin_exit_hole_v1.npz, cin_supply_filled_v1.npz, cin_thermal_T0_v1.npz: the covariances C on the inputs (supply first, then exit) in the
    channel convention (supply: particle occupations; exit: HOLE occupations, holes propagated as conjugate particles, so the exit block is
    read against conj(G_ee)); each checked 0 <= C <= I to 1e-12 (reviewer B bpchk2 sec. 2); no supply-exit cross block."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3b_cube as CU  # noqa: E402

PORT_TAG = "port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_cov(C, name):
    w = np.linalg.eigvalsh(0.5 * (C + C.conj().T))
    herm = float(np.max(np.abs(C - C.conj().T)))
    ok = herm < 1e-12 and w.min() > -1e-12 and w.max() < 1 + 1e-12
    return {"name": name, "herm_defect": herm, "eig_min": float(w.min()), "eig_max": float(w.max()), "valid_0_le_C_le_I": bool(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = np.load(os.path.join(a.prep_dir, "F1_s16_d0_prep.npz"))
    # the control preps re-solve the same retained block; degenerate levels make an independent eigensolve's basis an arbitrary rotation of
    # the same span, so the control preps are re-written with the BASE basis (reviewer A bpchk2 sec. 6: one input basis for every stage), after
    # checking that the fields on the retained sub-box agree and that the spans agree
    for t in ("F1_s16_d0_eta2", "F1_s16_d0_onset225"):
        z = np.load(os.path.join(a.prep_dir, t + "_prep.npz"))
        Hb, Hn, Hs_ = int(base["H"]), int(z["H"]), int(base["Hs"])
        sb = slice(Hb - Hs_, Hb + Hs_ + 1); sn = slice(Hn - Hs_, Hn + Hs_ + 1)
        dF = max(float(np.max(np.abs(z[k][sn, sn, sn] - base[k][sb, sb, sb]))) for k in ("N", "vx", "vy", "vz"))
        # the spans agree iff every singular value of U_base^dag U_new is 1 (no d x d projector is formed)
        if z["U_sub"].shape == base["U_sub"].shape:
            sv = np.linalg.svd(base["U_sub"].conj().T @ z["U_sub"], compute_uv=False)
            dP = float(np.max(np.abs(sv - 1.0)))
        else:
            dP = None
        d = {k: z[k] for k in z.files}
        for k in ("U_sub", "e", "n_supply", "Hs", "qF", "R_I"):
            d[k] = base[k]
        mm = json.loads(str(z["manifest"])); mm["basis_from"] = "F1_s16_d0_prep.npz (base basis)"; mm["subbox_field_max_diff"] = dF; mm["span_singular_value_max_dev_from_1"] = dP
        d["manifest"] = json.dumps(mm)
        np.savez_compressed(os.path.join(a.prep_dir, t + "_bb_prep.npz"), **d)
        json.dump(mm, open(os.path.join(a.prep_dir, t + "_bb_prep.json"), "w"), indent=1, default=str)
    tags = {"F1_s16_d0": "run 1 and 2c", "F1_s16_d0_eta2_bb": "run 2a", "F1_s16_d0_onset225_bb": "run 2b"}
    Z = {t: np.load(os.path.join(a.prep_dir, t + "_prep.npz")) for t in tags}
    ident = {}
    for t in tags:
        z = Z[t]
        same_shape = z["U_sub"].shape == base["U_sub"].shape
        ident[t] = {"md5": md5(os.path.join(a.prep_dir, t + "_prep.npz")), "K": int(len(z["e"])), "n_supply": int(z["n_supply"]),
                    "max_abs_dU": float(np.max(np.abs(z["U_sub"] - base["U_sub"]))) if same_shape else None,
                    "max_abs_de": float(np.max(np.abs(z["e"] - base["e"]))) if same_shape else None,
                    "eta_max": float(z["eta"].max()), "H": int(z["H"]), "R_I": float(z["R_I"])}
        if t != "F1_s16_d0":
            mm = json.loads(str(z["manifest"]))
            ident[t]["subbox_field_max_diff"] = mm.get("subbox_field_max_diff"); ident[t]["span_sv_dev_of_its_own_eigensolve"] = mm.get("span_singular_value_max_dev_from_1")
        ident[t]["basis_identical_1e-12"] = bool(same_shape and ident[t]["max_abs_dU"] < 1e-12 and ident[t]["max_abs_de"] < 1e-14)
    man = json.loads(str(base["manifest"]))
    ns = int(base["n_supply"]); K = len(base["e"]); e = base["e"]; T0 = float(base["T0"]); Hs = int(base["Hs"])
    roots = man["baseline_roots"]["axis"]
    r_mid = 0.5 * (roots[0] + roots[-1]) if len(roots) >= 2 else 0.75 * float(base["R_I"])
    _, _, _, rs = CU.grid(Hs)
    w = np.tile(((rs - r_mid) ** 2).ravel(), 4)
    Ue = base["U_sub"][:, ns:]
    M = Ue.conj().T @ (w[:, None] * Ue)
    lam, V = np.linalg.eigh(0.5 * (M + M.conj().T))
    c_exit = V[:, 0]
    c = np.zeros(K, complex); c[ns:] = c_exit
    spread = float(np.sqrt(lam[0]))
    e_packet = float(np.real(c.conj() @ (e * c)))
    np.savez_compressed(os.path.join(a.out, "hole_packet_v1.npz"), c=c, r_mid=r_mid, spread=spread, e_mean=e_packet)
    Cs = {}
    C = np.zeros((K, K), complex); C[ns:, ns:] = np.outer(c_exit, c_exit.conj()); Cs["exit_hole"] = C
    C = np.zeros((K, K), complex); C[np.arange(ns), np.arange(ns)] = 1.0; Cs["supply_filled"] = C
    f = lambda x: 1.0 / (np.exp(np.clip(x, -700, 700)) + 1.0)
    C = np.diag(np.concatenate([f(e[:ns] / T0), f(np.abs(e[ns:]) / T0)])).astype(complex); Cs["thermal_T0"] = C
    checks = []
    files = {}
    for k_, C in Cs.items():
        p = os.path.join(a.out, "cin_%s_v1.npz" % k_)
        np.savez_compressed(p, C=C, n_supply=ns, convention="supply: particle occupations; exit: hole occupations (conjugate particles)")
        checks.append(check_cov(C, k_)); files[k_] = {"file": os.path.basename(p), "md5": md5(p)}
    out = {"script": "stageA_declare_v1.py", "port_tag": PORT_TAG,
           "port_variants": {"run 1, 2c": PORT_TAG, "run 2a": PORT_TAG.replace("eta_max 1/a", "eta_max 2/a"),
                             "run 2b": PORT_TAG.replace("onset 2 L", "onset 2.25 L").replace("vacuum", "half-side 2.75 L, vacuum")},
           "hypotheses": {"killing_null": "supply empty, exit filled (zero holes): the held Killing vacuum on the span; every output identically zero; the ladder's origin, not a run",
                          "exit_hole": "one hole (the declared packet) in the Killing-filled exit window (H2); read at the measured transit time theta(chi)",
                          "supply_filled": "every (0, T0) supply direction occupied, exit at the Killing filling (H1); the in-run reader of every stage-A run",
                          "file:thermal_T0": "optional bracket: Fermi occupations at T0 on supply (particles) and exit (holes), block-diagonal; a hypothesis, not an Unruh state"},
           "conventions": "vacuum sink bath per channel; holes propagated as conjugate particles (exit block read against conj(G_ee)); no supply-exit cross block (the reader refuses one)",
           "no_preparation_claim": "G8 closed by scope: no massless preparation of the record exists in B'; no initial exterior correlations",
           "vocabulary": "not used for sink2L quantities: far width, greybody, emitted, radiation entropy, outgoing algebra, Page curve, accessible to infinity; 'accessible' only as 'delivered to the sink at 2 L'",
           "readout_meanings": {"p(T)": "p_2L(T): weight of a window direction delivered to the sink beyond 2 L by T", "N_acc, Gamma": "N_acc_2L, Gamma_2L: leakage to the sink; censoring unchanged; fitted-pole diagnostics only",
                                "S(t)": "returns inside the sub-2L cavity; the near-zone exterior return is absent by construction",
                                "bins_reader": "bath-record quantities of the sink2L dilation under the declared covariance",
                                "sectors": "direction-sector absorbed flux: a proxy, not kappa-channel content (C9 not built)",
                                "2b onset 2.25 L": "the onset derivative of p_2L: a result, not an error bar",
                                "2a eta x 2, 2c dt 2a": "error bar: pass if |dp| <= max(0.02 p, 1e-4) per direction; 2a failing = the eta-limited branch (not a HOLD)"},
           "B11_label": "B11 measured at 4 to 60 T0 on the chains (no chain level in the window)",
           "F1_labels": ["fails the hydrodynamic gate inside (sigma ||K|| ~ 1 at 0.30 L): results on given fields", "per valley: 16 + 16 of the 128 + 128 inputs (8 valleys)"],
           "hole_packet": {"r_mid_sites": r_mid, "radial_spread_sites": spread, "e_mean_over_T0": e_packet / T0, "file": "hole_packet_v1.npz",
                           "md5": md5(os.path.join(a.out, "hole_packet_v1.npz"))},
           "covariances": files, "covariance_checks": checks, "preps": ident,
           "all_bases_identical": all(v["basis_identical_1e-12"] for v in ident.values())}
    json.dump(out, open(os.path.join(a.out, "stageA_manifest_v1.json"), "w"), indent=1)
    print(json.dumps({"all_bases_identical": out["all_bases_identical"], "preps": ident, "hole_packet": out["hole_packet"], "checks": checks}, indent=1))


if __name__ == "__main__":
    main()
