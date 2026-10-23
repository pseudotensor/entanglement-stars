#!/usr/bin/env python
"""dust_tail_selftest_v1.py (busi lane, reviewer A): the selftest of dust_tail_v1.py on synthetic inputs with known answers. Items 1 to 4
test the operator module (dust_tail_ops_v1) on a 64-site chain with smooth random fields: Hermiticity; the anticommutation with
(-1)^i sigma_y K and its consequences (symmetric spectrum, the Killing vacuum's site density exactly 1, zero current through every cut);
the bond-block currents against the sparse-block formula at every cut, against the exact evolution e^{-i h_m t} by a central difference,
and against a closed form; the density identities and the exterior block's spectral sums. Item 5 runs process_slice on a synthetic slice
in the stored layout (the sea = the exact negative-energy eigenvectors, dust columns localized beyond the cuts with margin, one straddling
the inner cut) whose answers are known in closed form. Item 6 runs the whole program (run_main) on that slice with a synthetic run json and
fields npz built from the slice's own densities (the run checks must read zero) and with synthetic first-route jsons carrying this route's
own numbers (the comparison must print zero differences), and checks the written files."""
import json
import math
import os
import tempfile

import numpy as np
import scipy.linalg as sla

import dust_tail_ops_v1 as OPS


def synthetic_slice(d, R, m, N, v, rng):
    n = len(R)
    arrs = {"T": 128.0, "rs": 2.0, "R": R, "N_applied": N, "v_applied": v, "kappas": np.array([1, 2]), "weights": np.array([16.0, 32.0]), "dtype": "complex128"}
    known = {}
    for k in (1, 2):
        hk = OPS.killing_h(R, m, k, N, v); ek, Vk = sla.eigh(hk.toarray())
        A = np.zeros((2 * n, 1), complex); A[2 * 53:2 * 60, 0] = rng.standard_normal(14) + 1j * rng.standard_normal(14); A /= np.linalg.norm(A)
        if k == 2:
            B = np.zeros((2 * n, 1), complex); B[2 * 28:2 * 36, 0] = rng.standard_normal(16) + 1j * rng.standard_normal(16); B /= np.linalg.norm(B)
            dust = np.concatenate([A, B], axis=1); fk = np.array([1.0, 0.5])
        else:
            dust = A; fk = np.array([1.0])
        arrs["sea_%d" % k] = Vk[:, ek < 0]; arrs["dust_%d" % k] = dust; arrs["f_%d" % k] = fk; arrs["e_%d" % k] = ek
        known[k] = {"EA": float(np.real(A.conj().T @ (hk @ A))[0, 0]), "wB": (0.5 * float(np.sum(np.abs(dust[2 * 32:, 1]) ** 2))) if k == 2 else 0.0,
                    "FB32": float(OPS.cut_currents_all(N, v, dust, fk)[31]) if k == 2 else 0.0}
    path = os.path.join(d, "orbitals_selftest_T0064.00.npz"); np.savez(path, **arrs)
    return path, arrs, known


def synthetic_run(d, arrs, m, g=0.25):
    """a run json and fields npz whose stored shell totals are the slice's own weighted dust densities (pairs zero)."""
    R = arrs["R"]; N = arrs["N_applied"]; v = arrs["v_applied"]; n = len(R)
    eps = np.zeros(n); J = np.zeros(n)
    for k, w in zip(arrs["kappas"], arrs["weights"]):
        h0, Q = OPS.chain_operators(R, m, int(k))
        eps += w * OPS.op_density(h0, arrs["dust_%d" % k], arrs["f_%d" % k]); J += w * OPS.op_density(Q, arrs["dust_%d" % k], arrs["f_%d" % k])
    tag = "selftest"
    J_ = {"args": {"g": g, "rs": 2.0, "m": m, "nsites": n}, "records": [{"T": 128.0, "E": 10.0, "E_dust_shells": float(eps.sum()), "E_pairs_shells": 0.0, "R_t": None,
                                                                       "min_N_minus_absv": None, "continuation_label": "synthetic", "record_label": "synthetic"}]}
    json.dump(J_, open(os.path.join(d, "collapse_%s.json" % tag), "w"))
    np.savez(os.path.join(d, "collapse_%s_fields.npz" % tag), R=R, T=np.array([128.0]), N_applied=N[None, :], v_applied=v[None, :], eps_tot=eps[None, :], J_tot=J[None, :],
             eps_pairs=np.zeros((1, n)), J_pairs=np.zeros((1, n)))
    return os.path.join(d, tag)


def synthetic_first_route(d, row):
    """first-route jsons in the layout dust_tail_v1 reads (D_dust_tail_v1.json rows and a D v2 batch json), filled with this route's numbers."""
    W = row["weighted"]
    tail = {"rows": [{"T_over_rs": row["T_over_rs"], "F_dust_16_weighted": W["16.0"]["F"], "F_dust_16_per_channel": {str(c["kappa"]): c["beyond"]["16.0"]["F"] for c in row["channels"]}}]}
    for c, key in (("16.0", "beyond_16"), ("24.0", "beyond_24")):
        q = W[c]
        tail["rows"][0][key] = {"dust_weight": q["N"], "particles_dust": q["particles"], "E_particles_dust": q["E_particles"], "holes_dust": -q["holes"], "E_out_dust": q["E_sym"],
                                "E_per_particle_over_m": q["E_per_particle_over_m"]}
    json.dump(tail, open(os.path.join(d, "tail.json"), "w"))
    dv2 = {"slices": [{"T_over_rs": row["T_over_rs"], "channels": [{"kappa": c["kappa"], "regions": {k.rstrip("0").rstrip("."): {"dust_weight": r["N"], "particles_dust": r["particles"],
                                                                                                                             "E_particles_dust": r["E_particles"], "E_out_dust": r["E_sym"]}
                                                                                                     for k, r in c["beyond"].items()}} for c in row["channels"]]}]}
    json.dump(dv2, open(os.path.join(d, "dv2.json"), "w"))
    return os.path.join(d, "tail.json"), os.path.join(d, "dv2.json")


def selftest(MAIN):
    rng = np.random.default_rng(7)
    ok = True

    def chk(name, cond, val):
        nonlocal ok
        ok = ok and bool(cond)
        print("%s %s (%s)" % ("PASS" if cond else "FAIL", name, val))
    n = 64; R = np.arange(1, n + 1, dtype=float); m = 0.2
    N = 0.6 + 0.4 * R / n; v = -0.5 / np.sqrt(1.0 + R / 8.0)
    hm = OPS.killing_h(R, m, 2, N, v); H = hm.toarray()
    chk("1 h_m Hermitian", np.max(np.abs(H - H.conj().T)) < 1e-14, np.max(np.abs(H - H.conj().T)))
    Hc = OPS.antiunitary_conjugate(hm)
    chk("2 (-1)^i sigma_y K anticommutes with h_m", np.max(np.abs(Hc + H)) < 1e-14, np.max(np.abs(Hc + H)))
    e, V = sla.eigh(H); Vn = V[:, e < 0]
    chk("2b spectrum symmetric, half negative", np.max(np.abs(e + e[::-1])) < 1e-12 and int((e < 0).sum()) == n, (np.max(np.abs(e + e[::-1])), int((e < 0).sum())))
    rv = OPS.site_density(Vn)
    chk("2c the Killing vacuum's site density is 1", np.max(np.abs(rv - 1)) < 1e-12, np.max(np.abs(rv - 1)))
    Fv = OPS.cut_currents_all(N, v, Vn)
    chk("2d the vacuum's current through every cut vanishes", np.max(np.abs(Fv)) < 1e-12, np.max(np.abs(Fv)))
    cols = rng.standard_normal((2 * n, 6)) + 1j * rng.standard_normal((2 * n, 6)); cols /= np.linalg.norm(cols, axis=0); f = rng.uniform(0.2, 1.0, 6)
    Fa = OPS.cut_currents_all(N, v, cols, f); Fs = np.array([OPS.cut_current_at(hm, cols, f, p) for p in range(1, n)])
    chk("3a bond-block currents equal the sparse-block currents at every cut", np.max(np.abs(Fa - Fs)) < 1e-12, np.max(np.abs(Fa - Fs)))
    dt = 1e-4; p = 32
    Up = sla.expm(-1j * H * dt); Um = Up.conj().T
    NY = lambda c: float(np.sum(f * np.sum(np.abs(c[2 * p:]) ** 2, axis=0)))  # noqa: E731
    dN = (NY(Up @ cols) - NY(Um @ cols)) / (2 * dt)
    chk("3b F equals dN_Y/dt of the exact evolution (dt %g)" % dt, abs(dN - Fa[p - 1]) < 1e-7 * max(1.0, abs(Fa[p - 1])), (dN, Fa[p - 1]))
    c = np.zeros((2 * n, 1), complex); c[2 * (p - 1), 0] = 1 / math.sqrt(2); c[2 * p + 1, 0] = 1 / math.sqrt(2)
    Fc = OPS.cut_currents_all(N, v, c, [0.7])[p - 1]
    chk("3c closed form F = f (N_x + N_y)/4 for (e_x0 + e_y1)/sqrt2 across the cut", abs(Fc - 0.7 * (N[p - 1] + N[p]) / 4) < 1e-14, Fc)
    h0, Q = OPS.chain_operators(R, m, 2)
    eps0 = OPS.op_density(h0, cols, f); J = OPS.op_density(Q, cols, f); ks = OPS.op_density(hm, cols, f)
    tr = float(np.real(np.sum(f * np.sum(np.conj(cols) * (hm @ cols), axis=0))))
    chk("4a sum (N eps + v J) = sum k_sym = Re Tr(h_m C)", abs(np.sum(N * eps0 + v * J) - ks.sum()) < 1e-12 and abs(ks.sum() - tr) < 1e-12, (np.sum(N * eps0 + v * J), ks.sum(), tr))
    om, phi = OPS.exterior_spectrum(hm, p); nl = OPS.occupations_in_basis(phi, cols[2 * p:], f)
    chk("4b sum n_l = N(Y) and sum omega n = block energy", abs(nl.sum() - NY(cols)) < 1e-12 and abs((om * nl).sum() - OPS.block_energy(hm, cols[2 * p:], p, f)) < 1e-12, (nl.sum(), (om * nl).sum()))
    d = tempfile.mkdtemp(prefix="dust_selftest_")
    path, arrs, known = synthetic_slice(d, R, m, N, v, rng)
    row, _arrays = MAIN.process_slice(path, m, [16.0, 24.0], [8.0, 16.0, 24.0], "dust", False, log=lambda s: None)
    chk("5a cuts at R 32 and 48 (sites beyond 32 and 16)", row["cuts"][0]["first_site_index"] == 32 and row["cuts"][1]["first_site_index"] == 48, row["cuts"])
    chk("5b sea: density 1, no excess, no current, stored spectrum reproduced", row["checks"]["max_sea_density_dev"] < 1e-12 and row["weighted"]["16.0"]["max_sea_excess"] < 1e-12
        and row["weighted"]["16.0"]["max_sea_current"] < 1e-12 and row["checks"]["max_spectrum_vs_stored"] < 1e-12, (row["checks"]["max_sea_density_dev"], row["checks"]["max_spectrum_vs_stored"]))
    r24 = row["weighted"]["24.0"]; r16 = row["weighted"]["16.0"]
    E24 = 16 * known[1]["EA"] + 32 * known[2]["EA"]
    chk("5c beyond 24 r_s: N = 48 exactly, E_sym = K = E_block = E_particles + E_holes = sum w <A|h_m|A>, F = 0",
        abs(r24["N"] - 48) < 1e-12 and abs(r24["E_sym"] - E24) < 1e-10 and abs(r24["K_density"] - E24) < 1e-10 and abs(r24["E_block"] - E24) < 1e-10
        and abs(r24["E_particles"] + r24["E_holes"] - E24) < 1e-10 and abs(r24["F"]) < 1e-14, (r24["N"], r24["E_sym"], E24, r24["F"]))
    N16 = 16 + 32 * (1 + known[2]["wB"])
    chk("5d beyond 16 r_s: N = 16 + 32 (1 + 0.5 |Pi_Y B|^2), F = 32 F_B, the channel-1 current 0", abs(r16["N"] - N16) < 1e-12 and abs(r16["F"] - 32 * known[2]["FB32"]) < 1e-12
        and abs(row["channels"][0]["beyond"]["16.0"]["F"]) < 1e-14, (r16["N"], N16, r16["F"], 32 * known[2]["FB32"]))
    chk("5e totals: dust number 16 + 48 = 64 = nominal; the identities to 1e-10", abs(row["dust_number_total_weighted"] - 64) < 1e-12 and abs(row["checks"]["total_number_minus_nominal"]) < 1e-12
        and r16["max_sum_n_minus_N"] < 1e-12 and r16["max_block_identity"] < 1e-10 and r16["max_F_minus_F_sparse"] < 1e-12 and abs(row["checks"]["density_route_minus_killing_total"]) < 1e-10,
        (row["dust_number_total_weighted"], row["checks"]["density_route_minus_killing_total"]))
    st = MAIN.npz_structure(path)
    chk("5f the npz structure reader lists the members with their shapes", len(st) == len(arrs) and dict((x[0], x[1]) for x in st)["sea_1.npy"] == (2 * n, n), st[:3])
    # 6 the whole program on the synthetic slice
    run = synthetic_run(d, arrs, m); tail, dv2 = synthetic_first_route(d, row)
    out = os.path.join(d, "out")
    a = MAIN.default_args(slices=path, m=m, out_dir=out, tag="selftest", run=run, first_route_tail_json=tail, first_route_dv2_jsons=dv2)
    rows = MAIN.run_main(a, log=lambda s: None)
    rc = rows[0]["run_checks"]
    chk("6a run checks: the applied fields equal, K_density equals the stored K_dust beyond both cuts, the bare energy equals E_dust_shells",
        max(rc["applied_fields_max_dev"]) == 0.0 and abs(rc["beyond"]["16.0"]["diff"]) < 1e-12 and abs(rc["beyond"]["24.0"]["diff"]) < 1e-12 and abs(rc["E_dust_shells_diff"]) < 1e-12
        and abs(rc["m_over_TH"] - m * 8 * math.pi * 0.25 * 10.0) < 1e-12, rc["beyond"])
    files = [os.path.join(out, "selftest_%s" % s) for s in ("T0064.00.json", "T0064.00.npz", "all.json", "table.md")]
    chk("6b the per-slice json and npz, the all json and the table are written", all(os.path.exists(x) for x in files), files)
    md = open(files[-1]).read()
    chk("6c the table prints the first route beside with zero differences", ("| 64.000 | N (dust weight) | %.6e | %.6e | +0.000e+00 |" % (r16["N"], r16["N"])) in md and "## H." in md and "## G." in md, md.count("+0.000e+00"))
    Zo = np.load(files[1])
    chk("6d the npz holds the per-site and per-level arrays", all(k in Zo.files for k in ("R", "rho_dust_2", "F_dust_2", "omega_16_2", "n_16_2", "rho_sea_minus_1_1", "e_1")) and Zo["F_dust_2"].shape == (n - 1,), Zo.files)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok
