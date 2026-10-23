#!/usr/bin/env python
"""dust_tail_selftest_v2.py (busi lane, reviewer A): the selftest of dust_tail_v2.py's additions on small fixtures with known answers.
Items 1 to 3 test the pieces (the binary entropy and the spectral entropy; the 5-bin partition and its edges; the dust-only exterior block's
entropy against closed forms and the rank bound's equality case). Item 4 tests the bound d h2(n/d) as an inequality on random pure Gaussian
states (random Slater determinants: the subsystem's entropy against d h2(n/d) and N_f h2(n/N_f), its equality with the complement's entropy)
and the pinching form on random sectors with a random F_+, F_- split. Item 5 tests the number current's continuity identity on a fixture:
the site identity d rho_x/dt = F(x) - F(x+1) at every site, against the exact evolution's central difference, and the integrated form
N_Y(T) - N_Y(0) = integral F dt along the exact evolution. Item 6 runs process_slice_v2 with --deviation on a synthetic slice in the stored
layout (a 96-site chain with N = 3 inside the cut and 1 outside, the exact Killing vacuum as the sea, 2 positive eigenvectors of h_m as the
dust with occupations 0.5 and 1) whose escaped-sector answers are known in closed form (the pairs' counts 0; S_F = the sum of h2(f) over the
columns in F; n_+ = the sum of their f; n_- = 0; E_F = sum f omega; the block identities), then the whole program (run_main) with a synthetic
run json carrying a record at T 0 (E_0 12.5), the fields npz and synthetic first-route jsons that carry this route's own numbers under the
first route's key names, and checks the files, the E_0 handling, the v2 table and the npz; item 7 recomposes the v2 table from the outputs
alone. Writes only under TMPDIR."""
import json
import math
import os
import tempfile

import numpy as np
import scipy.linalg as sla
from scipy.integrate import simpson

import dust_tail_ops_v1 as OPS


def random_slater(rng, d, nf):
    X = rng.standard_normal((d, nf)) + 1j * rng.standard_normal((d, nf))
    Q, _ = np.linalg.qr(X)
    return Q


def fixture_fields(n):
    R = np.arange(1, n + 1, dtype=float)
    N = 2.0 - np.tanh((R - 32.5) / 3.0)          # 3 inside R <= 32, 1 outside: exterior levels with omega < 3 m are evanescent inside
    v = -0.2 / np.sqrt(1.0 + R / 16.0)
    return R, N, v


def synthetic_slice_v2(d, m, rng):
    n = 96
    R, N, v = fixture_fields(n)
    arrs = {"T": 128.0, "rs": 2.0, "R": R, "N_applied": N, "v_applied": v, "kappas": np.array([1, 2]), "weights": np.array([16.0, 32.0]), "dtype": "complex128"}
    known = {}
    for k in (1, 2):
        hk = OPS.killing_h(R, m, k, N, v); ek, Vk = sla.eigh(hk.toarray())
        pos = np.flatnonzero(ek > 0)
        w16 = np.sum(np.abs(Vk[64:, pos]) ** 2, axis=0)                       # the weight on the sites R > 32 (index 32 .. 95)
        j1 = int(pos[int(np.argmax(w16))]); j2 = int(pos[int(np.argmin(w16))])
        if k == 1:
            dust = Vk[:, [j1, j2]].copy(); fk = np.array([0.5, 1.0]); ej = ek[[j1, j2]]
        else:
            dust = Vk[:, [j1]].copy(); fk = np.array([0.5]); ej = ek[[j1]]
        arrs["sea_%d" % k] = Vk[:, ek < 0]; arrs["dust_%d" % k] = dust; arrs["f_%d" % k] = fk; arrs["e_%d" % k] = ek
        known[k] = {"e": ej, "f": fk, "dust": dust}
    path = os.path.join(d, "orbitals_selftest_v2_T0064.00.npz"); np.savez(path, **arrs)
    return path, arrs, known


def synthetic_run_v2(d, arrs, m, g=0.25):
    """a run json (with a record at T 0, E 12.5, and the slice's record) and a fields npz whose stored shell totals are the slice's own
    weighted dust densities (pairs zero)."""
    R = arrs["R"]; N = arrs["N_applied"]; v = arrs["v_applied"]; n = len(R)
    eps = np.zeros(n); J = np.zeros(n)
    for k, w in zip(arrs["kappas"], arrs["weights"]):
        h0, Q = OPS.chain_operators(R, m, int(k))
        eps += w * OPS.op_density(h0, arrs["dust_%d" % k], arrs["f_%d" % k]); J += w * OPS.op_density(Q, arrs["dust_%d" % k], arrs["f_%d" % k])
    tag = "selftest_v2"
    J_ = {"args": {"g": g, "rs": 2.0, "m": m, "nsites": n},
          "records": [{"T": 0.0, "E": 12.5, "E_dust_shells": None, "E_pairs_shells": None, "R_t": None, "min_N_minus_absv": None, "continuation_label": "synthetic T 0", "record_label": "synthetic"},
                      {"T": 128.0, "E": 10.0, "E_dust_shells": float(eps.sum()), "E_pairs_shells": 0.0, "R_t": None, "min_N_minus_absv": None, "continuation_label": "synthetic", "record_label": "synthetic"}]}
    json.dump(J_, open(os.path.join(d, "collapse_%s.json" % tag), "w"))
    np.savez(os.path.join(d, "collapse_%s_fields.npz" % tag), R=R, T=np.array([128.0]), N_applied=N[None, :], v_applied=v[None, :], eps_tot=eps[None, :], J_tot=J[None, :],
             eps_pairs=np.zeros((1, n)), J_pairs=np.zeros((1, n)))
    return os.path.join(d, tag)


def synthetic_first_route_v2(d, row):
    """first-route jsons in the layout the loaders read, filled with this route's numbers: the tail rows (weighted) and a D v2 batch json whose
    per-channel regions also carry S_hull, S_F, E_F, particles, holes, region_S_excess and escaped_dim (the key names dust_tail_table_v2 tries)."""
    W = row["weighted"]
    tail = {"rows": [{"T_over_rs": row["T_over_rs"], "F_dust_16_weighted": W["16.0"]["F"], "F_dust_16_per_channel": {str(c["kappa"]): c["beyond"]["16.0"]["F"] for c in row["channels"] if c["n_dust"]}}]}
    for c, key in (("16.0", "beyond_16"), ("24.0", "beyond_24")):
        q = W[c]
        tail["rows"][0][key] = {"dust_weight": q["N"], "particles_dust": q["particles"], "E_particles_dust": q["E_particles"], "holes_dust": -q["holes"], "E_out_dust": q["E_sym"],
                                "E_per_particle_over_m": q["E_per_particle_over_m"]}
    json.dump(tail, open(os.path.join(d, "tail_v2.json"), "w"))
    chans = []
    for c, c2 in zip(row["channels"], row["v2"]["channels"]):
        regions = {}
        for key in ("16.0", "24.0"):
            dv = c2["dev"][key]
            reg = {"S_hull": dv["S_F"], "S_F": dv["S_F"], "E_F": dv["E_F"], "particles": dv["particles_dev"], "holes": dv["holes_dev"], "region_S_excess": dv["region_excess"], "escaped_dim": dv["d_F"]}
            if c["n_dust"]:
                r = c["beyond"][key]
                reg.update({"dust_weight": r["N"], "particles_dust": r["particles"], "E_particles_dust": r["E_particles"], "holes_dust": -r["holes"], "E_holes_dust": r["E_holes"], "E_out_dust": r["E_sym"]})
            regions[key.rstrip("0").rstrip(".")] = reg
        chans.append({"kappa": c["kappa"], "w": c["w"], "regions": regions})
    dv2 = {"slices": [{"T_over_rs": row["T_over_rs"], "channels": chans}]}
    json.dump(dv2, open(os.path.join(d, "dv2_v2.json"), "w"))
    return os.path.join(d, "tail_v2.json"), os.path.join(d, "dv2_v2.json")


def selftest(MAIN):
    rng = np.random.default_rng(11)
    ok = True

    def chk(name, cond, val):
        nonlocal ok
        ok = ok and bool(cond)
        print("%s %s (%s)" % ("PASS" if cond else "FAIL", name, val))

    # 1 the entropy helpers
    chk("1a h2(0) = h2(1) = 0, h2(1/2) = ln 2", MAIN.h2(0.0) == 0.0 and MAIN.h2(1.0) == 0.0 and abs(MAIN.h2(0.5) - math.log(2)) < 1e-15, MAIN.h2(0.5))
    chk("1b spectral entropy of (1, 0, 1/2) is ln 2; of a projector 0", abs(MAIN.spectral_entropy([1.0, 0.0, 0.5]) - math.log(2)) < 1e-15 and MAIN.spectral_entropy(np.ones(7)) == 0.0,
        MAIN.spectral_entropy([1.0, 0.0, 0.5]))
    chk("1c d h2(n/d) equals sum h2 at equal occupations, exceeds it otherwise, and is 0 at d = 0",
        abs(MAIN.bound_dh2(1.2, 4) - 4 * MAIN.h2(0.3)) < 1e-14 and MAIN.spectral_entropy([0.5, 0.4, 0.2, 0.1]) <= MAIN.bound_dh2(1.2, 4) and MAIN.bound_dh2(0.3, 0) == 0.0, MAIN.bound_dh2(1.2, 4))
    # 2 the 5-bin partition
    m = 0.2; b = MAIN.band_top(m)
    chk("2a b = sqrt(m^2 + 1) at a = 1 (b/m 5.0990)", abs(b - math.sqrt(1.04)) < 1e-15 and abs(b / m - 5.0990195135927845) < 1e-12, b)
    om = np.array([-2.0, -1.02, -0.5, 0.0, 0.19999, 0.2, 1.0, b, 3.0]); nl = rng.uniform(0.1, 1.0, len(om))
    bins = MAIN.five_bins(om, nl, m, b)
    chk("2b the bins hold 2, 1, 2, 2, 2 levels (lower edges inclusive at -b, -m, m, b)", [x["levels"] for x in bins] == [2, 1, 2, 2, 2], [x["levels"] for x in bins])
    chk("2c the partition sums to the total n and E", abs(sum(x["n"] for x in bins) - nl.sum()) < 1e-14 and abs(sum(x["E"] for x in bins) - (om * nl).sum()) < 1e-14,
        (sum(x["n"] for x in bins), nl.sum()))
    om2 = rng.standard_normal(500) * 1.5; n2 = rng.uniform(0.0, 1.0, 500)
    bins2 = MAIN.five_bins(om2, n2, m, b)
    chk("2d 500 random levels: the 5 bins partition n and E; [m,b) + [b,inf) is the weight at or above m", abs(sum(x["n"] for x in bins2) - n2.sum()) < 1e-12
        and abs(sum(x["E"] for x in bins2) - (om2 * n2).sum()) < 1e-12 and abs(bins2[3]["n"] + bins2[4]["n"] - n2[om2 >= m].sum()) < 1e-12, sum(x["levels"] for x in bins2))
    # 3 the dust-only exterior block's entropy
    n = 64; p = 32
    c = np.zeros((2 * n, 1), complex); c[2 * 5, 0] = math.sqrt(0.7); c[2 * 40 + 1, 0] = math.sqrt(0.3)
    nu, S, Nn = MAIN.dust_exterior_entropy(c[2 * p:], [1.0])
    chk("3a one column with weight 0.3 beyond the cut: nu = 0.3, S = h2(0.3), N = 0.3", abs(nu[0] - 0.3) < 1e-14 and abs(S - MAIN.h2(0.3)) < 1e-14 and abs(Nn - 0.3) < 1e-14, (nu, S))
    c2_ = np.zeros((2 * n, 2), complex); c2_[2 * 3, 0] = math.sqrt(0.6); c2_[2 * 50, 0] = math.sqrt(0.4); c2_[2 * 4, 1] = math.sqrt(0.6); c2_[2 * 51 + 1, 1] = math.sqrt(0.4)
    nu2, S2, N2 = MAIN.dust_exterior_entropy(c2_[2 * p:], [1.0, 1.0])
    chk("3b two columns orthogonal beyond the cut with 0.4 each: S = 2 h2(0.4) = the rank bound 2 h2(N/2)", abs(S2 - 2 * MAIN.h2(0.4)) < 1e-14 and abs(S2 - MAIN.bound_dh2(N2, 2)) < 1e-14
        and abs(N2 - 0.8) < 1e-14, (nu2, S2))
    cols = random_slater(rng, 2 * n, 5); f = rng.uniform(0.2, 1.0, 5)
    nu3, S3, N3 = MAIN.dust_exterior_entropy(cols[2 * p:], f)
    Nsite = float(OPS.site_density(cols, f)[p:].sum())
    chk("3c random columns: sum nu = the site density beyond the cut; S <= the rank bound <= the dimension bound", abs(N3 - Nsite) < 1e-12 and S3 <= MAIN.bound_dh2(N3, 5) + 1e-12
        and MAIN.bound_dh2(N3, 5) <= MAIN.bound_dh2(N3, 2 * (n - p)) + 1e-12, (N3, S3, MAIN.bound_dh2(N3, 5), MAIN.bound_dh2(N3, 2 * (n - p))))
    # 4 the bound d h2(n/d) on random pure Gaussian states
    worst = 0.0; worst_rank = 0.0; purity = 0.0
    for _ in range(20):
        d_tot, nf, dY = 40, 13, 17
        Phi = random_slater(rng, d_tot, nf); perm = rng.permutation(d_tot); Y = perm[:dY]; X = perm[dY:]
        CY = Phi[Y] @ Phi[Y].conj().T; CX = Phi[X] @ Phi[X].conj().T
        SY_ = MAIN.spectral_entropy(MAIN.eigvals01(CY)); SX_ = MAIN.spectral_entropy(MAIN.eigvals01(CX)); nY = float(np.real(np.trace(CY)))
        worst = max(worst, SY_ / MAIN.bound_dh2(nY, dY)); worst_rank = max(worst_rank, SY_ / MAIN.bound_dh2(nY, nf)); purity = max(purity, abs(SY_ - SX_))
    chk("4a 20 random pure Slater states, 17 of 40 modes: S_Y <= d h2(n/d) and <= N_f h2(n/N_f); S_Y = S_X (purity)", worst <= 1 + 1e-12 and worst_rank <= 1 + 1e-12 and purity < 1e-9,
        (worst, worst_rank, purity))
    worst_F = 0.0; worst_blocks = 0.0
    for _ in range(30):
        d_tot, nf, dF, dp = 40, 13, 12, 5
        Phi = random_slater(rng, d_tot, nf); C = Phi @ Phi.conj().T
        Fm = random_slater(rng, d_tot, dF); M = Fm.conj().T @ C @ Fm
        dM = np.real(np.diag(M)); n_plus = float(dM[:dp].sum()); n_minus = float((dF - dp) - dM[dp:].sum())
        S_M = MAIN.spectral_entropy(MAIN.eigvals01(M))
        worst_F = max(worst_F, S_M / MAIN.bound_dh2(n_plus + n_minus, dF))
        worst_blocks = max(worst_blocks, S_M / (MAIN.bound_dh2(n_plus, dp) + MAIN.bound_dh2(n_minus, dF - dp)))
    chk("4b 30 random sectors F with a random split F_+, F_-: tr s(F^dag C F) <= d_+ h2(n_+/d_+) + d_- h2(n_-/d_-) <= d_F h2((n_+ + n_-)/d_F)", worst_F <= 1 + 1e-12 and worst_blocks <= 1 + 1e-12,
        (worst_F, worst_blocks))
    # 5 the number current's continuity identity
    R = np.arange(1, n + 1, dtype=float); N = 0.6 + 0.4 * R / n; v = -0.5 / np.sqrt(1.0 + R / 8.0)
    hm = OPS.killing_h(R, m, 2, N, v); H = hm.toarray()
    cols = rng.standard_normal((2 * n, 6)) + 1j * rng.standard_normal((2 * n, 6)); cols /= np.linalg.norm(cols, axis=0); f = rng.uniform(0.2, 1.0, 6)
    drho = MAIN.density_time_derivative(hm, cols, f); Fall = OPS.cut_currents_all(N, v, cols, f)
    chk("5a site continuity d rho_x/dt = F(x) - F(x+1) at every site (F(0) = F(n) = 0); sum_x d rho_x/dt = 0", MAIN.continuity_residual(drho, Fall) < 1e-12 and abs(drho.sum()) < 1e-12,
        (MAIN.continuity_residual(drho, Fall), drho.sum()))
    dt = 1e-4; Up = sla.expm(-1j * H * dt); Um = Up.conj().T
    cd = np.max(np.abs((OPS.site_density(Up @ cols, f) - OPS.site_density(Um @ cols, f)) / (2 * dt) - drho))
    chk("5b d rho/dt equals the exact evolution's central difference (dt %g)" % dt, cd < 1e-7, cd)
    T = 0.8; steps = 64; U = sla.expm(-1j * H * (T / steps)); cc = cols.copy(); Fs = [Fall[p - 1]]; N0 = float(OPS.site_density(cc, f)[p:].sum())
    for _ in range(steps):
        cc = U @ cc; Fs.append(OPS.cut_currents_all(N, v, cc, f)[p - 1])
    N1 = float(OPS.site_density(cc, f)[p:].sum()); integ = float(simpson(np.array(Fs), dx=T / steps))
    chk("5c the integrated form N_Y(T) - N_Y(0) = integral F dt along the exact evolution (T %g, Simpson over %d steps)" % (T, steps), abs(N1 - N0 - integ) < 1e-7, (N1 - N0, integ))
    # 6 process_slice_v2 and the whole program on the synthetic slice
    d = tempfile.mkdtemp(prefix="dust_selftest_v2_")
    path, arrs, known = synthetic_slice_v2(d, m, rng)
    eps_loc = 0.1
    row, arrays = MAIN.process_slice_v2(path, m, [16.0, 24.0], [8.0, 16.0, 24.0], "dust", True, eps_loc, log=lambda s: None)
    v2 = row["v2"]
    chk("6a version 1's readouts on the fixture: cuts at sites 32 and 48, sea density 1, no sea current, the stored spectrum reproduced from the v2 decomposition of every channel",
        row["cuts"][0]["first_site_index"] == 32 and row["cuts"][1]["first_site_index"] == 48 and row["checks"]["max_sea_density_dev"] < 1e-12 and row["weighted"]["16.0"]["max_sea_current"] < 1e-12
        and row["checks"]["max_spectrum_vs_stored"] is not None and row["checks"]["max_spectrum_vs_stored"] < 1e-11 and row["checks"]["n_negative"] == [96],
        (row["checks"]["max_sea_density_dev"], row["checks"]["max_spectrum_vs_stored"], row["checks"]["n_negative"]))
    okF = True; anyin = False; anyout = False; detail = []
    for c2, k in zip(v2["channels"], (1, 2)):
        kn = known[k]
        for key, p_ in (("16.0", 32), ("24.0", 48)):
            wj = np.sum(np.abs(kn["dust"][2 * p_:]) ** 2, axis=0); inF = wj >= 1.0 - eps_loc
            anyin = anyin or bool(inF.any()); anyout = anyout or bool((~inF).any())
            S_exp = float(sum(MAIN.h2(fj) for fj, i in zip(kn["f"], inF) if i)); n_exp = float(np.sum(kn["f"][inF])); E_exp = float(np.sum(kn["f"][inF] * kn["e"][inF]))
            dv = c2["dev"][key]; reg = c2["beyond"][key]
            good = (abs(dv["S_F"] - S_exp) < 1e-9 and abs(dv["n_plus"] - n_exp) < 1e-10 and abs(dv["n_minus"]) < 1e-10 and abs(dv["E_F"] - E_exp) < 1e-9 and abs(dv["n_plus_pairs"]) < 1e-10
                    and abs(dv["n_minus_pairs"]) < 1e-10 and abs(dv["S_F_dust"] - S_exp) < 1e-9 and dv["bound_F_respected"] and dv["bound_F_blocks_respected"]
                    and abs(reg["N"] - float(np.sum(kn["f"] * wj))) < 1e-12 and abs(dv["n_C_minus_n_P_minus_N"]) < 1e-9 and abs(dv["particles_dev"] - reg["particles"]) < 1e-9
                    and abs(dv["holes_dust_block"] - reg["holes"]) < 1e-9 and abs(dv["holes_dev"] - dv["holes_pairs"] + dv["holes_dust_block"]) < 1e-9   # builder fix (before any run): version 1's region "holes" is the dust's weight on the exterior block's negative levels, which is v2's holes_dust_block; v2's holes_dev = -sum over negative levels of the deviation = holes_pairs - holes_dust_block, so the first test's holes_dev = reg holes was a premise defect (it read 2 x the dust's negative-level weight)
                    and abs(dv["sum_ndev_minus_excess"]) < 1e-9 and dv["vacuum_plus_overlap_max"] < 1e-12 and dv["vacuum_minus_identity_dev"] < 1e-12
                    and abs(dv["particles_pairs"]) < 1e-9 and abs(dv["holes_pairs"]) < 1e-9)
            okF = okF and good
            detail.append((k, key, [round(float(x), 4) for x in wj], inF.tolist(), round(dv["S_F"], 9), round(S_exp, 9), dv["d_F"]))
    chk("6b the escaped sector on the fixture (the exact sea; dust = 2 positive eigenvectors with f 0.5 and 1): the pairs' counts 0; S_F = S_F,dust = sum over the columns in F of h2(f); "
        "n_+ = sum f; n_- = 0; E_F = sum f omega; the Killing-block deviation equals the dust's occupation; the block identities; both membership cases exercised", okF and anyin and anyout, detail)
    q16 = v2["weighted"]["16.0"]; q24 = v2["weighted"]["24.0"]
    chk("6c the 5-bin partition, nu sum = N, the continuity residual and the rank bound on the fixture (weighted)", max(q16["max_bins5_sum_n_minus_N"], q24["max_bins5_sum_n_minus_N"]) < 1e-12
        and max(q16["max_bins5_sum_E_minus_E_block"], q24["max_bins5_sum_E_minus_E_block"]) < 1e-10 and max(q16["max_nu_sum_minus_N"], q24["max_nu_sum_minus_N"]) < 1e-12
        and v2["max_continuity_residual"] < 1e-12 and q16["all_bound_rank_respected"] and q24["all_bound_rank_respected"] and abs(q16["S_dust"] - 16 * v2["channels"][0]["beyond"]["16.0"]["S_dust"]
                                                                                                                                 - 32 * v2["channels"][1]["beyond"]["16.0"]["S_dust"]) < 1e-12,
        (q16["max_bins5_sum_n_minus_N"], v2["max_continuity_residual"], q16["S_dust"], q16["bound_rank"]))
    run = synthetic_run_v2(d, arrs, m); tail, dv2 = synthetic_first_route_v2(d, row)
    out = os.path.join(d, "out")
    a = MAIN.default_args(slices=path, m=m, out_dir=out, tag="selftest_v2", run=run, first_route_tail_json=tail, first_route_dv2_jsons=dv2, deviation=True, eps_loc=eps_loc)
    rows = MAIN.run_main(a, log=lambda s: None)
    rc2 = rows[0]["run_checks_v2"]; w16 = rows[0]["v2"]["weighted"]["16.0"]; rc1 = rows[0]["run_checks"]
    chk("6d E_0 read from the record at T 0 (12.5), E_0/m = 62.5; the fraction of E_0 converts back to the count E_sym/m; version 1's run checks read zero",
        abs(rc2["E_0"] - 12.5) < 1e-12 and abs(rc2["E_0_over_m"] - 62.5) < 1e-12 and abs(w16["count_from_frac_E0"] - w16["E_sym_over_m"]) < 1e-10 and abs(w16["E_sym_frac_E0"] - w16["E_sym"] / 12.5) < 1e-14
        and max(rc1["applied_fields_max_dev"]) == 0.0 and abs(rc1["beyond"]["16.0"]["diff"]) < 1e-12 and abs(rc1["E_dust_shells_diff"]) < 1e-12, (rc2, w16["E_sym_frac_E0"]))
    files = [os.path.join(out, "selftest_v2_%s" % s) for s in ("T0064.00.json", "T0064.00.npz", "all.json", "table.md", "table_v2.md")]
    chk("6e the per-slice json and npz, the all json, version 1's table and the v2 table are written", all(os.path.exists(x) for x in files), files)
    md = open(files[-1]).read(); md1 = open(files[-2]).read()
    SFw = w16["dev"]["S_F"]
    chk("6f the v2 table prints the weighted S_F, the first route's S_hull beside with zero difference, E_0/m, and every section; version 1's table carries its table I (the pairs)",
        ("%.6e" % SFw) in md and "+0.000e+00" in md and "62.5000" in md and all(("## V2-%s" % s) in md for s in "ABCDEFGHI") and "## I." in md1, md.count("+0.000e+00"))
    Zo = np.load(files[1])
    need = ("nu_dust_16_1", "w_out_16_1", "F_plus_idx_16_1", "M_diag_16_1", "mu_F_16_1", "eig_CYY_16_2", "eig_PYY_24_1", "n_dev_24_2", "n_pairs_16_1", "drho_dust_1", "e_2", "ksym_pairs_1", "omega_16_1", "F_dust_1", "rho_sea_minus_1_2")
    chk("6g the npz holds version 1's arrays and v2's per-level arrays", all(x in Zo.files for x in need), [x for x in need if x not in Zo.files])
    # 7 the composer rereads only the outputs
    import dust_tail_table_v2 as TBL2
    out2, md2 = TBL2.compose_from_outputs(out, "selftest_v2", tail, dv2, out_path=os.path.join(d, "recomposed.md"))
    same = [l for l in md.splitlines() if l.startswith("|")] == [l for l in md2.splitlines() if l.startswith("|")]
    chk("7 dust_tail_table_v2 recomposes the same table rows from the outputs alone", same and os.path.exists(out2), out2)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok
