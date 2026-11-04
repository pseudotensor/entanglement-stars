#!/usr/bin/env python
"""engine_replay_v2.py (builder analyst; engine spec 20.1 and 21.4: COLLAPSE_records_v1 from the exact replay of a stored
engine_collapse_v17 run): engine_replay_v1's loop (the initial data re-prepared, the stored applied fields per record, R2 continued by the
run's own rule, the sources compared with the stored shell totals, the state propagated with the same eigendecomposition) plus, per record
on the replayed state, the handoff quantities that engine_collapse_v19 records natively (20.1): the stresses of engine_stress_v1 (the
Noether flux of the lane's declaration) for the dust and the pairs apart (the face flux F, the centrifugal force f, the mass-term density
b, the kappa-weighted sigma_y), the balance identity's largest difference and the virial check, the smoothed S_r,ell and S_t,ell of the
total at the coarse nodes with the trace identity's error, the K_0 averages <N^2>_n of the dense lapse re-solved from the stored shell
totals by the run's own field module (engine_fields_v12 for v17), the shell energies of the propagated sea and of R2 apart, per channel the
occupation mask (packed bits) and the eigenvalues; written to <out>_handoff.npz and <out>_spectra.npz with v19's keys, the exactness table
and the per-record summaries to --md and --out-json; the orbitals at --slices as engine_replay_v1. Args: --run dir/tag --gpu --slices
--save-dir --dtype --records --md --out-json --out-npz."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v17 as E17  # noqa: E402
import engine_fields_v12 as F12  # noqa: E402
import engine_stress_v1 as STR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--slices", default="")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--dtype", default="complex64", choices=["complex64", "complex128"])
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True, help="prefix: <prefix>_handoff.npz and <prefix>_spectra.npz")
    ap.add_argument("--tag", default=None, help="accepted for the GPU runner, unused")
    a = ap.parse_args()
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ra = J["args"]
    E17.OPT["collar_policy"] = ra.get("collar_policy", "flag"); E17.OPT["support_rel"] = ra.get("support_rel", 1e-14)
    E17.OPT["site_fields"] = ra.get("site_fields", "dense")
    E17.C5.backend(a.gpu)
    torch = E17.BK["torch"]
    rs = float(ra["rs"]); m = float(ra["m"]); g = float(ra["g"]); ell = float(ra["ell"]); tau = float(ra["tau"]); nsites = int(ra["nsites"])
    chs = [E17.Chan(k, m, nsites, bool(ra.get("half_offset", 0))) for k in range(1, int(ra["kappa_sea"]) + 1)]
    R = chs[0].R
    shell = [float(x) * rs for x in ra["shell"].split(",")] if ra.get("shell") else None
    t0 = time.time()
    Nc, vc, dust, info, head = E17.prepare_dust(chs, g, R, ell, rs / (2.0 * g * m), float(ra["R0_over_rs"]) * rs, m, ra["reading"], shell)
    Ncs = np.asarray(Z["N_cinit"], float); vcs = np.asarray(Z["v_cinit"], float)
    cinit_dN = float(np.max(np.abs(Nc - Ncs))); cinit_dv = float(np.max(np.abs(vc - vcs)))
    print("the initial data re-prepared (%.0f s): max |N - N_cinit| %.2e, max |v - v_cinit| %.2e" % (time.time() - t0, cinit_dN, cinit_dv), flush=True)
    E17.tm_reset()
    st = {}
    for ch in chs:
        e, U = E17.eigT(E17.clocked_gpu(ch, Ncs, vcs))
        neg = e < 0
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * nsites, 0), dtype=torch.complex128, device=E17.BK["dev"]), np.zeros(0)))
        en = e.real.cpu().numpy(); kk = int(neg.sum())
        st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": U[:, neg].clone(), "n_sea": kk,
                        "gap_between_prev": float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None,
                        "levels_prev": (np.asarray(en[:kk], float), np.asarray(en[kk:], float), 0)}
        del e, U
    cstr = {ch.kappa: STR.ChanStress(ch, torch, E17.BK["dev"]) for ch in chs}
    r_nodes = F12.coarse_grid(R, ell)
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    f_ref = (Ncs, vcs)
    Tz = np.asarray(Z["T"], float)
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    slices = [float(x) for x in a.slices.split(",") if x.strip()]
    if slices and a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    K = len(Tz) if a.records is None else min(len(Tz), a.records)
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc", "b", "sy_k"):
            hand["%s_%s" % (q, part)] = []
    spectra = {"T": []}
    for ch in chs:
        spectra["mask_%d" % ch.kappa] = []; spectra["e_%d" % ch.kappa] = []
    rows = []
    L = ["## %s replayed with the handoff quantities (engine_replay_v2.py; %s, r_s %g, tau %g, ell %g, site fields %s)" % (tag, J.get("script"), rs, tau, ell, E17.OPT["site_fields"]), "",
         "The initial data re-prepared: max |N - N_cinit| %.2e, max |v - v_cinit| %.2e; the stored initial and applied fields drive the replay; the dense lapse for <N^2>_n re-solved from the stored shell totals by engine_fields_v12 (the run's own module)." % (cinit_dN, cinit_dv), "",
         "| T/rs | eps rel. diff. | J rel. diff. | stress identity dust, pairs | virial rel. dust, pairs | trace identity | mask = negative set | orbitals | seconds |", "|---|---|---|---|---|---|---|---|---|"]
    t_all = time.time()

    def dump():
        np.savez_compressed(a.out_npz + "_handoff.npz", R=R, r_nodes=r_nodes, g=g, ell=ell, m=m, tau=tau, kappas=np.array([ch.kappa for ch in chs]),
                            weights=np.array([ch.w for ch in chs]), **{k: np.array(v_) for k, v_ in hand.items()})
        np.savez(a.out_npz + "_spectra.npz", **{k: np.array(v_) for k, v_ in spectra.items()})
        json.dump({"script": "engine_replay_v2.py", "run": a.run, "cinit": {"dN": cinit_dN, "dv": cinit_dv}, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    for k in range(K):
        tr = time.time()
        E17.tm_reset()
        T = float(Tz[k]); rec = recs.get(round(T, 6), {})
        N_app = np.asarray(Z["N_applied"][k], float); v_app = np.asarray(Z["v_applied"][k], float)
        force = bool(rec.get("verification_differs"))
        eig = E17.eigh_all(chs, N_app, v_app)
        cache = {}
        why = None
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            ref, ci = E17.continue_v17(ch, s_["ref"], f_ref, (N_app, v_app), e, U, s_["n_sea"], s_.get("gap_between_prev"), s_.get("levels_prev"), force_subdivide=force)
            if ref is None:
                why = "kappa %d: the continuation failed (%s)" % (ch.kappa, ci.get("why"))
                break
            cache[ch.kappa] = (e, U, ref, ci)
        if why:
            L.append("| %.3f | REPLAY STOPPED: %s |" % (T / rs, why)); print(why, flush=True)
            break
        eps = np.zeros(nsites); Jt = np.zeros(nsites); eps_sea = np.zeros(nsites); eps_R2 = np.zeros(nsites); eps_dust = np.zeros(nsites)
        Pd = Pp = None
        mask_neg = True
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            hs_ = E17.dens(ch.h, s_["sea"]); hr_ = E17.dens(ch.h, ref); hd_ = E17.dens(ch.h, s_["dust"], s_["f"])
            eps += ch.w * (hs_ - hr_ + hd_); eps_sea += ch.w * hs_; eps_R2 += ch.w * hr_; eps_dust += ch.w * hd_
            Jt += ch.w * (E17.dens(ch.Q, s_["sea"]) - E17.dens(ch.Q, ref) + E17.dens(ch.Q, s_["dust"], s_["f"]))
            cs = cstr[ch.kappa]
            pd_ = dict(STR.stress_parts(cs, s_["dust"], s_["f"]))
            pp_ = dict(STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])])))
            pd_["sy_k"] = ch.kappa * pd_["sy"]; pp_["sy_k"] = ch.kappa * pp_["sy"]
            Pd = {q: ch.w * np.asarray(x, float) for q, x in pd_.items()} if Pd is None else {q: Pd[q] + ch.w * np.asarray(pd_[q], float) for q in Pd}
            Pp = {q: ch.w * np.asarray(x, float) for q, x in pp_.items()} if Pp is None else {q: Pp[q] + ch.w * np.asarray(pp_[q], float) for q in Pp}
            # the occupation mask over the eigenbasis at the applied fields: an eigenvector is occupied when its weight in R2 exceeds 1/2
            wv = ((ref.conj().T @ U).abs() ** 2).sum(dim=0).cpu().numpy()
            mk = wv > 0.5
            mask_neg = mask_neg and bool(np.array_equal(mk, (e < 0).cpu().numpy()))
            spectra.setdefault("mask_%d" % ch.kappa, []).append(np.packbits(mk))
            spectra.setdefault("e_%d" % ch.kappa, []).append(e.cpu().numpy().astype(np.float64))
        spectra["T"].append(T)
        es = np.asarray(Z["eps_tot"][k], float); Js = np.asarray(Z["J_tot"][k], float)
        de = float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300)
        dj = float(np.max(np.abs(Jt - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
        SVd = STR.shell_values(Pd, R, m); SVp = STR.shell_values(Pp, R, m)
        Sr_l, St_l, tr_err = STR.smooth_stress(SVd["S_r"] + SVp["S_r"], SVd["S_t"] + SVp["S_t"], R, ell, r_nodes)
        _, _, _, inf_ = F12.solve_constraints_ivp(es, Js, g, R, ell, collar_policy=E17.OPT["collar_policy"], support_rel=E17.OPT["support_rel"])
        N2avg = F12.site_fields_kernel_average({"N_dense": (lambda r_: inf_["N_dense"](r_) ** 2), "v_dense": zero}, R, ell)[0] if not inf_.get("failed") else np.full(nsites, np.nan)
        vir = {}
        for lab_, SV in (("dust", SVd), ("pairs", SVp)):
            sc = max(float(np.max(np.abs(SV["virial_rhs"]))), 1e-300)
            vir[lab_] = float(np.max(np.abs(SV["virial_lhs"] - SV["virial_rhs"]))) / sc
        hand["T"].append(T); hand["N2avg"].append(np.asarray(N2avg, float)); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(eps_dust)
        hand["S_r_ell"].append(Sr_l); hand["S_t_ell"].append(St_l)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
            hand["b_%s" % part].append(np.asarray(P_["b"], float)); hand["sy_k_%s" % part].append(np.asarray(P_["sy_k"], float))
        saved = None
        if a.save_dir and any(abs(T / rs - sl) < 1e-6 for sl in slices):
            dt = np.complex64 if a.dtype == "complex64" else np.complex128
            arrs = {"T": T, "rs": rs, "N_applied": N_app, "v_applied": v_app, "R": R, "kappas": np.array([ch.kappa for ch in chs]), "weights": np.array([ch.w for ch in chs])}
            for ch in chs:
                e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
                arrs["sea_%d" % ch.kappa] = s_["sea"].cpu().numpy().astype(dt)
                arrs["dust_%d" % ch.kappa] = s_["dust"].cpu().numpy().astype(dt)
                arrs["f_%d" % ch.kappa] = np.asarray(s_["f"], float)
                arrs["R2_%d" % ch.kappa] = ref.cpu().numpy().astype(dt)
                arrs["e_%d" % ch.kappa] = e.cpu().numpy()
            saved = os.path.join(a.save_dir, "orbitals_%s_T%07.2f.npz" % (tag, T / rs))
            np.savez(saved, **arrs)
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            if "occ_e" in ci:
                s_["levels_prev"] = (ci.pop("occ_e"), ci.pop("emp_e"), ci.get("partial_pairs", 0))
            if ci.get("gap_between_blocks") is not None or "weyl" in ci:
                s_["gap_between_prev"] = ci.get("gap_between_blocks")
            ph = torch.exp(-1j * e.to(torch.complex128) * tau)
            prop = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        f_ref = (N_app, v_app)
        del cache, eig
        row = {"T_over_rs": T / rs, "eps_rel": de, "J_rel": dj, "identity_max_abs": [SVd["identity_max_abs"], SVp["identity_max_abs"]], "virial_rel": [vir["dust"], vir["pairs"]],
               "trace_identity_rel": tr_err, "mask_equals_negative_set": mask_neg, "saved": saved, "seconds": time.time() - tr,
               "F_total_last_face": float(SVd["F_face"][-1] + SVp["F_face"][-1]), "solve_failed": inf_.get("failed")}
        rows.append(row)
        L.append("| %.3f | %.2e | %.2e | %.1e, %.1e | %.1e, %.1e | %.1e | %s | %s | %.1f |" % (T / rs, de, dj, SVd["identity_max_abs"], SVp["identity_max_abs"], vir["dust"], vir["pairs"],
                                                                              tr_err, mask_neg, os.path.basename(saved) if saved else "-", row["seconds"]))
        print(L[-1], flush=True)
        if k % 10 == 0 or k == K - 1:
            dump()
    worst_e = max((r["eps_rel"] for r in rows), default=float("nan")); worst_j = max((r["J_rel"] for r in rows), default=float("nan"))
    L += ["", "- %d records replayed in %.0f s; the largest source differences from the stored shell totals: eps %.2e, J %.2e (relative to their largest values; acceptance 1e-8)." % (
        len(rows), time.time() - t_all, worst_e, worst_j),
          "- the mask column compares the occupied set (weight in R2 above 1/2 per eigenvector of h_m at the applied fields) with the negative-energy set, bitwise.",
          "- every number above is printed by the script."]
    dump()
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
