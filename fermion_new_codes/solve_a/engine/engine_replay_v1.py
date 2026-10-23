#!/usr/bin/env python
"""engine_replay_v1.py (builder analyst; the owner's handoffs addendum of 22:05 UTC, item 1: "if the running primary's saved state allows
these to be computed afterwards, do that"): an engine_collapse_v17 run's state replayed exactly from its stored history, record by record:
the initial data re-prepared (engine_collapse_v17.prepare_dust with the run's arguments: the dust's orbitals; its fields checked against
the stored N_cinit, v_cinit), the initial sea the negative eigenvectors of h_m at the stored initial fields; then per record k the stored
applied fields (N_applied, v_applied: the Hamiltonian that continued the reference and propagated the state), the channels' eigendecomposition
there, R2 continued from the previous record's applied fields by v17's own rule (continue_v17, the subdivided path where the record lists a
verification difference), the sources (the sea's and the dust's densities against R2 by v17's definitions) compared with the stored shell
totals eps_tot and J_tot (the exactness check, printed per record), the occupied orbitals saved at the requested slices (the sea, the dust
with its occupations and R2's columns per channel, in --dtype, with the applied fields), and the state propagated by exp(-i h_m tau) with the
same eigendecomposition. The stress densities and the evolution rows' residuals enter as version 2 once the lead fixes the operators.
Args: --run dir/tag --gpu --slices (T/rs list) --save-dir --dtype --records --md --out-json."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v17 as E17  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--slices", default="", help="T/rs values at which the orbitals are saved (comma list)")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--dtype", default="complex64", choices=["complex64", "complex128"])
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
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
    print("the initial data re-prepared (%.0f s): max |N - N_cinit| %.2e, max |v - v_cinit| %.2e (the stored fields are used below)" % (time.time() - t0, cinit_dN, cinit_dv), flush=True)
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
    f_ref = (Ncs, vcs)
    Tz = np.asarray(Z["T"], float)
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    slices = [float(x) for x in a.slices.split(",") if x.strip()]
    if slices and a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    K = len(Tz) if a.records is None else min(len(Tz), a.records)
    rows = []
    L = ["## %s replayed (engine_replay_v1.py; %s, r_s %g, tau %g, ell %g, site fields %s)" % (tag, J.get("script"), rs, tau, ell, E17.OPT["site_fields"]), "",
         "The initial data re-prepared: max |N - N_cinit| %.2e, max |v - v_cinit| %.2e; the stored initial and applied fields drive the replay." % (cinit_dN, cinit_dv), "",
         "| T/rs | max |eps - eps_tot| / max |eps_tot| | max |J - J_tot| / max |J_tot| | continuation (fast, blocks, subdivided) | orbitals saved | seconds |", "|---|---|---|---|---|---|"]
    t_all = time.time()
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
        eps = np.zeros(nsites); Jt = np.zeros(nsites)
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            eps += ch.w * (E17.dens(ch.h, s_["sea"]) - E17.dens(ch.h, ref) + E17.dens(ch.h, s_["dust"], s_["f"]))
            Jt += ch.w * (E17.dens(ch.Q, s_["sea"]) - E17.dens(ch.Q, ref) + E17.dens(ch.Q, s_["dust"], s_["f"]))
        es = np.asarray(Z["eps_tot"][k], float); Js = np.asarray(Z["J_tot"][k], float)
        de = float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300)
        dj = float(np.max(np.abs(Jt - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
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
        modes = {}
        for ch in chs:
            mode = (cache[ch.kappa][3].get("mode") or "-").split(" ")[0]
            modes[mode] = modes.get(mode, 0) + 1
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
        row = {"T_over_rs": T / rs, "eps_rel": de, "J_rel": dj, "modes": modes, "saved": saved, "seconds": time.time() - tr}
        rows.append(row)
        L.append("| %.3f | %.2e | %.2e | %s | %s | %.1f |" % (T / rs, de, dj, ", ".join("%s %d" % (k_, v_) for k_, v_ in sorted(modes.items())),
                                                          os.path.basename(saved) if saved else "-", row["seconds"]))
        print(L[-1], flush=True)
        if k % 10 == 0 or k == K - 1:
            json.dump({"script": "engine_replay_v1.py", "run": a.run, "cinit": {"dN": cinit_dN, "dv": cinit_dv}, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    worst_e = max((r["eps_rel"] for r in rows), default=float("nan")); worst_j = max((r["J_rel"] for r in rows), default=float("nan"))
    L += ["", "- %d records replayed in %.0f s; the largest source differences from the stored shell totals: eps %.2e, J %.2e (relative to their largest values)." % (
        len(rows), time.time() - t_all, worst_e, worst_j), "- every number above is printed by the script."]
    json.dump({"script": "engine_replay_v1.py", "run": a.run, "cinit": {"dN": cinit_dN, "dv": cinit_dv}, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
