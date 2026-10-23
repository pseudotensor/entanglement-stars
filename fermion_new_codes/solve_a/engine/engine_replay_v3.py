#!/usr/bin/env python
"""engine_replay_v3.py (builder analyst; engine spec 24.1 (11): "engine_replay_v2 compares its masks bitwise with the stored masks
channel by channel and record by record wherever one is stored (every v19 record) ... it adds the Gauss-closure column of 20.1, the dust's
and the pairs' shell totals separately against the stored ones, and names the initial covariance as the stored C-init state itself"): the
replay of a stored engine_collapse_v19 run on its own applied-field history. A run resumed from a v17 run (its args' resume_from) is first
brought to its resume record by engine_collapse_v19.resume_state (the v17 segment replayed by v17's own rule from the C-init state, which is
re-prepared from the stored C-init fields N_cinit, v_cinit and is the replay's input; acceptance 1e-8 in the sources, as in the run); a run
from T = 0 starts from the same C-init state. Then every native record of the run: the eigendecompositions at the record's stored applied
fields (engine_collapse_v19.eigh_all), R2 continued by engine_ops_v2.continue_sea_v2 from the run's window (the accepted references and
their fields, window_max deep, appended after each record exactly as the run does), the sources from the propagated sea and dust against
the stored shell totals (the total, the pairs' part and the dust's part = total - pairs, each relative to its own largest value), the
occupation mask bitwise against the run's stored mask per channel, and against the negative-energy set; the stresses (engine_stress_v1,
the run's module) of the dust and of the pairs against the stored face flux F and centrifugal force f (relative to the largest stored value;
the pairs' absolute scale beside, small while the sea follows R2); <N^2>_n from the dense lapse re-solved from the stored shell totals by
engine_fields_v14 (the run's module) and with it the Gauss identity E = sum_n eps_n <N^2>_n (the stored E) and, per interval, the balance's
closure dE - (evolution + reset + lapse) (20.1's column; the parts as in engine_collapse_records_v1); then the propagation by the record's
eigendecomposition for tau. Writes --md (the table and the summary), --out-json (per record) and <out-npz>_handoff.npz (the replay's shell
quantities with v19's keys). Args: --run dir/tag --resume-from (override of the stored path) --gpu --records --md --out-json --out-npz."""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v19 as E19  # noqa: E402
import engine_ops_v2 as OPS2  # noqa: E402
import engine_fields_v14 as F14  # noqa: E402
import engine_stress_v1 as STR  # noqa: E402


def rel(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.max(np.abs(x - y))) / max(float(np.max(np.abs(y))), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume-from", default=None, help="the stored v17 run's dir/tag on this host (overrides the run's own path)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--tag", default=None, help="accepted for the GPU runner, unused")
    x = ap.parse_args()
    d, tag = os.path.split(x.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    H = np.load(os.path.join(d, "collapse_%s_handoff.npz" % tag), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    a = SimpleNamespace(**J["args"])
    if x.resume_from:
        a.resume_from = x.resume_from
    E19.OPT["collar_policy"] = a.collar_policy; E19.OPT["support_rel"] = a.support_rel; E19.OPT["site_fields"] = a.site_fields
    E19.C5.backend(x.gpu)
    torch = E19.BK["torch"]
    rs = float(a.rs); tau = float(a.tau)
    chs = [E19.Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    shell = [float(q) * rs for q in a.shell.split(",")] if a.shell else None
    t0 = time.time()
    resume = None
    if a.resume_from:
        st, f_ref, stored, head, resume = E19.resume_state(a, chs, R, shell)
        start_txt = "resumed: %d records of %s replayed by v17's rule to T/rs %.3f, the largest source differences eps %.2e, J %.2e (acceptance 1e-8); the C-init state re-prepared from the stored C-init fields (%.2e, %.2e)" % (
            resume["records_replayed"], resume["from"], resume["T_over_rs"], resume["max_eps_rel"], resume["max_J_rel"], resume["cinit"]["dN"], resume["cinit"]["dv"])
    else:
        Nc, vc, dust, info0, head0 = E19.prepare_dust(chs, a.g, R, a.ell, rs / (2.0 * a.g * a.m), a.R0_over_rs * rs, a.m, a.reading, shell)
        Ncs = np.asarray(Z["N_cinit"], float); vcs = np.asarray(Z["v_cinit"], float)
        st = {}
        for ch in chs:
            e, U = E19.eigT(E19.clocked_gpu(ch, Ncs, vcs))
            neg = e < 0
            Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=E19.BK["dev"]), np.zeros(0)))
            en = e.real.cpu().numpy(); kk = int(neg.sum())
            gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
            ref0 = U[:, neg].clone()
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": kk, "window": [{"ref": ref0, "f": (Ncs.copy(), vcs.copy()), "gap": gap}]}
            del e, U
        f_ref = (Ncs, vcs)
        start_txt = "from T = 0: the C-init state re-prepared, the stored C-init fields driving it (max |N - N_cinit| %.2e, |v - v_cinit| %.2e)" % (
            float(np.max(np.abs(Nc - Ncs))), float(np.max(np.abs(vc - vcs))))
        # the record at T = 0 is native too: its applied fields are the C-init fields; the propagation starts after it
    print(start_txt, flush=True)
    Tz = np.asarray(Z["T"], float); TS = np.asarray(S["T"], float); TH = np.asarray(H["T"], float)
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    cstr = {ch.kappa: STR.ChanStress(ch, torch, E19.BK["dev"]) for ch in chs}
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    r_nodes = F14.coarse_grid(R, a.ell)
    K = len(TS) if x.records is None else min(len(TS), x.records)
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc"):
            hand["%s_%s" % (q, part)] = []
    rows = []
    L = ["## %s replayed on its applied-field history (engine_replay_v3.py; %s, r_s %g, tau %g, ell %g, site fields %s, c_LZ %g, window %d)" % (
        tag, J.get("script"), rs, tau, a.ell, a.site_fields, a.c_LZ, a.window_max), "",
         "Start: %s." % start_txt, "",
         "| T/rs | eps, J (total) | eps, J (dust) | eps, J (pairs) | mask = stored (channels) | mask = negative set | F, f dust | F, f pairs (pairs' max abs F) | Gauss identity | interval closure | seconds |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    prev = None
    t_all = time.time()

    def dump():
        np.savez_compressed(x.out_npz + "_handoff.npz", R=R, r_nodes=r_nodes, g=a.g, ell=a.ell, m=a.m, tau=tau, kappas=np.array([ch.kappa for ch in chs]),
                            weights=np.array([ch.w for ch in chs]), **{k: np.array(v_) for k, v_ in hand.items()})
        json.dump({"script": "engine_replay_v3.py", "run": x.run, "start": start_txt, "resume": resume, "rows": rows}, open(x.out_json, "w"), indent=1, default=float)
    for i in range(K):
        tr = time.time()
        E19.tm_reset()
        T = float(TS[i])
        kz = int(np.argmin(np.abs(Tz - T)))
        kh = int(np.argmin(np.abs(TH - T)))
        if abs(Tz[kz] - T) > 1e-9 or abs(TH[kh] - T) > 1e-9:
            L.append("| %.3f | the stored fields or handoff hold no record at this time: stopped |" % (T / rs)); break
        rec = recs.get(round(T, 6), {})
        N_app = np.asarray(Z["N_applied"][kz], float); v_app = np.asarray(Z["v_applied"][kz], float)
        eig = E19.eigh_all(chs, N_app, v_app)
        cache = {}
        nmask_equal = 0; mask_neg = True; mask_diff_ch = []
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            ref, ci, mask = OPS2.continue_sea_v2(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], tau, a.c_LZ, E19.eig_at, E19.clocked_host_sparse, a.window_max)
            mask = np.asarray(mask, bool)
            smask = np.unpackbits(np.asarray(S["mask_%d" % ch.kappa][i], np.uint8))[:mask.size].astype(bool)
            if np.array_equal(mask, smask):
                nmask_equal += 1
            else:
                mask_diff_ch.append("%d (%d bits)" % (ch.kappa, int(np.sum(mask != smask))))
            mask_neg = mask_neg and bool(np.array_equal(mask, (e < 0).cpu().numpy()))
            cache[ch.kappa] = (e, U, ref, ci)
        del eig
        eps = np.zeros(a.nsites); Jt = np.zeros(a.nsites); epsP = np.zeros(a.nsites); JP = np.zeros(a.nsites)
        eps_sea = np.zeros(a.nsites); eps_R2 = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites)
        Pd = Pp = None
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            hs_ = E19.dens(ch.h, s_["sea"]); hr_ = E19.dens(ch.h, ref); hd_ = E19.dens(ch.h, s_["dust"], s_["f"])
            qs_ = E19.dens(ch.Q, s_["sea"]); qr_ = E19.dens(ch.Q, ref); qd_ = E19.dens(ch.Q, s_["dust"], s_["f"])
            eps += ch.w * (hs_ - hr_ + hd_); Jt += ch.w * (qs_ - qr_ + qd_); epsP += ch.w * (hs_ - hr_); JP += ch.w * (qs_ - qr_)
            eps_sea += ch.w * hs_; eps_R2 += ch.w * hr_; eps_dust += ch.w * hd_
            cs = cstr[ch.kappa]
            pd_ = dict(STR.stress_parts(cs, s_["dust"], s_["f"]))
            pp_ = dict(STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])])))
            Pd = {q: ch.w * np.asarray(v_, float) for q, v_ in pd_.items()} if Pd is None else {q: Pd[q] + ch.w * np.asarray(pd_[q], float) for q in Pd}
            Pp = {q: ch.w * np.asarray(v_, float) for q, v_ in pp_.items()} if Pp is None else {q: Pp[q] + ch.w * np.asarray(pp_[q], float) for q in Pp}
        es = np.asarray(Z["eps_tot"][kz], float); Js = np.asarray(Z["J_tot"][kz], float)
        esP = np.asarray(Z["eps_pairs"][kz], float); JsP = np.asarray(Z["J_pairs"][kz], float)
        d_tot = (rel(eps, es), rel(Jt, Js)); d_dust = (rel(eps - epsP, es - esP), rel(Jt - JP, Js - JsP)); d_pairs = (rel(epsP, esP), rel(JP, JsP))
        SVd = STR.shell_values(Pd, R, a.m); SVp = STR.shell_values(Pp, R, a.m)
        s_dust = (rel(SVd["F_face"], H["F_face_dust"][kh]), rel(Pd["fc"], H["fc_dust"][kh]))
        s_pairs = (rel(SVp["F_face"], H["F_face_pairs"][kh]), rel(Pp["fc"], H["fc_pairs"][kh]))
        pmax = float(np.max(np.abs(np.asarray(H["F_face_pairs"][kh], float))))
        _, _, _, inf_ = F14.solve_constraints_ivp(es, Js, a.g, R, a.ell, collar_policy=E19.OPT["collar_policy"], support_rel=E19.OPT["support_rel"])
        if inf_.get("failed"):
            N2 = np.full(a.nsites, np.nan)
        else:
            N2 = F14.site_fields_kernel_average({"N_dense": (lambda r_: inf_["N_dense"](r_) ** 2), "v_dense": zero}, R, a.ell)[0]
        E = float(rec.get("E", np.nan))
        gid = (E - float(np.sum(es * N2))) / abs(E) if E == E else float("nan")
        eC = eps_sea + eps_dust
        clos = None
        if prev is not None:
            a_ = 0.5 * (prev["N2"] + N2)
            evo = float(np.sum(a_ * (eC - prev["eC"]))); rst = -float(np.sum(a_ * (eps_R2 - prev["eR"])))
            lap = float(np.sum((N2 - prev["N2"]) * 0.5 * (prev["eps"] + es)))
            clos = (E - prev["E"]) - (evo + rst + lap)
        prev = {"N2": N2, "eC": eC, "eR": eps_R2, "eps": es, "E": E}
        hand["T"].append(T); hand["N2avg"].append(N2); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(eps_dust)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            s_["window"].append({"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap_between_blocks")})
            s_["window"] = s_["window"][-a.window_max:]
            ph = torch.exp(-1j * e.to(torch.complex128) * tau)
            prop = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        del cache
        row = {"T_over_rs": T / rs, "eps_rel": d_tot[0], "J_rel": d_tot[1], "dust_rel": list(d_dust), "pairs_rel": list(d_pairs), "masks_equal_channels": nmask_equal,
               "masks_differ": mask_diff_ch, "mask_equals_negative_set": mask_neg, "stress_dust_rel": list(s_dust), "stress_pairs_rel": list(s_pairs), "pairs_F_max_abs": pmax,
               "gauss_identity": gid, "interval_closure": clos, "solve_failed": inf_.get("failed"), "seconds": time.time() - tr}
        rows.append(row)
        L.append("| %.3f | %.1e, %.1e | %.1e, %.1e | %.1e, %.1e | %d of %d%s | %s | %.1e, %.1e | %.1e, %.1e (%.1e) | %+.2e | %s | %.1f |" % (
            T / rs, d_tot[0], d_tot[1], d_dust[0], d_dust[1], d_pairs[0], d_pairs[1], nmask_equal, len(chs), (" (differ: %s)" % "; ".join(mask_diff_ch)) if mask_diff_ch else "",
            mask_neg, s_dust[0], s_dust[1], s_pairs[0], s_pairs[1], pmax, gid, ("%+.2e" % clos) if clos is not None else "-", row["seconds"]))
        print(L[-1], flush=True)
        if i % 10 == 0 or i == K - 1:
            dump()
    n = len(rows)
    eq_all = sum(1 for r in rows if r["masks_equal_channels"] == len(chs))
    w = lambda key, j=None: max(((r[key][j] if j is not None else r[key]) for r in rows), default=float("nan"))
    L += ["", "- %d native records replayed in %.0f s (from T/rs %.3f to %.3f); the masks equal the stored masks bitwise in every channel on %d of %d records; the masks equal the negative-energy set on %d records." % (
        n, time.time() - t_all, rows[0]["T_over_rs"] if rows else float("nan"), rows[-1]["T_over_rs"] if rows else float("nan"), eq_all, n, sum(1 for r in rows if r["mask_equals_negative_set"])),
          "- the largest source differences from the stored shell totals (each relative to its own largest value): total eps %.2e, J %.2e; dust eps %.2e, J %.2e; pairs eps %.2e, J %.2e (acceptance 1e-8)." % (
              w("eps_rel"), w("J_rel"), w("dust_rel", 0), w("dust_rel", 1), w("pairs_rel", 0), w("pairs_rel", 1)),
          "- the stresses against the stored ones (relative to the largest stored value): dust F %.2e, f %.2e; pairs F %.2e, f %.2e." % (
              w("stress_dust_rel", 0), w("stress_dust_rel", 1), w("stress_pairs_rel", 0), w("stress_pairs_rel", 1)),
          "- the Gauss identity E = sum_n eps_n <N^2>_n with the stored E and the replay's <N^2>_n (the dense lapse re-solved from the stored shell totals by engine_fields_v14): the largest |error| %.2e; the interval closure dE - (evolution + reset + lapse): the largest |value| %.2e." % (
              max((abs(r["gauss_identity"]) for r in rows if r["gauss_identity"] == r["gauss_identity"]), default=float("nan")),
              max((abs(r["interval_closure"]) for r in rows if r["interval_closure"] is not None), default=float("nan"))),
          "- the mask: an eigenvector of h_m at the applied fields is occupied when it is in R2's assignment (engine_ops_v2.continue_sea_v2's mask), compared bit by bit with the run's packed mask; every number above is printed by the script."]
    dump()
    open(x.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
