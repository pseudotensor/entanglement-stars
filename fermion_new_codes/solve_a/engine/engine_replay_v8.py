#!/usr/bin/env python
"""engine_replay_v8.py (builder analyst; engine spec 31.5: "engine_replay_v8 (a new file: v7 with engine_collapse_v27, v29, v30 and
v31 mapped to the v25 modules and ops v3, the preparation driven by the stored C-init fields as v7 does; the gate bitwise v7 on three
records of a v25 run)"): version 7 with the runs of engine_collapse_v27, v29, v30 and v31 (the fixed-preparation ladder's engines:
version 25 with --prep-ell, the field tolerances, the damped and the Newton-Krylov solves as options) replayed by engine_collapse_v25's
functions, engine_fields_v16, engine_stress_v2 and ops v3, and the dust prepared at the run's --prep-ell where it has one (the run's
own preparation scale: version 27 prepared the initial data at prep_ell and evolved at ell; version 7 prepared at ell), the sea still
built from the stored C-init fields. The replay re-solves no self-consistent iteration, so the run's solver (plain, damped or Newton) does
not enter. Nothing else changes; version 7's docstring follows.
engine_replay_v7.py (builder analyst): version 6 with the start line of a resumed v25 run: engine_collapse_v25.resume_state_v25
returns its v19 parent's resume (native records replayed, sources, masks) with the v17 grandparent's inside it ("parent_resume"), not
version 19's keys, so version 6 stopped at the start line after a correct resume (the c_LZ 0.5 extension on the second H100, 07:24 UTC);
here the line reads whichever form the resume returns. Nothing else changes; version 6's docstring follows.
engine_replay_v6.py (builder analyst): version 5 with the run's own continuation: an engine_collapse_v25 run continues R2 by
engine_ops_v3.continue_sea_v3 from a one-entry window that carries the Weyl gap and the mask (as v25 does), and a resumed v25 run is
brought to its resume record by engine_collapse_v25.resume_state_v25; v19 and v24 runs keep engine_ops_v2.continue_sea_v2 with their
window. (Version 5 replays every run with ops v2 and must not be used on a v25 run.) Nothing else changes; version 5's docstring follows.
engine_replay_v5.py (builder analyst; engine spec 28.6: "The ell 8 control's records are delivered as COLLAPSE_records_v1 in the set
members' form (the replay's exactness table added when box GPU 1 frees after vf10, \"replay pending\" until then)"). Version 5 = version 4
with the run's own modules: the engine, the field module and the stress module are chosen by the stored run's script (its json's "script":
engine_collapse_v19 -> engine_fields_v14, engine_stress_v1; engine_collapse_v24 and engine_collapse_v25 -> engine_fields_v16,
engine_stress_v2, the modules those versions import), or by --engine/--fields/--stress; the dense lapse of the Gauss identity is
re-solved by that field module. Nothing else changes; version 4's docstring follows.
engine_replay_v4.py (builder analyst; engine spec 28.2, S1 of REVIEWA_REVIEW_web40e: "engine_replay_v4 (new; GPU) prints [the moving-reference
terms] with their N- and v-weighted sums beside the lane's reset term ... Collar budget per interval (engine_replay_v4): M_R ..., N_R Delta
M_R, the cut current times Delta T as F_R, the remainder printed as the measured D_R ... beside the reset term, Tr(h_k Delta P_k) (the
record's Hamiltonian ... expected near zero) and Tr(h_R Delta P_k) (weights zeroed outside R); the first-adiabatic residual F_R[C^(1)] + D_R
with C^(1)_ab = i P_dot_ab/(eps_a - eps_b) on opposite-occupation pairs per channel (P_dot = Delta P/Delta t), pairs below delta dropped at
delta 1e-6 and 1e-4 (both printed), their weight the diagnosed equal-energy failure"). Version 4 = version 3 (unchanged replay of a v19 run;
its checks and table kept) plus, at every native record after the first: (a) per shell the changes Delta eps_R2 and Delta J_R2 of R2's
energy and current densities since the previous record (Tr(Delta P a_x(h)), Tr(Delta P a_x(Q)) with a_x(B) = {e_x, B}/2, the engine's site
densities) and their weighted sums sum_x [N_x Delta eps_R2,x + v_x Delta J_R2,x] with the record's applied fields, which equal Tr(h_k
Delta P_k) (the density route), beside the same trace by the record's spectrum (the spectral route: sum_a e_a [(U^+ P_k U)_aa - (U^+ P_k-1
U)_aa]); (b) Tr(h_R Delta P_k): the density route over the sites inside the record's stored cut (M1_parts' cut index; the engine's inside
set, sites below the cut); (c) the first-adiabatic collar current F_R[C^(1)] per channel at the record: C^(1) = U X U^+, X_ab = i P_dot_ab/
(e_a - e_b) on the pairs of opposite occupation by the record's mask (P_dot = (P_k - P_k-1)/tau in the record's eigenbasis), pairs with
|e_a - e_b| below delta dropped (delta 1e-6 and 1e-4; the dropped weight sum |P_dot_ab|^2 printed), the current by the engine's
energy_cut_current form (the h^2 window of +-8 sites about the cut) with the density matrix in place of the occupied columns; (d) from the
stored records: R the cut site's radius, M_R = R w^2/(2 g) from the stored site fields, N_R the interval mean, F_R Delta T the stored
trapezoid of the collar's Killing cut current (M1), D_R Delta T = -N_R Delta M_R - F_R Delta T, and the two residuals F_R[C^(1)] + D_R
(D_R the measured rate) and F_R[C^(1)] + Tr(h_R Delta P_k)/tau, with the Gauss reset of version 3's closure beside. A run resumed from a
v17 parent: the columns start at its first native record (the previous reference the resume record's). Writes version 3's outputs and the
columns per record (json, md table) and per shell (the handoff npz: deps_R2, dJ_R2, J_R2). Version 3's docstring follows.
engine_replay_v3.py (builder analyst; engine spec 24.1 (11): "engine_replay_v2 compares its masks bitwise with the stored masks
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
import importlib  # noqa: E402

import engine_ops_v2 as OPS2  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402

# the run's engine -> (engine, field module, stress module), as each engine version imports them
MODULES = {"engine_collapse_v19": ("engine_collapse_v19", "engine_fields_v14", "engine_stress_v1"),
           "engine_collapse_v24": ("engine_collapse_v24", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v25": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           # version 8: the fixed-preparation ladder's engines, version 25's modules (engine spec 31.5)
           "engine_collapse_v27": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v29": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v30": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v31": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2")}
CONT = {"engine_collapse_v19": "ops v2", "engine_collapse_v24": "ops v2", "engine_collapse_v25": "ops v3"}  # each engine's own continuation
E19 = F14 = STR = None  # set in main from the stored run's script


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
    ap.add_argument("--engine", default=None); ap.add_argument("--fields", default=None); ap.add_argument("--stress", default=None)
    x = ap.parse_args()
    d, tag = os.path.split(x.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    H = np.load(os.path.join(d, "collapse_%s_handoff.npz" % tag), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    a = SimpleNamespace(**J["args"])
    global E19, F14, STR
    run_eng = os.path.basename(str(J.get("script", "engine_collapse_v19.py"))).replace(".py", "")
    if run_eng not in MODULES and not (x.engine and x.fields and x.stress):
        raise SystemExit("engine_replay_v5: the run's script %s has no module set; pass --engine, --fields and --stress" % run_eng)
    eng_m, fld_m, str_m = MODULES.get(run_eng, (None, None, None))
    eng_m = x.engine or eng_m; fld_m = x.fields or fld_m; str_m = x.stress or str_m
    E19 = importlib.import_module(eng_m); F14 = importlib.import_module(fld_m); STR = importlib.import_module(str_m)
    cont = CONT.get(os.path.basename(eng_m), "ops v2")
    print("engine_replay_v8: the run's script %s; replaying with %s, %s, %s; the continuation %s" % (run_eng, eng_m, fld_m, str_m, cont), flush=True)
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
        st, f_ref, stored, head, resume = (E19.resume_state_v25 if cont == "ops v3" else E19.resume_state)(a, chs, R, shell)
        if "records_replayed" in resume:
            start_txt = "resumed: %d records of %s replayed by v17's rule to T/rs %.3f, the largest source differences eps %.2e, J %.2e (acceptance 1e-8); the C-init state re-prepared from the stored C-init fields (%.2e, %.2e)" % (
                resume["records_replayed"], resume["from"], resume["T_over_rs"], resume["max_eps_rel"], resume["max_J_rel"], resume["cinit"]["dN"], resume["cinit"]["dv"])
        else:
            pr = resume.get("parent_resume") or {}
            start_txt = ("resumed (v25 from a v19 parent): the grandparent's %s records replayed by v17's rule to T/rs %s (sources eps %s, J %s; the C-init state %s, %s); "
                         "then %d native records of %s replayed by its own rule to T/rs %.3f, the largest source differences eps %.2e, J %.2e, the masks equal in %d of %d channel-records (acceptance 1e-8)") % (
                pr.get("records_replayed", "-"), ("%.3f" % pr["T_over_rs"]) if pr.get("T_over_rs") is not None else "-", ("%.2e" % pr["max_eps_rel"]) if pr.get("max_eps_rel") is not None else "-",
                ("%.2e" % pr["max_J_rel"]) if pr.get("max_J_rel") is not None else "-", ("%.2e" % pr["cinit"]["dN"]) if pr.get("cinit") else "-", ("%.2e" % pr["cinit"]["dv"]) if pr.get("cinit") else "-",
                resume["native_records_replayed"], resume["from"], resume["T_over_rs"], resume["native_max_eps_rel"], resume["native_max_J_rel"], resume["native_masks_equal"][0], resume["native_masks_equal"][1])
    else:
        prep_ell = getattr(a, "prep_ell", None) or a.ell  # version 8: the run's own preparation scale
        Nc, vc, dust, info0, head0 = E19.prepare_dust(chs, a.g, R, prep_ell, rs / (2.0 * a.g * a.m), a.R0_over_rs * rs, a.m, a.reading, shell)
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
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "J_R2": [], "deps_R2": [], "dJ_R2": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc"):
            hand["%s_%s" % (q, part)] = []
    rows = []
    L = ["## %s replayed on its applied-field history (engine_replay_v8.py, version 3's table; %s with %s and %s, the continuation %s; r_s %g, tau %g, ell %g, site fields %s, c_LZ %g, window %d)" % (
        tag, J.get("script"), fld_m, str_m, cont, rs, tau, a.ell, a.site_fields, a.c_LZ, a.window_max), "",
         "Start: %s." % start_txt, "",
         "| T/rs | eps, J (total) | eps, J (dust) | eps, J (pairs) | mask = stored (channels) | mask = negative set | F, f dust | F, f pairs (pairs' max abs F) | Gauss identity | interval closure | seconds |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    prev = None
    t_all = time.time()
    s1rows = []
    prevS1 = None
    T_ = lambda X: torch.as_tensor(X, device=E19.BK["dev"])

    def cut_current_D(Hhost, DW, cut, lo, hi):
        """the engine's energy_cut_current form with a density-matrix window DW (rows and columns lo:hi) in place of the occupied columns."""
        h2W = T_((Hhost[lo:hi, :] @ Hhost[:, lo:hi]).toarray()).to(torch.complex128)
        P = torch.zeros(hi - lo, dtype=torch.complex128, device=E19.BK["dev"]); P[: 2 * cut - lo] = 1.0
        Cm = h2W * P[None, :] - P[:, None] * h2W
        return float((0.5j * (DW.T * Cm).sum()).real)

    def dump():
        np.savez_compressed(x.out_npz + "_handoff.npz", R=R, r_nodes=r_nodes, g=a.g, ell=a.ell, m=a.m, tau=tau, kappas=np.array([ch.kappa for ch in chs]),
                            weights=np.array([ch.w for ch in chs]), **{k: np.array(v_) for k, v_ in hand.items()})
        json.dump({"script": "engine_replay_v8.py", "modules": [eng_m, fld_m, str_m], "continuation": cont, "run": x.run, "start": start_txt, "resume": json.loads(json.dumps(resume, default=lambda o: float(o) if np.isscalar(o) else str(o))), "rows": rows, "s1": s1rows}, open(x.out_json, "w"), indent=1, default=float)
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
            if cont == "ops v3":
                ref, ci, mask = OPS3.continue_sea_v3(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], tau, a.c_LZ, E19.eig_at, E19.clocked_host_sparse)
            else:
                ref, ci, mask = OPS2.continue_sea_v2(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], tau, a.c_LZ, E19.eig_at, E19.clocked_host_sparse, a.window_max)
            mask = np.asarray(mask, bool)
            ci["_mask"] = mask
            smask = np.unpackbits(np.asarray(S["mask_%d" % ch.kappa][i], np.uint8))[:mask.size].astype(bool)
            if np.array_equal(mask, smask):
                nmask_equal += 1
            else:
                mask_diff_ch.append("%d (%d bits)" % (ch.kappa, int(np.sum(mask != smask))))
            mask_neg = mask_neg and bool(np.array_equal(mask, (e < 0).cpu().numpy()))
            cache[ch.kappa] = (e, U, ref, ci)
        del eig
        eps = np.zeros(a.nsites); Jt = np.zeros(a.nsites); epsP = np.zeros(a.nsites); JP = np.zeros(a.nsites)
        eps_sea = np.zeros(a.nsites); eps_R2 = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_R2 = np.zeros(a.nsites)
        Pd = Pp = None
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            hs_ = E19.dens(ch.h, s_["sea"]); hr_ = E19.dens(ch.h, ref); hd_ = E19.dens(ch.h, s_["dust"], s_["f"])
            qs_ = E19.dens(ch.Q, s_["sea"]); qr_ = E19.dens(ch.Q, ref); qd_ = E19.dens(ch.Q, s_["dust"], s_["f"])
            eps += ch.w * (hs_ - hr_ + hd_); Jt += ch.w * (qs_ - qr_ + qd_); epsP += ch.w * (hs_ - hr_); JP += ch.w * (qs_ - qr_)
            eps_sea += ch.w * hs_; eps_R2 += ch.w * hr_; eps_dust += ch.w * hd_; J_R2 += ch.w * qr_
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
        clos = None; clos_parts = None
        if prev is not None:
            a_ = 0.5 * (prev["N2"] + N2)
            evo = float(np.sum(a_ * (eC - prev["eC"]))); rst = -float(np.sum(a_ * (eps_R2 - prev["eR"])))
            lap = float(np.sum((N2 - prev["N2"]) * 0.5 * (prev["eps"] + es)))
            clos = (E - prev["E"]) - (evo + rst + lap)
            clos_parts = {"evolution": evo, "reset": rst, "lapse": lap}
        prev = {"N2": N2, "eC": eC, "eR": eps_R2, "eps": es, "E": E}
        # engine 28.2 (S1): the moving reference since the previous record, the collar budget and the first-adiabatic current
        s1 = None
        cutk = int(rec["M1_parts"]["cut"]) if rec.get("M1_parts") else None
        if prevS1 is not None and cutk is not None:
            t_s1 = time.time()
            deps = eps_R2 - prevS1["eps_R2"]; dJ = J_R2 - prevS1["J_R2"]
            trh_dens = float(np.sum(N_app * deps + v_app * dJ))
            trhR = float(np.sum((N_app * deps + v_app * dJ)[:cutk]))
            trh_spec = 0.0; FC1 = {1e-6: 0.0, 1e-4: 0.0}; drop = {1e-6: 0.0, 1e-4: 0.0}; pd_tot = 0.0
            for ch in chs:
                e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
                en = e.real.to(torch.float64)
                Ak = U.conj().T @ ref; Bk = U.conj().T @ s_["ref"]
                dk = (Ak.abs() ** 2).sum(dim=1) - (Bk.abs() ** 2).sum(dim=1)
                trh_spec += ch.w * float((en * dk).sum())
                occ = torch.as_tensor(np.asarray(ci["_mask"], bool), device=E19.BK["dev"])
                oi = torch.nonzero(occ).ravel(); ei = torch.nonzero(~occ).ravel()
                if len(oi) == 0 or len(ei) == 0:
                    continue
                Pdot = ((Ak[oi] @ Ak[ei].conj().T) - (Bk[oi] @ Bk[ei].conj().T)) / tau
                pd_tot += ch.w * float((Pdot.abs() ** 2).sum())
                de = en[oi][:, None] - en[ei][None, :]
                Hk = E19.clocked_host_sparse(ch, N_app, v_app)
                lo, hi = 2 * max(cutk - 8, 0), 2 * min(cutk + 8, Hk.shape[0] // 2)
                UWo = U[lo:hi][:, oi]; UWe = U[lo:hi][:, ei]
                for dl in (1e-6, 1e-4):
                    keep = de.abs() >= dl
                    X = torch.where(keep, 1j * Pdot / torch.where(keep, de, torch.ones_like(de)).to(Pdot.dtype), torch.zeros_like(Pdot))
                    drop[dl] += ch.w * float((Pdot.abs() ** 2 * (~keep)).sum())
                    DW = UWo @ X @ UWe.conj().T; DW = DW + DW.conj().T
                    FC1[dl] += ch.w * cut_current_D(Hk, DW, cutk, lo, hi)
                del Ak, Bk, Pdot, de
            NRk = float(Z["N"][kz][cutk]); vRk = float(Z["v"][kz][cutk]); Rk = float(R[cutk])
            MR = Rk * (vRk / NRk) ** 2 / (2.0 * a.g)
            NR = 0.5 * (NRk + prevS1["N_at"](cutk)); MR0 = prevS1["M_at"](cutk)
            bal = rec.get("M1_balance") or {}
            FRdT = float(bal.get("evolution_trapezoid", float("nan")))
            DRdT = -NR * (MR - MR0) - FRdT
            rst_g = clos_parts.get("reset") if isinstance(clos_parts, dict) else None
            s1 = {"T_over_rs": T / rs, "cut": cutk, "R": Rk, "tr_h_dP_spectral": trh_spec, "tr_h_dP_density": trh_dens, "tr_hR_dP": trhR,
                  "F_C1_1e-6": FC1[1e-6], "F_C1_1e-4": FC1[1e-4], "dropped_1e-6": drop[1e-6], "dropped_1e-4": drop[1e-4], "P_dot_opposite_weight": pd_tot,
                  "N_R": NR, "dM_R": MR - MR0, "F_R_dT": FRdT, "D_R_dT": DRdT, "D_R_rate": DRdT / tau,
                  "residual_C1_plus_DR_1e-6": FC1[1e-6] + DRdT / tau, "residual_C1_plus_DR_1e-4": FC1[1e-4] + DRdT / tau,
                  "residual_C1_plus_trhR_1e-6": FC1[1e-6] + trhR / tau, "gauss_reset": rst_g, "seconds": time.time() - t_s1}
            s1rows.append(s1)
        prevS1 = {"eps_R2": eps_R2.copy(), "J_R2": J_R2.copy(), "N_at": (lambda c, kz=kz: float(Z["N"][kz][c])),
                  "M_at": (lambda c, kz=kz: float(R[c]) * (float(Z["v"][kz][c]) / float(Z["N"][kz][c])) ** 2 / (2.0 * a.g))}
        hand["J_R2"].append(J_R2)
        hand["deps_R2"].append(eps_R2 - (hand["eps_R2"][-1] if hand["eps_R2"] else eps_R2)); hand["dJ_R2"].append(J_R2 - (hand["J_R2"][-2] if len(hand["J_R2"]) > 1 else J_R2))
        hand["T"].append(T); hand["N2avg"].append(N2); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(eps_dust)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            if cont == "ops v3":
                s_["window"].append({"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap_between_blocks"),
                                     "weyl_gap": ci.get("weyl_gap", ci.get("gap_between_blocks")), "mask": np.asarray(ci["_mask"], bool)})
                s_["window"] = s_["window"][-1:]
            else:
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
          "- the Gauss identity E = sum_n eps_n <N^2>_n with the stored E and the replay's <N^2>_n (the dense lapse re-solved from the stored shell totals by %s): the largest |error| %.2e; the interval closure dE - (evolution + reset + lapse): the largest |value| %.2e." % (
              fld_m,
              max((abs(r["gauss_identity"]) for r in rows if r["gauss_identity"] == r["gauss_identity"]), default=float("nan")),
              max((abs(r["interval_closure"]) for r in rows if r["interval_closure"] is not None), default=float("nan"))),
          "- the mask: an eigenvector of h_m at the applied fields is occupied when it is in R2's assignment (the run's own continuation's mask), compared bit by bit with the run's packed mask; every number above is printed by the script."]
    L += ["", "### Engine 28.2 (S1): the moving reference, the collar budget and the first-adiabatic current per interval", "",
          "| T/rs | Tr(h dP) spectral | Tr(h dP) density | Tr(h_R dP) | F_R[C1] (1e-6) | F_R[C1] (1e-4) | dropped weight (1e-6, 1e-4) | N_R dM_R | F_R dT | D_R dT | F_R[C1] + D_R (1e-6) | F_R[C1] + Tr(h_R dP)/tau | Gauss reset |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g_ = lambda v_, fm: (fm % v_) if v_ is not None and v_ == v_ else "-"
    for q in s1rows:
        L.append("| %.3f | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %.2e, %.2e | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %s |" % (
            q["T_over_rs"], q["tr_h_dP_spectral"], q["tr_h_dP_density"], q["tr_hR_dP"], q["F_C1_1e-6"], q["F_C1_1e-4"], q["dropped_1e-6"], q["dropped_1e-4"],
            q["N_R"] * q["dM_R"], q["F_R_dT"], q["D_R_dT"], q["residual_C1_plus_DR_1e-6"], q["residual_C1_plus_trhR_1e-6"], g_(q["gauss_reset"], "%+.4e")))
    if s1rows:
        sm = lambda key: float(np.nansum([q[key] for q in s1rows if q[key] is not None]))
        L += ["", "- sums over the %d intervals: Tr(h dP) spectral %+.6e, density %+.6e; Tr(h_R dP) %+.6e; N_R dM_R %+.6e; F_R dT %+.6e; D_R dT %+.6e; tau F_R[C1] (1e-6) %+.6e, (1e-4) %+.6e; the Gauss reset %+.6e." % (
            len(s1rows), sm("tr_h_dP_spectral"), sm("tr_h_dP_density"), sm("tr_hR_dP"), float(np.nansum([q["N_R"] * q["dM_R"] for q in s1rows])), sm("F_R_dT"), sm("D_R_dT"),
            tau * sm("F_C1_1e-6"), tau * sm("F_C1_1e-4"), sm("gauss_reset")),
              "- every number above is printed by the script; the identification of D_R with Tr(H_R P_dot) and of the first adiabatic correction's current is the review's 1.26, tested here, not assumed."]
    dump()
    open(x.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
