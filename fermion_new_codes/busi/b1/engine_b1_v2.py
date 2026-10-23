#!/usr/bin/env python
"""engine_b1_v2.py (busi lane, B1 build: the self-consistent small-object evaporation; inbox #309, #315; design
busi/notes/B1_design_reviewa_v1.md (reviewer A), review deliveries/lead_gaps/REVIEWA_B1_design_review_v1.md (write lane, BUILD WITH MUST-FIXES), both
applied: M1 to M8, R1, R2, R3, R4, R5, R8, R9).

engine_b1_v2 = engine_collapse_v19.run() (md5 9e4e6abf, lines 1077 to 1580) copied verbatim into run_b1() with marked hooks (every added or
changed line carries "# B1 H<k>"), plus the light channel of engine_b1_light_v2 (page_history_v6q's massless chain on its own long chain, its
Gauss source normal-ordered against its own continued sea on a truncated chain, fed back into the heavy sources). With --light off and
--io v19 the run is v19's arithmetic and v19's files (test 4.1, bitwise); with --io chunked the same arithmetic and the chunked files of M7.

Hooks (anchors in v19's run()):
 H0  after R = chs[0].R: the LightSet and the B1 extras (schedule, I/O, resume, stop);
 H1  after cstr = ...: --b1-resume (the b1 state) or LightSet.start (the T_0 vacua, Delta_0, Delta S_0, the conventions of 4.2 and M8);
     H1s after nsteps = ...: the record schedule (--tau-schedule, M3 a); H1r after unconv_state: the sticky labels restored on resume;
 H2  after the base block: the light sources at record k - 1's accepted fields and charge (the lagged light reference);
 H3  inside sources(cache_): eps += eps_light, J += J_light (--light-ref-iterate 1: the trial's reference);
 H4  M1: the light's old-configuration terms (before f_ref is replaced) and its new terms, I_E^L, the light balance;
 H5  the handoff: the light stresses (interior factorization) into S_r, S_t and the hand arrays;
     H6s before the Simpson block: the next interval's tau (schedule);
 H6  before recs.append(rec): the light record (M2 ledger, stop rule), the readouts (M5), N_heavy_in (4.9), the label (R9);
 H7  after the heavy propagation: the light's step (flows, L_at_t, propagate_lean) and the b1 state saves (M7);
 H8  the writes: v19's json and save_fields with --io v19, the chunked writer (M7) otherwise; the stop rule, --benchmark-records,
     --T-end-before-return.

Also: --ladder LIST (2.2.1, the trapping-threshold ladder with the light off; R5's conservation readout per run), main() with v19's options
verbatim plus B1's. The self-tests are engine_b1_tests_v2.py.

v2 over v1 (e070c01e): imports engine_b1_light_v2 (the M2 a, M2 b, M6 and M8 gaps closed there), the option --light-init-offload; the copied
loop and the hooks are unchanged (test 4.1 is rerun on v2).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import time

for _p in (os.environ.get("B1_PATHS") or "<internal>/code_v6").split(":"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp

import engine_collapse_v19 as E19
import engine_collapse_v5 as C5
import engine_fields_v14 as FLD
import engine_mixing_v1 as MIX
import engine_ops_v2 as OPS2
import engine_stress_v1 as STR

# v19's module names, bound to v19's objects (the same dicts OPT, TM, BK: v19's helpers read and write them)
BK = E19.BK
T_ = E19.T_
eigT = E19.eigT
dens = E19.dens
Chan = E19.Chan
logistic = E19.logistic
fill_lowest = E19.fill_lowest
threshold_radii = E19.threshold_radii
energy_cut_current = E19.energy_cut_current
OPT = E19.OPT
TM = E19.TM
solve = E19.solve
kernel_identities = E19.kernel_identities
tail_screen = E19.tail_screen
indicator = E19.indicator
prepare_dust = E19.prepare_dust
surface_jets = E19.surface_jets
sync_dev = E19.sync_dev
tm_reset = E19.tm_reset
eigT_t = E19.eigT_t
clocked_gpu = E19.clocked_gpu
eigh_all = E19.eigh_all
commutator_norm = E19.commutator_norm
eig_at = E19.eig_at
clocked_host_sparse = E19.clocked_host_sparse
dev_csr = E19.dev_csr
inside_energy_sp = E19.inside_energy_sp
energy_cut_current_sp = E19.energy_cut_current_sp
resume_state = E19.resume_state

ENGINE = "engine_b1_v2"


def _jdefault(o):
    try:
        if np.ndim(o) == 0:
            return float(o)
        return np.asarray(o).tolist()
    except Exception:                                                              # noqa: BLE001
        return str(o)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Schedule:
    """M3 a: tau = --tau to --tau-switch-rs, then --tau-late (records at the accumulated clocks); nsteps from the schedule."""

    def __init__(self, a, rs):
        self.t1 = float(a.tau); self.t2 = float(a.tau_late); self.Tsw = float(a.tau_switch_rs) * rs
        n1 = int(round(min(self.Tsw, a.T_over_rs * rs) / self.t1))
        rest = a.T_over_rs * rs - n1 * self.t1
        self.nsteps = n1 + max(0, int(round(rest / self.t2)))
        self.n1 = n1
        self.label = "record step %g to T/r_s %g, then %g (--tau-schedule; the heavy by exact phases, the light by one Chebyshev step per record)" % (self.t1, a.tau_switch_rs, self.t2)

    def tau_next(self, step):
        return self.t1 if step < self.n1 else self.t2


def heavy_number(chs, st, cache, cut):
    """4.9: N_heavy_in = sum_kappa w [sum_j f_j ||P_in dust_j||^2 + ||P_in sea||_F^2 - ||P_in ref||_F^2] (P_in the first cut sites)."""
    n2 = 2 * cut
    dust = pairs = 0.0
    for ch in chs:
        s_ = st[ch.kappa]; ref = cache[ch.kappa][3]
        if s_["dust"].shape[1]:
            dust += ch.w * float(((s_["dust"][:n2].abs() ** 2).sum(dim=0).cpu().numpy() * np.asarray(s_["f"], float)).sum())
        pairs += ch.w * (float((s_["sea"][:n2].abs() ** 2).sum()) - float((ref[:n2].abs() ** 2).sum()))
    return {"total": dust + pairs, "dust": dust, "pairs": pairs, "cut": cut}


LEAN_KEYS = ("T", "T_over_rs", "E", "E_over_E0", "E_dust_shells", "E_pairs_shells", "pairs_inside_Rt", "R_t", "r_in", "min_N_minus_absv", "kappa_T", "N_min",
             "R_sh", "R_c", "n_pocket", "min_occupied_weight", "nsub_max", "M1_I_E", "H_in", "iterations", "eigendecompositions", "converged", "unconverged",
             "iteration_history", "continuation_label", "mask_equals_negative_set", "inversions_total", "seconds", "timing", "M1_balance", "dE_dT",
             "light", "M1_light", "N_heavy_in", "tau_record", "record_label", "engine", "b1_wall_s")   # v2: b1_wall_s (the D2 wall row reads it)


class ChunkIO:
    """M7: records appended as jsonl (the lean dict per record, the full v19 dict at the readout clocks), a summary json every --json-every
    records, the fields, handoff and spectra arrays in chunks of --fields-chunk-records records (v19's keys plus eps_light, J_light), a manifest
    with md5 per file."""

    def __init__(self, a):
        self.a = a; self.out = a.out; self.tag = a.tag
        self.jl = os.path.join(a.out, "collapse_%s_records.jsonl" % a.tag)
        self.man = os.path.join(a.out, "collapse_%s_manifest.tsv" % a.tag)
        self.chunk = 0; self.n_rec = 0
        if a.b1_resume:
            st_ = json.load(open(a.b1_resume + ".io.json")) if os.path.exists(a.b1_resume + ".io.json") else {}
            self.chunk = int(st_.get("chunk", 0)); self.n_rec = int(st_.get("n_rec", 0))

    def manifest(self, p):
        with open(self.man, "a") as fh:
            fh.write("%s\t%s\t%d\t%s\n" % (os.path.basename(p), md5_file(p), os.path.getsize(p), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    def record(self, rec, full):
        d = rec if (full or self.a.lean_records == "off") else {k_: rec.get(k_) for k_ in LEAN_KEYS if k_ in rec}
        with open(self.jl, "a") as fh:
            fh.write(json.dumps(d, default=_jdefault) + "\n")
        self.n_rec += 1

    def flush(self, R, fields, hand, spectra, meta, final=False):
        if not fields["T"] or (len(fields["T"]) < self.a.fields_chunk_records and not final):
            return None
        p = os.path.join(self.out, "collapse_%s_fields_%04d.npz" % (self.tag, self.chunk))
        arr = {}
        for k_, v_ in fields.items():
            if k_ in ("r_nodes", "N_nodes", "v_nodes"):
                arr[k_] = np.array(v_, dtype=object)
            else:
                arr[k_] = np.array(v_)
        np.savez_compressed(p, R=R, **arr, **meta)
        self.manifest(p)
        if hand.get("T"):
            ph = os.path.join(self.out, "collapse_%s_handoff_%04d.npz" % (self.tag, self.chunk))
            np.savez_compressed(ph, R=R, **{k_: np.array(v_) for k_, v_ in hand.items()})
            self.manifest(ph)
        if spectra.get("T"):
            ps = os.path.join(self.out, "collapse_%s_spectra_%04d.npz" % (self.tag, self.chunk))
            np.savez(ps, **{k_: np.array(v_) for k_, v_ in spectra.items()})
            self.manifest(ps)
        for D in (fields, hand, spectra):
            for k_ in D:
                D[k_].clear()
        self.chunk += 1
        return p

    def summary(self, a, head, recs, extra=None):
        p = os.path.join(self.out, "collapse_%s.json" % self.tag)
        with open(p, "w") as fh:
            json.dump({"script": "engine_b1_v2.py", "args": vars(a), "head": head, "n_records": self.n_rec, "records_tail": recs[-5:], "jsonl": os.path.basename(self.jl),
                       "chunks": self.chunk, "extra": extra or {}}, fh, default=_jdefault)
        return p


class B1Extras:
    """the B1 options that act on the copied loop: the schedule, the chunked I/O, the readout clocks, the b1 state saves and resume, the stops."""

    def __init__(self, a, rs):
        self.a = a; self.rs = rs
        self.sched = Schedule(a, rs) if a.tau_schedule == "on" else None
        io = a.io if a.io != "auto" else ("chunked" if a.light == "on" else "v19")
        self.io = ChunkIO(a) if io == "chunked" else None
        self.next_ro = 0.0; self.next_rows = 0.0; self.next_imp = 0.0; self.next_contour = 0.0; self.next_save = None
        self.saves = []; self.resumed = None; self.t_start = time.time(); self.walls = []
        self.label = ("engine_b1_v2 over engine_collapse_v19; light sources fed back (B1); " if a.light == "on" else "engine_b1_v2 over engine_collapse_v19; light off; ")

    def due(self, T, key, every_rs):
        if every_rs is None or every_rs <= 0:
            return False
        nxt = getattr(self, key)
        if T + 1e-9 >= nxt * self.rs:
            setattr(self, key, (math.floor(T / self.rs / every_rs + 1e-9) + 1) * every_rs)
            return True
        return False

    def save_state(self, a, st, f_ref, recs, step_next, T_next, head, N_cinit, v_cinit, LIGHT, tau_next, labels, force=False):
        import torch
        if not a.b1_save_every_rs or a.b1_save_every_rs <= 0:
            return None
        if self.next_save is None:
            self.next_save = (math.floor(T_next / self.rs / a.b1_save_every_rs + 1e-9) + 1) * a.b1_save_every_rs
            if not force:
                return None
        if not force and T_next + 1e-9 < self.next_save * self.rs:
            return None
        self.next_save = (math.floor(T_next / self.rs / a.b1_save_every_rs + 1e-9) + 1) * a.b1_save_every_rs
        st_ = os.statvfs(a.out); free_gb = st_.f_bavail * st_.f_frsize / 1e9
        if free_gb < a.disk_min_gb:
            print("B1: the b1 state at T/r_s %.3f NOT saved: %.1f GB free below --disk-min-gb %g" % (T_next / self.rs, free_gb, a.disk_min_gb), flush=True)
            return None
        p = os.path.join(a.out, "collapse_%s_b1state_T%08.2f.pt" % (a.tag, T_next / self.rs))
        d = {"st": st, "f_ref": f_ref, "recs": recs[-16:], "step_next": step_next, "T_next": T_next, "head": head, "N_cinit": N_cinit, "v_cinit": v_cinit,
             "tau_next": tau_next, "labels": labels, "light": (LIGHT.state_dict() if LIGHT is not None else None), "engine": ENGINE}
        torch.save(d, p)
        if self.io is not None:
            json.dump({"chunk": self.io.chunk, "n_rec": self.io.n_rec}, open(p + ".io.json", "w"))
            self.io.manifest(p)
        self.saves.append(p)
        while len(self.saves) > a.b1_keep_states:
            old = self.saves.pop(0)
            for q in (old, old + ".io.json"):
                if os.path.exists(q):
                    os.remove(q)
        print("B1: the b1 state saved at T/r_s %.3f: %s (%.1f GB free)" % (T_next / self.rs, p, free_gb), flush=True)
        return p

    def resume(self, a, LIGHT):
        import torch
        d = torch.load(a.b1_resume, map_location=BK["dev"], weights_only=False)
        if LIGHT is not None:
            LIGHT.load_state_dict(d["light"])
        a.tau = d["tau_next"]
        self.resumed = d
        self.next_save = None
        print("B1: resumed from %s at T/r_s %.3f (record %d)" % (a.b1_resume, d["T_next"] / self.rs, d["step_next"]), flush=True)
        return d


def run_b1(a):                                                                  # B1: v19 run() copied verbatim (lines 1077 to 1580), hooks marked
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)
    label = ("quasi-static (tau = %g); ell = %d; reference: %s (%s); site fields: %s; engine_fields_v14 (one outward initial-value problem, the momentum "
             "equation kept everywhere, the flat-slice gauge continued and labelled); economy: lagged first iterate, stop |dE/E| < %g and fields < %g, "
             "%s iteration, at most %d iterations (the smallest-residual iterate returned, flagged); tau = %g; collar policy %s; %s") % (
        a.tau, a.ell, a.reference, ("the continued sea, Landau-Zener matched maximum overlap (c_LZ = %g, window %d)" % (a.c_LZ, a.window_max)) if a.reference == "R2" else "the propagated sea",
        a.site_fields, a.stop_E, a.stop_fields, a.mix, a.maxit, a.tau, OPT["collar_policy"], FLD.DICTIONARY)
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m)
    chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    LIGHT = None                                                                                   # B1 H0
    if a.light == "on":                                                                            # B1 H0
        import engine_b1_light_v2 as B1L                                                           # B1 H0
        LIGHT = B1L.LightSet(a, R, chs, E19)                                                       # B1 H0
    B1X = B1Extras(a, rs)                                                                          # B1 H0: the schedule, the I/O, the readout clocks, the saves, the stops
    eps_light = J_light = None; m1L = None                                                         # B1 H0
    shell = [float(x) * rs for x in a.shell.split(",")] if a.shell else None
    kid = kernel_identities(R, a.ell) if a.site_fields == "averaged" else None
    if kid:
        print("kernel identities (site fields averaged): max |K_0 average of 1 - 1| %.2e; max |K_1 average of c r - c R_n| %.2e (relative %.2e); K_0 weight sums %.15f to %.15f" % (
            kid["K0_constant_max_abs"], kid["K1_linear_max_abs"], kid["K1_linear_max_rel"], kid["K0_weight_sums"][0], kid["K0_weight_sums"][1]), flush=True)
    t0 = time.time()
    resume = None
    if a.resume_from:
        st, f_ref, stored, head, resume = resume_state(a, chs, R, shell)
        head = dict(head); head["resume"] = resume; head["label_v19"] = label; head["kernel_identities_v19"] = kid
        E0 = head["E0"]
        N_cinit, v_cinit = stored["N_cinit"], stored["v_cinit"]
        recs = stored["records"]; fields = stored["fields"]
        step0 = stored["k_res"] + 1; T = stored["T_next"]
    else:
        N, v, dust, info, head = prepare_dust(chs, a.g, R, a.ell, NB, a.R0_over_rs * rs, a.m, a.reading, shell)
        N_cinit, v_cinit = N.copy(), v.copy()
        head.update({"label": label, "rs_rest_label": rs, "R0": a.R0_over_rs * rs, "seconds": time.time() - t0, "kernel_identities": kid})
        print("initial data (%s): N_B %.2f E0 %.6f passes %d f_kin %.4f (bulk frame %.4f) discarded %.2e Gram cond %.2e stationarity %.2e; the spline's distance from the dense fields at the sites N %.3e v %.3e; residuals W %.1e 1/N %.1e; source splines %s; centre series %.6f; collar %s (%.0f s)" % (
            label, head["N_B"], head["E0"], head["passes"], head["f_kin"], head["f_kin_bulk_frame"], head["projection"]["discarded_norm_max"],
            head["projection"]["gram_cond_max"], head["projection"]["stationarity_max"], head["interp_floor_N"] or float("nan"), head["interp_floor_v"] or float("nan"),
            head["residual_W"] or float("nan"), head["residual_invN"] or float("nan"), head["source_spline_error"], head["centre_series_ratio"] or float("nan"),
            head["collar_margin"], head["seconds"]), flush=True)
        E0 = head["E0"]
        st = {}
        for ch in chs:
            e, U = eigT(clocked_gpu(ch, N, v))
            neg = e < 0
            Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), np.zeros(0)))
            _en = e.real.cpu().numpy(); _k = int(neg.sum())
            gap = float(_en[_k] - _en[_k - 1]) if 0 < _k < len(_en) else None
            ref0 = U[:, neg].clone()
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": _k, "window": [{"ref": ref0, "f": (N.copy(), v.copy()), "gap": gap}]}
            del e, U
        f_ref = (N.copy(), v.copy())
        recs = []
        fields = {"T": [], "N": [], "v": [], "eps_tot": [], "J_tot": [], "eps_pairs": [], "J_pairs": [], "N_applied": [], "v_applied": [], "r_nodes": [], "N_nodes": [], "v_nodes": []}
        step0 = 0; T = 0.0
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    if a.b1_resume:                                                                                # B1 H1: the b1 state (M7) replaces the initial data
        _rd = B1X.resume(a, LIGHT)                                                                 # B1 H1
        st, f_ref, recs, step0, T, head = _rd["st"], _rd["f_ref"], list(_rd["recs"]), _rd["step_next"], _rd["T_next"], _rd["head"]   # B1 H1
        N_cinit, v_cinit, E0 = _rd["N_cinit"], _rd["v_cinit"], _rd["head"]["E0"]                  # B1 H1
    elif LIGHT is not None:                                                                        # B1 H1
        LIGHT.start(f_ref, E0, head)                                                               # B1 H1: the T_0 vacua, Delta_0, Delta S_0, the conventions (4.2, M8)
    head["b1"] = {"engine": "engine_b1_v2 (v19 run() verbatim + hooks H0-H8)", "light": a.light, "schedule": (B1X.sched.label if B1X.sched else None),   # B1 H1
                  "io": ("chunked" if B1X.io is not None else "v19"), "label": B1X.label + (LIGHT.label if LIGHT is not None else "")}   # B1 H1
    r_nodes_fixed = FLD.coarse_grid(R, a.ell)
    drtol_last = {"value": None, "T_over_rs": None}
    for r_ in reversed(recs):
        ind_ = r_.get("indicator") or {}
        if ind_.get("delta_rtol_source") == "this record" and ind_.get("delta_rtol") is not None:
            drtol_last = {"value": ind_["delta_rtol"], "T_over_rs": r_["T"] / rs}
            break
    nsteps = int(round(a.T_over_rs * rs / a.tau))
    if B1X.sched is not None:                                                                      # B1 H1s: the record schedule (M3 a)
        nsteps = B1X.sched.nsteps                                                                  # B1 H1s
    if a.m1_simpson_steps == "auto":
        q8 = max(1, nsteps // 8)
        simpson_steps = sorted({min(q8, nsteps - 1), nsteps // 2, max(nsteps - q8, 1)}) if nsteps >= 2 else []
    else:
        simpson_steps = [int(x) for x in a.m1_simpson_steps.split(",") if x.strip()]
    sl128 = [float(x) for x in a.slices_c128.split(",") if x.strip()]
    sl64 = [float(x) for x in a.slices_c64.split(",") if x.strip()]
    if (sl128 or sl64) and a.slice_dir:
        os.makedirs(a.slice_dir, exist_ok=True)
    path = os.path.join(a.out, "collapse_%s.json" % a.tag)
    prevW = None
    ind_state = {"first_failure": None}
    for r_ in recs:
        if (r_.get("indicator") or {}).get("indicator_pass") is False:
            ind_state["first_failure"] = r_["T"] / rs
            break
    lz_state = {}
    deg_state = {}
    unconv_state = {"first": None}
    if B1X.resumed is not None:                                                                    # B1 H1r: the sticky labels' first clocks restored
        _lb = B1X.resumed.get("labels") or {}                                                      # B1 H1r
        lz_state.update(_lb.get("lz", {})); deg_state.update(_lb.get("deg", {})); unconv_state.update(_lb.get("unconv", {}))   # B1 H1r
    t_run0 = time.time()
    solver_meta = {"rtol": 1e-12, "atol": 1e-16, "collar_policy": OPT["collar_policy"], "support_rel": OPT["support_rel"], "module": "engine_fields_v14",
                   "solver": "solve_constraints_ivp", "site_fields": a.site_fields, "engine": "engine_collapse_v19", "stop_E": a.stop_E, "stop_fields": a.stop_fields,
                   "mix": a.mix, "maxit": a.maxit, "c_LZ": a.c_LZ, "window_max": a.window_max, "delta_rtol_every": a.delta_rtol_every, "streams": int(a.streams),
                   "resumed": resume is not None}
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc", "b", "sy_k"):
            hand["%s_%s" % (q, part)] = []
    spectra = {"T": []}
    for ch in chs:
        spectra["mask_%d" % ch.kappa] = []; spectra["e_%d" % ch.kappa] = []

    def save_fields():
        np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),
                            eps_tot=np.array(fields["eps_tot"]), J_tot=np.array(fields["J_tot"]), eps_pairs=np.array(fields["eps_pairs"]), J_pairs=np.array(fields["J_pairs"]),
                            N_applied=np.array(fields["N_applied"]), v_applied=np.array(fields["v_applied"]), N_cinit=N_cinit, v_cinit=v_cinit,
                            g=a.g, ell=a.ell, tau=a.tau, solver=json.dumps(solver_meta),
                            r_nodes=np.array(fields["r_nodes"], dtype=object), N_nodes=np.array(fields["N_nodes"], dtype=object),
                            v_nodes=np.array(fields["v_nodes"], dtype=object), label=label)
        if hand["T"]:
            np.savez_compressed(os.path.join(a.out, "collapse_%s_handoff.npz" % a.tag), R=R, r_nodes=r_nodes_fixed, g=a.g, ell=a.ell, m=a.m, tau=a.tau,
                                kappas=np.array([ch.kappa for ch in chs]), weights=np.array([ch.w for ch in chs]), **{k: np.array(v_) for k, v_ in hand.items()})
        if spectra["T"]:
            np.savez(os.path.join(a.out, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})
    last_prop_s = 0.0
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    for step in range(step0, nsteps + 1):
        tr = time.time()
        tm_reset()
        TM["propagation_prev_s"] = last_prop_s
        mixer = MIX.Anderson(beta=1.0) if a.mix == "anderson" else None
        converged = False
        it_hist = []
        t_d = time.time()
        base = {}
        for ch in chs:
            s_ = st[ch.kappa]
            base[ch.kappa] = (dens(ch.h, s_["sea"]), dens(ch.Q, s_["sea"]), dens(ch.h, s_["dust"], s_["f"]), dens(ch.Q, s_["dust"], s_["f"]))
        TM["densities_s"] += time.time() - t_d
        if LIGHT is not None:                                                                      # B1 H2: the light sources at record k - 1's accepted fields and charge
            E_rec_prev = recs[-1]["E"] if (recs and recs[-1].get("E") is not None) else E0         # B1 H2
            eps_light, J_light, _ldiag = LIGHT.sources(f_ref, E_rec_prev, T, step)                 # B1 H2

        def sources(cache_):
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            if LIGHT is not None:                                                                  # B1 H3: the light's Gauss source fed back
                if a.light_ref_iterate and cache_[chs[0].kappa][1] is not None:                    # B1 H3: --light-ref-iterate 1: the trial's reference
                    _el, _jl = LIGHT.sources_trial((N_app, v_app), E_rec_prev)                     # B1 H3
                    eps = eps + _el; J = J + _jl                                                   # B1 H3
                else:                                                                              # B1 H3
                    eps = eps + eps_light; J = J + J_light                                         # B1 H3
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def do_solve(srcs_):
            t_s = time.time()
            out = solve(srcs_[0], srcs_[1], a.g, R, a.ell, None)
            TM["solve_s"] += time.time() - t_s; TM["solve_n"] += 1
            if out[3].get("ivp"):
                TM["solve_nfev"] += int(out[3]["ivp"]["nfev"])
            return out
        N_app, v_app = f_ref[0].copy(), f_ref[1].copy()
        cache = {ch.kappa: (None, None, None, (st[ch.kappa]["ref"] if a.reference == "R2" else st[ch.kappa]["sea"]), {"mode": "lagged"}) for ch in chs}
        TM["lagged"] += len(chs)
        srcs = sources(cache)
        Nn, vn, Wn, info = do_solve(srcs)
        E_prev = FLD.charge(info, a.g) if not info.get("failed") else None
        if not info.get("failed"):
            it_hist.append([None, float(np.max(np.abs(Nn - N_app))), float(np.max(np.abs(vn - v_app)))])
        k_it = 0
        best = None
        for k_it in (range(1, a.maxit + 1) if not info.get("failed") else ()):
            if mixer is not None:
                x = mixer.step(np.concatenate([N_app, v_app]), np.concatenate([Nn, vn]))
                N_app, v_app = x[:a.nsites].copy(), x[a.nsites:].copy()
            else:
                N_app, v_app = Nn.copy(), vn.copy()
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)
            for ch in chs:
                e, U = eig.pop(ch.kappa)
                if a.reference == "R2":
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = OPS2.continue_sea_v2(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], a.tau, a.c_LZ, eig_at, clocked_host_sparse, a.window_max)
                    cinfo["_mask"] = mask
                    TM["continuation_s"] += time.time() - t_c
                    TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
                else:
                    ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
                cache[ch.kappa] = (None, e, U, ref, cinfo)
            del eig
            srcs = sources(cache)
            Nn, vn, Wn, info = do_solve(srcs)
            if info.get("failed"):
                break
            E = FLD.charge(info, a.g)
            dE = abs(E - E_prev) / abs(E)
            E_prev = E
            dN = float(np.max(np.abs(Nn - N_app))); dv = float(np.max(np.abs(vn - v_app)))
            it_hist.append([float(dE), dN, dv])
            resid = max(dN, dv)
            if best is None or resid < best["resid"]:
                best = {"resid": resid, "k_it": k_it, "cache": cache, "srcs": srcs, "sol": (Nn, vn, Wn, info), "app": (N_app.copy(), v_app.copy()), "dE": dE}
            if dE < a.stop_E and dN < a.stop_fields and dv < a.stop_fields:
                converged = True
                break
        if info.get("failed"):
            recs.append({"T": T, "failed": info["failed"], "label": label, "iteration_history": it_hist})
            print("T/rs %.2f: the re-solve is REJECTED (%s); the run stops with the record written" % (T / rs, info["failed"]), flush=True)
            break
        if k_it == 0 or cache[chs[0].kappa][1] is None:
            recs.append({"T": T, "failed": "no eigendecomposition at the accepted fields", "label": label})
            print("T/rs %.2f: no eigendecomposition at the accepted fields; the run stops" % (T / rs), flush=True)
            break
        unconv = None
        if not converged and best is not None:
            cache, srcs = best["cache"], best["srcs"]
            Nn, vn, Wn, info = best["sol"]
            N_app, v_app = best["app"]
            unconv = {"residual": best["resid"], "iteration": best["k_it"], "dE": best["dE"]}
            if unconv_state["first"] is None:
                unconv_state["first"] = T / rs
        best = None
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)
        # the reference's diagnostics on the accepted state; ||[h_m, R2]||_F per channel; the continuation's record
        wmin, npocket, nrefill, dR2R0, nsub_max = 1.0, {}, {}, 0.0, 1
        comm = {}
        cont = {}
        n_inter = n_inv = 0
        mask_neg_equal = True
        masks = {}
        if a.reference == "R2":
            for ch in chs:
                hm_, e, U, ref, cinfo = cache[ch.kappa]
                s_ = st[ch.kappa]
                masks[ch.kappa] = np.asarray(cinfo.pop("_mask"), bool)
                eneg = (e < 0).cpu().numpy()
                same = bool(np.array_equal(masks[ch.kappa], eneg))
                mask_neg_equal = mask_neg_equal and same
                if cinfo.get("min_occupied_weight") is not None:
                    wmin = min(wmin, cinfo["min_occupied_weight"])
                npocket[ch.kappa] = cinfo["n_pocket"]
                nrefill[ch.kappa] = cinfo["nneg"] - s_["n_sea"]
                nsub_max = max(nsub_max, int(cinfo.get("nsub") or 1))
                n_inter += int(cinfo.get("n_intermediate") or 0); n_inv += int(cinfo.get("inversions") or 0)
                ov = float(((U[:, e < 0].conj().T @ ref).abs() ** 2).sum())
                dR2R0 += math.sqrt(max(cinfo["nneg"] + s_["n_sea"] - 2 * ov, 0.0))
                cont[str(ch.kappa)] = {k_: cinfo.get(k_) for k_ in ("mode", "nsub", "m", "m_eff", "verified", "margin_min", "s_T", "tau_LZ", "norm_dH_G", "window_consistency",
                                                                     "n_intermediate", "min_occupied_weight", "max_empty_weight", "inversions", "n_pocket",
                                                                     "pockets_occupied_positive", "holes_empty_negative", "tol_block", "rotated_blocks",
                                                                     "gap_between_blocks", "partly_occupied_blocks")}
                cont[str(ch.kappa)]["mask_equals_negative_set"] = same
                if not cinfo.get("verified") and ch.kappa not in lz_state:
                    lz_state[ch.kappa] = T / rs
                if cinfo.get("min_occupied_weight") is not None and cinfo["min_occupied_weight"] < 0.5 and ch.kappa not in deg_state:
                    deg_state[ch.kappa] = T / rs
            if a.commutator_every > 0 and step % a.commutator_every == 0:
                t_q = time.time()
                for ch in chs:
                    comm[str(ch.kappa)] = commutator_norm(ch, N_app, v_app, cache[ch.kappa][3])
                TM["commutator_s"] += time.time() - t_q
        else:
            for ch in chs:
                masks[ch.kappa] = np.asarray(cache[ch.kappa][4].pop("_mask"), bool)
        if lz_state:
            first = min(lz_state.values())
            cont_label = "LZ convention (c_LZ = %g) from T/rs %.3f (channels %s); inversions %d; n_pocket %d; intermediate weights %d" % (
                a.c_LZ, first, ",".join(str(k_) for k_ in sorted(lz_state)), n_inv, sum(npocket.values()) if npocket else 0, n_inter)
        else:
            cont_label = "verified (Weyl)"
        if deg_state:
            cont_label += "; reference degraded from T/rs %.3f (channels %s)" % (min(deg_state.values()), ",".join(str(k_) for k_ in sorted(deg_state)))
        unconv_label = ("unconverged (residual %.2e at iteration %d) from T/rs %.3f" % (unconv["residual"], unconv["iteration"], unconv_state["first"])) if unconv else (
            ("converged (unconverged records from T/rs %.3f)" % unconv_state["first"]) if unconv_state["first"] is not None else "converged")
        # the indicator, the r_0 check and the collar (engine 18.1, 20.2)
        t_ce = time.time()
        fresh = bool(step % max(1, a.delta_rtol_every) == 0 or drtol_last["value"] is None)
        ind = indicator(eps, J, a.g, R, a.ell, info, N, v, fresh, drtol_last)
        if fresh and ind["delta_rtol"] is not None:
            drtol_last.update({"value": ind["delta_rtol"], "T_over_rs": T / rs})
        r0c = FLD.r0_check(eps, J, a.g, R, a.ell, info, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"]) if fresh else None
        TM["indicator_s"] += time.time() - t_ce
        if not ind["indicator_pass"] and ind_state["first_failure"] is None:
            ind_state["first_failure"] = T / rs
        ind_label = "indicator pass" if ind_state["first_failure"] is None else ("indicator fail from T/rs %.2f" % ind_state["first_failure"])
        gauge = info.get("gauge") or {}
        gauge_label = "intact" if not gauge.get("intervals") else ("continued on %s" % ", ".join("[%.2f, %.2f]" % (x_, y_) for x_, y_ in gauge["intervals"]))
        # M1's balance (17.3 (4)): the old configuration's terms before the reference and h are replaced
        cut = int(min(max(round(info["r_collar"]) - 1, 2), a.nsites - 10))
        Y1 = Y2 = IEend = 0.0
        t_m1 = time.time()
        if recs and recs[-1].get("E") is not None:
            for ch in chs:
                s_ = st[ch.kappa]; H_old = clocked_host_sparse(ch, f_ref[0], f_ref[1]); Hr_old = dev_csr(H_old[:2 * cut, :]); ref_old = s_["ref"]; ref_new = cache[ch.kappa][3]
                Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
                Y1 += ch.w * inside_energy_sp(Hr_old, Ccols, fC, 2 * cut)
                Y2 += ch.w * inside_energy_sp(Hr_old, ref_new, np.ones(ref_new.shape[1]), 2 * cut)
                colsE = torch.cat([s_["sea"], s_["dust"], ref_old], dim=1)
                fE = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref_old.shape[1])])
                IEend += ch.w * energy_cut_current_sp(H_old, colsE, fE, cut)
                del Hr_old, colsE
        TM["M1_s"] += time.time() - t_m1
        if LIGHT is not None and recs and recs[-1].get("E") is not None:                          # B1 H4: the light's old-configuration terms (f_ref still record k - 1's)
            LIGHT.m1_old(f_ref, recs[-1]["E"], cut)                                                # B1 H4
        for ch in chs:
            ci_ = cache[ch.kappa][4]
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks")})
            st[ch.kappa]["window"] = st[ch.kappa]["window"][-a.window_max:]
        f_ref = (N_app.copy(), v_app.copy())
        tz = FLD.trapping_radius(N, v, R, info)
        Rt = tz["zeros"][-1] if tz["zeros"] else None
        r_in = tz["zeros"][0] if len(tz["zeros"]) >= 2 else None
        jets = surface_jets(info, Rt, a.g, E)
        kT = jets["kappa_T"] if jets else None
        M_wg = FLD.mass_function(info, a.g, Rt) if Rt else None
        M_ms = FLD.misner_sharp(info, a.g, Rt) if Rt else None
        Rc = Rsh = None
        tot = eps_dust.sum()
        if tot > 0:
            cum = np.cumsum(eps_dust); Rsh = float(R[min(int(np.searchsorted(cum, 0.99 * tot)), a.nsites - 1)]); Rc = float(eps_dust @ R / tot)
        late = None
        if Rt:
            rr = np.linspace(max(Rt - 5 * a.ell, 0.0), Rt + 5 * a.ell, 101)
            rho_l = np.abs(info["rho_ell"](rr)); J_l = np.abs(info["J_ell"](rr))
            rmax = max(float(np.max(np.abs(info["rho_ell"](np.linspace(0, info["r_collar"], 2001))))), 1e-300)
            jmax = max(float(np.max(np.abs(info["J_ell"](np.linspace(0, info["r_collar"], 2001))))), 1e-300)
            late = {"rho_ratio": float(rho_l.max() / rmax), "J_ratio": float(J_l.max() / jmax)}
            late["flag"] = bool(late["rho_ratio"] < 1e-6 and late["J_ratio"] < 1e-6)
        IE = Hin = 0.0
        X1 = X2 = 0.0
        t_m1 = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]
            H_new = clocked_host_sparse(ch, N_app, v_app); Hr = dev_csr(H_new[:2 * cut, :])
            s_ = st[ch.kappa]
            cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
            fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
            IE += ch.w * energy_cut_current_sp(H_new, cols, fs, cut)
            Hin += ch.w * inside_energy_sp(Hr, cols, fs, 2 * cut)
            Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
            X1 += ch.w * inside_energy_sp(Hr, Ccols, fC, 2 * cut)
            X2 += ch.w * inside_energy_sp(Hr, ref, np.ones(ref.shape[1]), 2 * cut)
            del Hr, cols
        TM["M1_s"] += time.time() - t_m1
        if LIGHT is not None:                                                                      # B1 H4: X1L, X2L, I_E^L at the accepted fields; the light balance
            m1L = LIGHT.m1_new((N_app, v_app), E, cut)                                             # B1 H4
        if B1X.sched is not None:                                                                  # B1 H6s: the next interval's tau (the Simpson half step, the propagations)
            a.tau = B1X.sched.tau_next(step)                                                       # B1 H6s
        IE_mid = None
        if step in simpson_steps and step < nsteps and not a.benchmark:
            t_sm = time.time()
            IE_mid = 0.0
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]
                ph2 = torch.exp(-1j * e.to(torch.complex128) * (0.5 * a.tau))
                s_ = st[ch.kappa]
                halfp = lambda X: U @ (ph2[:, None] * (U.conj().T @ X))
                cols = torch.cat([halfp(s_["sea"]), halfp(s_["dust"]) if s_["dust"].shape[1] else s_["dust"], ref], dim=1)
                fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                IE_mid += ch.w * energy_cut_current_sp(clocked_host_sparse(ch, N_app, v_app), cols, fs, cut)
                del cols
            TM["simpson_s"] += time.time() - t_sm
        # the handoff quantities (20.1): the stresses (dust and pairs apart), <N^2>_n, the sea's and R2's shell energies
        t_h = time.time()
        Pd = None; Pp = None
        eps_sea = np.zeros(a.nsites); eps_R2 = np.zeros(a.nsites)
        for ch in chs:
            s_ = st[ch.kappa]; ref = cache[ch.kappa][3]; cs = cstr[ch.kappa]
            pd_ = STR.stress_parts(cs, s_["dust"], s_["f"])
            pp_ = STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])]))
            for P_, p_ in ((0, pd_), (1, pp_)):
                p_ = dict(p_); p_["sy_k"] = ch.kappa * p_["sy"]
                if P_ == 0:
                    Pd = {k_: ch.w * np.asarray(v_, float) for k_, v_ in p_.items()} if Pd is None else {k_: Pd[k_] + ch.w * np.asarray(p_[k_], float) for k_ in Pd}
                else:
                    Pp = {k_: ch.w * np.asarray(v_, float) for k_, v_ in p_.items()} if Pp is None else {k_: Pp[k_] + ch.w * np.asarray(p_[k_], float) for k_ in Pp}
            eps_sea += ch.w * base[ch.kappa][0]; eps_R2 += ch.w * dens(ch.h, ref)
        SVd = STR.shell_values(Pd, R, a.m); SVp = STR.shell_values(Pp, R, a.m)
        Sr_tot = SVd["S_r"] + SVp["S_r"]; St_tot = SVd["S_t"] + SVp["S_t"]
        SVl = None                                                                                 # B1 H5
        if LIGHT is not None:                                                                      # B1 H5: the light stresses (interior factorization, T_0 offset subtracted)
            Pl = LIGHT.stresses(); SVl = STR.shell_values(Pl, R, 0.0)                              # B1 H5
            Sr_tot = Sr_tot + SVl["S_r"]; St_tot = St_tot + SVl["S_t"]                             # B1 H5
            hand.setdefault("F_face_light", []).append(SVl["F_face"])                              # B1 H5
            for q_ in ("fc", "b", "sy_k"):                                                         # B1 H5
                hand.setdefault("%s_light" % q_, []).append(np.asarray(Pl[q_], float))             # B1 H5
            hand.setdefault("eps_light", []).append(np.asarray(eps_light, float).copy())           # B1 H5
        Sr_l, St_l, tr_err = STR.smooth_stress(Sr_tot, St_tot, R, a.ell, r_nodes_fixed)
        N2avg, _vz, _nn = FLD.site_fields_kernel_average({"N_dense": (lambda r_: info["N_dense"](r_) ** 2), "v_dense": zero}, R, a.ell)
        vir = {}
        for lab_, SV in (("dust", SVd), ("pairs", SVp)):
            sc = max(float(np.max(np.abs(SV["virial_rhs"]))), 1e-300)
            vir[lab_] = float(np.max(np.abs(SV["virial_lhs"] - SV["virial_rhs"]))) / sc
        stress_rec = {"identity_max_abs": {"dust": SVd["identity_max_abs"], "pairs": SVp["identity_max_abs"]}, "virial_rel": vir, "trace_identity_rel": tr_err,
                      "F_total_at_last_face": float(SVd["F_face"][-1] + SVp["F_face"][-1]), "seconds": time.time() - t_h}
        hand["T"].append(T); hand["N2avg"].append(np.asarray(N2avg, float)); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(np.asarray(eps_dust, float))
        hand["S_r_ell"].append(Sr_l); hand["S_t_ell"].append(St_l)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
            hand["b_%s" % part].append(np.asarray(P_["b"], float)); hand["sy_k_%s" % part].append(np.asarray(P_["sy_k"], float))
        spectra["T"].append(T)
        for ch in chs:
            spectra["mask_%d" % ch.kappa].append(np.packbits(masks[ch.kappa]))
            spectra["e_%d" % ch.kappa].append(cache[ch.kappa][1].cpu().numpy().astype(np.float64))
        TM["handoff_s"] = time.time() - t_h
        saved = None
        tslice = [x for x in sl128 + sl64 if abs(T / rs - x) < 1e-6]
        if tslice and a.slice_dir:
            dt_ = np.complex128 if any(abs(T / rs - x) < 1e-6 for x in sl128) else np.complex64
            arrs = {"T": T, "rs": rs, "N_applied": N_app, "v_applied": v_app, "R": R, "kappas": np.array([ch.kappa for ch in chs]), "weights": np.array([ch.w for ch in chs]),
                    "dtype": str(np.dtype(dt_))}
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]; s_ = st[ch.kappa]
                arrs["sea_%d" % ch.kappa] = s_["sea"].cpu().numpy().astype(dt_)
                arrs["dust_%d" % ch.kappa] = s_["dust"].cpu().numpy().astype(dt_)
                arrs["f_%d" % ch.kappa] = np.asarray(s_["f"], float)
                arrs["R2_%d" % ch.kappa] = ref.cpu().numpy().astype(dt_)
                arrs["e_%d" % ch.kappa] = e.cpu().numpy()
                arrs["mask_%d" % ch.kappa] = np.packbits(masks[ch.kappa])
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(saved, **arrs)
            del arrs
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "E_dust_shells": float(eps_dust.sum()), "E_pairs_shells": float(eps_pairs.sum()),
               "pairs_inside_Rt": (float(eps_pairs[R < Rt].sum()) if Rt else None), "pairs_outside_Rt": (float(eps_pairs[R >= Rt].sum()) if Rt else float(eps_pairs.sum())),
               "R_t": Rt, "R_t_over_2gE": (Rt / (2 * a.g * E) if Rt else None), "W_over_2g_at_Rt": M_wg, "misner_sharp_at_Rt": M_ms,
               "R_t_over_2g_MS": (Rt / (2 * a.g * M_ms) if (Rt and M_ms) else None), "kappa_T": kT, "four_gE_kappa_T": (4.0 * a.g * E * kT if kT is not None else None),
               "surface_jets": jets, "r_in": r_in, "trapping_zeros": tz["zeros"], "tangencies": tz["tangencies"], "min_N_minus_absv": tz["min"], "r_min_N_minus_absv": tz["r_min"],
               "late_surface": late, "N_min": float(N.min()), "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(info["v_dense"](Rc)[0]) if Rc else None),
               "threshold_radius": threshold_radii(N, v, R, a.m, [1, 2, a.kappa_sea]), "n_pocket": npocket, "n_refill_R0": nrefill, "R2_minus_R0_F": dR2R0,
               "min_occupied_weight": wmin, "nsub_max": nsub_max, "M1_I_E": IE, "H_in": Hin,
               "iterations": k_it + 1, "eigendecompositions": TM["eigh_n"], "converged": converged, "unconverged": unconv, "iteration_history": it_hist,
               "continuation": cont, "continuation_label": cont_label, "mask_equals_negative_set": mask_neg_equal, "inversions_total": n_inv, "intermediate_weights_total": n_inter,
               "commutator_h_R2_F": comm, "gauge": {k_: gauge.get(k_) for k_ in ("label", "intervals", "min_U_over_U_ref", "delta_L_bound", "lapse_unresolved", "segments", "step_failure_zeros", "t_stretches")},
               "regime": info.get("regime"), "r0_check": r0c, "stress": stress_rec, "slice_saved": saved,
               "solver": {"module": "engine_fields_v14.solve_constraints_ivp", "nfev": (info.get("ivp") or {}).get("nfev"), "N_centre": (info.get("ivp") or {}).get("N_centre"),
                          "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "residual_matching_region": info.get("residual_matching_region"),
                          "residual_ladder": info.get("residual_ladder"), "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"),
                          "centre_series_check": info.get("centre_series_check"), "collar_margin": info.get("collar_margin"), "W_tail": info.get("W_tail"),
                          "W_roundoff_negative_points": info.get("W_roundoff_negative_points"), "W_negative": info.get("W_negative"), "site_fields": a.site_fields, "kernel_norms": info.get("kernel_norms")},
               "spectrum": info["spectrum"], "r_collar": info["r_collar"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
               "label": label, "seconds": time.time() - tr, "indicator": ind,
               "record_label": "engine_fields_v14; tau = %g; stresses: Noether (lane's declaration); rows: flat-slice ADM, K^ADM = -K; J_ell = <P_r> = -J_r of the goal's display; gauge: %s; continuation: %s; %s; %s; collar: %s" % (
                   a.tau, gauge_label, cont_label, unconv_label, ind_label, ind["collar_label"]),
               "timing": dict(TM),
               "M1_parts": {"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2, "I_E_end_prev_interval": IEend, "I_E_start": IE, "cut": cut, "I_E_mid": IE_mid}}
        if resume is not None and not any(q.get("engine") == "engine_collapse_v19" for q in recs):
            rec["resumed_from"] = {"run": resume["from"], "T_over_rs": resume["T_over_rs"]}
        rec["engine"] = "engine_collapse_v19"
        prevM = recs[-1].get("M1_parts") if (recs and recs[-1].get("E") is not None) else None
        if prevM is not None:
            dT = T - recs[-1]["T"]
            a_exact = Y1 - prevM["X1"]; b_work = (X1 - X2) - (Y1 - Y2); c_reset = -(Y2 - prevM["X2"]); dH = (X1 - X2) - (prevM["X1"] - prevM["X2"])
            rec["M1_balance"] = {"dHin": dH, "evolution_exact": a_exact, "evolution_trapezoid": 0.5 * dT * (prevM["I_E_start"] + IEend), "field_work": b_work,
                                 "reference_reset": c_reset, "closure": a_exact + b_work + c_reset - dH, "cut_changed": bool(prevM.get("cut") != cut),
                                 "per_unit_time": {"dHin": dH / dT, "evolution": a_exact / dT, "field_work": b_work / dT, "reference_reset": c_reset / dT}}
            if prevM.get("I_E_mid") is not None:
                rec["M1_balance"]["evolution_simpson"] = dT / 6.0 * (prevM["I_E_start"] + 4.0 * prevM["I_E_mid"] + IEend)
                rec["M1_balance"]["trapezoid_minus_exact"] = rec["M1_balance"]["evolution_trapezoid"] - a_exact
                rec["M1_balance"]["simpson_minus_exact"] = rec["M1_balance"]["evolution_simpson"] - a_exact
        if recs and recs[-1].get("E") is not None:
            dT = T - recs[-1]["T"]
            rec["dE_dT"] = (E - recs[-1]["E"]) / dT
            rec["dHin_dT"] = (Hin - recs[-1]["H_in"]) / dT
            rec["balance_d_E_minus_Hin_dT"] = rec["dE_dT"] - rec["dHin_dT"]
            if Rc is not None and recs[-1].get("R_c") is not None:
                rec["dRc_dT"] = (Rc - recs[-1]["R_c"]) / dT
        if step % 10 == 0 and prevW is not None and T > prevW[1]:
            rec["M9"] = {}
            for lab, rr_ in (("0.5 R_sh", 0.5 * Rsh if Rsh else None), ("R_t", Rt), ("collar", float(info["r_collar"]) - 1.0)):
                if rr_:
                    ci = int(min(max(round(rr_), 2), a.nsites - 10))
                    I_c = 0.0
                    for ch in chs:
                        _hm, e, U, ref, _ci = cache[ch.kappa]
                        H_new = clocked_host_sparse(ch, N_app, v_app)
                        s_ = st[ch.kappa]
                        cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
                        fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                        I_c += ch.w * energy_cut_current_sp(H_new, cols, fs, ci)
                        del cols
                    rec["M9"][lab] = {"r": rr_, "cut_site": ci, "dW_over_2g_dT": float((info["W_field"](rr_)[0] - prevW[0](rr_)[0]) / (2 * a.g * (T - prevW[1]))),
                                      "I_E_cut": I_c}
        if step % 10 == 0:
            prevW = (info["W_field"], T)
        rec["engine"] = ENGINE; rec["record_label"] = B1X.label + rec["record_label"]               # B1 H6: the lineage (R9)
        rec["tau_record"] = a.tau                                                                  # B1 H6: the next interval's step
        rec["N_heavy_in"] = heavy_number(chs, st, cache, cut)                                      # B1 H6: 4.9
        _tw = time.time(); rec["b1_wall_s"] = _tw - getattr(B1X, "t_last", tr); B1X.t_last = _tw   # B1 H6: record to record (4.15)
        B1X.walls.append([step, T / rs, rec["b1_wall_s"], rec["seconds"]])                         # B1 H6
        ro_due = B1X.due(T, "next_ro", a.light_readout_every_rs)                                   # B1 H6
        if LIGHT is not None:                                                                      # B1 H6: the light record (M2 ledger, the stop rule), the readouts (M5)
            rec["light"] = LIGHT.record(T, E, Rt, kT, cut, step, eps_light, converged)             # B1 H6
            rec["M1_light"] = m1L                                                                  # B1 H6
            if SVl is not None:                                                                    # B1 H6
                rec["stress"]["light_identity_max_abs"] = SVl["identity_max_abs"]                 # B1 H6
            if ro_due:                                                                             # B1 H6
                rec["light"]["readout"] = LIGHT.readout(T, E, Rt, kT, (N_app, v_app), step, a.out, a.tag, impurity=B1X.due(T, "next_imp", a.light_impurity_every_rs),   # B1 H6
                                                        contour=B1X.due(T, "next_contour", a.light_contour_every_rs), rows=B1X.due(T, "next_rows", a.light_rows_every_rs))   # B1 H6
            fields.setdefault("eps_light", []).append(np.asarray(eps_light, float).copy())         # B1 H6
            fields.setdefault("J_light", []).append(np.asarray(J_light, float).copy())             # B1 H6
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy()); fields["eps_tot"].append(eps.copy()); fields["J_tot"].append(J.copy())
        fields["eps_pairs"].append(eps_pairs.copy()); fields["J_pairs"].append(J_pairs.copy())
        fields["N_applied"].append(N_app.copy()); fields["v_applied"].append(v_app.copy())
        fields["r_nodes"].append(info["r_nodes"]); fields["N_nodes"].append(info["N_nodes"]); fields["v_nodes"].append(info["v_nodes"])
        oe = jets["over_exact"] if jets else None
        mb = rec.get("M1_balance")
        pc = ind.get("collar_padded") or {}
        last_h = it_hist[-1] if it_hist else [None, float("nan"), float("nan")]
        cmin = [c_.get("min_occupied_weight") for c_ in cont.values() if c_.get("min_occupied_weight") is not None]
        cmax = [c_.get("max_empty_weight") for c_ in cont.values() if c_.get("max_empty_weight") is not None]
        tlz = [c_.get("tau_LZ") for c_ in cont.values() if c_.get("tau_LZ") is not None]
        print("T/rs %.3f E/E0 %.7f dust %.4f pairs %.3e R_t %s (2gE %.3f; d1..d4 of N+v over exact %s; r_in %s) min(N-|v|) %.4f R_sh %s R_c %s pockets %s inversions %d intermediate %d mask=neg %s nsub max %d it %d (eigh %d)%s M1 %.3e dE/dT %s res %.1e %.1e; B_estimate %s; r0 check dE %s; gauge %s; regime max %.2e; continuation [%s]; weights min occ %s max empty %s; tau_LZ min %s; [h,R2] max %s; stress identity %.1e %.1e virial %.1e %.1e trace %.1e; M1 balance: evolution %s work %s reset %s closure %s [%s] last iteration dE %s dN %.1e dv %.1e (%.0f s; eigh %d in %.1f s, continuation %.1f s, densities %.1f s, solves %d in %.1f s, indicator %.1f s, handoff %.1f s, M1 %.1f s, propagation %.1f s)" % (
            T / rs, E / E0, eps_dust.sum(), eps_pairs.sum(), ("%.3f" % Rt) if Rt else None, 2 * a.g * E, (", ".join("%.4f" % q for q in oe)) if oe else None,
            ("%.3f" % r_in) if r_in else None, tz["min"], Rsh, ("%.2f" % Rc) if Rc else None, sum(npocket.values()) if npocket else 0, n_inv, n_inter, mask_neg_equal, nsub_max,
            k_it + 1, TM["eigh_n"], ("" if converged else " UNCONVERGED (residual %.1e)" % (unconv["residual"] if unconv else float("nan"))), IE,
            ("%.3e" % rec["dE_dT"]) if "dE_dT" in rec else "-", info.get("residual_W") or 0.0, info.get("residual_invN") or 0.0,
            ("%.1e" % ind["B_estimate"]) if ind["B_estimate"] is not None else "none", ("%.1e" % r0c["dE_rel"]) if (r0c and "dE_rel" in r0c) else "-",
            gauge_label, (info.get("regime") or {}).get("max_ratio", float("nan")), cont_label, ("%.3e" % min(cmin)) if cmin else "-", ("%.3e" % max(cmax)) if cmax else "-",
            ("%.2f" % min(tlz)) if tlz else "-", ("%.1e" % max(comm.values())) if comm else "-",
            SVd["identity_max_abs"], SVp["identity_max_abs"], vir["dust"], vir["pairs"], tr_err,
            ("%.3e" % mb["evolution_exact"]) if mb else "-", ("%.3e" % mb["field_work"]) if mb else "-", ("%.3e" % mb["reference_reset"]) if mb else "-",
            ("%.1e" % mb["closure"]) if mb else "-", rec["record_label"],
            ("%.1e" % last_h[0]) if last_h[0] is not None else "-", last_h[1], last_h[2], time.time() - tr, TM["eigh_n"], TM["eigh_s"], TM["continuation_s"],
            TM["densities_s"], TM["solve_n"], TM["solve_s"], TM["indicator_s"], TM.get("handoff_s", 0.0), TM["M1_s"], TM["propagation_prev_s"]), flush=True)
        if step > step0 and (step - step0) % 10 == 0:
            last = [q["seconds"] for q in recs[-10:] if "seconds" in q]
            mean = float(np.mean(last)) if last else 0.0
            remain = (nsteps - step) * mean
            end = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + remain))
            print("projection at T/rs %.2f: %.0f s per re-solve over the last 10; %d re-solves left; the end about %s UTC" % (T / rs, mean, nsteps - step, end), flush=True)
        if B1X.io is None and (step % 5 == 0 or step == nsteps or step == step0):                 # B1 H8: v19's writes with --io v19
            with open(path, "w") as fh:
                json.dump({"script": "engine_collapse_v19.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
            save_fields()
        b1_stop = None                                                                             # B1 H8
        if LIGHT is not None and LIGHT.stop_reason:                                                # B1 H8: the stop rule (M2 d)
            b1_stop = LIGHT.stop_reason                                                            # B1 H8
        if a.benchmark_records and step - step0 + 1 >= a.benchmark_records:                        # B1 H8
            b1_stop = "--benchmark-records %d reached" % a.benchmark_records                       # B1 H8
        if LIGHT is not None and a.T_end_before_return == "on":                                    # B1 H8: the chain end's reflection (or the first arrival with the sourced exterior)
            _fr = LIGHT.first_return_rs(E)                                                         # B1 H8
            if T / rs >= (_fr["front_at_chain_end_rs"] if a.light_exterior == "sourced" else _fr["reflection_at_collar_rs"]):   # B1 H8
                b1_stop = "--T-end-before-return: T/r_s %.2f at the chain end's return clock %s" % (T / rs, _fr)   # B1 H8
        if B1X.io is not None:                                                                     # B1 H8: the chunked writer (M7)
            if not ro_due and spectra["T"]:                                                        # B1 H8: the spectra at the readout clocks only
                for k_ in spectra:                                                                 # B1 H8
                    spectra[k_].pop()                                                              # B1 H8
            B1X.io.record(rec, full=ro_due)                                                        # B1 H8
            B1X.io.flush(R, fields, hand, spectra, {"N_cinit": N_cinit, "v_cinit": v_cinit, "g": a.g, "ell": a.ell, "tau": a.tau, "solver": json.dumps(solver_meta), "label": label},   # B1 H8
                         final=bool(b1_stop or step >= nsteps or a.benchmark))                     # B1 H8
            if step % max(1, a.json_every) == 0 or b1_stop or step >= nsteps:                     # B1 H8
                B1X.io.summary(a, head, recs, {"stop": b1_stop})                                   # B1 H8
            del recs[:-16]                                                                         # B1 H8: the records live in the jsonl
        if b1_stop:                                                                                # B1 H8
            print("B1: the run stops at T/r_s %.3f: %s" % (T / rs, b1_stop), flush=True)           # B1 H8
            break                                                                                  # B1 H8
        if a.benchmark or step >= nsteps:
            break
        t_p = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[ch.kappa]
            prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
            cache[ch.kappa] = (None, None, None, ref, _ci)
            del e, U
        sync_dev()
        last_prop_s = time.time() - t_p
        if LIGHT is not None:                                                                      # B1 H7: the light's step under record k's h (flows, L_at_t, propagate_lean)
            LIGHT.step((N_app, v_app), E, cut, a.tau, a.ell)                                       # B1 H7
        B1X.save_state(a, st, f_ref, recs, step + 1, T + a.tau, head, N_cinit, v_cinit, LIGHT, a.tau,   # B1 H7: the b1 state (M7)
                       {"lz": lz_state, "deg": deg_state, "unconv": unconv_state})                # B1 H7
        T += a.tau
    if B1X.io is None:                                                                             # B1 H8
        with open(path, "w") as fh:
            json.dump({"script": "engine_collapse_v19.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
        save_fields()
    else:                                                                                          # B1 H8: the wall table (records 21 to 100 and the quasi-static window) in the summary
        B1X.io.summary(a, head, recs, {"stop": locals().get("b1_stop"), "wall": B1X.walls})        # B1 H8


def ladder(a):
    """2.2.1: the trapping-threshold ladder with the light off: per factor f of --ladder, E_0 = f E_max with E_max = c_E m_P^3/m^2 (m_P =
    g^(-1/2), c_E = --ladder-cE, the production's 0.135), r_s = 2 g E_0, a run of --T-over-rs; the root's existence (min N - |v| < 0) and R5's
    heavy conservation per run (max |E - E_0|/E_0 through the run and over its second half, pockets, inversions, LZ labels) printed as one line
    each and into collapse_<tag>_ladder.json. A build input only (owner #306), not a paper result."""
    mP = a.g ** -0.5
    Emax = a.ladder_cE * mP ** 3 / a.m ** 2
    rows = []
    for fs in [x for x in a.ladder.split(",") if x.strip()]:
        f = float(fs)
        b = copy.copy(a)
        b.rs = 2.0 * a.g * f * Emax
        if a.ladder_tau_over_rs and a.ladder_tau_over_rs > 0:                      # v2: the record step 0.125 r_s per member (the review's 160 records per 20 r_s)
            b.tau = a.ladder_tau_over_rs * b.rs
        b.tag = "%s_f%s" % (a.tag, fs.strip())
        b.light = "off"; b.ladder = None
        t0 = time.time()
        run_b1(b)
        rows.append(ladder_row(b, f, Emax, time.time() - t0))
        print("LADDER f %.3f E0_rest %.4f r_s %.3f: %s" % (f, f * Emax, b.rs, json.dumps(rows[-1], default=_jdefault)), flush=True)
    trap = [r_ for r_ in rows if r_.get("root_formed")]
    out = {"E_max_nominal": Emax, "c_E_nominal": a.ladder_cE, "m_P": mP, "rows": rows, "m": a.m, "g": a.g, "nsites": a.nsites, "kappa_sea": a.kappa_sea,
           "tau_rule": ("%g r_s per member" % a.ladder_tau_over_rs) if a.ladder_tau_over_rs else ("tau %g for every member" % a.tau),
           "E_trap_rest": (min(r_["E0_rest"] for r_ in trap) if trap else None),
           "rule": "c_E := E_trap/(m_P^3/m^2) against 0.135: within 0.7 to 1.4 -> E_0 = 2.5 E_trap; outside -> the fallback point (ma 0.35, sigma 4a); a build input only"}
    if out["E_trap_rest"]:
        out["c_E_measured"] = out["E_trap_rest"] / (mP ** 3 / a.m ** 2)
        out["c_E_ratio"] = out["c_E_measured"] / 0.135
        out["E0_B1"] = 2.5 * out["E_trap_rest"]
        out["rs_B1"] = 2.0 * a.g * out["E0_B1"]
        out["decision"] = ("B1 point: E_0 = 2.5 E_trap, --rs %.6f" % out["rs_B1"]) if 0.7 <= out["c_E_ratio"] <= 1.4 else "c_E outside 0.7 to 1.4: the fallback point (ma 0.35, sigma 4a)"
    else:
        out["decision"] = "no member formed a root: the ladder is extended upward before E_0 is fixed"
    with open(os.path.join(a.out, "collapse_%s_ladder.json" % a.tag), "w") as fh:
        json.dump(out, fh, indent=1, default=_jdefault)
    print("LADDER summary: E_trap %s, c_E %s (ratio %s against 0.135); E_0(B1) = 2.5 E_trap = %s" % (out.get("E_trap_rest"), out.get("c_E_measured"), out.get("c_E_ratio"), out.get("E0_B1")), flush=True)
    return out


def load_records(out, tag):
    """the records of a run in either layout (v19's json, or the chunked jsonl)."""
    jl = os.path.join(out, "collapse_%s_records.jsonl" % tag)
    if os.path.exists(jl):
        return [json.loads(x) for x in open(jl) if x.strip()]
    return json.load(open(os.path.join(out, "collapse_%s.json" % tag)))["records"]


def ladder_row(b, f, Emax, secs):
    recs = [r_ for r_ in load_records(b.out, b.tag) if r_.get("E") is not None]
    if not recs:
        return {"f": f, "E0_rest": f * Emax, "failed": True}
    E = np.array([r_["E"] for r_ in recs]); T = np.array([r_["T"] for r_ in recs]) / b.rs
    mn = [r_.get("min_N_minus_absv") for r_ in recs]
    Rt = [r_.get("R_t") for r_ in recs]
    half = T >= 0.5 * T[-1]
    E0 = E[0]
    return {"f": f, "E0_rest": f * Emax, "E0_collar": float(E0), "rs": b.rs, "records": len(recs), "root_formed": any(x is not None and x < 0 for x in mn),
            "first_root_T_over_rs": next((float(t) for t, x in zip(T, mn) if x is not None and x < 0), None), "min_min_N_minus_absv": float(np.nanmin([x for x in mn if x is not None])),
            "R_t_last": Rt[-1], "max_rel_dE": float(np.max(np.abs(E - E0)) / abs(E0)), "max_rel_dE_second_half": float(np.max(np.abs(E[half] - E0)) / abs(E0)),
            "unconverged": int(sum(1 for r_ in recs if not r_.get("converged"))), "n_pocket_max": max((sum((r_.get("n_pocket") or {}).values()) for r_ in recs), default=0),
            "inversions_max": max((r_.get("inversions_total") or 0 for r_ in recs), default=0), "continuation_label_last": recs[-1].get("continuation_label"), "seconds": secs}


def build_parser():
    ap = argparse.ArgumentParser()
    # v19's options (verbatim)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--shell", default=None, help="R_in,R_out in units of r_s (S-shell)")
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--reference", default="R2", choices=["R2", "R1"])
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--collar-policy", default="flag", choices=["flag", "fail"])
    ap.add_argument("--support-rel", type=float, default=1e-14)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--site-fields", default="averaged", choices=["averaged", "dense"])
    ap.add_argument("--stop-E", type=float, default=1e-7)
    ap.add_argument("--stop-fields", type=float, default=1e-6)
    ap.add_argument("--maxit", type=int, default=20)
    ap.add_argument("--mix", default="plain", choices=["plain", "anderson"])
    ap.add_argument("--delta-rtol-every", type=int, default=10)
    ap.add_argument("--commutator-every", type=int, default=1)
    ap.add_argument("--streams", type=int, default=0)
    ap.add_argument("--m1-simpson-steps", default="auto")
    ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0)
    ap.add_argument("--window-max", type=int, default=4)
    ap.add_argument("--resume-from", default=None, help="dir/tag of a stored engine_collapse_v17 run")
    ap.add_argument("--resume-T-over-rs", type=float, default=None)
    ap.add_argument("--slices-c128", default="")
    ap.add_argument("--slices-c64", default="")
    ap.add_argument("--slice-dir", default=None)
    # B1's options (design 3.2 with the review's amendments)
    ap.add_argument("--light", default="off", choices=["on", "off"])
    ap.add_argument("--light-kappas", default="1,2,3", help="the sourced channels")
    ap.add_argument("--light-guard-kappas", default="4", help="propagated and read, not sourced (the guard; R10)")
    ap.add_argument("--light-kappa-zero", default="off", choices=["on", "off"], help="R8: a kappa = 0 conformal copy among the guards")
    ap.add_argument("--light-nsites", type=int, default=16896, help="n_L (M3 c: 16896 for D2 and D4, 33792 for the production if D4 says so)")
    ap.add_argument("--light-nv", type=int, default=0, help="n_v of the truncated reference chain (0: n_e + --light-pad)")
    ap.add_argument("--light-pad", type=int, default=128)
    ap.add_argument("--light-ref", default="continued", choices=["continued", "killing"])
    ap.add_argument("--light-ref-every", type=int, default=1, help="R4: >1 reuses the reference between records after --light-ref-every-after-rs")
    ap.add_argument("--light-ref-every-after-rs", type=float, default=50.0)
    ap.add_argument("--light-ref-iterate", type=int, default=0, help="1: the reference at each trial field (the 4.11 control)")
    ap.add_argument("--light-exterior", default="vacuum", choices=["vacuum", "sourced"])
    ap.add_argument("--light-exterior-work", default="on", choices=["on", "off"], help="the exterior's work on the escaped radiation per record (vacuum exterior)")
    ap.add_argument("--light-cheb-tol", type=float, default=1e-14)
    ap.add_argument("--light-chunk", type=int, default=2048)
    ap.add_argument("--light-sub", type=int, default=1)
    ap.add_argument("--light-init", default="device", choices=["device", "host"], help="M6: the T_0 vacuum by torch eigh on the card (host: zheevr)")
    ap.add_argument("--light-init-cache", default=None, help="a directory caching Phi_+(T_0) per channel by the md5 of h_+(T_0)")
    ap.add_argument("--light-init-impurity", default="on", choices=["on", "off"])
    ap.add_argument("--light-init-offload", default="on", choices=["on", "off"], help="v2 M6: park each channel's Phi_+ on the host while the next eigh runs")
    ap.add_argument("--light-threads", type=int, default=8)
    ap.add_argument("--light-readout-every-rs", type=float, default=10.0)
    ap.add_argument("--light-cuts-rs", default="2,4,8,16", help="M5: fixed exterior cuts in r_s0 (the root cut is always added)")
    ap.add_argument("--light-impurity-every-rs", type=float, default=100.0)
    ap.add_argument("--light-rows-every-rs", type=float, default=100.0, help="Phi_+[:n_cut] rows saved (0: never)")
    ap.add_argument("--light-contour-every-rs", type=float, default=100.0, help="the long chain's contour vacuum block, the certificate of the truncated one (0: never)")
    ap.add_argument("--light-comparator", default="on", choices=["on", "off"])
    ap.add_argument("--b1-save-every-rs", type=float, default=500.0, help="M7: the b1 state (resume) every x r_s and at the end (0: never)")
    ap.add_argument("--b1-keep-states", type=int, default=2)
    ap.add_argument("--b1-resume", default=None, help="a collapse_<tag>_b1state_T*.pt file")
    ap.add_argument("--b1-stop-rule", default="on", choices=["on", "off"])
    ap.add_argument("--b1-stop-dEend", type=float, default=1e-3, help="M2 d: |Delta E_end| per record above this times E_0 halts")
    ap.add_argument("--b1-stop-inside", type=float, default=0.1, help="M2 d: the fed inside-root total changing by more than this times E_0 per record halts")
    ap.add_argument("--disk-min-gb", type=float, default=20.0)
    ap.add_argument("--io", default="auto", choices=["auto", "v19", "chunked"], help="auto: chunked with --light on, v19 with it off")
    ap.add_argument("--json-every", type=int, default=200)
    ap.add_argument("--fields-chunk-records", type=int, default=800)
    ap.add_argument("--lean-records", default="on", choices=["on", "off"])
    ap.add_argument("--benchmark-records", type=int, default=0)
    ap.add_argument("--tau-schedule", default="off", choices=["on", "off"])
    ap.add_argument("--tau-switch-rs", type=float, default=50.0)
    ap.add_argument("--tau-late", type=float, default=4.2)
    ap.add_argument("--T-end-before-return", default="on", choices=["on", "off"])
    ap.add_argument("--ladder", default=None, help="factors of E_max (2.2.1), e.g. 0.5,0.7,0.85,1.0,1.15,1.3,1.6,2.0")
    ap.add_argument("--ladder-cE", type=float, default=0.135)
    ap.add_argument("--ladder-tau-over-rs", type=float, default=0.0, help="v2: >0 sets each member's tau to this times its r_s (0: --tau for all)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    OPT["collar_policy"] = a.collar_policy; OPT["support_rel"] = a.support_rel; OPT["site_fields"] = a.site_fields
    C5.backend(a.gpu)
    if a.ladder:
        return ladder(a)
    run_b1(a)


if __name__ == "__main__":
    main()
