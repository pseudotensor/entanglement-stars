#!/usr/bin/env python
"""page_recursion_v2.py. Changes against page_recursion_v1.py (builder analyst, BRIEF_page_debug_v1.md): imports page_ops_v2,
page_core_v2, page_selftest_v2 and page_scatter_v2 (the numerical fixes behind T4b, T4d, T4e and T4g live there); new option --engine-path
(T1a: the directory of engine_ops_v1.py for the self-test's operator comparison; without it: $PAGE_ENGINE_PATH, then ../engine,
../engine/code and ../../engine/code beside this file, then <internal>/code); the Checks block adds fifo_overlap_after_max
(the largest overlap with the FIFO after the projection; fifo_overlap_max stays the overlap before it); checkpoint stores each FIFO mode
at its own collection time tq (v1 stored it propagated to t_ref under the label tq, so a resumed run advanced its phase a second time; the
icloud lane's finding, SPEC_final4b_engine.md 29.9; the transition's time origin, the other finding, is fixed in page_core_v2); resume
marks where the stage record's stored modes restart (rec["cQ_from"], read by page_core_v2.stage_direct); the file names in the printed
text. Nothing else changes.

page_recursion_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the Page curve of the model's radiating object by
the retained-channel recursion (REVIEWA_REVIEW_web40e.md S5; GAP03_integrated_v1.md). The sea of the initial configuration is carried through
the declared formation and through the exact exterior sequence (N = 1, v = -sqrt(2GE/r) outside a declared regular ball) by one-particle
propagation; per channel kappa (weight 16 kappa) the memory covariance obeys M_{k+1} = D_k M_k D_k^dagger + Xi_k, S_esc = tr s(M), the
new block's entropy S(B_k) = tr s(Z_k), I(R_<k : B_k) = tr s(M_k) + tr s(Z_k) - tr s(M_{k+1}), turnover iff I > S(B_k). A stage ends when
the collected Killing energy reaches dE_frac E_k (the review's E_{n+1} = 0.995 E_n), the charge drops by the measured energy and the state is
carried through a smooth ramp. Nothing thermal is put in. Outputs: <out>/<tag>_summary.json, <tag>_tables.md, <tag>_curve.npz, the
checkpoint <tag>_ckpt.pt (resume with --resume). Defaults are the r_s/a = 32 pilot (G/a^2 = 1/4, massless).

Usage: python page_recursion_v2.py --out DIR --tag TAG [--rs 32 --g 0.25 --nsites 8192 --channels 1,2 ...]
       python page_recursion_v2.py --selftest [--device cpu] [--engine-path DIR]
       python page_recursion_v2.py --validate-scattering --out DIR --tag TAG [--rs 32 ...]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v2 as OPS            # noqa: E402
import page_core_v2 as CORE          # noqa: E402

CD = OPS.CD
RD = OPS.RD
L_CONT_RS2 = 0.0052373      # REVIEWA_REVIEW_web40e.md 1.11 (continuum massless luminosity, 16 two-component species)
B_CONT = 1.6391             # REVIEWA_REVIEW_web40e.md 1.11 (continuum stationary entropy ratio)
T_START = time.time()


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T_START, msg), flush=True)


# ------------------------------------------------------------------------------------------------------------------------ config
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs", type=float, default=32.0, help="r_s/a of the first stage")
    ap.add_argument("--g", type=float, default=0.25, help="G/a^2")
    ap.add_argument("--m", type=float, default=0.0, help="m a (the pilot is massless; 4 pi m r_s is printed beside the collapse's 160)")
    ap.add_argument("--nsites", type=int, default=8192)
    ap.add_argument("--half-offset", dest="half_offset", type=int, default=0)
    ap.add_argument("--channels", default="1,2", help="kappa list, weights 16 kappa; 3 is the truncation check")
    ap.add_argument("--ell", type=float, default=4.0)
    ap.add_argument("--fields", default="averaged", choices=["averaged", "point"])
    ap.add_argument("--Rb", type=float, default=0.5, help="ball radius over r_s (0.8 control)")
    ap.add_argument("--Rb0", type=float, default=4.0, help="formation start radius over r_s (8 control)")
    ap.add_argument("--formation", default="dust", choices=["dust", "ramp"])
    ap.add_argument("--t-ramp", dest="t_ramp", type=float, default=2.0, help="ramp formation duration over r_s")
    ap.add_argument("--dt-form", dest="dt_form", type=float, default=0.5)
    ap.add_argument("--form-chunk", dest="form_chunk", type=int, default=4096, help="column chunk of the formation propagation (device memory)")
    ap.add_argument("--R-cut", dest="R_cut", type=float, default=32.0, help="collection radius over r_s,1 (fixed through the sequence)")
    ap.add_argument("--R-collar", dest="R_collar", type=float, default=16.0, help="collar radius over r_s,1 for the current readout")
    ap.add_argument("--tau-s", dest="tau_s", type=float, default=4.0, help="checkpoint spacing over r_s,1")
    ap.add_argument("--m-cap", dest="m_cap", type=int, default=64, help="largest number of collected directions per checkpoint and channel")
    ap.add_argument("--eps-coll", dest="eps_coll", type=float, default=1e-5)
    ap.add_argument("--eps-rank", dest="eps_rank", type=float, default=1e-9)
    ap.add_argument("--eps-form", dest="eps_form", type=float, default=1e-7)
    ap.add_argument("--eps-orth", dest="eps_orth", type=float, default=1e-12)
    ap.add_argument("--fifo-depth", dest="fifo_depth", type=int, default=12)
    ap.add_argument("--fifo-prune-weight", dest="fifo_prune_weight", type=float, default=1e-10, help="a collected mode leaves the FIFO below this weight near the cut")
    ap.add_argument("--dE-frac", dest="dE_frac", type=float, default=0.005, help="stage ends when E_rad reaches dE_frac E_k")
    ap.add_argument("--interior", default="selfsimilar", choices=["selfsimilar", "fixed"])
    ap.add_argument("--tau-tr", dest="tau_tr", type=float, default=4.0, help="ramp duration over r_s,k at a transition")
    ap.add_argument("--dt-tr", dest="dt_tr", type=float, default=0.5)
    ap.add_argument("--max-stages", dest="max_stages", type=int, default=200)
    ap.add_argument("--E-stop-frac", dest="E_stop_frac", type=float, default=0.5)
    ap.add_argument("--t-stage-max", dest="t_stage_max", type=float, default=3000.0, help="stage cap over r_s,1")
    ap.add_argument("--stall-frac", dest="stall_frac", type=float, default=0.01)
    ap.add_argument("--stall-window", dest="stall_window", type=int, default=10)
    ap.add_argument("--t-stall-min", dest="t_stall_min", type=float, default=200.0, help="no stall verdict before this time over r_s,1")
    ap.add_argument("--force-steps", dest="force_steps", type=int, default=0, help="labelled control: step the charge by dE_frac E_k on a stall")
    ap.add_argument("--store-cap", dest="store_cap", type=int, default=6000)
    ap.add_argument("--store-modes", dest="store_modes", type=int, default=1)
    ap.add_argument("--direct-every", dest="direct_every", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.9145, help="the lane's sphere coefficient at mesh 4 (REVIEWA_REVIEW_web40e.md)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-wall-hours", dest="max_wall_hours", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--engine-path", dest="engine_path", default=None, help="directory holding engine_ops_v1.py for the self-test's T1a")
    ap.add_argument("--validate-scattering", dest="validate_scattering", action="store_true")
    a = ap.parse_args(argv)
    finish_cfg(a)
    return a


def finish_cfg(a):
    a.E1 = a.rs / (2.0 * a.g)
    a.TH = 1.0 / (4.0 * math.pi * a.rs)
    a.kappa_surf = 1.0 / (2.0 * a.rs)
    a.R_cut_abs = a.R_cut * a.rs
    a.R_collar_abs = a.R_collar * a.rs
    a.tau_s_abs = a.tau_s * a.rs
    a.m_over_TH = 4.0 * math.pi * a.m * a.rs
    a.Rbf = a.Rb
    a.channel_list = [int(x) for x in str(a.channels).split(",") if x.strip()]
    a.store_modes = bool(a.store_modes)
    return a


def levels(a, rs):
    return {"A_over_4G": math.pi * rs * rs / a.g, "eta_A_over_a2": 4.0 * math.pi * a.eta * rs * rs,
            "memory_budget_modes": 4.0 * math.pi * a.eta / math.log(2.0) * rs * rs, "L_cont": L_CONT_RS2 / (rs * rs),
            "S_dot_cont": B_CONT * L_CONT_RS2 / (rs * rs) * 4.0 * math.pi * rs}


def Rb_of(a, E):
    rs = 2.0 * a.g * E
    return a.Rb * rs if a.interior == "selfsimilar" else a.Rb * a.rs


# ---------------------------------------------------------------------------------------------------------------------- driver
class Run:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
        torch.set_num_threads(a.threads)
        os.makedirs(a.out, exist_ok=True)
        self.prefix = os.path.join(a.out, a.tag)
        self.chans = [CORE.Channel(k, a.m, a.nsites, bool(a.half_offset), self.dev) for k in a.channel_list]
        for ch in self.chans:
            ch.set_averager(a.ell, a.fields)
        self.curve = []; self.stages = []; self.transitions = []; self.formation = {}; self.inversion = {}
        self.k = 0; self.E = a.E1; self.t = 0.0; self.t_stage0 = 0.0; self.n_block = 0; self.E_rad_stage = 0.0; self.E_rad_total = 0.0
        self.L_hist = []; self.L_peak = 0.0; self.Rb_now = Rb_of(a, a.E1); self.stop_reason = None
        self.checks = {"Z_identity_max": 0.0, "energy_balance_max_rel": 0.0, "clip_excursion_max": 0.0, "fifo_overlap_max": 0.0, "fifo_overlap_after_max": 0.0,
                       "Xi_min_eig_min": 0.0, "I_minus_DD_minus_Xi_min": 0.0, "direct_vs_memory_max_rel": 0.0}

    # ------------------------------------------------------------------------------------------------------------ start
    def start(self):
        a = self.a
        for ch in self.chans:
            Phi0, frec = ch.formation(a)
            log("kappa %d: formation %s t_form %.2f r_s, n0 %d, zero modes %d, u_max %.3f, unitarity %.2e (eigh %.0f s, cf4 %.0f s)" % (
                ch.kappa, a.formation, frec["t_form_over_rs"], frec["n0"], frec["n_zero_modes_initial"], frec["u_max_initial"], frec["unitarity"], frec["eigh_s"], frec["cf4_s"]))
            st = ch.build_stage(a, a.E1, self.Rb_now, label="stage 1")
            irec = ch.initial_memory(Phi0, a)
            del Phi0
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
            log("kappa %d: stage 1 r_s %.3f r_t %s r_in %s n_neg %d (eigh %.0f s); Delta_0 rank %d (sin theta max %.3e), memory rank %d; "
                "N+ %.4f N- %.4f E_inv/E %.3e; E_inv per bin of T_H %s" % (
                    ch.kappa, st["rs"], st["r_t"], st["r_in"], st["nneg"], st["eigh_s"], irec["rank_kept"], irec["sin_theta_max"], irec["memory_rank"],
                    irec["N_plus"], irec["N_minus"], irec["E_inv_over_E"], ["%.3f" % x for x in irec["inverted_by_bin_TH"]]))
            self.formation[str(ch.kappa)] = frec; self.inversion[str(ch.kappa)] = irec
        self.t = 0.0; self.t_stage0 = 0.0; self.k = 0; self.E = a.E1; self.E_rad_stage = 0.0

    # ------------------------------------------------------------------------------------------------------------- block
    def block(self):
        a = self.a
        self.t += a.tau_s_abs
        self.n_block += 1
        rs_k = 2.0 * a.g * self.E
        row = {"t": self.t, "t_over_rs1": self.t / a.rs, "t_inf": OPS.chart_retarded_time(self.t, rs_k, a.R_cut_abs) - OPS.chart_retarded_time(0.0, a.rs, a.R_cut_abs),
               "stage": self.k, "E": self.E, "E_over_E1": self.E / a.E1, "per_channel": {}}
        S_tot = S_B_tot = I_tot = E_B_tot = Lc_tot = E_mem_tot = 0.0; rank_tot = 0.0; wall = 0.0; S_dir_tot = 0.0; have_dir = True
        for ch in self.chans:
            r = ch.block_step(a, self.t)
            w = ch.w
            S_tot += w * r["S_new"]; S_B_tot += w * r["S_B"]; I_tot += w * r["I_RB"]; E_B_tot += w * r["E_B"]
            Lc_tot += w * r["L_collar"]; E_mem_tot += w * r["E_mem"]; rank_tot += w * r["r_after"]; wall += r["wall_s"]
            if r["m_coll"]:
                self.checks["Z_identity_max"] = max(self.checks["Z_identity_max"], abs(r["S_B_Z"] - r["S_B"]))
                self.checks["clip_excursion_max"] = max(self.checks["clip_excursion_max"], r.get("clip_excursion_max", 0.0))
                self.checks["fifo_overlap_max"] = max(self.checks["fifo_overlap_max"], r.get("fifo_overlap_max", 0.0))
                self.checks["fifo_overlap_after_max"] = max(self.checks.get("fifo_overlap_after_max", 0.0), r.get("fifo_overlap_after", 0.0))
                self.checks["Xi_min_eig_min"] = min(self.checks["Xi_min_eig_min"], r.get("Xi_min_eig", 0.0))
                self.checks["I_minus_DD_minus_Xi_min"] = min(self.checks["I_minus_DD_minus_Xi_min"], r.get("I_minus_DD_minus_Xi_min_eig", 0.0))
            # energy balance within the stage: E_mem + E_rad(stage) against the stage's initial excess
            rec = ch.rec
            bal = (r["E_mem"] + rec["E_rad"] - rec["E_delta_init"]) / max(abs(rec["E_delta_init"]), 1e-300)
            self.checks["energy_balance_max_rel"] = max(self.checks["energy_balance_max_rel"], abs(bal))
            r["energy_balance_rel"] = bal
            if self.n_block % a.direct_every == 0:
                d = ch.stage_direct(a)
                if d is not None and d["m"] == rec["n_coll"]:
                    r["S_direct"] = d["S_direct"]; S_dir_tot += w * d["S_direct"]
                    self.checks["direct_vs_memory_max_rel"] = max(self.checks["direct_vs_memory_max_rel"], abs(d["S_direct"] - r["S_new"]) / max(r["S_new"], 1e-12))
                else:
                    have_dir = False
            else:
                have_dir = False
            row["per_channel"][str(ch.kappa)] = {k_: v_ for k_, v_ in r.items() if not isinstance(v_, (torch.Tensor,))}
        self.E_rad_stage += E_B_tot; self.E_rad_total += E_B_tot
        L_now = E_B_tot / a.tau_s_abs
        self.L_hist.append(L_now)
        row.update({"S_esc_total": S_tot, "S_direct_total": S_dir_tot if have_dir else None, "S_B_total": S_B_tot, "I_total": I_tot,
                    "turnover_total": bool(I_tot > S_B_tot + 1e-9), "E_rad_stage": self.E_rad_stage, "E_rad_total": self.E_rad_total,
                    "L_collected": L_now, "L_collar": Lc_tot, "L_cont": levels(a, rs_k)["L_cont"], "E_mem_total": E_mem_tot,
                    "memory_rank_weighted": rank_tot, "wall_s": wall,
                    "gpu_mem_GB": (torch.cuda.max_memory_allocated() / 1e9) if self.dev.type == "cuda" else None})
        self.curve.append(row)
        if self.n_block % max(1, a.direct_every) == 0 or self.n_block <= 3:
            log("stage %d block %d t %.1f r_s: S_esc %.5f (direct %s) S_B %.3e I %.3e turnover %s; E_rad(stage) %.4e of dE %.4e; L %.3e (collar %.3e, cont %.3e); rank %.0f; %.2f s" % (
                self.k, self.n_block, self.t / a.rs, S_tot, ("%.5f" % S_dir_tot) if have_dir else "-", S_B_tot, I_tot, row["turnover_total"],
                self.E_rad_stage, a.dE_frac * self.E, L_now, Lc_tot, row["L_cont"], rank_tot, wall))
        return row

    def stage_verdict(self):
        """normal: E_rad >= dE; stall: late and the recent flux below stall_frac of its peak; cap: the stage duration cap."""
        a = self.a
        dE = a.dE_frac * self.E
        t_stage = self.t - self.t_stage0
        if self.E_rad_stage >= dE:
            return "reached dE"
        n = a.stall_window
        if len(self.L_hist) >= n + 2:
            L_recent = float(np.mean(self.L_hist[-n:]))
            self.L_peak = max(self.L_peak, L_recent)
            if t_stage >= a.t_stall_min * a.rs and self.L_peak > 0 and L_recent < a.stall_frac * self.L_peak:
                return "stalled"
        if t_stage >= a.t_stage_max * a.rs:
            return "stage cap"
        return None

    def end_stage(self, reason):
        a = self.a
        t_stage = self.t - self.t_stage0
        rs_k = 2.0 * a.g * self.E
        row = {"k": self.k, "E_k": self.E, "rs_k": rs_k, "duration": t_stage, "duration_over_rs1": t_stage / a.rs, "E_rad": self.E_rad_stage,
               "dE_target": a.dE_frac * self.E, "L_mean": self.E_rad_stage / max(t_stage, 1e-300), "L_cont": levels(a, rs_k)["L_cont"],
               "L_peak_recent": self.L_peak, "end": reason, "n_blocks": self.n_block, "S_esc_end": self.curve[-1]["S_esc_total"] if self.curve else 0.0,
               "levels_now": levels(a, rs_k)}
        self.stages.append(row)
        log("stage %d ends (%s): emitted %.4e of %.4e over %.1f r_s; L_mean %.3e against the continuum thermal %.3e" % (
            self.k, reason, self.E_rad_stage, a.dE_frac * self.E, t_stage / a.rs, row["L_mean"], row["L_cont"]))

    def do_transition(self, E_new, forced=False):
        a = self.a
        Rb_new = Rb_of(a, E_new)
        trs = {}
        for ch in self.chans:
            tr = ch.transition(a, E_new, Rb_new)
            trs[str(ch.kappa)] = tr
            log("kappa %d transition: r_s %.3f -> %.3f, released up %d down %d, ambiguous %d, adiabatic leak max %.2e, S continuity %.2e, Xi_tr %.2e, rank %d (%.0f s)" % (
                ch.kappa, tr["rs_old"], tr["rs_new"], tr["released_pushed_up"], tr["released_new_pockets"], tr["ambiguous_continuations"],
                tr["adiabatic_leak_max"], tr["S_continuity_defect"], tr["Xi_tr_norm"], tr["memory_rank"], tr["wall_s"]))
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.transitions.append({"k_from": self.k, "E_from": self.E, "E_to": E_new, "forced": forced, "per_channel": trs,
                                 "t": self.t, "tau_tr": a.tau_tr * 2.0 * a.g * self.E})
        self.t += a.tau_tr * 2.0 * a.g * self.E
        self.E = E_new; self.Rb_now = Rb_new; self.k += 1
        self.t_stage0 = self.t; self.E_rad_stage = 0.0; self.L_hist = []; self.L_peak = 0.0

    # -------------------------------------------------------------------------------------------------------------- loop
    def main_loop(self):
        a = self.a
        while True:
            self.block()
            if self.n_block % a.ckpt_every == 0:
                self.checkpoint(); self.write_outputs(final=False)
            if (time.time() - T_START) / 3600.0 > a.max_wall_hours:
                self.stop_reason = "wall clock cap"; self.end_stage("wall clock cap"); break
            verdict = self.stage_verdict()
            if verdict is None:
                continue
            self.end_stage(verdict)
            E_new = self.E - self.E_rad_stage
            if verdict != "reached dE":
                if a.force_steps:
                    E_new = self.E - a.dE_frac * self.E
                    log("forced step (labelled control, not the model's balance): E -> %.6f" % E_new)
                    forced = True
                else:
                    self.stop_reason = verdict; break
            else:
                forced = False
            if self.k + 1 >= a.max_stages or E_new <= a.E_stop_frac * a.E1 or 2.0 * a.g * E_new <= 1.05 * (self.Rb_now if a.interior == "fixed" else 0.0):
                self.stop_reason = "sequence end (stages %d, E/E1 %.4f)" % (self.k + 1, E_new / a.E1); break
            self.do_transition(E_new, forced=forced)
            self.checkpoint()
        self.finish()

    # --------------------------------------------------------------------------------------------- checkpoint, resume
    def checkpoint(self):
        a = self.a
        state = {"cfg": {k_: v_ for k_, v_ in vars(a).items() if not callable(v_)}, "k": self.k, "E": self.E, "t": self.t, "t_stage0": self.t_stage0,
                 "n_block": self.n_block, "E_rad_stage": self.E_rad_stage, "E_rad_total": self.E_rad_total, "L_hist": self.L_hist, "L_peak": self.L_peak,
                 "Rb_now": self.Rb_now, "curve": self.curve, "stages": self.stages, "transitions": self.transitions, "formation": self.formation,
                 "inversion": self.inversion, "checks": self.checks, "channels": {}}
        for ch in self.chans:
            V = ch.stage["V"]; e = ch.stage["e"].to(CD)
            D = (V @ ch.c).cpu()
            # v2: each FIFO mode is stored at its own collection time tq, the time saved beside it; v1 stored it propagated to t_ref with
            # the label tq, so resume's block_step propagated it from tq a second time (the icloud lane's finding, SPEC 29.9)
            fifo = [((V @ cq).cpu(), hq.cpu(), tq) for cq, hq, tq in ch.fifo]
            rec = ch.rec
            recs = None
            if rec is not None:
                recs = {"D_init": (V @ rec["c_init"]).cpu(), "MD_init": rec["MD_init"].cpu(), "t_init": rec["t_init"],
                        "a": [x.cpu() for x in rec["a"]], "halfQ": [x.cpu() for x in rec["halfQ"]], "tQ": rec["tQ"], "n_coll": rec["n_coll"],
                        "E_rad": rec["E_rad"], "E_delta_init": rec["E_delta_init"]}
            state["channels"][str(ch.kappa)] = {"D": D, "half": ch.half.cpu(), "MD": ch.MD.cpu(), "t_ref": ch.t_ref, "fifo": fifo, "rec": recs,
                                                 "ent_bound": ch.ent_bound}
        tmp = self.prefix + "_ckpt.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.prefix + "_ckpt.pt")

    def resume(self):
        a = self.a
        state = torch.load(self.prefix + "_ckpt.pt", map_location="cpu", weights_only=False)
        for key in ("k", "E", "t", "t_stage0", "n_block", "E_rad_stage", "E_rad_total", "L_hist", "L_peak", "Rb_now", "curve", "stages", "transitions",
                    "formation", "inversion", "checks"):
            setattr(self, key, state[key])
        for ch in self.chans:
            s = state["channels"][str(ch.kappa)]
            st = ch.build_stage(a, self.E, self.Rb_now, label="stage %d (resumed)" % (self.k + 1))
            V, neg = st["V"], st["neg"]
            def to_eig(Dcpu, half):
                c = V.mH @ Dcpu.to(self.dev)
                half = half.to(self.dev)
                for hsign in (1, -1):
                    cols = (half == hsign).nonzero().ravel()
                    if cols.numel():
                        rows = neg if hsign > 0 else ~neg
                        sub = c[:, cols]; sub[rows] = 0.0; c[:, cols] = sub
                nrm = torch.linalg.norm(c, dim=0).clamp_min(1e-300)
                return c / nrm[None, :].to(CD), half
            ch.c, ch.half = to_eig(s["D"], s["half"])
            ch.MD = s["MD"].to(self.dev); ch.t_ref = s["t_ref"]; ch.ent_bound = s["ent_bound"]
            ch.fifo = []
            for Dq, hq, tq in s["fifo"]:
                cq, hq2 = to_eig(Dq, hq)
                ch.fifo.append((cq, hq2, tq))
            r = s["rec"]
            if r is not None:
                # the stored D_init columns are split; recover their halves from the support
                c_raw = V.mH @ r["D_init"].to(self.dev)
                wneg = (c_raw.abs().pow(2) * neg.to(RD)[:, None]).sum(dim=0)
                hinit = torch.where(wneg > 0.5, -torch.ones_like(wneg, dtype=torch.int64), torch.ones_like(wneg, dtype=torch.int64))
                c_init, _ = to_eig(r["D_init"], hinit.cpu())
                ch.rec = {"c_init": c_init, "MD_init": r["MD_init"].to(self.dev), "t_init": r["t_init"], "a": [x.to(self.dev) for x in r["a"]],
                          "cQ": [], "cQ_from": (len(r["tQ"]), r["n_coll"]),     # v2: the stored modes restart here (page_core_v2.stage_direct)
                          "halfQ": [x.to(self.dev) for x in r["halfQ"]], "tQ": r["tQ"], "n_coll": r["n_coll"], "E_rad": r["E_rad"],
                          "E_delta_init": r["E_delta_init"]}
            else:
                ch.init_stage_record()
        log("resumed at stage %d, block %d, t %.1f r_s, E %.6f" % (self.k, self.n_block, self.t / a.rs, self.E))

    # ------------------------------------------------------------------------------------------------------------ outputs
    def finish(self):
        a = self.a
        self.readouts = {}
        for ch in self.chans:
            d = ch.stage_direct(a, with_spectrum=True)
            self.readouts[str(ch.kappa)] = {"stage_direct": {k_: v_ for k_, v_ in (d or {}).items() if k_ != "spectrum_omega_over_TH_occupation_half"},
                                            "spectrum": (d or {}).get("spectrum_omega_over_TH_occupation_half"),
                                            "partners": ch.partner_readout(a), "cross_surface": ch.cross_surface(a), "ent_bound": ch.ent_bound}
        self.checkpoint()
        self.write_outputs(final=True)
        log("done: %s; outputs %s_summary.json, _tables.md, _curve.npz" % (self.stop_reason, self.prefix))

    def write_outputs(self, final):
        a = self.a
        lv1 = levels(a, a.rs)
        summary = {"settings": {k_: v_ for k_, v_ in vars(a).items() if not callable(v_)},
                   "derived": {"E1": a.E1, "T_H": a.TH, "kappa_surface": a.kappa_surf, "m_over_TH": a.m_over_TH, "collapse_m_over_TH": 160.0,
                               "levels_initial": lv1, "L_cont_rs2": L_CONT_RS2, "b_cont": B_CONT, "device": str(self.dev)},
                   "formation": self.formation, "inversion": self.inversion, "stages": self.stages, "transitions": self.transitions,
                   "checks": self.checks, "stop_reason": self.stop_reason, "n_blocks": self.n_block, "wall_s": time.time() - T_START,
                   "final": final, "readouts": getattr(self, "readouts", None), "curve": self.curve}
        with open(self.prefix + "_summary.json", "w") as f:
            json.dump(summary, f, default=_jsonable, indent=1)
        self.write_npz()
        self.write_md(summary)

    def write_npz(self):
        a = self.a
        rows = self.curve
        if not rows:
            return
        arr = {"t": np.array([r["t"] for r in rows]), "t_inf": np.array([r["t_inf"] for r in rows]), "stage": np.array([r["stage"] for r in rows]),
               "E": np.array([r["E"] for r in rows]), "S_esc_total": np.array([r["S_esc_total"] for r in rows]),
               "S_direct_total": np.array([r["S_direct_total"] if r["S_direct_total"] is not None else np.nan for r in rows]),
               "S_B_total": np.array([r["S_B_total"] for r in rows]), "I_total": np.array([r["I_total"] for r in rows]),
               "E_rad_total": np.array([r["E_rad_total"] for r in rows]), "L_collected": np.array([r["L_collected"] for r in rows]),
               "L_collar": np.array([r["L_collar"] for r in rows]), "L_cont": np.array([r["L_cont"] for r in rows]),
               "memory_rank_weighted": np.array([r["memory_rank_weighted"] for r in rows]),
               "A_over_4G_initial": levels(a, a.rs)["A_over_4G"], "eta_A_over_a2_initial": levels(a, a.rs)["eta_A_over_a2"],
               "memory_budget_initial": levels(a, a.rs)["memory_budget_modes"], "rs1": a.rs, "g": a.g, "eta": a.eta, "TH": a.TH, "E1": a.E1}
        for ch in self.chans:
            key = str(ch.kappa)
            arr["S_esc_kappa%s" % key] = np.array([r["per_channel"][key]["S_new"] for r in rows])
            arr["rank_kappa%s" % key] = np.array([r["per_channel"][key]["r_after"] for r in rows])
        arr["stage_E"] = np.array([s["E_k"] for s in self.stages]); arr["stage_duration"] = np.array([s["duration"] for s in self.stages])
        arr["stage_L_mean"] = np.array([s["L_mean"] for s in self.stages]); arr["stage_E_rad"] = np.array([s["E_rad"] for s in self.stages])
        np.savez_compressed(self.prefix + "_curve.npz", **arr)

    def write_md(self, summary):
        a = self.a
        lv = levels(a, a.rs)
        L = []
        L.append("# %s: the Page curve by the retained-channel recursion (%s)\n" % (a.tag, "final" if summary["final"] else "running"))
        L.append("Settings: r_s/a %g, G/a^2 %g, m a %g (m/T_H = 4 pi m r_s = %.3f beside the collapse's 160), channels %s (weights 16 kappa), n_sites %d, ell %g (%s), ball R_b %g r_s (%s), formation %s from %g r_s, R_cut %g r_s, tau_s %g r_s, dE/E %g, eta %g, device %s. Exterior: on the exact exterior sequence (no certified solved exterior exists)." % (
            a.rs, a.g, a.m, a.m_over_TH, a.channel_list, a.nsites, a.ell, a.fields, a.Rb, a.interior, a.formation, a.Rb0, a.R_cut, a.tau_s, a.dE_frac, a.eta, summary["derived"]["device"]))
        L.append("Derived: E a %.4f, T_H a %.4e, kappa a %.4e, A/(4G) %.1f, eta A/a^2 %.1f, memory budget %.0f modes, L_cont %.4e (0.0052373/r_s^2), S_dot_cont %.4e per unit time (b = 1.6391).\n" % (
            a.E1, a.TH, a.kappa_surf, lv["A_over_4G"], lv["eta_A_over_a2"], lv["memory_budget_modes"], lv["L_cont"], lv["S_dot_cont"]))
        L.append("## Formation and inversion per channel\n")
        L.append("| kappa | t_form/r_s | n0 | unitarity | Delta_0 rank | memory rank | N+ | N- | E_inv/E | E_inv by bin of T_H [0,1,2,3,4,6,8,12,25,inf) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k_, f in summary["formation"].items():
            i = summary["inversion"].get(k_, {})
            L.append("| %s | %.3f | %d | %.2e | %s | %s | %.4f | %.4f | %.3e | %s |" % (k_, f["t_form_over_rs"], f["n0"], f["unitarity"], i.get("rank_kept"), i.get("memory_rank"),
                                                                              i.get("N_plus", 0), i.get("N_minus", 0), i.get("E_inv_over_E", 0), ["%.3f" % x for x in i.get("inverted_by_bin_TH", [])]))
        L.append("\n## Stage table\n")
        L.append("| k | E_k | r_s,k | duration/r_s1 | E_rad | dE target | L_mean | L_cont | end | S_esc at end | A/(4G) now | eta A/a^2 now |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in summary["stages"]:
            L.append("| %d | %.5f | %.4f | %.1f | %.4e | %.4e | %.3e | %.3e | %s | %.4f | %.1f | %.1f |" % (s["k"], s["E_k"], s["rs_k"], s["duration_over_rs1"], s["E_rad"], s["dE_target"],
                                                                                           s["L_mean"], s["L_cont"], s["end"], s["S_esc_end"], s["levels_now"]["A_over_4G"], s["levels_now"]["eta_A_over_a2"]))
        L.append("\n## Transitions\n")
        for tr in summary["transitions"]:
            for k_, t in tr["per_channel"].items():
                L.append("- stage %d -> %d, kappa %s: r_s %.4f -> %.4f, released pushed-up %d, new pockets %d, ambiguous %d, adiabatic leak max %.2e, S continuity defect %.2e, |Xi_tr| %.2e, rank %d%s" % (
                    tr["k_from"], tr["k_from"] + 1, k_, t["rs_old"], t["rs_new"], t["released_pushed_up"], t["released_new_pockets"], t["ambiguous_continuations"],
                    t["adiabatic_leak_max"], t["S_continuity_defect"], t["Xi_tr_norm"], t["memory_rank"], " (forced step, labelled control)" if tr["forced"] else ""))
        L.append("\n## The curve (every row while there are at most 200, else every 10th)\n")
        L.append("| t/r_s1 | t_inf/r_s1 | stage | E/E1 | S_esc total | S_direct | S_B total | I total | turnover | E_rad total | L collected | L collar | L_cont | rank (weighted) | s per block |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = summary["curve"]
        step = 1 if len(rows) <= 200 else 10
        for r in rows[::step]:
            L.append("| %.1f | %.1f | %d | %.5f | %.5f | %s | %.3e | %.3e | %s | %.4e | %.3e | %.3e | %.3e | %.0f | %.2f |" % (
                r["t_over_rs1"], r["t_inf"] / a.rs, r["stage"], r["E_over_E1"], r["S_esc_total"], ("%.5f" % r["S_direct_total"]) if r["S_direct_total"] is not None else "-",
                r["S_B_total"], r["I_total"], r["turnover_total"], r["E_rad_total"], r["L_collected"], r["L_collar"], r["L_cont"], r["memory_rank_weighted"], r["wall_s"]))
        L.append("\n## Saturation comparison\n")
        if rows:
            smax = max(r["S_esc_total"] for r in rows)
            L.append("Largest S_esc total %.4f nats at t %.1f r_s1; against eta A/a^2 (initial) %.1f: ratio %.4e; against A/(4G) (initial) %.1f: ratio %.4e. The thermal comparator over the same time would give S_dot_cont t = %.4f nats (continuum, stationary channel; not an input)." % (
                smax, [r for r in rows if r["S_esc_total"] == smax][0]["t_over_rs1"], lv["eta_A_over_a2"], smax / lv["eta_A_over_a2"], lv["A_over_4G"], smax / lv["A_over_4G"], lv["S_dot_cont"] * rows[-1]["t"]))
        L.append("\n## Checks\n")
        for k_, v_ in summary["checks"].items():
            L.append("- %s: %.3e" % (k_, v_))
        if summary.get("readouts"):
            for k_, rd in summary["readouts"].items():
                L.append("- kappa %s: accumulated truncation entropy bound %.3e nats; stage direct route %s" % (k_, rd["ent_bound"], json.dumps(rd["stage_direct"], default=_jsonable)))
                for p in rd["partners"][:4]:
                    L.append("  - partner: n %.4e, ||w||^2 %.4e against n(1-n) %.4e; weights core %.3e annulus %.3e outside r_t %.3e beyond cut %.3e; negative-half %.3e" % (
                        p["n"], p["norm2_w"], p["n_1_minus_n"], p["w_core"], p["w_annulus"], p["w_outside_rt"], p["w_beyond_cut"], p["w_negative_half"]))
                for cs in rd["cross_surface"]:
                    L.append("  - cross-surface d %d: I physical %.4e (bound %.3e), Killing vacuum %.4e (bound %.3e)" % (cs["d"], cs["physical"]["I"], cs["physical"]["bound_2K2"], cs["killing_vacuum"]["I"], cs["killing_vacuum"]["bound_2K2"]))
                spec = rd.get("spectrum")
                if spec:
                    L.append("  - energy-resolved collected occupations of the last stage (omega/T_H bins; mean excess occupation; count):")
                    edges = [0, 1, 2, 3, 4, 6, 8, 12, 25, 1e9]
                    for i in range(len(edges) - 1):
                        sel = [s for s in spec if edges[i] <= abs(s[0]) < edges[i + 1]]
                        if sel:
                            L.append("    - [%g, %g): particles %.3e (n %d), holes %.3e (n %d)" % (edges[i], edges[i + 1], np.mean([s[1] for s in sel if s[2] > 0] or [0.0]), len([s for s in sel if s[2] > 0]),
                                                                                           np.mean([s[1] for s in sel if s[2] < 0] or [0.0]), len([s for s in sel if s[2] < 0])))
        L.append("\nStop: %s. Wall %.0f s. Every number above is computed by page_recursion_v2.py; nothing thermal is an input." % (summary["stop_reason"], summary["wall_s"]))
        with open(self.prefix + "_tables.md", "w") as f:
            f.write("\n".join(L) + "\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, torch.device):
        return str(o)
    return str(o)


def main(argv=None):
    a = parse(argv)
    if a.selftest:
        import page_selftest_v2 as ST
        ok = ST.run_all(a)
        sys.exit(0 if ok else 1)
    if a.out is None or a.tag is None:
        raise SystemExit("--out and --tag are required")
    if a.validate_scattering:
        import page_scatter_v2 as SC
        SC.validate(a)
        return
    run = Run(a)
    log("settings: r_s/a %g, g %g, m a %g (m/T_H %.3f; collapse 160), channels %s, nsites %d, ell %g %s, R_b %g r_s (%s), formation %s, R_cut %g r_s, tau_s %g r_s; E a %.4f, T_H a %.4e, A/(4G) %.1f, eta A/a^2 %.1f, budget %.0f modes, L_cont %.3e" % (
        a.rs, a.g, a.m, a.m_over_TH, a.channel_list, a.nsites, a.ell, a.fields, a.Rb, a.interior, a.formation, a.R_cut, a.tau_s, a.E1, a.TH,
        levels(a, a.rs)["A_over_4G"], levels(a, a.rs)["eta_A_over_a2"], levels(a, a.rs)["memory_budget_modes"], levels(a, a.rs)["L_cont"]))
    if a.resume and os.path.exists(run.prefix + "_ckpt.pt"):
        run.resume()
    else:
        run.start()
        run.checkpoint()
    run.main_loop()


if __name__ == "__main__":
    main()
