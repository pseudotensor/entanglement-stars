#!/usr/bin/env python
"""build_page_history_v6n.py: page_history_v6n.py built from page_history_v6m.py by an exact replacement list (each "old" required to
match exactly once in the evolving text; abort otherwise); v6m is not touched. Usage:
    python build_page_history_v6n.py [--src code_v6/page_history_v6m.py] [--dst page_history_v6n.py]
The built file is compiled (py_compile) and the replacements are listed. page_history_v6n imports page_d13_readout_v1 (same directory or
$PAGE_CODE_V6 beside page_ops_v6, page_core_v6, page_scatter_v5)."""
import argparse
import hashlib
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

V6N_DOC = r'''"""page_history_v6n.py (reviewer A for the solve builder, the busi lane; reorientation 4, the solve entry's item 2): page_history_v6m.py
with the collector-free run as the default (--collector off) and v6m's code paths kept behind --collector on (then every v6m option is
v6m's; the records gain the keys cut_flow and cut_jump, empty). Built from v6m by build_page_history_v6n.py's exact replacement list.
With --collector off:
(1) no archive, no selection, no memory representation (principal_delta), no Z (energy/collar) identity with an archive, no transfer
    ledgers; E_inv = Tr[h C] - Tr min(h, 0) with Tr min(h, 0) from the banded operator's eigenvalues alone (page_d13_readout_v1.tr_min_h0,
    LAPACK zhbevd); the dense eigendecomposition of h only at the baseline clocks (--vac-every-rs 10, the reference clocks 12, 20, 40 and
    --vac-extra-rs), where E_out, W_hdot, the exterior-cut vacuum entropies and the deviation's density are read (None elsewhere; the
    weighted sums propagate None, W (finite difference) only between two baseline rows). The massless chain's conjugate-pair structure
    (h = i A, A real antisymmetric) is available for that eigendecomposition as --vac-route pair (the real eigenproblem of h^2, half the
    work; dense is the default, bit for bit v6m's).
(2) item 1's readouts inline at every checkpoint (page_d13_readout_v1.ChannelReadout.checkpoint): the exterior-cut entropies at 16, 24,
    30 and 48 r_s with the vacuum's at the baseline clocks; the ledger's inventories at every checkpoint with the per-record flows and
    jumps at the ledger cuts (step_record: records' per-channel cut_flow and cut_jump) and the windows (<tag>_d13.json "windows"); the
    exterior shells' hard/soft content; the collar flow's soft/hard split (v6k's windowed reading at the collar, the P_- block of the
    nearest baseline); the source modes at the reference clocks (propagated with the sea) and their transfer; the join's departure and
    the first-return estimate on every row; the enclosed source at the root; the probes (d) at every readout clock (every 4 r_s from 40)
    back-propagated to the references through the stored fields (the same sub-steps as step_record, the cubic window included).
(3) the occupied columns saved at the reference clocks and every --columns-every-rs (10) to <out>/<tag>_columns_T<T/r_s>_k<kappa>.pt
    (torch.save {"Phi": cpu complex128 n2 x N_occ, metadata}; .json sidecar; --columns-format npz for numpy), the references' vacuum
    columns kept on the host (--ref-cache host, within --ref-cache-GB), on disk (<tag>_vacuum_...pt) or recomputed.
(4) channels sequential: --kappas 1-16 run one channel per HistRun (tag <tag>_k<kappa>, each resumable with --resume and --save-every),
    the 16 kappa weights in the ksum table (<tag>_ksum.json/.md, page_d13_readout_v1.ksum_table) written after every channel: E_inv, the
    collar flow (soft and hard separately), the interval current and L/L_H, the exterior-cut entropies and their excess, the hard shells,
    the enclosed source; the sum extended beyond the last requested channel while the hard outgoing collar flow of the last channel exceeds
    --kappa-hard-frac (1 percent) of the total at any readout clock (up to --kappa-hard-max).
(5) the record-interpolation control: --cubic-window T0,T1 (v6m's cubic sub-steps) on --kappas 1 as a second run (the RUN file), compared
    by --compare TAG_A TAG_B (page_d13_readout_v1.compare_runs).
(6) a memory figure for one channel at n = 8192 printed at start (estimated from the shapes; the measured device peaks stay in the rows);
    --max-wall-hours is the wall-clock limit of the whole sequence (the running channel saves its state and the sequence stops, resumable).
Not combined with --collector off: --freeze-at-rs, --ledger-anchor-rs, --transfer-at-rs, --transfer-at-peak, --col-diag, --collar-split,
--init-state (the parser refuses). The summary json carries "collector" and "d13"; the per-channel tables.md keeps v6m's columns (the
archive's columns read 0) and <tag>_k<kappa>_d13.json/.npz hold every D13 readout and raw array.

v6m's docstring follows.

page_history_v6m.py (solve builder 7'''

V6N_BLOCK = r'''    # ------------------------------------------------------------------------------------------------- v6n: the collector-free run
    def d13_setup(self):
        """v6n: the D13 readouts' configuration (page_d13_readout_v1.cfg_from_args): the column, reference, readout and vacuum clocks
        added to the checkpoint set, T_H^ref from the record at --TH-ref-rs, the formation's root, the store <tag>_d13.json/.npz."""
        a = self.a
        self.d13_on = (getattr(a, "collector", "on") == "off")
        self.col_k = set(); self.ref_k = {}; self.probe_k = set(); self.vac_k = set(); self.d13_store = None; self.d13_cfg = None
        self.TH_ref = None; self.d13_dens = None; self.d13_form = None; self.d13_inline = False
        if not self.d13_on:
            return
        cfg = D13.cfg_from_args(a)
        cfg["chunk"] = int(a.chunk); cfg["cheb_tol"] = float(a.cheb_tol); cfg["trace_chunk"] = int(getattr(a, "trace_chunk", 1024)); cfg["g"] = self.g
        self.d13_cfg = cfg
        self.d13_inline = (getattr(a, "d13", "on") == "on")
        rs = self.rs_nom; H = self.H
        T0r = float(H.T[self.k0]) / rs; T1r = self._T_of_k(self.kend) / rs
        if not getattr(a, "no_columns", False):
            for x in cfg["columns_at_rs"]:
                if T0r - 1e-9 <= x <= T1r + 1e-9:
                    self.col_k.add(self._k_of(x * rs))
            if cfg["columns_every_rs"] > 0:
                x = math.ceil(T0r / cfg["columns_every_rs"] - 1e-9) * cfg["columns_every_rs"]
                while x <= T1r + 1e-9:
                    self.col_k.add(self._k_of(x * rs)); x += cfg["columns_every_rs"]
        for x in cfg["ref_clocks_rs"]:
            if T0r - 1e-9 <= x <= T1r + 1e-9:
                k_ = self._k_of(x * rs); self.ref_k[k_] = float(x)
                if not getattr(a, "no_columns", False):
                    self.col_k.add(k_)
        x = cfg["probe_from_rs"]
        while x <= T1r + 1e-9:
            if x >= T0r - 1e-9:
                self.probe_k.add(self._k_of(x * rs))
            x += cfg["probe_every_rs"]
        x = math.ceil(T0r / cfg["vac_every_rs"] - 1e-9) * cfg["vac_every_rs"]
        while x <= T1r + 1e-9:
            self.vac_k.add(self._k_of(x * rs)); x += cfg["vac_every_rs"]
        for x in cfg["vac_extra_rs"]:
            if T0r - 1e-9 <= x <= T1r + 1e-9:
                self.vac_k.add(self._k_of(x * rs))
        self.vac_k |= set(self.ref_k)
        self.ck |= self.col_k | set(self.ref_k) | self.probe_k | self.vac_k
        k_TH = self._k_of(cfg["TH_ref_rs"] * rs)
        self.TH_ref = float(cfg["TH_ref"]) if cfg.get("TH_ref") else 1.0 / (4.0 * math.pi * 2.0 * self.g * float(H.E[k_TH]))
        k_form = self._k_of(cfg["form_rs"] * rs)
        Nf, vf = H.fields(k_form)
        r_form = D13.outer_root(Nf, vf, H.R)
        self.d13_form = {"t": float(H.T[k_form]), "k": int(k_form), "r_root": r_form if r_form is not None else 2.0 * self.g * float(H.E[k_form]), "root_present": r_form is not None}
        self.d13_store = D13.Store(self.prefix + "_d13", settings={"code": "page_history_v6n.py", "args": vars(a), "d13": {k_: (list(v_) if isinstance(v_, (list, tuple)) else v_) for k_, v_ in cfg.items()}},
                                   label=LABEL_A % "-")
        for c in self.chs:
            self.d13_init_channel(c)
        log("v6n: the collector-free run (D13 readouts inline: %s): cuts %s r_s (ledger %s, probes %s); the dense vacuum at %d clocks (every %g r_s, the references %s, route %s); the exterior shells "
            "every %g r_s beyond R_cut %g r_s with T_H^ref %.6e from the record at T/r_s %.3f (25 T_H^ref = %.4e; the band top %g); %d readout clocks (every %g r_s from %g); the columns at %d clocks (%s); "
            "the formation at T/r_s %.4f, its root %.3f (present: %s)" % (
                self.d13_inline, cfg["cuts_rs"], cfg["ledger_cuts_rs"], cfg["probe_cuts_rs"], len(self.vac_k), cfg["vac_every_rs"], cfg["ref_clocks_rs"], cfg["vac_route"], cfg["ext_every_rs"], cfg["R_cut_rs"],
                self.TH_ref, H.T[k_TH] / rs, cfg["hard_TH"] * self.TH_ref, cfg["band_top"], len(self.probe_k), cfg["probe_every_rs"], cfg["probe_from_rs"], len(self.col_k),
                ", ".join("%.3f" % (self._T_of_k(k_) / rs) for k_ in sorted(self.col_k)), self.d13_form["t"] / rs, self.d13_form["r_root"], self.d13_form["root_present"]))

    def d13_init_channel(self, c):
        if getattr(self, "d13_on", False) and c.d13 is None:
            c.d13 = D13.ChannelReadout(c.ops.kappa, self.H.R, self.dev, self.rs_nom, self.d13_cfg)
            c.d13.TH_ref = self.TH_ref
            c.d13.ref_loader = (lambda k_ref, c=c: self.d13_load_reference(c, k_ref))

    def d13_cut_masks(self, c):
        if not getattr(self, "d13_on", False):
            return {}
        self.d13_init_channel(c)
        return c.d13.masks()

    def d13_columns_path(self, kappa, trs):
        return "%s_columns_T%09.4f_k%d.%s" % (self.prefix, trs, int(kappa), "npz" if self.d13_cfg["columns_format"] == "npz" else "pt")

    def d13_vacuum_path(self, kappa, trs):
        return "%s_vacuum_T%09.4f_k%d.pt" % (self.prefix, trs, int(kappa))

    def d13_densities(self, k):
        """the stored heavy densities of record k (eps_tot, J_tot of the fields npz) for (g)'s D_R check; None where absent."""
        if not getattr(self, "d13_on", False):
            return None
        if self.d13_dens is None:
            self.d13_dens = {"eps": None}
            try:
                if self.a.fields:
                    Z = np.load(self.a.fields, allow_pickle=True)
                    if "eps_tot" in Z.files and "J_tot" in Z.files:
                        self.d13_dens = {"eps": np.asarray(Z["eps_tot"], float), "J": np.asarray(Z["J_tot"], float), "R": np.asarray(Z["R"], float)}
            except Exception as ex:                                            # noqa: BLE001
                log("v6n: the fields' densities could not be read (%s): (g)'s D_R check is skipped" % ex)
        d = {"g": self.g}
        if self.d13_dens.get("eps") is not None and k < len(self.d13_dens["eps"]):
            d.update({"eps": self.d13_dens["eps"][k], "J": self.d13_dens["J"][k], "R": self.d13_dens["R"]})
        return d

    def d13_record_steps(self, c, k):
        """record k's [(h_csr, dt)] in forward order, the sub-steps of step_record (the cubic window's included)."""
        H = self.H
        t0 = H.T[k]; t1 = H.T[k + 1] if k + 1 < len(H) else t0 + self.tau
        sub = [(t0, t1, None)]
        if self.cw is not None and self.cw[0] <= t0 < self.cw[1]:
            nsub = max(1, int(round(t1 - t0)))
            sub = [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic") for i in range(nsub)]
        out = []
        for (sa, sb, mode) in sub:
            N, v = H.cubic_fields(0.5 * (sa + sb)) if mode == "cubic" else H.fields(k)
            out.append((c.ops.h_csr(N, v), sb - sa))
        return out

    def d13_load_reference(self, c, k_ref):
        """(Phi, Vneg) at the reference record on the device: the columns from their file, the vacuum from the host cache, its file or the
        dense eigendecomposition (cached afterwards per --ref-cache)."""
        cr = c.d13; cfg = self.d13_cfg
        t = float(self.H.T[k_ref]); trs = t / self.rs_nom
        path = (cr.refs.get(int(k_ref)) or {}).get("columns") or self.d13_columns_path(c.ops.kappa, trs)
        if k_ref == self.k and c.Phi is not None and not os.path.exists(path):
            Phi = c.Phi
        else:
            if not os.path.exists(path):
                raise FileNotFoundError("the reference columns %s are missing (the run saves them at the reference clocks; --no-columns forbids the probes)" % path)
            Phi = D13.load_columns(path, self.dev)
        Vneg = cr.ref_vacuum_host(k_ref)
        if Vneg is None:
            vpath = self.d13_vacuum_path(c.ops.kappa, trs)
            if os.path.exists(vpath):
                Vneg = D13.load_columns(vpath, self.dev, key="Vneg")
            else:
                vac = D13.killing_vacuum(c.ops.h_csr(*self.H.fields(k_ref)), self.dev, cfg["vac_route"], cfg["vac_cluster_tol"])
                Vneg = vac["Vneg"]
                cached = cr.cache_ref_vacuum(k_ref, Vneg)
                if cfg["ref_cache"] == "disk":
                    D13.save_columns(vpath, Vneg, {"kappa": c.ops.kappa, "k": int(k_ref), "t": t, "t_over_rs": trs, "code": "page_history_v6n.py", "what": "the occupied columns of P_-(h(t)), the held Killing vacuum"}, key="Vneg")
                log("v6n: kappa %d: the vacuum at the reference T/r_s %.4f recomputed (%s, %.1f s; host cache %s)" % (c.ops.kappa, trs, vac["route"], vac["seconds"], cached))
        return Phi, Vneg

    def d13_reference(self, c, k, t, Phi, h, vac, N, v, E_k):
        """at a reference clock: the source modes (e), the reference's vacuum cached per --ref-cache, the reference registered."""
        cr = c.d13; cfg = self.d13_cfg; trs = t / self.rs_nom
        if vac is None:
            log("v6n: kappa %d: no dense vacuum at the reference T/r_s %.4f (the vacuum is off): no source modes" % (c.ops.kappa, trs))
        if int(k) not in cr.src:
            r_region = c.outer_root(N, v); region = "inside the outer root"
            if r_region is None or cfg["source_region"] == "2gE":
                r_region = 2.0 * self.g * E_k; region = "inside 2gE (no root at the reference clock)" if r_region is None or cfg["source_region"] != "2gE" else "inside 2gE"
            meta, arrays = cr.add_sources(k, t, Phi, h, vac, r_region, region)
            self.d13_store.put_sources(c.ops.kappa, k, meta, arrays)
            log("v6n: kappa %d: reference T/r_s %.4f: %d source modes of %d candidates above %.2f %s (R < %.2f = %.3f r_s; %d interior modes); occupation min %s; the vacuum's occupation %s" % (
                c.ops.kappa, trs, meta["n_modes"], meta["n_candidates_above"], cfg["source_weight"], region, r_region, r_region / self.rs_nom, meta["n_in_modes"], _f(meta.get("occ_C_min"), "%.8f"), meta.get("occ_vac_extremes")))
        cr.register_reference(k, t, columns=self.d13_columns_path(c.ops.kappa, trs) if not getattr(self.a, "no_columns", False) else None)
        if vac is not None:
            cached = cr.cache_ref_vacuum(k, vac["Vneg"])
            if cfg["ref_cache"] == "disk":
                vpath = self.d13_vacuum_path(c.ops.kappa, trs)
                D13.save_columns(vpath, vac["Vneg"], {"kappa": c.ops.kappa, "k": int(k), "t": t, "t_over_rs": trs, "code": "page_history_v6n.py", "what": "the occupied columns of P_-(h(t)), the held Killing vacuum"}, key="Vneg")
                cr.register_reference(k, t, vacuum_path=vpath)
            log("v6n: kappa %d: the reference's vacuum at T/r_s %.4f kept on the host: %s (%s)" % (c.ops.kappa, trs, cached, cfg["ref_cache"]))

    def save_columns(self, k):
        """v6n (3): the occupied columns at record k to named files per channel (and the reference registered)."""
        if not getattr(self, "d13_on", False) or getattr(self.a, "no_columns", False):
            return
        t = float(self.H.T[k]); trs = t / self.rs_nom
        for c in self.chs:
            if self.d13_cfg["columns_kappas"] and c.ops.kappa not in self.d13_cfg["columns_kappas"]:
                continue
            path = self.d13_columns_path(c.ops.kappa, trs)
            meta = {"kappa": c.ops.kappa, "k": int(k), "t": t, "t_over_rs": trs, "nsites": int(self.H.nsites), "n2": int(c.ops.n2), "n_occ": int(c.Phi.shape[1]), "rs_nominal": self.rs_nom,
                    "E": float(self.H.E[k]), "two_gE": 2.0 * self.g * float(self.H.E[k]), "g": self.g, "history_label": self.H.label, "record_json": self.a.record, "fields_npz": self.a.fields,
                    "code": "page_history_v6n.py", "what": "the occupied columns Phi of the propagated pure Gaussian state (C = Phi Phi^dagger) at the record's clock, before step_record(k)",
                    "layout": "flattened index 2 n + a (site n from 0, component a), R_n = n + 1; the chain's h = (1/2){N, h_0} + (1/2){v, Q} of page_ops_v6 at m = 0, kappa; columns orthonormal to rounding"}
            if os.path.exists(path) and not getattr(self.a, "force", False):
                log("v6n: kappa %d: the columns at T/r_s %.4f exist (%s): not rewritten" % (c.ops.kappa, trs, path))
            else:
                D13.save_columns(path, c.Phi, meta, key="Phi", fmt=self.d13_cfg["columns_format"])
                log("v6n: kappa %d: the occupied columns at T/r_s %.4f (record %d) saved to %s (%d x %d complex128, %.2f GiB)" % (c.ops.kappa, trs, k, path, c.Phi.shape[0], c.Phi.shape[1], c.Phi.numel() * 16 / 2.0 ** 30))
            if k in self.ref_k:
                self.d13_init_channel(c); c.d13.register_reference(k, t, columns=path)

    def checkpoint_lean(self, k, t=None):
        """v6n: the collector-free checkpoint: no archive, no selection, no memory representation, no Z identity; E_inv = Tr[h C] - Tr min(h, 0)
        with Tr min(h, 0) from the banded operator's eigenvalues; the dense vacuum P_-(h(t)) only at the baseline clocks (cached in the channel's
        readout), where E_out, W_hdot, the exterior-cut vacuum entropies and the deviation's density are read; item 1's readouts at every
        checkpoint; the source modes and the reference's vacuum at the reference clocks; the probes after the row at the readout clocks."""
        a, H = self.a, self.H
        t = float(H.T[k]) if t is None else float(t)
        E_k = float(H.E[k]); rs_k = 2.0 * self.g * E_k; TH = 1.0 / (4.0 * math.pi * rs_k)
        N, v = H.fields(k)
        kn = k + 1 if k + 1 < len(H) else k - 1
        Nn, vn = H.fields(kn)
        sgn_d = 1.0 if kn > k else -1.0
        dN, dv = sgn_d * (Nn - N), sgn_d * (vn - v)
        lo, hi = [float(x) * self.rs_nom for x in a.join_rs.split(",")]
        jN, jv = H.join_check(k, lo, hi)
        rj = H.rec[k] or {}
        row = {"k": k, "t": t, "t_over_rs": t / self.rs_nom, "E": E_k, "two_gE": rs_k, "T_H": TH, "join_max_dN": jN, "join_max_rel_dv": jv,
               "kappa_T": rj.get("kappa_T"), "four_gE_kappa_T": rj.get("four_gE_kappa_T"), "R_t_record": rj.get("R_t"), "n_pocket_record": rj.get("n_pocket"),
               "min_N_minus_absv": float(np.min(N - np.abs(v))), "collect": False, "collector": "off", "per_channel": {}}
        t_ck = time.time()
        if self.dev.type == "cuda":
            row["device_peak_before_GiB"] = {"allocated": torch.cuda.max_memory_allocated(self.dev) / 2.0 ** 30, "reserved": torch.cuda.max_memory_reserved(self.dev) / 2.0 ** 30}
            torch.cuda.reset_peak_memory_stats(self.dev)
        dens = self.d13_densities(k)
        for c in self.chs:
            tc = time.time()
            self.d13_init_channel(c)
            cr = c.d13
            h = c.ops.h_csr(N, v); Bh = c.ops.banded(h)
            Phi = c.Phi; n0 = Phi.shape[1]
            E_hC = c.trace_h(Bh, Phi)
            E_ref, mininfo = D13.tr_min_h0(h)
            E_inv = E_hC - E_ref
            Gs = Phi.mH @ Phi
            Gs.diagonal().sub_(1.0)
            unit = float(torch.linalg.norm(Gs)); del Gs
            c.unit_defect = max(c.unit_defect, unit)
            chi_c = (c.ops.R2 > self.R_collar_abs).to(CD)[:, None]
            E_out_C = tr_blocked(Bh, Phi, chi_c)
            vac = cr.vacuum(h, t, k, due=(k in self.vac_k))
            r_t = c.outer_root(N, v)
            E_out = None; W_hdot = None; E_ref_dense = None
            if vac is not None:
                Vneg = vac["Vneg"]
                E_ref_dense = vac["E_ref"]
                E_out = E_out_C - tr_blocked(Bh, Vneg, chi_c)
                Bd = OPS.Banded(c.ops.hdot_csr(dN, dv, self.tau), self.dev)
                W_hdot = c.trace_h(Bd, Phi) - c.trace_h(Bd, Vneg)
                del Bd
            pc = {"n0": n0, "n_neg": mininfo["n_neg"], "E_hC": E_hC, "E_ref": E_ref, "E_ref_dense": E_ref_dense, "E_ref_banded_minus_dense": (E_ref - E_ref_dense) if E_ref_dense is not None else None,
                  "E_ref_route": mininfo["route"], "E_inv": E_inv, "E_out": E_out, "E_out_C": E_out_C, "E_inv_inside": (E_inv - E_out) if E_out is not None else None, "W_hdot": W_hdot,
                  "sea_unitarity_F": unit, "memory_rank": None, "principal_method": None, "sin_max": None, "dropped_pairs": None, "dropped_pair_ledger": 0.0, "pair_ledger": None,
                  "collar_flow_acc": c.collar_flow_acc, "cheb_K_max": c.cheb_K_max, "cheb_tail_acc": c.cheb_tail_acc, "vacuum_dense": vac is not None,
                  "vacuum_route": vac["route"] if vac is not None else None, "vacuum_seconds": vac["seconds"] if vac is not None else None,
                  "S_esc": 0.0, "archive": 0, "collected": 0, "S_vac_archive": 0.0, "S_esc_minus_S_vac": 0.0, "S_B": None, "S_M_minus_S_Mprime": None,
                  "E_K": E_inv, "E_F": 0.0, "E_cross": 0.0, "E_remove": 0.0, "M_chi": E_out, "energy_identity_residual": 0.0, "collar_identity_residual": 0.0, "r_t": r_t,
                  "collector": "off (v6n): no archive, no selection, no Z identity; the escaping entropy is the exterior-cut entropy of the D13 readouts"}
            pc["work_jumps_since_last"] = c.work_jump_since
            if c.E_hC_last is not None:
                pc["E_hC_change"] = E_hC - c.E_hC_last
                pc["jump_check"] = (E_hC - c.E_hC_last) - c.work_jump_since
                pc["W_acc_interval"] = c.work_jump_since - (E_ref - c.E_ref_last)
            c.E_hC_last = E_hC; c.E_ref_last = E_ref; c.work_jump_since = 0.0
            if k in self.ref_k:
                self.d13_reference(c, k, t, Phi, h, vac, N, v, E_k)
            if self.d13_inline:
                sc, arrays = cr.checkpoint(Phi, h, Bh, t, k, N, v, E_k, rs_k, TH, vac, dens=dens, T_start=float(H.T[self.k0]),
                                           join={"max_dN": jN, "max_rel_dv": jv, "window_rs": a.join_rs}, form=self.d13_form)
                sc.update({"E_inv": E_inv, "E_hC": E_hC, "E_ref_banded": E_ref, "collar_flow_acc": c.collar_flow_acc, "L_at_t": (self.records[-1]["per_channel"][str(c.ops.kappa)]["L_at_t"] if self.records else None)})
                log(D13.format_line(c.ops.kappa, sc, E_inv))
                D13.split_arrays(sc, arrays, ())
                pc["d13"] = sc
                self.d13_store.put_clock(t / self.rs_nom, c.ops.kappa, sc, arrays, common={"k": k, "t": t, "E": E_k, "two_gE": rs_k, "T_H": TH, "join_max_dN": jN, "join_max_rel_dv": jv,
                                                                                            "min_N_minus_absv": row["min_N_minus_absv"], "r_t": r_t})
            cr.release_vacuum()
            pc["seconds"] = time.time() - tc
            row["per_channel"][str(c.ops.kappa)] = pc
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        row["vacuum_dense"] = all(pc_["vacuum_dense"] for pc_ in row["per_channel"].values())
        if self.dev.type == "cuda":
            row["device_peak_GiB"] = {"allocated": torch.cuda.max_memory_allocated(self.dev) / 2.0 ** 30, "reserved": torch.cuda.max_memory_reserved(self.dev) / 2.0 ** 30}
            log("device peak (GiB, torch): this checkpoint allocated %.2f, reserved %.2f; since the previous checkpoint allocated %.2f, reserved %.2f" % (
                row["device_peak_GiB"]["allocated"], row["device_peak_GiB"]["reserved"], row["device_peak_before_GiB"]["allocated"], row["device_peak_before_GiB"]["reserved"]))
        row["ledger_anchored"] = None; row["ledger_peak"] = None
        self.finish_row(row, t_ck)
        self.d13_probes_if_due(k, t, N, v)

    def d13_probes_if_due(self, k, t, N, v):
        """(d) at the readout clocks: the probes of every probe cut back-propagated to the references at or before this clock."""
        if not getattr(self, "d13_on", False) or not self.d13_inline or k not in self.probe_k:
            return
        refs = sorted([(k_, t_ * self.rs_nom) for k_, t_ in self.ref_k.items() if k_ <= k], key=lambda z: -z[0])
        if not refs:
            return
        row = self.rows[-1]
        for c in self.chs:
            tp = time.time()
            h = c.ops.h_csr(N, v)
            res = c.d13.probes(t, k, c.Phi, h, lambda kk, c=c: self.d13_record_steps(c, kk), refs)
            fr = ((row["per_channel"][str(c.ops.kappa)].get("d13") or {}).get("first_return") or {})
            for r_ in res:
                r_["before_first_return"] = (fr.get(r_["cut"]) or {}).get("before_first_return")
                r_["first_return_rs"] = (fr.get(r_["cut"]) or {}).get("first_return_rs")
                self.d13_store.put_probe(dict(r_, t_over_rs=t / self.rs_nom, k=k, kappa=c.ops.kappa))
                for q in r_["refs"]:
                    log("D13 kappa %d probes at T/r_s %.3f, cut %s r_s, reference T/r_s %.3f: L_direct %.6e, back-propagated L %.6e (check %.1e; %d records, Chebyshev K max %d), L_old %.6e, L_sea %.6e, L_src %s (%d source modes); rank %d, ||J||_1 %.3e, pointwise bound %.3e; before the first return %s (%s r_s)" % (
                        c.ops.kappa, t / self.rs_nom, r_["cut"], q["t_ref_over_rs"], r_["L_direct"], q["L_total_backpropagated"], q["check_L_minus_direct"], q["records_back_propagated"], q["cheb_K_max"], q["L_old"], q["L_sea"],
                        _f(q.get("L_src"), "%.6e"), q["n_source_modes"], r_["rank"], q["bound_rank16_trace_norm"], q["bound_old_pointwise"], r_["before_first_return"], _f(r_["first_return_rs"], "%.1f")))
            row["per_channel"][str(c.ops.kappa)]["d13_probes"] = [{k_: v_ for k_, v_ in r_.items() if k_ != "refs"} | {"refs": [{k2: v2 for k2, v2 in q.items() if not isinstance(v2, np.ndarray)} for q in r_["refs"]]} for r_ in res]
            row["per_channel"][str(c.ops.kappa)]["d13_probes_seconds"] = time.time() - tp

    def d13_flush(self, final):
        """the ledger windows (b), the hard fluence windows (c) and the per-channel D13 table from the rows and the records; the store flushed."""
        if not getattr(self, "d13_on", False) or self.d13_store is None:
            return
        rs = self.rs_nom; cfg = self.d13_cfg
        R_cuts = {("%g" % x): x * rs for x in cfg["cuts_rs"]}
        lines = []
        for c in self.chs:
            kap = str(c.ops.kappa)
            rows_ch = []
            for r in self.rows:
                pc = r["per_channel"].get(kap) or {}
                sc = pc.get("d13") or {}
                rows_ch.append({"t_over_rs": r["t_over_rs"], "two_gE": r["two_gE"], "vacuum_dense": bool(pc.get("vacuum_dense")), "inv": sc.get("inventory"),
                                "shells": (sc.get("shells") or {}).get("totals"), "E_inv": pc.get("E_inv"), "cuts": sc.get("cuts"), "collar_split": sc.get("collar_split"), "L_interval": r.get("L_interval"), "L_over_LH": r.get("L_over_LH_kappaT")})
            recs_ch = [{"t_over_rs": q["t"] / rs, "cut_flow": q["per_channel"].get(kap, {}).get("cut_flow") or {}, "cut_jump": q["per_channel"].get(kap, {}).get("cut_jump") or {}} for q in self.records]
            self.d13_store.J["windows"][kap] = D13.ledger_windows(rows_ch, recs_ch, ["%g" % x for x in cfg["ledger_cuts_rs"]], cfg["window_rs"], rs, R_cuts)
            self.d13_store.J["hard_windows"][kap] = D13.hard_fluence_windows(rows_ch, cfg["window_rs"], rs, cfg["R_cut_rs"] * rs)
            cl = ["%g" % x for x in cfg["cuts_rs"]]
            lines += ["", "## kappa %s: the printable columns per checkpoint (this channel, unweighted)" % kap, "",
                      "| T/r_s | E_inv | L (interval) | L/L_H(kappa_T) | collar out soft | collar out hard | shells hard out | shells hard in | above band N | " + " | ".join("S_X %s" % x for x in cl) + " | " + " | ".join("excess %s" % x for x in cl) + " | vacuum |",
                      "|---|" + "---|" * (9 + 2 * len(cl))]
            for r in rows_ch:
                cs = r.get("collar_split") or {}; sh = r.get("shells") or {}; cu = r.get("cuts") or {}
                lines.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["t_over_rs"], _f(r["E_inv"], "%.6e"), _f(r["L_interval"], "%.4e"), _f(r["L_over_LH"], "%.4f"), _f(cs.get("E_flux_out_soft"), "%.4e"), _f(cs.get("E_flux_out_hard"), "%.4e"),
                    _f(sh.get("hard_E_out"), "%.4e"), _f(sh.get("hard_E_in"), "%.4e"), _f(sh.get("above_N_total"), "%.3e"),
                    " | ".join(_f((cu.get(x) or {}).get("S_X"), "%.6f") for x in cl), " | ".join(_f((cu.get(x) or {}).get("excess"), "%.6f") for x in cl), "dense" if r["vacuum_dense"] else "-"))
            lines += ["", "### kappa %s: the ledger windows (b)" % kap, "", "| cut | t0 | t1 | kind | inventory D at t0 | crossing: current | crossing: jumps | change of the C inventory | telescoping residual | inventory D at t1 | second term on D |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for w_ in self.d13_store.J["windows"][kap]:
                if "cut" in w_ and "crossing_current" in w_:
                    lines.append("| %s | %.3f | %.3f | %s | %s | %s | %s | %s | %s | %s | %s |" % (w_["cut"], w_["t0_over_rs"], w_["t1_over_rs"], w_["kind"], _f(w_.get("inventory_D_start"), "%.6e"), _f(w_.get("crossing_current"), "%.6e"),
                                                                                            _f(w_.get("crossing_jumps"), "%.6e"), _f(w_.get("change_C"), "%.6e"), _f(w_.get("telescoping_residual"), "%.2e"), _f(w_.get("inventory_D_end"), "%.6e"), _f(w_.get("second_term_D"), "%.6e")))
            lines += ["", "### kappa %s: the probes (d)" % kap, "", "| T/r_s | cut | reference | L_direct | L (back-propagated) | L_old | L_sea | L_src | ||J||_1 | before the first return |", "|---|---|---|---|---|---|---|---|---|---|"]
            for p in self.d13_store.J["probes"]:
                if str(p["kappa"]) == kap:
                    for q in p["refs"]:
                        lines.append("| %.3f | %s | %.3f | %.6e | %.6e | %.6e | %.6e | %s | %.3e | %s |" % (p["t_over_rs"], p["cut"], q["t_ref_over_rs"], p["L_direct"], q["L_total_backpropagated"], q["L_old"], q["L_sea"], _f(q.get("L_src"), "%.6e"), q["bound_rank16_trace_norm"], p.get("before_first_return")))
        self.d13_store.J["final"] = bool(final)
        self.d13_store.J["label"] = self.label or (LABEL_A % "-")
        self.d13_store.flush()
        with open(self.prefix + "_d13_tables.md", "w") as f:
            f.write("# %s: the D13 readouts (page_history_v6n, collector-free; %s)\n" % (self.a.tag, "final" if final else "running") + "\n".join(lines) + "\n")

    def d13_summary(self):
        if not getattr(self, "d13_on", False):
            return None
        return {"on": True, "inline": self.d13_inline, "cfg": {k_: (list(v_) if isinstance(v_, (list, tuple)) else v_) for k_, v_ in self.d13_cfg.items()}, "TH_ref": self.TH_ref, "formation": self.d13_form,
                "columns_clocks_over_rs": sorted(self._T_of_k(k_) / self.rs_nom for k_ in self.col_k), "reference_records": {str(k_): v_ for k_, v_ in self.ref_k.items()},
                "readout_clocks_over_rs": sorted(self._T_of_k(k_) / self.rs_nom for k_ in self.probe_k), "vacuum_clocks_over_rs": sorted(self._T_of_k(k_) / self.rs_nom for k_ in self.vac_k),
                "store": {"json": self.d13_store.pj if self.d13_store else None, "npz": self.d13_store.pz if self.d13_store else None}}

'''

V6N_MAIN = r'''def parse_kappas(s):
    s = str(s).strip()
    if "-" in s and "," not in s:
        a_, b_ = s.split("-"); return list(range(int(a_), int(b_) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def run_one(a):
    """one HistRun (v6m's main): resumed when --resume and its state exists; returns (the run, finished)."""
    run = HistRun(a)
    if a.resume and os.path.exists(run.prefix + "_state.pt"):
        run.load_state()
        run.run(resumed=True)
    else:
        run.run()
    sj = run.prefix + "_summary.json"
    done = os.path.exists(sj) and bool(json.load(open(sj)).get("final"))
    return run, done


def main(argv=None):
    a = parse(argv)
    if a.compare:
        res = D13.compare_runs(a.out, a.compare[0], a.compare[1])
        log("v6n: compared %s against %s at %d common clocks: %s" % (a.compare[0], a.compare[1], len(res["rows"]), os.path.join(a.out, "%s_vs_%s.md" % tuple(a.compare))))
        return
    fig = D13.memory_figure(8192, None, a.chunk, a.R_cut_rs, 64.0, getattr(a, "source_max", 128))
    log("v6n: " + D13format_memory_figure_placeholder(fig))
    if a.memory_figure_only:
        return
    if a.collector == "on" or a.sequential == "off":
        run_one(a)
        return
    kappas = parse_kappas(a.kappas)
    done = []
    i = 0
    while True:
        if i < len(kappas):
            kappa = kappas[i]
        else:
            kappa = (done[-1] if done else kappas[-1]) + 1
        i += 1
        a_k = argparse.Namespace(**vars(a)); a_k.channels = str(kappa); a_k.tag = "%s_k%d" % (a.tag, kappa)
        sj = os.path.join(a.out, a_k.tag + "_summary.json")
        if os.path.exists(sj) and json.load(open(sj)).get("final") and not a.force:
            log("v6n: kappa %d is complete (%s): skipped" % (kappa, sj)); done.append(kappa)
        else:
            log("v6n: channel kappa %d (weight %g) starts; tag %s" % (kappa, 16.0 * kappa, a_k.tag))
            run, finished = run_one(a_k)
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not finished:
                log("v6n: kappa %d did not finish (the wall cap or a follow timeout): the sequence stops here; rerun with --resume" % kappa)
                D13.ksum_table(a.out, a.tag, done + [kappa], a.kappa_hard_frac, getattr(a, "probe_from_rs", 40.0))
                return
            done.append(kappa)
        res = D13.ksum_table(a.out, a.tag, done, a.kappa_hard_frac, getattr(a, "probe_from_rs", 40.0))
        if res is not None:
            log("v6n: the convergence table in kappa after channels %s: the last channel's hard outgoing collar flow share %.4f of the total (rule: %s; %s)" % (
                done, res["last_channel_hard_share_max"], res["rule"], "the hard part still moves" if res["hard_moves_above_frac"] else "converged at this channel"))
        if (time.time() - T_START) / 3600.0 > a.max_wall_hours:
            log("v6n: the wall cap after kappa %d; rerun with --resume to continue the sequence" % kappa)
            return
        if i >= len(kappas):
            if not a.kappa_extend or res is None or not res["hard_moves_above_frac"]:
                break
            if done[-1] + 1 > a.kappa_hard_max:
                log("v6n: the hard part still moves at kappa %d but --kappa-hard-max %d is reached: the sum stops here (printed with the last channel's share)" % (done[-1], a.kappa_hard_max))
                break
            log("v6n: the hard part moves at kappa %d by more than %g of its total: the sum is extended to kappa %d" % (done[-1], a.kappa_hard_frac, done[-1] + 1))
    log("v6n: done; channels %s; %s" % (done, os.path.join(a.out, a.tag + "_ksum.md")))


if __name__ == "__main__":
    main()
'''.replace("D13format_memory_figure_placeholder", "D13.format_memory_figure")

V6N_ARGS = r'''    # v6n: the collector-free run, the channel sequence, the columns, the control's comparison
    ap.add_argument("--collector", default="off", choices=["on", "off"], help="v6n: off (default) the collector-free run with the D13 readouts inline; on: v6m's code paths")
    ap.add_argument("--kappas", default="1-16", help="v6n: the channel sequence (a-b or a list), one channel per HistRun, the weights 16 kappa in the ksum table")
    ap.add_argument("--sequential", default="on", choices=["on", "off"], help="v6n: off runs --channels in one HistRun (v6m's layout)")
    ap.add_argument("--kappa-hard-max", dest="kappa_hard_max", type=int, default=64, help="v6n: the sum is not extended beyond this channel")
    ap.add_argument("--kappa-hard-frac", dest="kappa_hard_frac", type=float, default=0.01, help="v6n: the hard part's share of the total below which the sum stops")
    ap.add_argument("--no-kappa-extend", dest="kappa_extend", action="store_false", help="v6n: never extend beyond --kappas")
    ap.add_argument("--columns-at-rs", dest="columns_at_rs", default="12,20,40", help="v6n: the occupied columns saved at these clocks (r_s)")
    ap.add_argument("--columns-every-rs", dest="columns_every_rs", type=float, default=10.0, help="v6n: and every this many r_s (0: none)")
    ap.add_argument("--columns-format", dest="columns_format", default="pt", choices=["pt", "npz"])
    ap.add_argument("--no-columns", dest="no_columns", action="store_true", help="v6n: save no columns (the probes then need --ref-cache and cannot load references)")
    ap.add_argument("--d13", default="on", choices=["on", "off"], help="v6n: the inline D13 readouts (off: propagation, E_inv and the columns only)")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("TAG_A", "TAG_B"), help="v6n: compare two runs' per-channel outputs in --out and exit")
    ap.add_argument("--force", action="store_true", help="v6n: recompute complete channels and rewrite existing column files")
    ap.add_argument("--memory-figure-only", dest="memory_figure_only", action="store_true")
    D13.add_arguments(ap, common=False)
    a = ap.parse_args(argv)
    if a.collector == "off":
        for name in ("freeze_at_rs", "ledger_anchor_rs", "transfer_at_rs", "init_state"):
            if getattr(a, name, None) is not None:
                ap.error("--%s is not combined with --collector off (v6m's code paths need --collector on)" % name.replace("_", "-"))
        if a.transfer_at_peak or a.col_diag or a.collar_split:
            ap.error("--transfer-at-peak, --col-diag and --collar-split are v6m's collector readouts: --collector on")'''

PAIRS = [
    # 1. the docstring head
    ('"""page_history_v6m.py (solve builder 7', V6N_DOC),
    # 2. the import
    ("import page_core_v6 as CORE", "import page_core_v6 as CORE          # noqa: E402\nimport page_d13_readout_v1 as D13   # noqa: E402\n# (v6n) CORE imported as v6m's:"),
    # 3. HistChannel: the readout handle
    ("        self.ext_t = None\n", "        self.ext_t = None\n        self.d13 = None                                 # v6n: the channel's D13 readout (its caches and source modes)\n"),
    # 4. step_record: the per-record flows and jumps at the ledger cuts, the source modes with the sea
    ("            flow = 0.0; L0 = None; Kmax = 0; tail = 0.0; jump = 0.0\n",
     "            flow = 0.0; L0 = None; Kmax = 0; tail = 0.0; jump = 0.0\n"
     "            chis_x = self.d13_cut_masks(c); flow_x = {lab: 0.0 for lab in chis_x}; jump_x = {lab: 0.0 for lab in chis_x}   # v6n (b): the ledger cuts\n"),
    ("                    jump += tr_blocked(c.ops.banded(h - h_cur), c.Phi); h_cur = h\n",
     "                    Bdh = c.ops.banded(h - h_cur)\n"
     "                    jump += tr_blocked(Bdh, c.Phi)\n"
     "                    for lab in chis_x:                                           # v6n (b): the sub-step's jump at the ledger cuts\n"
     "                        jump_x[lab] += tr_blocked(Bdh, c.Phi, chis_x[lab])\n"
     "                    del Bdh; h_cur = h\n"),
    ("                Eo0 = tr_blocked(Bh, c.Phi, chi)\n",
     "                Eo0 = tr_blocked(Bh, c.Phi, chi)\n"
     "                Ex0 = {lab: tr_blocked(Bh, c.Phi, chis_x[lab]) for lab in chis_x}      # v6n (b)\n"),
    ("                if c.F.shape[1]:\n                    c.F, _, _ = propagate(Bh, c.F, sb - sa, a.cheb_tol, a.chunk)\n",
     "                if c.F.shape[1]:\n                    c.F, _, _ = propagate(Bh, c.F, sb - sa, a.cheb_tol, a.chunk)\n"
     "                if c.d13 is not None:                                            # v6n (e): the source modes travel with the sea\n"
     "                    c.d13.propagate_sources(Bh, sb - sa, a.cheb_tol, a.chunk)\n"),
    ("                Eo1 = tr_blocked(Bh, c.Phi, chi)\n                flow += Eo1 - Eo0\n",
     "                Eo1 = tr_blocked(Bh, c.Phi, chi)\n                flow += Eo1 - Eo0\n"
     "                for lab in chis_x:                                               # v6n (b): the exact flow at the ledger cuts over the sub-step\n"
     "                    flow_x[lab] += tr_blocked(Bh, c.Phi, chis_x[lab]) - Ex0[lab]\n"),
    ("                jump += tr_blocked(c.ops.banded(c.ops.h_csr(*H.fields(k + 1)) - h_cur), c.Phi)\n",
     "                Bdh = c.ops.banded(c.ops.h_csr(*H.fields(k + 1)) - h_cur)\n"
     "                jump += tr_blocked(Bdh, c.Phi)\n"
     "                for lab in chis_x:                                               # v6n (b): the jump at the ledger cuts into the next record's h\n"
     "                    jump_x[lab] += tr_blocked(Bdh, c.Phi, chis_x[lab])\n"
     "                del Bdh\n"),
    ('            rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L0, "cheb_K": Kmax, "cheb_tail": tail, "work_jump": jump}\n',
     '            rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L0, "cheb_K": Kmax, "cheb_tail": tail, "work_jump": jump,\n'
     '                                                    "cut_flow": dict(flow_x), "cut_jump": dict(jump_x)}                      # v6n (b)\n'),
    # 5. the checkpoint's dispatch
    ("    def checkpoint(self, k, t=None, collect_override=None):\n        a, H = self.a, self.H\n",
     "    def checkpoint(self, k, t=None, collect_override=None):\n"
     '        if getattr(self.a, "collector", "on") == "off":                       # v6n: the collector-free checkpoint\n'
     "            return self.checkpoint_lean(k, t)\n"
     "        a, H = self.a, self.H\n"),
    # 6. the new methods before col_diag
    ("    def col_diag(self, cdg, KD, Q, Phi, Vneg):\n", V6N_BLOCK + "    def col_diag(self, cdg, KD, Q, Phi, Vneg):\n"),
    # 7. finish_row: None propagates; W only between two baseline rows; the log formats
    ('        wsum = lambda key: sum(c.ops.w * (row["per_channel"][str(c.ops.kappa)].get(key) or 0.0) for c in self.chs)\n',
     "        def wsum(key):                                                            # v6n: a present None (no dense vacuum at this clock) propagates; a missing key is 0 as in v6m\n"
     '            vals = [row["per_channel"][str(c.ops.kappa)].get(key, 0.0) for c in self.chs]\n'
     "            if any(v_ is None for v_ in vals):\n"
     "                return None\n"
     "            return sum(c.ops.w * v_ for c, v_ in zip(self.chs, vals))\n"),
    ('            row["W_creation"] = (row["E_inv_inside_weighted"] - self.last_ck["E_inv_inside_weighted"]) / dt + row["L_interval"]\n'
     '            row["W_hdot_trapezoid"] = 0.5 * (row["W_hdot_weighted"] + self.last_ck["W_hdot_weighted"])\n',
     '            if row.get("E_inv_inside_weighted") is not None and self.last_ck.get("E_inv_inside_weighted") is not None:     # v6n: both ends with the dense vacuum\n'
     '                row["W_creation"] = (row["E_inv_inside_weighted"] - self.last_ck["E_inv_inside_weighted"]) / dt + row["L_interval"]\n'
     '                row["W_hdot_trapezoid"] = 0.5 * (row["W_hdot_weighted"] + self.last_ck["W_hdot_weighted"])\n'
     "            else:\n"
     '                row["W_creation"] = None; row["W_hdot_trapezoid"] = None\n'),
    ("the checkpoints' trapezoid %.4e; the jump check %.3e (weighted)\" % (", "the checkpoints' trapezoid %s; the jump check %.3e (weighted)\" % ("),
    ('row["W_acc_rate"], row["W_hdot_trapezoid"], row["jump_check_weighted"]))', 'row["W_acc_rate"], _f(row.get("W_hdot_trapezoid"), "%.4e"), row["jump_check_weighted"]))'),
    ("E_inv %.6e (inside the collar %.6e, beyond %.6e), W %s against ", "E_inv %.6e (inside the collar %s, beyond %s), W %s against "),
    ('row["E_inv_weighted"], row["E_inv_inside_weighted"], row["E_out_weighted"], _f(row.get("W_creation"), "%.4e"), _f(row["W_hdot_weighted"], "%.4e"),',
     'row["E_inv_weighted"], _f(row.get("E_inv_inside_weighted"), "%.6e"), _f(row.get("E_out_weighted"), "%.6e"), _f(row.get("W_creation"), "%.4e"), _f(row["W_hdot_weighted"], "%.4e"),'),
    ('E_remove %.6e, M_chi %.6e, the identities %.2e, %.2e; collar flow %.6e; %.0f s" % (', 'E_remove %.6e, M_chi %s, the identities %.2e, %.2e; collar flow %.6e; %.0f s" % ('),
    ('row["E_remove_weighted"], row["M_chi_weighted"], row["energy_identity_residual_weighted"], row["collar_identity_residual_weighted"],',
     'row["E_remove_weighted"], _f(row.get("M_chi_weighted"), "%.6e"), row["energy_identity_residual_weighted"], row["collar_identity_residual_weighted"],'),
    # 8. write_outputs: the table's formats, the code name, the store flushed
    ('"| %.3f | %.4f | %.6e | %.6e | %.6e | %s | %s | %.4e | %s | %s | %s | %s | %s | %.6f | %.2e | %.4e | %.6f | %s | %s | %s | %s | %s | %s | %.6e | %.6e | %.2e | %.2e | %s | %s |" % (',
     '"| %.3f | %.4f | %.6e | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.6f | %.2e | %.4e | %.6f | %s | %s | %s | %s | %s | %s | %.6e | %s | %.2e | %.2e | %s | %s |" % ('),
    ('r["t_over_rs"], r["two_gE"], r["E_inv_weighted"], r["E_inv_inside_weighted"], r["E_out_weighted"], _f(r.get("W_creation"), "%.4e"), _f(r.get("W_acc_rate"), "%.4e"), r["W_hdot_weighted"],',
     'r["t_over_rs"], r["two_gE"], r["E_inv_weighted"], _f(r.get("E_inv_inside_weighted"), "%.6e"), _f(r.get("E_out_weighted"), "%.6e"), _f(r.get("W_creation"), "%.4e"), _f(r.get("W_acc_rate"), "%.4e"), _f(r.get("W_hdot_weighted"), "%.4e"),'),
    ('r["E_remove_weighted"], r["M_chi_weighted"], r["energy_identity_residual_weighted"], r["collar_identity_residual_weighted"], _f(r.get("join_max_dN"), "%.2e"),',
     'r["E_remove_weighted"], _f(r.get("M_chi_weighted"), "%.6e"), r["energy_identity_residual_weighted"], r["collar_identity_residual_weighted"], _f(r.get("join_max_dN"), "%.2e"),'),
    ('summary = {"code": "page_history_v6k.py", "settings": vars(a),', 'summary = {"code": "page_history_v6n.py", "collector": getattr(a, "collector", "on"), "d13": self.d13_summary(), "settings": vars(a),'),
    ('        with open(self.prefix + "_tables.md", "w") as f:\n            f.write("\\n".join(L) + "\\n")\n',
     '        with open(self.prefix + "_tables.md", "w") as f:\n            f.write("\\n".join(L) + "\\n")\n        self.d13_flush(final)                                                     # v6n\n'),
    # 9. start(): the T_0 vacuum cached (the same eigendecomposition)
    ("            del V, e\n            if self.dev.type == \"cuda\":\n                torch.cuda.empty_cache()\n        self.min_N_minus_absv_T0 = mnv\n",
     '            if getattr(a, "collector", "on") == "off":                            # v6n: the T_0 vacuum (the ground state\'s own columns) for the first checkpoint\n'
     "                self.d13_init_channel(c)\n"
     '                c.d13.set_vacuum(float(H.T[self.k0]), self.k0, e.cpu().numpy(), c.Phi, "the ground state\'s eigendecomposition at T_0 (dense, v6m\'s)", float(e[neg].sum()))\n'
     "            del V, e\n            if self.dev.type == \"cuda\":\n                torch.cuda.empty_cache()\n        self.min_N_minus_absv_T0 = mnv\n"),
    # 10. the state: the readout's host state
    ('"work_jump_since": c.work_jump_since, "E_hC_last": c.E_hC_last, "E_ref_last": c.E_ref_last}',
     '"work_jump_since": c.work_jump_since, "E_hC_last": c.E_hC_last, "E_ref_last": c.E_ref_last,\n'
     '                                                 "d13": (c.d13.state_host() if c.d13 is not None else None)}'),
    ('            c.work_jump_since = s.get("work_jump_since", 0.0); c.E_hC_last = s.get("E_hC_last"); c.E_ref_last = s.get("E_ref_last")   # v6g: kept on resume\n',
     '            c.work_jump_since = s.get("work_jump_since", 0.0); c.E_hC_last = s.get("E_hC_last"); c.E_ref_last = s.get("E_ref_last")   # v6g: kept on resume\n'
     '            if s.get("d13") is not None:                                             # v6n: the source modes, the references, the collar block\n'
     "                self.d13_init_channel(c); c.d13.load_state_host(s[\"d13\"], self.dev)\n"),
    # 11. run(): the columns at their clocks
    ('            if k in self.save_at_k:\n                self.save_state(k, path="%s_state_at_%07.3f.pt" % (self.prefix, self.H.T[k] / self.rs_nom))\n',
     '            if k in self.save_at_k:\n                self.save_state(k, path="%s_state_at_%07.3f.pt" % (self.prefix, self.H.T[k] / self.rs_nom))\n'
     "            if k in self.col_k:                                                   # v6n (3): the occupied columns at the reference clocks and every --columns-every-rs\n"
     "                self.save_columns(k)\n"),
    # 12. __init__: the D13 setup
    ("        self.anchor = None; self.anchor_started = False; self.follow_end = None\n",
     "        self.anchor = None; self.anchor_started = False; self.follow_end = None\n        self.d13_setup()                                                          # v6n\n"),
    # 13. parse(): the arguments and the checks
    ("    a = ap.parse_args(argv)\n", V6N_ARGS + "\n"),
    # 14. main()
    ("def main(argv=None):\n    a = parse(argv)\n    run = HistRun(a)\n    if a.resume and os.path.exists(run.prefix + \"_state.pt\"):\n        run.load_state()\n        run.run(resumed=True)\n    else:\n        run.run()\n\n\nif __name__ == \"__main__\":\n    main()\n",
     V6N_MAIN),
]


def build(src, dst):
    text = open(src, encoding="utf-8").read()
    print("source %s: %d bytes, md5 %s" % (src, len(text.encode("utf-8")), hashlib.md5(text.encode("utf-8")).hexdigest()))
    for i, (old, new) in enumerate(PAIRS, 1):
        n = text.count(old)
        if n != 1:
            sys.exit("ABORT: replacement %d matches %d times (must be exactly 1): %r" % (i, n, old[:120]))
        text = text.replace(old, new)
        print("  %2d. applied (%d -> %d chars): %s" % (i, len(old), len(new), old.strip().splitlines()[0][:100]))
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    py_compile.compile(dst, doraise=True)
    print("built %s: %d replacements, %d bytes, md5 %s; compiled" % (dst, len(PAIRS), len(text.encode("utf-8")), hashlib.md5(text.encode("utf-8")).hexdigest()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cand = [os.path.join(HERE, "page_history_v6m.py"), os.path.join(HERE, "code_v6", "page_history_v6m.py"), os.path.join(HERE, "..", "code_v6", "page_history_v6m.py")]
    default_src = next((c for c in cand if os.path.isfile(c)), cand[0])
    ap.add_argument("--src", default=default_src); ap.add_argument("--dst", default=os.path.join(HERE, "page_history_v6n.py"))
    a = ap.parse_args()
    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        sys.exit("ABORT: --dst must differ from --src (v6m is not touched)")
    build(a.src, a.dst)


if __name__ == "__main__":
    main()
