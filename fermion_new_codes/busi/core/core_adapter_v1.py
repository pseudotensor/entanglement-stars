#!/usr/bin/env python3
"""
core_adapter_v1.py  (busi lane, inbox #284 (c)/(f) and #286: the core-slice adapter; SKELETON, version 1)

Purpose. qh_branch_witness_v1b and hf_star_variance_v3 read only a static-star family json (S10.load_star and
S10.rebuild_channels: eigenchannels of h_m with a zero-temperature filling). The core's numbers (rows 8, 19, 22) need
the same per-slice, per-channel objects built from the CORE_well v1.2 slices (the production c19 r11 and the sigma/a 8
member c27 ell8; clocks T/r_s 12 to 40). This module builds exactly the objects the witness's --core text names
(QH_S10_spec_v1 sec. 7):
  - the fields N, v on the slice's sites (the slice's applied fields = the record's accepted site fields);
  - the channel set with its multiplicities (the slice's kappas, weights 16 kappa);
  - the state per channel: the occupied orbital columns Phi = [sea | dust] with occupations f (a projector C =
    Phi Phi^dagger when every f is 0 or 1: the PURE-STATE path, complete here) or, when some f is fractional, the
    covariance of a Gaussian mixture (the MIXED-STATE path: a marked stub until REVIEWA_core_adapter_spec_v1 lands);
  - the continued reference P_ref per channel: the slice's stored R2 columns (P_ref = R2 R2^dagger), NOT the
    instantaneous negative spectral projector of h_m (the difference is measured and printed);
  - the radii (CORE_well v1's per-slice json: R_c, R_50, R_90, the dense and applied roots, 2gE) and the slice's
    hydrodynamic-gate status (reported beside every number, never repaired).
The channel operators are engine_ops_v1's (chain_parts, clocked, site_density, sources_dynamic), as in S10's rebuild.
The slices are read with slices144_io_v1.SliceReader (memory-mapped columns; complex64 slices carry their floor).

Witness-shaped channels. Every pure channel is also returned in the key layout the witness's per-channel routines use
(qh_branch_witness_v1b.prepare_channel's outputs): kappa, w, h, Q, hm, h_sp, Q_sp, U = [Uocc | Uemp], f, Uocc, Uemp,
Uo (= the reference columns), Uu (their complement), Ud (the dust columns, orthonormalized with the sea), n_occ,
closed. So static_identity, gram_channel, var_block, vertex_sparse, fock_singles run on them unchanged. The keys the
star's spectral response needs (e: the sea's spectrum, W_ou: energy denominators) are NOT supplied: the reference is a
continued one, so P^A for the core is the spec's to define; e is None and e_stored (the slice's e_k) is given apart.

Verified state convention (probe on prod T12, complex128): sum_kappa 16 kappa Tr[(C - R2 R2^dagger) hcal_n] with
engine_ops_v1.sources_dynamic reproduces the fields npz's eps_tot to 1.9e-13 and J_tot to 6.5e-15 (max relative);
the dust part alone reproduces eps_tot - eps_pairs to 1.8e-16.

Modes: --interface (print what is built and stop), --selftest (the checks on one slice; exit 1 on a FAIL), --summary
(per slice: the fields, channel set, state kind, radii, gate and the per-channel cheap checks, one json per slice).
CPU only (numpy, scipy); at most 16 threads; renices itself. Never edits its inputs. Run on the box only.
"""
import os
import sys


def _set_threads():
    t = 8
    if '--threads' in sys.argv:
        try:
            t = int(sys.argv[sys.argv.index('--threads') + 1])
        except (ValueError, IndexError):
            t = 8
    t = str(max(1, min(t, 16)))
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = t


_set_threads()
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402

VERSION = 'core_adapter_v1'
MIXED_MSG = 'mixed-state variance: per REVIEWA_core_adapter_spec_v1'

# ============================================================================ locations (box)
ENGINE_DIR = '<internal>/code'          # engine_ops_v1.py
SLICES_DIR = '<internal>/slices144'      # slices144_io_v1.py
WITNESS_DIR = '<internal>/hf'          # qh_branch_witness_v1b.py, hf_star_variance_v3.py
PAGE_DIR = '<internal>/v6'                 # page_ops_v6.py (cross-check of the channel ops only)
WELL_DIR = '<internal>/well'      # CORE_well v1 per-slice jsons (radii, gate)
HISTORIES = {
    # the production: the H100b states copy was deleted on the box after the Drive check (19 Oct); two box copies remain
    'prod': dict(tag='c19_rs64_A_R2_tau8_ell4_avg_r11', clocks=[12, 15, 16, 20, 24, 25, 28, 30, 35, 40],
                 slice_dirs=['<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11',
                             '<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11',
                             '<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11'],
                 records=['<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
                          '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json'],
                 fields=['<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz',
                         '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz']),
    # the sigma/a 8 member: its only box copy (H100b/states) was deleted after the Drive check (19 Oct); --slice-dirs
    # must point at a restored copy (MANIFEST: <internal>/MANIFEST_H100b.md5)
    's8': dict(tag='c27_rs64_A_R2_tau8_ell8_prep4_avg', clocks=[12, 13, 14, 16, 20, 25, 30, 40],
               slice_dirs=['<internal>/c27_rs64_A_R2_tau8_ell8_prep4_avg'],
               records=['<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json'],
               fields=['<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz']),
}
RADIUS_LABELS = ('R_c', 'R_50', 'R_90', 'dense_outer', 'dense_inner', 'applied_outer', 'applied_inner', 'two_gE')


# ============================================================================ helpers
def log(*a):
    print(*a, flush=True)


def fnorm(X):
    return float(np.linalg.norm(np.asarray(X).ravel()))


def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    return o


def import_mods(engine_dir=ENGINE_DIR, slices_dir=SLICES_DIR):
    for d in (slices_dir, engine_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    import engine_ops_v1 as OPS
    import slices144_io_v1 as IO
    OPS.GPU['dev'] = None
    return OPS, IO


def slice_name(tag, t):
    return 'orbitals_%s_T%07.2f.npz' % (tag, float(t))


def locate_slice(history, t, extra_dirs=()):
    H = HISTORIES[history]
    return first_existing([os.path.join(d, slice_name(H['tag'], t)) for d in list(extra_dirs) + H['slice_dirs']])


# ============================================================================ the slice context
class CoreSlice:
    """One stored core slice with its history's records (m, g, ell, rs), its fields-npz row, the CORE_well v1 radii and
    gate. Channels are built on demand (channel(kappa)); nothing is held beyond the reader's memory maps."""

    def __init__(self, history, t, OPS, IO, slice_path=None, records=None, fields=None, well_dir=WELL_DIR, orthonormalize=True):
        H = HISTORIES[history]
        self.history, self.t, self.OPS, self.IO = history, float(t), OPS, IO
        self.path = slice_path or locate_slice(history, t)
        if self.path is None:
            raise FileNotFoundError('%s T/r_s %g: no box copy of %s (searched %s)' % (history, t, slice_name(H['tag'], t), H['slice_dirs']))
        self.rd = IO.SliceReader(self.path, log=log)
        self.records_path = records or first_existing(H['records'])
        J = json.load(open(self.records_path))
        A = J['args']
        self.m, self.g, self.ell, self.rs = float(A['m']), float(A['g']), float(A['ell']), float(A['rs'])
        self.half = bool(A.get('half_offset', 0))
        self.record = next((r for r in J['records'] if r.get('E') is not None and abs(float(r['T']) - self.rd.T) < 1e-6), {})
        self.fields_path = fields or first_existing(H['fields'])
        Z = np.load(self.fields_path, allow_pickle=True)
        Tz = np.asarray(Z['T'], float)
        k = int(np.argmin(np.abs(Tz - self.rd.T)))
        self.fields_row = {key: np.asarray(Z[key][k], float) for key in ('eps_tot', 'J_tot', 'eps_pairs', 'J_pairs', 'N', 'v', 'N_applied', 'v_applied')
                           if key in Z.files} if abs(Tz[k] - self.rd.T) < 1e-6 else {}
        self.fields_T = float(Tz[k])
        self.nsites = self.rd.nsites
        self.R = np.asarray(self.rd.R, float)
        self.N = np.asarray(self.rd.N_applied, float)
        self.v = np.asarray(self.rd.v_applied, float)
        self.kappas = list(self.rd.kappas)
        self.weights = {k_: float(self.rd.weights[k_]) for k_ in self.kappas}
        self.orthonormalize = orthonormalize
        self.well_path = os.path.join(well_dir, 'well_%s_T%07.2f.json' % (history, self.t))
        self.radii, self.gate, self.limit_label, self.well_dust = self._radii()

    def _radii(self):
        if not os.path.exists(self.well_path):
            return {lab: None for lab in RADIUS_LABELS}, None, 'unknown (no CORE_well json)', None
        W = json.load(open(self.well_path))
        ro, du, re = W.get('roots', {}), W.get('dust', {}), W.get('record', {})
        src = dict(R_c=re.get('R_c'), R_50=du.get('R_50'), R_90=du.get('R_90'), dense_outer=ro.get('dense_outer'), dense_inner=ro.get('dense_inner'),
                   applied_outer=ro.get('applied_outer'), applied_inner=ro.get('applied_inner'), two_gE=re.get('two_gE'))
        radii = {}
        for lab in RADIUS_LABELS:
            r = src.get(lab)
            radii[lab] = None if r is None else (float(r), min(int(np.searchsorted(self.R, float(r))), self.nsites - 1))
        return radii, W.get('gate'), W.get('limit_label'), du.get('total_weighted')

    def state_kind(self):
        """'pure' when every dust occupation is 0 or 1 in every channel (the sea's are 1 by construction), else 'mixed'."""
        for k_ in self.kappas:
            f = np.asarray(self.rd.small('f_%d' % k_), float).reshape(-1)
            if f.size and not np.all((np.abs(f) < 1e-12) | (np.abs(f - 1.0) < 1e-12)):
                return 'mixed'
        return 'pure'

    def operators(self, kappa):
        """engine_ops_v1's channel operators on the slice's chain and applied fields: h (unit clock), Q, h_m."""
        h, Q, Rc = self.OPS.chain_parts(self.nsites, self.m, float(kappa), self.half)
        if np.max(np.abs(Rc - self.R)) > 1e-12:
            raise RuntimeError('engine_ops_v1 chain radii differ from the slice R (half_offset %s)' % self.half)
        hm = self.OPS.clocked(h, Q, self.N, self.v)
        return np.asarray(h), np.asarray(Q), np.asarray(hm)

    def channel(self, kappa, witness_shape=True):
        """The pure-state channel (projector C = Uocc Uocc^dagger, P_ref = Uo Uo^dagger) in the witness's key layout."""
        t0 = time.time()
        ch_raw = self.rd.channel(kappa, orthonormalize=self.orthonormalize)
        f = np.asarray(ch_raw['f'], float)
        if not np.all((np.abs(f) < 1e-12) | (np.abs(f - 1.0) < 1e-12)):
            return self.mixed_channel(kappa, ch_raw)
        h, Q, hm = self.operators(kappa)
        Phi = ch_raw['Phi']
        occ = f > 0.5
        Uocc = np.ascontiguousarray(Phi[:, occ])
        n_sea = int(ch_raw['n_sea'])
        dust_cols = occ.copy()
        dust_cols[:n_sea] = False
        Ud = np.ascontiguousarray(Phi[:, dust_cols])
        Uemp = complement(Uocc)
        R2 = np.asarray(self.rd.R2(kappa), np.complex128)
        Uo, ref_e0, ref_e1, _ = self.IO.orthonormalize_columns(R2) if self.orthonormalize else (R2, None, None, 'none')
        Uu = complement(Uo)
        ch = dict(kappa=int(kappa), w=self.weights[kappa], h=h, Q=Q, hm=hm, h_sp=sp.csr_matrix(h), Q_sp=sp.csr_matrix(Q),
                  U=np.concatenate([Uocc, Uemp], axis=1), f=np.concatenate([np.ones(Uocc.shape[1]), np.zeros(Uemp.shape[1])]),
                  Uocc=Uocc, Uemp=Uemp, Uo=np.ascontiguousarray(Uo), Uu=Uu, Ud=Ud, n_occ=int(Uocc.shape[1]), closed=True,
                  e=None, fsea=None, e_stored=self.rd.e(kappa), mask_stored=self.rd.mask(kappa), state_kind='pure',
                  n_sea=n_sea, n_dust=int(ch_raw['n_dust']), f_dust=np.asarray(ch_raw['f_dust'], float), complex64=bool(ch_raw['complex64']),
                  checks=dict(stored_orthonormality=ch_raw['stored_orthonormality'], loewdin_orthonormality=ch_raw['loewdin_orthonormality'],
                              orthonormalization=ch_raw['orthonormalization'], dust_sea_overlap_max=ch_raw['dust_sea_overlap_max'],
                              ref_stored_orthonormality=ref_e0, ref_loewdin_orthonormality=ref_e1,
                              occ_emp_overlap=float(np.max(np.abs(Uocc.conj().T @ Uemp[:, :min(64, Uemp.shape[1])]))) if Uemp.shape[1] else 0.0),
                  seconds=time.time() - t0)
        return ch

    def mixed_channel(self, kappa, ch_raw):
        """The Gaussian-mixture channel: the covariance C = Phi diag(f) Phi^dagger and its variance formula."""
        raise NotImplementedError(MIXED_MSG)

    def channels(self, kappas=None):
        for k_ in (kappas or self.kappas):
            yield self.channel(k_)

    def interface(self):
        """The witness --core items for this slice (without the channel matrices)."""
        return dict(version=VERSION, history=self.history, T_over_rs=self.rd.T_over_rs, T=self.rd.T, slice=self.path, dtype=self.rd.dtype_str,
                    complex64_floor=self.rd.floor if self.rd.complex64 else None, nsites=self.nsites, m=self.m, g=self.g, ell=self.ell, rs=self.rs,
                    fields=dict(source='the slice N_applied, v_applied (the record accepted site fields)', N_min=float(self.N.min()), v_min=float(self.v.min())),
                    channel_set=dict(kappas=self.kappas, weights=[self.weights[k_] for k_ in self.kappas], n_sea=self.rd.n_sea, n_dust=self.rd.n_dust),
                    state=self.state_kind(), reference='P_ref = R2 R2^dagger per channel (the slice stored continued reference)',
                    radii={lab: (None if r is None else r[0]) for lab, r in self.radii.items()}, gate=self.gate, limit_label=self.limit_label,
                    record=dict(E=self.record.get('E'), R_c=self.record.get('R_c'), trapping_zeros=self.record.get('trapping_zeros')),
                    records_json=self.records_path, fields_npz=self.fields_path, well_json=self.well_path if os.path.exists(self.well_path) else None)


def complement(U):
    """an orthonormal basis of the orthogonal complement of the orthonormal columns U (full Householder QR)."""
    n, k = U.shape
    if k == n:
        return np.zeros((n, 0), np.complex128)
    Qf, _ = sla.qr(U, mode='full')
    return np.ascontiguousarray(Qf[:, k:])


# ============================================================================ sources and pure-state variances
def channel_sources(OPS, ch):
    """w Tr[(C - P_ref) hcal_n] (eps, J) and the dust part, engine_ops_v1's site densities."""
    e, j = OPS.sources_dynamic(ch['h'], ch['Q'], ch['Uocc'], None, ch['Uo'], ch['w'])
    ed = ch['w'] * OPS.site_density(ch['h'], ch['Ud']) if ch['Ud'].shape[1] else np.zeros(len(e))
    jd = ch['w'] * OPS.site_density(ch['Q'], ch['Ud']) if ch['Ud'].shape[1] else np.zeros(len(e))
    return e, j, ed, jd


def pure_variances(ch, A_sp):
    """Var_C(dGamma(A)) = ||(1-C) A Uocc||_F^2 and Var_{P_ref} = ||(1-P_ref) A Uo||_F^2 for one copy (pure states)."""
    AU = A_sp @ ch['Uocc']
    M = ch['Uemp'].conj().T @ AU
    AR = A_sp @ ch['Uo']
    Mr = ch['Uu'].conj().T @ AR
    return float(np.vdot(M, M).real), float(np.vdot(Mr, Mr).real)


def mixed_variances(ch, A_sp):
    """sum_kk' f_k (1 - f_k') |A_kk'|^2 in the covariance's eigenbasis and the reference part, for a Gaussian mixture."""
    raise NotImplementedError(MIXED_MSG)


# ============================================================================ the self-test
def selftest(args):
    OPS, IO = import_mods(args.engine_dir, args.slices_dir)
    sys.path.insert(0, args.witness_dir)
    import qh_branch_witness_v1b as W
    res = []

    def rep(name, val, tol, ok=None, info=False):
        ok = (val <= tol) if ok is None else ok
        tag = 'INFO' if info else ('PASS' if ok else 'FAIL')
        res.append(dict(name=name, value=val, tol=tol, status=tag))
        log('  [%s] %s: %s (tol %s)' % (tag, name, val, tol))

    t_all = time.time()
    cs = CoreSlice(args.history, args.clock, OPS, IO, slice_path=args.slice, records=args.records, fields=args.fields, well_dir=args.well_dir)
    c64 = cs.rd.complex64
    src_tol = 1e-4 if c64 else 1e-10  # complex64: the inventory floor 6.2e-6 to 8.7e-6 (sources from the columns vs the stored totals)
    log('slice %s: T/r_s %.3f, %d sites, %s, kappas %s, dust %s, state %s, m %g g %g ell %g'
        % (cs.path, cs.rd.T_over_rs, cs.nsites, cs.rd.dtype_str, cs.kappas, cs.rd.n_dust, cs.state_kind(), cs.m, cs.g, cs.ell))
    # S1 the fields
    rep('S1 slice applied fields vs fields-npz applied row (max abs)',
        float(max(np.max(np.abs(cs.fields_row['N_applied'] - cs.N)), np.max(np.abs(cs.fields_row['v_applied'] - cs.v)))), 0.0, ok=None)
    rep('S1b engine_ops_v1 chain radii vs slice R (max abs)', float(np.max(np.abs(OPS.chain_parts(cs.nsites, cs.m, 1.0, cs.half)[2] - cs.R))), 1e-12)
    # S2 channel ops against page_ops_v6 (the CORE_well route)
    try:
        sys.path.insert(0, args.page_dir)
        import page_ops_v6 as PO
        h0p, Qp = PO.chain_sparse(cs.R, cs.m, 1)
        Hp = PO.clocked_csr(h0p, Qp, cs.N, cs.v)
        h, Q, hm = cs.operators(1)
        rep('S2 engine_ops_v1 h_m vs page_ops_v6 clocked_csr, kappa 1 (max abs)', float(np.max(np.abs(hm - Hp.toarray()))), 1e-13)
    except Exception as ex:  # noqa: BLE001
        rep('S2 page_ops_v6 cross-check not run (%s: %s)' % (type(ex).__name__, ex), float('nan'), 0.0, ok=True, info=True)
    # S3 to S6 the state, every channel
    eps = np.zeros(cs.nsites)
    jj = np.zeros(cs.nsites)
    epd = np.zeros(cs.nsites)
    eps_raw = np.zeros(cs.nsites)
    jj_raw = np.zeros(cs.nsites)
    dust_total = 0.0
    worst_orth = worst_ref = 0.0
    keep = {}
    for k_ in cs.kappas:
        ch = cs.channel(k_)
        e1, j1, ed, jd = channel_sources(OPS, ch)
        eps += e1
        jj += j1
        epd += ed
        # the stored columns as written (no Loewdin step): the form the engine's sources were made from
        d_raw, f_raw = cs.rd.dust(k_)
        Phi_raw = np.concatenate([cs.rd.sea(k_), d_raw], axis=1)
        er, jr = OPS.sources_dynamic(ch['h'], ch['Q'], Phi_raw, np.concatenate([np.ones(cs.rd.n_sea[k_]), f_raw]), cs.rd.R2(k_), ch['w'])
        eps_raw += er
        jj_raw += jr
        del Phi_raw, d_raw
        dust_total += ch['w'] * float(np.sum(ch['f_dust']))
        G = ch['Uocc'].conj().T @ ch['Uocc']
        worst_orth = max(worst_orth, float(np.max(np.abs(G - np.eye(G.shape[0])))))
        Gr = ch['Uo'].conj().T @ ch['Uo']
        worst_ref = max(worst_ref, float(np.max(np.abs(Gr - np.eye(Gr.shape[0])))))
        log('  kappa %d: n_occ %d (sea %d, dust %d), stored orthonormality %.1e, ref stored %.1e, %.0f s'
            % (k_, ch['n_occ'], ch['n_sea'], ch['n_dust'], ch['checks']['stored_orthonormality'], ch['checks']['ref_stored_orthonormality'] or 0.0, ch['seconds']))
        if k_ in (args.wit_kappas or []):
            keep[k_] = ch
        else:
            del ch
    nrm = lambda x: float(np.max(np.abs(x)))  # noqa: E731
    rep('S3 stored columns: sum_kappa 16k Tr[(C - P_ref) h_n] vs fields-npz eps_tot (max rel)', nrm(eps_raw - cs.fields_row['eps_tot']) / nrm(cs.fields_row['eps_tot']), src_tol * 0.1)
    rep('S3b stored columns: the same for Q_n vs J_tot (max rel)', nrm(jj_raw - cs.fields_row['J_tot']) / nrm(cs.fields_row['J_tot']), src_tol * 0.1)
    # the adapter's (Loewdin-orthonormalized) state moves the sources by about the stored orthonormality defect times the sea's size
    rep('S3c adapter state (Loewdin): eps vs eps_tot (max rel)', nrm(eps - cs.fields_row['eps_tot']) / nrm(cs.fields_row['eps_tot']), src_tol * 10)
    rep('S3d adapter state (Loewdin): J vs J_tot (max rel)', nrm(jj - cs.fields_row['J_tot']) / nrm(cs.fields_row['J_tot']), src_tol * 10)
    if 'eps_pairs' in cs.fields_row:
        ref = cs.fields_row['eps_tot'] - cs.fields_row['eps_pairs']
        rep('S4 dust part vs eps_tot - eps_pairs (max rel)', nrm(epd - ref) / nrm(ref), src_tol)
    if cs.well_dust is not None:
        rep('S5 sum 16k f_dust vs CORE_well dust total_weighted (abs)', abs(dust_total - float(cs.well_dust)), 1e-9)
    rep('S6 occupied columns orthonormality after Loewdin (max abs)', worst_orth, 1e-10)
    rep('S6b reference columns orthonormality after Loewdin (max abs)', worst_ref, 1e-10)
    # S7, S8 information: the stored e_k against h_m's spectrum; P_ref against the instantaneous negative projector
    ch = keep.get(cs.kappas[0]) or cs.channel(cs.kappas[0])
    ev, Ue = sla.eigh(ch['hm'], driver='evr', check_finite=False)
    if ch['e_stored'] is not None:
        rep('S7 stored e_k vs eigvalsh(h_m(applied)), kappa %d (max abs)' % ch['kappa'], float(np.max(np.abs(np.sort(ch['e_stored']) - ev))), 1e-8, info=True)
    Upos = Ue[:, ev > 0]
    leak = float(np.vdot(Upos.conj().T @ ch['Uo'], Upos.conj().T @ ch['Uo']).real)
    rep('S8 P_ref leakage into the positive spectrum of h_m, ||P_+ R2||_F^2, kappa %d (continued reference, not instantaneous)' % ch['kappa'], leak, 0.0, info=True)
    del Ue, Upos
    # W1 the witness interface: every item of qh_branch_witness_v1b --core
    itf = cs.interface()
    have = dict(fields=cs.N.shape == (cs.nsites,) and cs.v.shape == (cs.nsites,), channel_set=bool(itf['channel_set']['kappas']),
                state=itf['state'] in ('pure', 'mixed'), P_ref=all(('R2_%d' % k_) in cs.rd.members for k_ in cs.kappas),
                radii=any(r is not None for r in itf['radii'].values()), gate=itf['limit_label'] is not None)
    out = subprocess.run([sys.executable, '-B', os.path.join(args.witness_dir, 'qh_branch_witness_v1b.py'), '--core', '--nice', '0'],
                         capture_output=True, text=True, timeout=300)
    printed = out.stdout.strip()
    rep('W1 witness --core prints its interface (rc %d) and every item is supplied: %s' % (out.returncode, have),
        float(sum(not v for v in have.values())), 0.0, ok=out.returncode == 0 and 'core slices' in printed and all(have.values()))
    # W2 the witness's per-channel routines on the adapter's channels (pure-state part)
    rng = np.random.default_rng(7)
    for k_, ch in keep.items():
        x = rng.normal(size=2 * cs.nsites)
        A = W.vertex_sparse(ch['h_sp'], ch['Q_sp'], x, cs.nsites)
        si = W.static_identity(ch, A, cs.nsites)
        rep('W2 kappa %d static identity Var_eig vs (1/2)||[A,C]||^2 (rel)' % k_, si['rel_diff_eig_comm'], 1e-10)
        rep('W2b kappa %d static identity Var_site vs (1/2)||[A,C]||^2 (rel)' % k_, si['rel_diff_site_comm'], 1e-10)
        vC, vR = pure_variances(ch, A)
        rep('W3 kappa %d adapter Var_C vs witness var_block(A, Uemp, Uocc) (rel)' % k_, abs(vC - W.var_block(A, ch['Uemp'], ch['Uocc'])) / max(vC, 1e-300), 1e-12)
        G, dft = W.gram_channel(ch, cs.nsites)
        s2 = float(np.real(x @ (G @ x)))
        rep('W4 kappa %d witness source Gram x^T G x vs Var_C (rel); Hermiticity defect %.1e' % (k_, dft), abs(s2 - vC) / max(vC, 1e-300), 1e-10)
        ih = W.static_identity(ch, sp.csr_matrix(ch['hm']), cs.nsites)
        rep('W5 kappa %d Var_C(dGamma h_m) of the core state (nonzero: not stationary), Var_P_ref %.4e' % (k_, pure_variances(ch, sp.csr_matrix(ch['hm']))[1]),
            ih['var_site'], 0.0, info=True)
        del G
    # M1 the mixed-state stub
    try:
        mixed_variances({}, None)
        rep('M1 mixed-state stub raises NotImplementedError', 1.0, 0.0, ok=False)
    except NotImplementedError as ex:
        rep('M1 mixed-state stub raises NotImplementedError("%s")' % ex, 0.0, 0.0, ok=str(ex) == MIXED_MSG)
    nfail = sum(r['status'] == 'FAIL' for r in res)
    summary = dict(version=VERSION, slice=itf, results=res, n_fail=nfail, n_pass=sum(r['status'] == 'PASS' for r in res), seconds=time.time() - t_all)
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, 'core_adapter_selftest_%s_T%05.2f.json' % (args.history, args.clock))
    with open(p, 'w') as fh:
        json.dump(jsonable(summary), fh, indent=1)
    log('self-test: %d PASS, %d FAIL, %d INFO (%.0f s); %s' % (summary['n_pass'], nfail, sum(r['status'] == 'INFO' for r in res), summary['seconds'], p))
    return 1 if nfail else 0


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='interface', choices=['interface', 'selftest', 'summary'])
    ap.add_argument('--history', default='prod', choices=sorted(HISTORIES))
    ap.add_argument('--clock', type=float, default=12.0, help='T/r_s of the slice (selftest)')
    ap.add_argument('--clocks', default='all', help='comma list of T/r_s or "all" (summary)')
    ap.add_argument('--slice', default=None, help='explicit slice npz (overrides the search)')
    ap.add_argument('--slice-dirs', nargs='*', default=[], help='extra directories searched first (a restored s8 copy)')
    ap.add_argument('--records', default=None)
    ap.add_argument('--fields', default=None)
    ap.add_argument('--well-dir', default=WELL_DIR)
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--slices-dir', default=SLICES_DIR)
    ap.add_argument('--witness-dir', default=WITNESS_DIR)
    ap.add_argument('--page-dir', default=PAGE_DIR)
    ap.add_argument('--wit-kappas', type=lambda s: [int(x) for x in s.split(',') if x.strip()], default=[1, 6],
                    help='channels kept for the witness-routine checks (selftest)')
    ap.add_argument('--out', default='<internal>/core_adapter')
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    if args.nice > 0:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    if args.mode == 'selftest':
        if args.slice is None:
            args.slice = locate_slice(args.history, args.clock, args.slice_dirs)
        sys.exit(selftest(args))
    OPS, IO = import_mods(args.engine_dir, args.slices_dir)
    clocks = HISTORIES[args.history]['clocks'] if args.clocks == 'all' else [float(x) for x in args.clocks.split(',') if x.strip()]
    os.makedirs(args.out, exist_ok=True)
    for t in clocks:
        p = args.slice if (args.slice and len(clocks) == 1) else locate_slice(args.history, t, args.slice_dirs)
        if p is None:
            log('%s T/r_s %g: MISSING on the box (%s)' % (args.history, t, slice_name(HISTORIES[args.history]['tag'], t)))
            continue
        cs = CoreSlice(args.history, t, OPS, IO, slice_path=p, records=args.records, fields=args.fields, well_dir=args.well_dir)
        itf = cs.interface()
        if args.mode == 'summary':
            itf['channels'] = []
            for k_ in cs.kappas:
                ch = cs.channel(k_)
                itf['channels'].append(dict(kappa=k_, w=ch['w'], n_occ=ch['n_occ'], n_sea=ch['n_sea'], n_dust=ch['n_dust'], checks=ch['checks'], seconds=ch['seconds']))
                del ch
            with open(os.path.join(args.out, 'core_slice_%s_T%07.2f.json' % (args.history, t)), 'w') as fh:
                json.dump(jsonable(itf), fh, indent=1)
        log(json.dumps(jsonable(itf), indent=1))


if __name__ == '__main__':
    main()
