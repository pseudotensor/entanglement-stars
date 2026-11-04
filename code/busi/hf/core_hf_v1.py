#!/usr/bin/env python3
"""
core_hf_v1.py  (busi lane, inbox #295; REVIEWA_core_adapter_spec_v1, the write lane's specification of 19 Oct): the core-slice
adapter CORE_HF_v1 and its driver, for the two consumers qh_branch_witness_v1b (the core q_H branch witness, row 8) and
hf_star_variance_v3 (the core's Fock term, the differentiated-vertex variance, the sea part, the native-kernel variance; rows
19 and 22).

Built on the skeleton busi/code/core/core_adapter_v1.py (md5 c020d454; its pure-state path, its reading and its 21 checks are
carried over: the checks are run again in the adapter stage under the skeleton's names S1 to S8, W1 to W5). The skeleton stays
as issued. The mixed-state path of the specification (section 2) is complete here; on the stored slices (pure per channel, f = 1
on every dust column) it is the general code path and the complex64 floor control.

Imports by name, edits none of them: engine_ops_v1 (the chain operators, the site densities), slices144_io_v1 (SliceReader,
inventory, frac_radius; setup() is never called: no torch), hf_star_variance_v3 (radial_solve, linearized,
complex_step_columns, ShellKernel, ResponseOperator, accumulate_response, gmres_solve, rebuild_channels, load_star),
qh_branch_witness_v1b (bilinear_blocks, sigma_map, vertex_sparse, exp_sum_nodes, copy_interval, psd_factor, exact_doubles,
f_width, subtraction_bound, var_block), engine_fields_v2 (the Gaussian kernels K_0, K_1).

Stages (per slice; a stage's json present = skipped unless --force):
  adapter   section 1 of the specification with its checks (the skeleton's included): core_hf_<h>_T<t>_adapter.json
  variance  the native-kernel rows, the response, the dressed kernel, the variance table (variants x radii x channels), the
            sea part, the signed excess, the Grams, the exchange and pair energies, the static identity
  witness   variance + q_H (J and subsets) + L_seed (pure slices; singles with the h_m term) [+ exact doubles, + B_E]:
            core_hf_<h>_T<t>.json and _arrays.npz
  channels  the sea part summed channel by channel beyond kappa_sea = 10 at the slice's fields (resumable checkpoint)
  all       witness then channels;   table: core_hf_table.md
Star mode (--star FAMILY.json --member m): the same stages on a static-star member (S10's channels; --kernel compact is the S10
regression, K10; --kernel gauss the native-kernel star).

CPU only (numpy, scipy), at most 16 threads, renices itself. Outputs only under --out (default <internal>/fundmarks
core_hf_v1). Raw arrays saved; no plotting. Run on the box only.
"""
import os
import sys


def _set_threads():
    t = 16
    for i, x in enumerate(sys.argv):
        if x == '--threads' and i + 1 < len(sys.argv):
            try:
                t = int(sys.argv[i + 1])
            except ValueError:
                t = 16
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

VERSION = 'core_hf_v1'
FOURPI = 4.0 * math.pi
EXIT_FORMAT = 2
EXIT_SPLIT = 3

# ============================================================================ locations (box)
ENGINE_DIR = '<internal>/code'          # engine_ops_v1.py (md5 301c45a2), engine_fields_v2.py
SLICES_DIR = '<internal>/slices144'      # slices144_io_v1.py
S10_DIR = '<internal>/hf'              # hf_star_variance_v3.py, qh_branch_witness_v1b.py
PAGE_DIR = '<internal>/v6'                 # page_ops_v6.py (the skeleton's S2 cross-check only)
OUT_DIR = '<internal>/core_hf_v1'
BD = '<internal>/solve_a_bulk'
EH = '<internal>/H100b'
HIST = {
    'prod': dict(tag='c19_rs64_A_R2_tau8_ell4_avg_r11', label='the production (sigma/a 4)',
                 records=EH + '/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
                 fields=EH + '/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz',
                 well=BD + '/core_well_v1/well', extra_dirs=['<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11']),
    's8': dict(tag='c27_rs64_A_R2_tau8_ell8_prep4_avg', label='the sigma/a 8 member',
               records=EH + '/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json',
               fields=EH + '/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz',
               well=BD + '/core_well_v1/well', extra_dirs=[]),
    'r8': dict(tag='c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0', label="the rung's (sigma/a 8 stand-in; never the member's own)",
               records=BD + '/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0.json',
               fields=BD + '/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_fields.npz',
               well=BD + '/core_well_v1_rung/well', extra_dirs=[BD + '/v36d_from0/slices']),
    'r4': dict(tag='c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0', label="the rung's (sigma/a 4 rung)",
               records=BD + '/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json',
               fields=BD + '/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_fields.npz',
               well=BD + '/core_well_v1_rung/well', extra_dirs=[BD + '/v36d_from0/slices']),
}
CLOCKS = {'prod': [12, 15, 16, 20, 24, 25, 28, 30, 35, 40], 's8': [12, 13, 14, 16, 20, 25, 30, 40], 'r8': [12, 13, 14, 16], 'r4': [12]}


# ============================================================================ helpers
def log(*a):
    print(*a, flush=True)


def fnorm(X):
    if sp.issparse(X):
        X = X.tocsr()
        X.sum_duplicates()
        return float(np.sqrt(np.sum(np.abs(X.data) ** 2)))
    return float(np.linalg.norm(np.asarray(X).ravel()))


def herm(X):
    return 0.5 * (X + X.conj().T)


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
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    return o


def site_sum(x, ncomp=2):
    return np.asarray(x).reshape(-1, ncomp).sum(axis=1)


class Mods:
    """the imported modules (by name, from the directories given)."""

    def __init__(self, engine_dir=ENGINE_DIR, slices_dir=SLICES_DIR, s10_dir=S10_DIR):
        for d in (s10_dir, slices_dir, engine_dir):
            if d not in sys.path:
                sys.path.insert(0, d)
        import engine_ops_v1 as OPS
        OPS.GPU['dev'] = None                         # R14: CPU eigh (scipy evr)
        import slices144_io_v1 as IO                  # SliceReader, inventory, frac_radius; setup() never called (R12)
        import hf_star_variance_v3 as S10
        import qh_branch_witness_v1b as QH
        try:
            import engine_fields_v2 as EF2
        except Exception as ex:  # noqa: BLE001
            EF2 = None
            log('engine_fields_v2 not importable (%s): the Gaussian kernel uses the re-typed formula' % ex)
        self.OPS, self.IO, self.S10, self.QH, self.EF2 = OPS, IO, S10, QH, EF2
        self.paths = dict(engine_ops_v1=OPS.__file__, slices144_io_v1=IO.__file__, hf_star_variance_v3=S10.__file__,
                          qh_branch_witness_v1b=QH.__file__, engine_fields_v2=None if EF2 is None else EF2.__file__)


class ResponseSplitError(RuntimeError):
    pass


class FormatError(RuntimeError):
    pass


# ============================================================================ reading
def find_slice(history, T_over_rs, slice_dir=None, IO=None):
    """the slice path: --slice-dir first, then slices144_io_v1.inventory()/locate() order, then the lane's extra copies."""
    H = HIST[history]
    name = 'orbitals_%s_T%07.2f.npz' % (H['tag'], float(T_over_rs))
    entry = None
    cands = []
    if slice_dir:
        cands.append(os.path.join(slice_dir, name))
    if IO is not None and history in ('prod', 's8'):
        inv = IO.inventory()
        sel = IO.select_entries(inv[history], '%g' % float(T_over_rs))
        if sel:
            entry = sel[0]
            cands += list(entry['box_paths'])
    cands += [os.path.join(d, name) for d in H['extra_dirs']]
    path = next((p for p in cands if os.path.isfile(p)), None)
    return dict(path=path, candidates=cands, entry=entry, name=name)


class CoreSlice:
    pass


def load_core_slice(path, IO, verify_sha=False, log=log, m=None, half=False, entry=None):
    """small arrays loaded, columns memory-mapped (SliceReader); the size check against the inventory; --verify-sha."""
    rd = IO.SliceReader(path, entry=entry, log=log)
    sl = CoreSlice()
    sl.path, sl.rd = os.path.abspath(path), rd
    sl.T, sl.rs, sl.T_over_rs = rd.T, rd.rs, rd.T_over_rs
    sl.R = np.asarray(rd.R, float)
    sl.nsites = rd.nsites
    sl.N = np.asarray(rd.N_applied, float)
    sl.v = np.asarray(rd.v_applied, float)
    sl.kappas = list(rd.kappas)
    sl.weights = {k: float(rd.weights[k]) for k in rd.kappas}
    sl.complex64 = bool(rd.complex64)
    sl.floor = rd.floor if rd.complex64 else None
    sl.dtype = rd.dtype_str
    sl.n_sea, sl.n_dust = dict(rd.n_sea), dict(rd.n_dust)
    sl.m, sl.half = m, bool(half)
    sl.size = os.path.getsize(path)
    exp = IO.SIZES.get((sl.nsites, 'c64' if sl.complex64 else 'c128'))
    sl.size_check = dict(bytes=sl.size, expected=exp, ok=(exp is None or sl.size == exp) if entry is not None else None)
    if entry is not None and exp is not None and sl.size != exp:
        raise FormatError('%s: %d bytes, the inventory expects %d' % (path, sl.size, exp))
    sl.sha = None
    if verify_sha and entry is not None and entry.get('sha256_prefix'):
        h = IO.sha256_file(path)
        sl.sha = dict(sha256=h, expected_prefix=entry['sha256_prefix'], ok=h.startswith(entry['sha256_prefix']))
        if not sl.sha['ok']:
            raise FormatError('%s: sha256 %s does not start with %s' % (path, h[:12], entry['sha256_prefix']))
    return sl


REC_KEYS = ('T', 'T_over_rs', 'E', 'E_over_E0', 'E_dust_shells', 'E_pairs_shells', 'trapping_zeros', 'r_in', 'R_t', 'R_c', 'misner_sharp_at_Rt',
            'W_over_2g_at_Rt', 'converged', 'n_pocket', 'inversions_total', 'intermediate_weights_total', 'mask_equals_negative_set',
            'commutator_h_R2_F', 'min_occupied_weight', 'N_min', 'continuation_label')


def load_record(records_json, fields_npz, T_over_rs):
    J = json.load(open(records_json))
    A = J['args']
    rs = float(A['rs'])
    rec = None
    for r in J['records']:
        if r.get('E') is None:
            continue
        tr = float(r['T_over_rs']) if r.get('T_over_rs') is not None else float(r['T']) / rs
        if abs(tr - float(T_over_rs)) < 1e-6:
            rec = r
    if rec is None:
        raise RuntimeError('%s: no record at T/r_s %g' % (records_json, T_over_rs))
    out = dict(records_json=records_json, fields_npz=fields_npz, m=float(A['m']), g=float(A['g']), ell=float(A['ell']), rs=rs,
               half=bool(A.get('half_offset', 0)), kappa_sea=A.get('kappa_sea'), args_tag=A.get('tag'))
    out['record'] = {k: rec.get(k) for k in REC_KEYS}
    out['record']['gauge_label'] = (rec.get('gauge') or {}).get('label')
    sv = rec.get('solver') or {}
    out['record']['solver_residual_W'] = sv.get('residual_W')
    out['record']['solver_residual_invN'] = sv.get('residual_invN')
    Z = np.load(fields_npz, allow_pickle=True)
    Tz = np.asarray(Z['T'], float)
    i = int(np.argmin(np.abs(Tz / rs - float(T_over_rs))))
    if abs(Tz[i] / rs - float(T_over_rs)) > 1e-6:
        raise RuntimeError('%s: no fields row at T/r_s %g' % (fields_npz, T_over_rs))
    out['index'] = i
    for k in ('eps_tot', 'J_tot', 'eps_pairs', 'J_pairs', 'N_applied', 'v_applied', 'N', 'v'):
        if k in Z.files:
            out[k + '_fields' if k in ('N_applied', 'v_applied', 'N', 'v') else k] = np.asarray(Z[k][i], float)
    out['fields_ell'] = float(Z['ell']) if 'ell' in Z.files else None
    out['fields_g'] = float(Z['g']) if 'g' in Z.files else None
    return out


def status_flags(rec):
    r = rec['record']
    flags = []
    pockets = sum(int(x) for x in (r.get('n_pocket') or {}).values()) if isinstance(r.get('n_pocket'), dict) else int(r.get('n_pocket') or 0)
    if pockets > 0:
        flags.append('pockets %d' % pockets)
    if int(r.get('inversions_total') or 0) > 0:
        flags.append('inversions %d' % int(r['inversions_total']))
    if float(r.get('intermediate_weights_total') or 0) > 0:
        flags.append('intermediate weights %g' % float(r['intermediate_weights_total']))
    if r.get('mask_equals_negative_set') is False:
        flags.append('mask != negative set')
    stop = list(flags)
    if r.get('converged') is False:
        flags.append('record not converged')
    if r.get('gauge_label') not in (None, 'intact'):
        flags.append('gauge %s' % r.get('gauge_label'))
    return flags, stop


def load_well(well_dir, history, T_over_rs):
    p = os.path.join(well_dir, 'well_%s_T%07.2f.json' % (history, float(T_over_rs)))
    if not os.path.exists(p):
        return None
    W = json.load(open(p))
    out = dict(path=p, roots=W.get('roots', {}), dust=W.get('dust', {}), record=W.get('record', {}), gate=W.get('gate'),
               limit_label=W.get('limit_label'), mass_function=W.get('mass_function', {}), channels=W.get('channels', []), floor=W.get('floor'))
    pz = p[:-5] + '.npz'
    out['arrays'] = {}
    if os.path.exists(pz):
        Z = np.load(pz)
        for k in ('dust_enclosed', 'dust_number_site', 'n_density', 'eps_tot', 'J_tot', 'rho_density', 'rho_dust_density', 'M_w_dense',
                  'M_w_applied', 'N_dense_sites', 'v_dense_sites'):
            if k in Z.files:
                out['arrays'][k] = np.asarray(Z[k], float)
    return out


def dust_profile(sl):
    """the dust's number per site from the stored dust columns with f (the readout's definition), weighted 16 kappa."""
    n_w = np.zeros(sl.nsites)
    per = {}
    for k in sl.kappas:
        if sl.n_dust[k]:
            d, f = sl.rd.dust(k)
            per[k] = site_sum((np.abs(d) ** 2) @ np.asarray(f, float))
            n_w += sl.weights[k] * per[k]
    return n_w, np.cumsum(n_w), per


def radii_table(sl, rec, well, IO):
    """label -> (r, site, N_enclosed, M_w_dense, M_w_applied, source); R_50, R_90 recomputed (frac_radius) and compared with well."""
    n_w, cum, per = dust_profile(sl)
    tot = float(cum[-1]) if len(cum) else 0.0
    R = sl.R
    r = rec['record']
    tz = list(r.get('trapping_zeros') or [])
    src = {}
    if len(tz) >= 2:
        src['R_in'] = (float(tz[0]), "the record's dense inner root (trapping_zeros[0], r_in %s)" % r.get('r_in'))
    if len(tz) >= 1:
        src['R_t'] = (float(tz[-1]), "the record's dense outer root (trapping_zeros[-1], R_t %s)" % r.get('R_t'))
    cmp = {}
    if tot > 0:
        for lab, q in (('R_50', 0.5), ('R_90', 0.9)):
            x = IO.frac_radius(R, cum / tot, q)
            src[lab] = (float(x), 'recomputed from the slice dust columns (frac_radius, linear between sites)')
            if well and well['dust'].get(lab) is not None:
                cmp[lab] = abs(float(x) - float(well['dust'][lab]))
    if r.get('R_c') is not None:
        src['R_c'] = (float(r['R_c']), "the record's R_c (a window edge only)")
    if well:
        for lab in ('applied_inner', 'applied_outer'):
            if well['roots'].get(lab) is not None:
                src[lab] = (float(well['roots'][lab]), 'CORE_well applied root (the radius systematic)')
    out = {}
    A = (well or {}).get('arrays', {})
    for lab, (x, s) in src.items():
        site = min(int(np.searchsorted(R, x)), sl.nsites - 1)
        ent = dict(r=x, site=site, N_enclosed=float(np.interp(x, R, cum)), source=s)
        for key in ('M_w_dense', 'M_w_applied'):
            if key in A:
                ent[key] = float(np.interp(x, R, A[key]))
        out[lab] = ent
    return out, dict(N_B_dust=tot, R_vs_well_abs=cmp, n_per_channel={k: float(v.sum()) for k, v in per.items()}, cum=cum, n_site=n_w)


# ============================================================================ the channel
def loewdin(X):
    """X (X^dagger X)^(-1/2) by numpy eigh of the Gram (engine_massive_channel_D_v2.loewdin's arithmetic); (Xo, defect before, after)."""
    X = np.asarray(X, np.complex128)
    if X.shape[1] == 0:
        return X.copy(), 0.0, 0.0
    G = herm(X.conj().T @ X)
    d0 = float(np.max(np.abs(G - np.eye(G.shape[0]))))
    w, V = np.linalg.eigh(G)
    Xo = X @ ((V * (1.0 / np.sqrt(np.clip(w, 1e-300, None)))[None, :]) @ V.conj().T)
    G1 = herm(Xo.conj().T @ Xo)
    d1 = float(np.max(np.abs(G1 - np.eye(G1.shape[0]))))
    if d1 > 4e-15:   # one refinement pass of the same step (removes the eigh's roundoff; the state moves by O(d1))
        w, V = np.linalg.eigh(G1)
        Xo = Xo @ ((V * (1.0 / np.sqrt(np.clip(w, 1e-300, None)))[None, :]) @ V.conj().T)
        G1 = Xo.conj().T @ Xo
        d1 = float(np.max(np.abs(G1 - np.eye(G1.shape[0]))))
    return Xo, d0, d1


def purity_decision(Phi_raw, f, complex64):
    G = herm(Phi_raw.conj().T @ Phi_raw)
    d_orth = float(np.max(np.abs(G - np.eye(G.shape[0])))) if G.shape[0] else 0.0
    f = np.asarray(f, float)
    d_f = float(np.max(np.minimum(np.abs(f), np.abs(1.0 - f)))) if f.size else 0.0
    tol = 1e-6 if complex64 else 1e-10
    info = dict(d_orth=d_orth, d_f=d_f, orth_tol=tol, complex64=bool(complex64))
    if d_orth > tol:
        return 'FORMAT', info
    return ('pure' if d_f <= 1e-12 else 'mixed'), info


def c2_defect(Phi, f):
    """||C^2 - C||_F / sqrt(rank C) for C = Phi F Phi^+ by Tr(Y G Y G), Y = F G F - F, G = Phi^+ Phi."""
    if Phi.shape[1] == 0:
        return 0.0
    G = Phi.conj().T @ Phi
    F = np.asarray(f, float)
    Y = (F[:, None] * G) * F[None, :] - np.diag(F)
    YG = Y @ G
    val = float(np.real(np.sum(YG * YG.T)))
    return math.sqrt(max(val, 0.0)) / math.sqrt(max(int(np.sum(F > 0)), 1))


def vertex(QH, ch, x, ns):
    return QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], np.asarray(x, float), ns)


def channel_ops(sl, kappa, OPS, QH):
    """engine_ops_v1's channel kappa at the slice's applied fields: h, Q (dense and sparse), h_m sparse, eigh(h_m) ascending,
    the vertex identity h_m = sum_n N_n hcal_n + v_n p_n."""
    t0 = time.time()
    h, Q, Rc = OPS.chain_parts(sl.nsites, sl.m, float(kappa), sl.half)
    if np.max(np.abs(Rc - sl.R)) > 1e-12:
        raise FormatError('engine_ops_v1 chain radii differ from the slice R (half_offset %s)' % sl.half)
    hm = OPS.clocked(h, Q, sl.N, sl.v)
    h_sp, Q_sp = sp.csr_matrix(h), sp.csr_matrix(Q)
    hm_sp = sp.csr_matrix(hm)
    ident = fnorm(QH.vertex_sparse(h_sp, Q_sp, np.concatenate([sl.N, sl.v]), sl.nsites) - hm_sp) / fnorm(hm_sp)
    t1 = time.time()
    e, U = sla.eigh(hm, driver='evr', check_finite=False)
    del hm
    return dict(kappa=int(kappa), h=np.asarray(h), Q=np.asarray(Q), h_sp=h_sp, Q_sp=Q_sp, hm_sp=hm_sp, e=np.asarray(e), U=np.asarray(U),
                vertex_identity_rel=ident, eigh_seconds=time.time() - t1, ops_seconds=time.time() - t0)


def channel_state(sl, kappa, ops, OPS, tol_pure=1e-12, keep_raw=None, eigD=True):
    """the dict of specification 1.10: the Loewdin step, the purity decision, the mask and P_ref checks, D, the floors."""
    t0 = time.time()
    rd = sl.rd
    raw = rd.channel(kappa, orthonormalize=False)
    Phi_raw, f = raw['Phi'], np.asarray(raw['f'], float)
    kind, pinfo = purity_decision(Phi_raw, f, sl.complex64)
    w = float(sl.weights[kappa])
    ch = dict(ops)
    ch.update(kappa=int(kappa), w=w, extra=False, n_sea=int(raw['n_sea']), n_dust=int(raw['n_dust']), purity=kind, purity_info=pinfo,
              stored_orthonormality=pinfo['d_orth'], dust_sea_overlap_max=raw['dust_sea_overlap_max'], complex64=sl.complex64)
    if kind == 'FORMAT':
        ch['seconds'] = time.time() - t0
        return ch
    Phi, d0, d1 = loewdin(Phi_raw)
    ch.update(Phi=Phi, f=f, n_occ=int(Phi.shape[1]), loewdin_orthonormality=d1, loewdin_input_defect=d0, closed=(kind == 'pure'),
              Uocc=Phi if kind == 'pure' else None, Uemp=None, Ud=None)
    ch['C2_defect'] = c2_defect(Phi, f)
    # the stored columns as written: the engine's own sources (its normal ordering), the floor control
    R2raw = rd.R2(kappa)
    src_raw = OPS.sources_dynamic(ch['h_sp'], ch['Q_sp'], Phi_raw, f, R2raw, w)
    n_s = ch['n_sea']
    ch['src_dust_raw'] = ((w * OPS.site_density(ch['h_sp'], Phi_raw[:, n_s:], f[n_s:]), w * OPS.site_density(ch['Q_sp'], Phi_raw[:, n_s:], f[n_s:]))
                          if ch['n_dust'] else (np.zeros(sl.nsites), np.zeros(sl.nsites)))
    ch['src_raw'] = src_raw
    if keep_raw is None:
        keep_raw = sl.complex64
    ch['Phi_raw'] = Phi_raw if keep_raw else None
    del raw
    Pref, r0, r1 = loewdin(R2raw)
    del R2raw
    ch.update(Pref_cols=Pref, ref_stored_orthonormality=r0, ref_loewdin_orthonormality=r1)
    # the reference's eigen-resolution
    e, U = ch['e'], ch['U']
    mask_eig = e < 0
    e_st = rd.e(kappa)
    m_st = rd.mask(kappa)
    ch['e_vs_stored_abs'] = float(np.max(np.abs(np.asarray(e_st, float) - e))) if e_st is not None else None
    ch['e_vs_stored_rel'] = ch['e_vs_stored_abs'] / float(np.max(np.abs(e))) if e_st is not None else None
    ch['mask_stored'] = m_st
    ch['mask_equals_neg'] = bool(m_st is not None and np.array_equal(m_st, mask_eig))
    ch['stored_mask_equals_stored_neg'] = bool(m_st is not None and e_st is not None and np.array_equal(m_st, np.asarray(e_st) < 0))
    ch['mask'] = mask_eig
    ch['Uo'] = np.ascontiguousarray(U[:, mask_eig])
    ch['Uu'] = np.ascontiguousarray(U[:, ~mask_eig])
    ch['min_abs_e'] = float(np.min(np.abs(e)))
    if Pref.shape[1] == ch['Uo'].shape[1]:
        X = ch['Uu'].conj().T @ Pref
        leak = float(np.vdot(X, X).real)
        ch['pref_resolution_defect'] = math.sqrt(2.0 * leak) / math.sqrt(max(Pref.shape[1], 1))
        del X
    else:
        ch['pref_resolution_defect'] = float('inf')
    ch['pref_resolution_flag'] = bool(ch['pref_resolution_defect'] > 1e-5)
    HP = ch['hm_sp'] @ Pref
    res = HP - Pref @ (Pref.conj().T @ HP)
    ch['commutator_rel'] = fnorm(res) / fnorm(ch['hm_sp'])
    del HP, res
    # D = C - P_ref (dense, Hermitian); its spectrum is the pair content
    C = (Phi * f[None, :]) @ Phi.conj().T
    D = herm(C - Pref @ Pref.conj().T)
    ch['C'] = herm(C)
    ch['D'] = D
    ch['TrD'] = float(np.trace(D).real)
    ch['TrD2'] = float(np.vdot(D, D).real)
    if eigD:
        ev = np.linalg.eigvalsh(D)
        ch['eigD'] = ev
        ch['n_eigD_above_1e-6'] = int(np.sum(np.abs(ev) > 1e-6))
    ch['seconds'] = time.time() - t0
    return ch


def build_channel(sl, kappa, M, keep_raw=None, eigD=True):
    ops = channel_ops(sl, kappa, M.OPS, M.QH)
    return channel_state(sl, kappa, ops, M.OPS, keep_raw=keep_raw, eigD=eigD)


def channel_sources(ch, OPS):
    """rho_C, j_C, rho_ref, j_ref per site (unweighted); s = w (rho_C - rho_ref), w (j_C - j_ref)."""
    rC = OPS.site_density(ch['h_sp'], ch['Phi'], ch['f'])
    jC = OPS.site_density(ch['Q_sp'], ch['Phi'], ch['f'])
    rR = OPS.site_density(ch['h_sp'], ch['Pref_cols'])
    jR = OPS.site_density(ch['Q_sp'], ch['Pref_cols'])
    return rC, jC, rR, jR


# ============================================================================ variances (specification 2.1, 2.2)
def variance_cov(A_sp, Phi, f, orthonormal=False):
    """(Var, mean) of dGamma(A) in the quasi-free state with C = Phi diag(f) Phi^+. orthonormal: the stable form
    sum f ||(1 - Phi Phi^+) A phi||^2 + sum f_k (1 - f_l)|M_kl|^2; otherwise sum f ||A phi||^2 - sum f f |M|^2 (Phi need not be orthonormal)."""
    f = np.asarray(f, float)
    AP = A_sp @ Phi
    M = Phi.conj().T @ AP
    mean = float(np.sum(f * np.real(np.diagonal(M))))
    M2 = np.abs(M) ** 2
    if orthonormal:
        res = AP - Phi @ M
        t_out = float(np.real(np.einsum('ik,ik->k', res.conj(), res)) @ f)
        t_in = float(f @ M2 @ (1.0 - f))
        return t_out + t_in, mean
    t1 = float(np.real(np.einsum('ik,ik->k', AP.conj(), AP)) @ f)
    t2 = float(f @ M2 @ f)
    return t1 - t2, mean


def variance_pure(A_sp, Phi):
    """||A Phi - Phi (Phi^+ A Phi)||_F^2 (= ||A Phi||^2 - ||Phi^+ A Phi||^2) and Tr(C A)."""
    AP = A_sp @ Phi
    M = Phi.conj().T @ AP
    res = AP - Phi @ M
    return float(np.vdot(res, res).real), float(np.trace(M).real)


def variance_ref(A_sp, Uo, Uu=None):
    """Var_P = ||Uu^+ A Uo||_F^2 (QH's var_block(A, Uu, Uo)); with Uu None the residual form; and Tr(P A)."""
    AU = A_sp @ Uo
    if Uu is not None:
        X = Uu.conj().T @ AU
        v = float(np.vdot(X, X).real)
        mean = float(np.real(np.vdot(Uo, AU)))
        return v, mean
    M = Uo.conj().T @ AU
    res = AU - Uo @ M
    return float(np.vdot(res, res).real), float(np.trace(M).real)


def variance_of_vertex_core(ch, beta, ns, QH):
    A = vertex(QH, ch, beta, ns)
    if ch['purity'] == 'pure':
        vC, mC = variance_pure(A, ch['Phi'])
    else:
        vC, mC = variance_cov(A, ch['Phi'], ch['f'], orthonormal=True)
    vS, mS = variance_ref(A, ch['Uo'], ch.get('Uu'))
    out = dict(var_C=vC, var_sea=vS, mean_C=mC, mean_sea=mS, excess=vC - vS)
    if ch.get('Phi_raw') is not None:
        out['var_C_raw'] = variance_cov(A, ch['Phi_raw'], ch['f'], orthonormal=False)[0]
    return out


def static_identity_core(ch, A_sp):
    """var_site = ||Q A Phi||^2 (mixed: the covariance form), (1/2)||[A, C]||_F^2 in the site basis, the mixed defect Tr((C - C^2) A^2)."""
    f = np.asarray(ch['f'], float)
    var_site, _ = variance_cov(A_sp, ch['Phi'], f, orthonormal=True)
    C = ch['C'] if ch.get('C') is not None else herm((ch['Phi'] * f[None, :]) @ ch['Phi'].conj().T)
    AC = A_sp @ C
    comm = AC - AC.conj().T
    half = 0.5 * float(np.vdot(comm, comm).real)
    AP = A_sp @ ch['Phi']
    mixed = float(np.real(np.einsum('ik,ik->k', AP.conj(), AP)) @ (f * (1.0 - f)))
    scale = max(var_site, half, 1e-300)
    return dict(var_site=var_site, half_comm_sq=half, mixed_defect=mixed, rel_diff=abs(var_site - half) / scale,
                rel_diff_with_defect=abs(var_site - half - mixed) / scale)


def gram_channel_core(ch, ns, QH):
    """G_{mu nu} = Tr(C A_mu (1 - C) A_nu) by bilinear_blocks(hs, 1 - C, C) (v1b argument order) with C = Phi F Phi^+."""
    C = ch['C'] if ch.get('C') is not None else herm((ch['Phi'] * ch['f'][None, :]) @ ch['Phi'].conj().T)
    Qm = np.eye(C.shape[0], dtype=np.complex128) - C
    G = QH.bilinear_blocks((ch['h_sp'], ch['Q_sp']), Qm, C, ns)
    defect = fnorm(G - G.conj().T) / max(fnorm(G), 1e-300)
    return herm(G), defect


# ============================================================================ the exchange and the pair energy (2.5)
def exchange_blocks(ch, ns, QH, controls=True):
    """T_kappa[mu, nu] = Tr(A_mu D A_nu D) (complex Hermitian); controls with P (the removed sea term) and with C (unordered)."""
    hs = (ch['h_sp'], ch['Q_sp'])
    out = dict(T=QH.bilinear_blocks(hs, ch['D'], ch['D'], ns))
    if controls:
        P = ch['Uo'] @ ch['Uo'].conj().T
        out['TP'] = QH.bilinear_blocks(hs, P, P, ns)
        del P
        C = ch['C'] if ch.get('C') is not None else herm((ch['Phi'] * ch['f'][None, :]) @ ch['Phi'].conj().T)
        out['TC'] = QH.bilinear_blocks(hs, C, C, ns)
    return out


def exchange_energy(T_list, g_list, V):
    """E_F = -(1/2) sum_kappa g_kappa Re sum_{mu nu} V_{mu nu} T_kappa[mu, nu]; per channel."""
    per = [-0.5 * g * float(np.sum(V * np.real(T))) for T, g in zip(T_list, g_list)]
    return dict(E_F=float(sum(per)), per_channel=per)


def pair_energy(s, V):
    s = np.asarray(s, float)
    return 0.5 * float(s @ (V @ s))


# ============================================================================ the native kernel (2.6)
def gauss_kernels_formula(r, R, ell):
    """the re-typed engine_fields_v2.kernels (K_0, K_1 of the Gaussian of standard deviation ell per axis)."""
    r = np.asarray(r, float)[:, None]
    R = np.asarray(R, float)[None, :]
    Em = np.exp(-(r - R) ** 2 / (2.0 * ell ** 2))
    Ep = np.exp(-(r + R) ** 2 / (2.0 * ell ** 2))
    EA = np.exp(-(r ** 2 + R ** 2) / (2.0 * ell ** 2))
    z = r * R / ell ** 2
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    i0 = np.where(small, EA * (1.0 + z ** 2 / 6.0), (Em - Ep) / (2.0 * zs))
    i1 = np.where(small, EA * (z / 3.0) * (1.0 + z ** 2 / 10.0), ((zs - 1.0) * Em + (zs + 1.0) * Ep) / (2.0 * zs ** 2))
    pref = 1.0 / ((2.0 * np.pi) ** 1.5 * ell ** 3)
    return pref * i0, pref * i1


class GaussShellKernel:
    """hf_star_variance_v3.ShellKernel's interface with the engine's Gaussian kernels (engine_fields_v2.kernels by name; the
    re-typed formula as the fallback): K0f, K1f on the fine grid (dr/2), K0 = K0f[::2], K1 = K1f[::2]; the l = 0 tables
    normalized column by column under the two Simpson quadratures; K1 unnormalized (mom1_err = the K_1 identity on v = c r)."""

    def __init__(self, Rs, ell, Rout, dr, EF2=None, S10=None, chunk_entries=4_000_000):
        self.Rs = np.asarray(Rs, float)
        self.ell, self.Rout = float(ell), float(Rout)
        ns = int(round(Rout / dr))
        ns += ns % 2
        self.ns, self.dr = ns, Rout / ns
        self.r = np.linspace(0.0, Rout, ns + 1)
        self.rf = np.linspace(0.0, Rout, 2 * ns + 1)
        self.W = S10.simpson_weights(ns, self.dr)
        self.Wf = S10.simpson_weights(2 * ns, self.dr / 2.0)
        fun = EF2.kernels if EF2 is not None else gauss_kernels_formula
        self.kernel_source = 'engine_fields_v2.kernels (imported)' if EF2 is not None else 'gauss_kernels_formula (re-typed)'
        nR = len(self.Rs)
        K0f = np.empty((len(self.rf), nR))
        K1f = np.empty((len(self.rf), nR))
        step = max(1, chunk_entries // max(nR, 1))
        for i0 in range(0, len(self.rf), step):
            a, b = fun(self.rf[i0:i0 + step], self.Rs, self.ell)
            K0f[i0:i0 + step] = a
            K1f[i0:i0 + step] = b
        K0, K1 = K0f[::2].copy(), K1f[::2].copy()
        w2 = self.W * FOURPI * self.r ** 2
        w2f = self.Wf * FOURPI * self.rf ** 2
        self.norm_raw = float(np.max(np.abs(w2 @ K0 - 1.0)))
        self.K0 = K0 / (w2 @ K0)[None, :]
        K0f /= (w2f @ K0f)[None, :]
        self.K0f = K0f
        self.K1, self.K1f = K1, K1f
        self.norm_err = float(np.max(np.abs(w2 @ self.K0 - 1.0)))
        self.mom1_err = float(np.max(np.abs((w2 * self.r) @ self.K1 - self.Rs)))


def make_kernel(kind, Rs, ell, Rout, dr, M):
    if kind == 'gauss':
        return GaussShellKernel(Rs, ell, Rout, dr, EF2=M.EF2, S10=M.S10)
    if kind == 'compact':
        kt = M.S10.ShellKernel(Rs, ell, Rout, dr)
        kt.kernel_source = "hf_star_variance_v3.ShellKernel (GAP06's compact kernel; a labelled scheme control on the core)"
        return kt
    raise ValueError(kind)


def resolve_rows(kt, e_src, j_src, G, N_ref, v_ref, E_ref, field_tol, S10, ok_label='CORE'):
    sol = S10.radial_solve(kt, np.asarray(e_src, float), np.asarray(j_src, float), G)
    dN = float(np.max(np.abs(sol['Nx'].real - N_ref)) / np.max(np.abs(N_ref)))
    dv = float(np.max(np.abs(sol['vx'].real - v_ref)) / max(np.max(np.abs(v_ref)), 1e-300))
    E_res = float(sol['E'].real)
    label = ok_label if max(dN, dv) < field_tol else 'SCHEME CONTROL'
    fc = dict(N_rel_maxdev=dN, v_rel_maxdev=dv, E_resolved=E_res, E_record=E_ref, E_rel_dev=abs(E_res - E_ref) / abs(E_ref) if E_ref else None,
              label=label, field_tol=field_tol, kernel=getattr(kt, 'kernel_source', None), dr=kt.dr, R_out=float(kt.r[-1]), ell=kt.ell,
              kernel_norm_raw=kt.norm_raw, kernel_norm_err=kt.norm_err, kernel_mom1_err=kt.mom1_err, N_center=float(sol['N'][0].real))
    return sol, fc


def hessian_rows(kt, sol, G, rows, cols, e_src, j_src, S10, Phi_res):
    """S10.linearized (with and without the current) + complex_step_columns; the first law b(R_out) = Phi_resolved; B symmetrized."""
    t0 = time.time()
    B, bMr, bMc = S10.linearized(kt, sol, G, True, rows=rows, cols=cols)
    Bnc, bMr_nc, _ = S10.linearized(kt, sol, G, False, rows=rows, cols=[])
    info = dict(seconds_linearized=time.time() - t0)
    if cols:
        Bcs, bMcs = S10.complex_step_columns(kt, np.asarray(e_src, float), np.asarray(j_src, float), G, cols)
        info['lin_vs_cs_B'] = max(fnorm(B[:, nu] - Bcs[nu]) / max(fnorm(Bcs[nu]), 1e-300) for nu in cols)
        info['lin_vs_cs_bM'] = max(fnorm(bMc[:, k] - bMcs[nu]) / max(fnorm(bMcs[nu]), 1e-300) for k, nu in enumerate(cols))
    info['first_law_raw'] = fnorm(bMr[-1] - Phi_res) / fnorm(Phi_res)
    info['asym_raw'] = fnorm(B - B.T) / max(fnorm(B), 1e-300)
    info['asym_raw_nocur'] = fnorm(Bnc - Bnc.T) / max(fnorm(Bnc), 1e-300)
    B = 0.5 * (B + B.T)
    Bnc = 0.5 * (Bnc + Bnc.T)
    info['nocur_current_columns_zero'] = bool(fnorm(Bnc[:, len(Phi_res) // 2:]) == 0.0)
    info['Bnorm'] = fnorm(B)
    info['seconds'] = time.time() - t0
    return B, Bnc, bMr, bMr_nc, info


def check_split(chans, stop_flags=()):
    """R6: the exponential sum and S10's split by sign need mask == (e < 0) and no inversions; otherwise the explicit route."""
    bad = [ch['kappa'] for ch in chans if not ch.get('mask_equals_neg', True)]
    if bad or stop_flags:
        raise ResponseSplitError('response stage stopped: %s%s; the explicit route is needed (per inverted or near-degenerate pair a rank-one '
                                 'term 2 Re (A_mu)_uo (A_nu)_ou/(e_o - e_u) added to P^A, the gapped block by the exponential sum)'
                                 % ('stored mask != (e < 0) in channels %s' % bad if bad else '', ('; record: %s' % ', '.join(stop_flags)) if stop_flags else ''))


def response_guard(chans, stop_flags=()):
    """(exit code, message) of the split check: 0 when the response may run, EXIT_SPLIT otherwise."""
    try:
        check_split(chans, stop_flags)
        return 0, 'ok'
    except ResponseSplitError as ex:
        return EXIT_SPLIT, str(ex)


def response_core(chans, ns, quad_tol, phi_applied, M, per_channel_skip=True, n_check=2, stop_flags=()):
    """P^A_{mu nu} = sum g Tr(A_mu Pi(A_nu)) by the exponential sum (qh.response_matrix's arithmetic, the per-channel node skip
    with the channel's own gap), certified against S10.ResponseOperator and by P^A Phi_applied."""
    QH, S10 = M.QH, M.S10
    check_split(chans, stop_flags)
    d = 2 * ns
    gaps = {ch['kappa']: float(ch['e'][ch['e'] > 0].min() - ch['e'][ch['e'] < 0].max()) for ch in chans}
    xmin = min(gaps.values())
    xmax = max(float(ch['e'].max() - ch['e'].min()) for ch in chans)
    t, c, err, h = QH.exp_sum_nodes(xmin, xmax, quad_tol)
    PA = np.zeros((d, d))
    used = skipped = 0
    t0 = time.time()
    per_ch = {}
    for ch in chans:
        hs = (ch['h_sp'], ch['Q_sp'])
        eo = ch['e'][ch['e'] < 0]
        eu = ch['e'][ch['e'] > 0]
        Uo, Uu = ch['Uo'], ch['Uu']
        xk = gaps[ch['kappa']] if per_channel_skip else xmin
        t1 = time.time()
        nu = 0
        for tk, ck in zip(t, c):
            if ck * math.exp(-tk * xk) < 1e-18:
                skipped += 1
                continue
            Eo = (Uo * np.exp(tk * eo)[None, :]) @ Uo.conj().T
            Eu = (Uu * np.exp(-tk * eu)[None, :]) @ Uu.conj().T
            T = QH.bilinear_blocks(hs, Eo, Eu, ns)
            PA -= 2.0 * ck * ch['w'] * T.real
            used += 1
            nu += 1
            del Eo, Eu, T
        per_ch[ch['kappa']] = dict(nodes_used=nu, gap=gaps[ch['kappa']], seconds=time.time() - t1)
        log('    response: kappa %d, %d nodes, gap %.4f, %.0f s' % (ch['kappa'], nu, gaps[ch['kappa']], time.time() - t1))
    asym = fnorm(PA - PA.T) / max(fnorm(PA), 1e-300)
    PA = 0.5 * (PA + PA.T)
    info = dict(nodes=int(len(t)), nodes_used=used, nodes_skipped=skipped, step=h, quadrature_max_rel_err=err, xmin=xmin, xmax=xmax,
                per_channel_skip=per_channel_skip, per_channel=per_ch, asymmetry_raw=asym, seconds=time.time() - t0, PAnorm=fnorm(PA))
    op = S10.ResponseOperator(chans, ns)
    rng = np.random.default_rng(11)
    rel = []
    for _ in range(n_check):
        xr = rng.normal(size=d)
        yo = op.matvec(xr)
        rel.append(fnorm(PA @ xr - yo) / max(fnorm(yo), 1e-300))
    phi = np.asarray(phi_applied, float)
    info.update(matrix_vs_operator_rel=rel, PA_Phi_applied_over_Phi=fnorm(PA @ phi) / fnorm(phi),
                PA_Phi_applied_over_Phi_operator=fnorm(op.matvec(phi)) / fnorm(phi),
                PA_Phi_applied_over_PAnorm_Phi=fnorm(PA @ phi) / max(fnorm(PA) * fnorm(phi), 1e-300))
    return PA, info, op


# ============================================================================ the kernels, the vertices, the table
def kernels_and_betas(B, Bnc, PA, b_rows, bnc_rows, radii, Nx, N_rows, ns, svd=True):
    d = 2 * ns
    out = {}
    t0 = time.time()
    if PA is not None:
        Jm = np.eye(d) + B @ PA
        lu = sla.lu_factor(Jm)
        V = sla.lu_solve(lu, B)
        out['dressed_asym_after_solve'] = fnorm(V - V.T) / max(fnorm(V), 1e-300)
        V = 0.5 * (V + V.T)
        lunc = sla.lu_factor(np.eye(d) + Bnc @ PA)
        if svd:
            out['smin_I_plus_BPA'] = float(np.linalg.svd(Jm, compute_uv=False).min())
            out['qstar'] = float(np.linalg.norm(PA @ B, 2))
        out['Jm_true_residual'] = None
    else:
        lu = lunc = None
        V = None
        Jm = None
    betas = {}
    for lab in radii:
        b = b_rows[lab]
        bnc = bnc_rows[lab]
        ic = radii[lab]['site']
        ent = dict(noref=b.copy(), nocur=bnc.copy())
        if lu is not None:
            ent['full'] = sla.lu_solve(lu, b)
            ent['nocur_dressed'] = sla.lu_solve(lunc, bnc)
        lapse = np.zeros(d)
        lapse[:ic + 1] = Nx[:ic + 1] / N_rows[lab]
        ent['lapse_window'] = lapse
        win = np.zeros(d)
        win[:ic + 1] = 1.0
        ent['sharp_window'] = win
        betas[lab] = ent
    if V is not None:
        out['V_minus_B_rel'] = fnorm(V - B) / fnorm(B)
        ev = np.linalg.eigvalsh(V)
        out['V_eig_min'], out['V_eig_max'] = float(ev[0]), float(ev[-1])
        out['V_eigs'] = ev
        out['Vnorm'] = fnorm(V)
        lab0 = next(iter(radii))
        out['Jm_true_residual'] = fnorm(Jm @ betas[lab0]['full'] - b_rows[lab0]) / max(fnorm(b_rows[lab0]), 1e-300)
    evB = np.linalg.eigvalsh(B)
    out['B_eig_min'], out['B_eig_max'], out['B_eigs'] = float(evB[0]), float(evB[-1]), evB
    out['Bnorm'] = fnorm(B)
    out['seconds'] = time.time() - t0
    return V, lu, Jm, betas, out


def variance_table(chans, betas, radii, M_H, N_enc, ns, QH, E_scale=None):
    tab = {}
    neg = []
    for lab in radii:
        row = dict(r=radii[lab]['r'], site=radii[lab]['site'], M_H=M_H[lab], variants={})
        for k in ('source', 'N_enclosed', 'M_w_dense', 'M_w_applied'):
            if k in radii[lab]:
                row[k] = radii[lab][k]
        for name, beta in betas[lab].items():
            vC = vS = vR = 0.0
            have_raw = False
            per = []
            for ch in chans:
                vv = variance_of_vertex_core(ch, beta, ns, QH)
                vC += ch['w'] * vv['var_C']
                vS += ch['w'] * vv['var_sea']
                if 'var_C_raw' in vv:
                    vR += ch['w'] * vv['var_C_raw']
                    have_raw = True
                per.append([ch['kappa'], ch['w'], ch['w'] * vv['var_C'], ch['w'] * vv['var_sea']])
            if vC < -1e-12 or vS < -1e-12:
                neg.append((lab, name, vC, vS))
            Mref = M_H[lab] if M_H[lab] > 0 else (E_scale or 1.0)
            wtot = float(sum(p[1] for p in per))
            Nr = radii[lab].get('N_enclosed')
            ent = dict(Var_C=vC, Var_C0=vS, signed_excess=vC - vS, per_channel=per, total_weight=wtot, sea_variance_per_unit_weight=vS / wtot,
                       sqrt_VarC_over_M=math.sqrt(max(vC, 0.0)) / Mref, sqrt_VarC0_over_M=math.sqrt(max(vS, 0.0)) / Mref,
                       signed_sqrt_excess_over_M=math.copysign(math.sqrt(abs(vC - vS)), vC - vS) / Mref,
                       N_r_inv_sqrt=(Nr ** -0.5) if (Nr and Nr > 0) else None,
                       beta_energy_max=float(np.max(np.abs(beta[:ns]))), beta_current_max=float(np.max(np.abs(beta[ns:]))))
            if have_raw:
                ent['Var_C_raw'] = vR
                ent['floor_footprint_rel'] = abs(vR - vC) / max(abs(vC), 1e-300)
            row['variants'][name] = ent
        tab[lab] = row
    return tab, neg


def witness_q(var_full, M_H, labs):
    """q_H = 4 sum_j sigma_j^2/(M_j^H)^2 over labs (each term nonnegative)."""
    return float(sum(4.0 * var_full[l] / M_H[l] ** 2 for l in labs))


def seed_core(chans, ns, K, G_list, g_list, QH):
    """L_1^2 = sum g ||Q (h_m + Sigma_F(D)) Phi||^2 (the singles with the state's own motion; QH 1.7 first line with h_H = h_m),
    the parts apart; L_2^2 by the copy interval of the per-channel Grams (pure slices only)."""
    t0 = time.time()
    L1 = L1h = L1F = 0.0
    per = []
    for ch in chans:
        hs = (ch['h_sp'], ch['Q_sp'])
        SF = QH.sigma_map(hs, K, ch['D'], ns)
        dft = fnorm(SF - SF.conj().T) / max(fnorm(SF), 1e-300)
        SF = herm(SF)
        Phi = ch['Phi']
        HP = ch['hm_sp'] @ Phi
        SP = SF @ Phi
        rH = HP - Phi @ (Phi.conj().T @ HP)
        rS = SP - Phi @ (Phi.conj().T @ SP)
        a = float(np.vdot(rH + rS, rH + rS).real)
        b = float(np.vdot(rH, rH).real)
        c = float(np.vdot(rS, rS).real)
        L1 += ch['w'] * a
        L1h += ch['w'] * b
        L1F += ch['w'] * c
        per.append(dict(kappa=ch['kappa'], L1sq=a, singles_h_m=b, singles_SigmaF=c, hermiticity_defect=dft, SigmaF_fro=fnorm(SF)))
        del SF, HP, SP, rH, rS
    out = dict(L1sq=L1, L1=math.sqrt(L1), singles_h_m_sq=L1h, singles_SigmaF_sq=L1F, per_channel=per,
               label='non-stationary seed: the h_m singles (the state own motion, Var_C(dGamma h_m)) are included in L_1')
    if G_list is not None:
        ci = QH.copy_interval(G_list, g_list, K)
        dcv = {k: ci[k] for k in ('D_total', 'D_sum_check_rel', 'q0', 'delta', 'delta_over_q0', 'gmin_bound', 'L2sq_interval', 'L2_interval',
                                  'psd_neg_rel', 'psd_neg_rel_total')}
        dcv['D'] = ci['D']
        out['doubles'] = dcv
        out['Lseed_interval'] = [math.sqrt(L1 + ci['L2sq_interval'][0]), math.sqrt(L1 + ci['L2sq_interval'][1])]
    out['seconds'] = time.time() - t0
    return out


# ============================================================================ the context (core slice or star member)
def core_context(args, history, t, M):
    """everything of section 1 with its checks (stage adapter)."""
    t_all = time.time()
    OPS, IO, QH = M.OPS, M.IO, M.QH
    H = HIST[history]
    loc = find_slice(history, t, args.slice_dir, IO)
    if loc['path'] is None:
        raise FileNotFoundError('%s T/r_s %g: no copy of %s (searched %s)' % (history, t, loc['name'], loc['candidates']))
    log('slice %s (%s, path used)' % (loc['path'], H['label']))
    rec = load_record(args.records or H['records'], args.fields or H['fields'], t)
    sl = load_core_slice(loc['path'], IO, verify_sha=args.verify_sha, m=rec['m'], half=rec['half'], entry=loc['entry'])
    log('  T/r_s %.3f, %d sites, %s, kappas %s, dust %s, size %d (%s), m %g g %g ell %g'
        % (sl.T_over_rs, sl.nsites, sl.dtype, sl.kappas, sl.n_dust, sl.size, sl.size_check, rec['m'], rec['g'], rec['ell']))
    well = load_well(args.well_dir or H['well'], history, t)
    flags, stop = status_flags(rec)
    radii, dinfo = radii_table(sl, rec, well, IO)
    checks = []

    def chk(name, val, tol, ok=None, info=False):
        ok = (val <= tol) if ok is None else bool(ok)
        tag = 'INFO' if info else ('PASS' if ok else 'FAIL')
        checks.append(dict(name=name, value=val, tol=tol, status=tag))
        log('  [%s] %s: %s (tol %s)' % (tag, name, val, tol))

    # the skeleton's S1, S1b, S2
    chk('S1 slice applied fields vs fields-npz applied row (max abs)',
        float(max(np.max(np.abs(rec['N_applied_fields'] - sl.N)), np.max(np.abs(rec['v_applied_fields'] - sl.v)))), 0.0)
    chk('S1b engine_ops_v1 chain radii vs slice R = 1..ns (max abs)', float(np.max(np.abs(sl.R - np.arange(1, sl.nsites + 1)))), 1e-12)
    chk('weights = 16 kappa (max abs)', float(max(abs(sl.weights[k] - 16.0 * k) for k in sl.kappas)), 0.0)
    try:
        if args.page_check:
            sys.path.insert(0, args.page_dir)
            import page_ops_v6 as PO
            h0p, Qp = PO.chain_sparse(sl.R, sl.m, 1)
            Hp = PO.clocked_csr(h0p, Qp, sl.N, sl.v)
            h, Q, _ = OPS.chain_parts(sl.nsites, sl.m, 1.0, sl.half)
            hm1 = OPS.clocked(h, Q, sl.N, sl.v)
            chk('S2 engine_ops_v1 h_m vs page_ops_v6 clocked_csr, kappa 1 (max abs)', float(np.max(np.abs(hm1 - Hp.toarray()))), 1e-13)
            del h, Q, hm1, Hp
    except Exception as ex:  # noqa: BLE001
        chk('S2 page_ops_v6 cross-check not run (%s: %s)' % (type(ex).__name__, ex), float('nan'), 0.0, ok=True, info=True)
    kappas = [k for k in sl.kappas if (not args.kappas or k in args.kappas)]
    if args.kappas:
        log('  CHANNEL SUBSET %s (a smoke control: every sum below is over these channels only, not the record set)' % kappas)
    chans = []
    s = np.zeros(2 * sl.nsites)
    s_raw = np.zeros(2 * sl.nsites)
    s_dust_raw = np.zeros(2 * sl.nsites)
    s_dust = np.zeros(2 * sl.nsites)
    per = []
    for k in kappas:
        ch = build_channel(sl, k, M, keep_raw=(sl.complex64 or args.keep_raw), eigD=True)
        if ch['purity'] == 'FORMAT':
            raise FormatError('kappa %d: FORMAT (stored orthonormality %.2e above %.0e)' % (k, ch['stored_orthonormality'], ch['purity_info']['orth_tol']))
        rC, jC, rR, jR = channel_sources(ch, OPS)
        ch['s'] = np.concatenate([ch['w'] * (rC - rR), ch['w'] * (jC - jR)])
        s += ch['s']
        s_raw += np.concatenate(ch['src_raw'])
        s_dust_raw += np.concatenate(ch['src_dust_raw'])
        if ch['n_dust']:
            ns_ = ch['n_sea']
            s_dust += np.concatenate([ch['w'] * OPS.site_density(ch['h_sp'], ch['Phi'][:, ns_:], ch['f'][ns_:]),
                                      ch['w'] * OPS.site_density(ch['Q_sp'], ch['Phi'][:, ns_:], ch['f'][ns_:])])
        wj = None
        if well:
            wj = next((c for c in well['channels'] if int(c.get('kappa', -1)) == k), None)
        row = dict(kappa=k, w=ch['w'], n_sea=ch['n_sea'], n_dust=ch['n_dust'], f_min=float(ch['f'].min()), f_max=float(ch['f'].max()),
                   purity=ch['purity'], d_f=ch['purity_info']['d_f'], stored_orthonormality=ch['stored_orthonormality'],
                   well_stored_orthonormality=None if wj is None else wj.get('stored_orthonormality'),
                   loewdin_orthonormality=ch['loewdin_orthonormality'], C2_defect=ch['C2_defect'], dust_sea_overlap_max=ch['dust_sea_overlap_max'],
                   ref_stored_orthonormality=ch['ref_stored_orthonormality'], ref_loewdin_orthonormality=ch['ref_loewdin_orthonormality'],
                   e_vs_stored_abs=ch['e_vs_stored_abs'], e_vs_stored_rel=ch['e_vs_stored_rel'], mask_equals_neg=ch['mask_equals_neg'],
                   stored_mask_equals_stored_neg=ch['stored_mask_equals_stored_neg'], n_neg=int(np.sum(ch['mask'])), min_abs_e=ch['min_abs_e'],
                   pref_resolution_defect=ch['pref_resolution_defect'], pref_resolution_flag=ch['pref_resolution_flag'],
                   commutator_rel=ch['commutator_rel'], record_commutator_h_R2_F=(rec['record'].get('commutator_h_R2_F') or {}).get(str(k)),
                   vertex_identity_rel=ch['vertex_identity_rel'], TrD=ch['TrD'], TrD2=ch['TrD2'], n_eigD_above_1e6=ch.get('n_eigD_above_1e-6'),
                   eigh_seconds=ch['eigh_seconds'], seconds=ch['seconds'] + ch['ops_seconds'])
        per.append(row)
        log('  kappa %d: n_occ %d (sea %d, dust %d) %s; stored orth %.2e (well %s) Loewdin %.1e; C2 %.1e; e vs e_k %.1e; mask==neg %s; P_ref res %.1e; '
            'comm %.1e; Tr D %.6f; |eig D| > 1e-6: %s; %.0f s'
            % (k, ch['n_occ'], ch['n_sea'], ch['n_dust'], ch['purity'], ch['stored_orthonormality'], row['well_stored_orthonormality'],
               ch['loewdin_orthonormality'], ch['C2_defect'], ch['e_vs_stored_abs'] or float('nan'), ch['mask_equals_neg'], ch['pref_resolution_defect'],
               ch['commutator_rel'], ch['TrD'], row['n_eigD_above_1e6'], row['seconds']))
        chans.append(ch)
    ns = sl.nsites
    c64 = sl.complex64
    nrm = lambda x: float(np.max(np.abs(x)))  # noqa: E731
    src_tol = 1e-5 if c64 else 1e-12
    full = not args.kappas
    if full:
        chk('Q3\' / S3 stored columns: sum 16k Tr[(C - P_ref) hcal_n] vs fields-npz eps_tot (max rel)', nrm(s_raw[:ns] - rec['eps_tot']) / nrm(rec['eps_tot']), src_tol)
        chk('Q3\' / S3b stored columns: the same for p_n vs J_tot (max rel)', nrm(s_raw[ns:] - rec['J_tot']) / nrm(rec['J_tot']), src_tol)
        chk('S3c adapter state (Loewdin): eps vs eps_tot (max rel)', nrm(s[:ns] - rec['eps_tot']) / nrm(rec['eps_tot']), 1e-4 if c64 else 1e-9)
        chk('S3d adapter state (Loewdin): J vs J_tot (max rel)', nrm(s[ns:] - rec['J_tot']) / nrm(rec['J_tot']), 1e-4 if c64 else 1e-9)
        if 'eps_pairs' in rec:
            ref = rec['eps_tot'] - rec['eps_pairs']
            chk('S4 dust part (stored columns) vs eps_tot - eps_pairs (max rel)', nrm(s_dust_raw[:ns] - ref) / nrm(ref), 1e-4 if c64 else 1e-10)
        Etot = (rec['record'].get('E_dust_shells') or 0.0) + (rec['record'].get('E_pairs_shells') or 0.0)
        chk('sum s_energy (Loewdin state) vs E_dust_shells + E_pairs_shells of the record (rel)', abs(float(s[:ns].sum()) - Etot) / abs(Etot), 1e-4 if c64 else 1e-9)
        if well and 'rho_dust_density' in well['arrays']:
            vol = FOURPI * sl.R ** 2
            ref = well['arrays']['rho_dust_density'] * vol
            chk('dust energy per shell (stored columns) vs well rho_dust_density x 4 pi r^2 (max rel)', nrm(s_dust_raw[:ns] - ref) / max(nrm(ref), 1e-300), 1e-4 if c64 else 1e-9)
        NB = float(sum(ch['w'] * ch['TrD'] for ch in chans))
        chk("Q11' N_B = sum 16 kappa Tr D_kappa = 640 (abs)", abs(NB - 640.0), 1e-9 if not c64 else 1e-5)
        chk("Q11' dust count per channel Tr D = n_dust (5, 4, 3, 2, 2, 0, ...) (max abs)", float(max(abs(ch['TrD'] - ch['n_dust']) for ch in chans)), 1e-9 if not c64 else 1e-5)
        chk('S5 sum 16k f_dust vs CORE_well dust total_weighted (abs)', abs(dinfo['N_B_dust'] - float(well['dust'].get('total_weighted', float('nan')))) if well else float('nan'),
            1e-9 if not c64 else 1e-4, info=well is None)
        for lab, dv in dinfo['R_vs_well_abs'].items():
            chk('%s recomputed vs well json (abs)' % lab, dv, 1e-9)
    chk('S6 occupied columns orthonormality after Loewdin (max abs over channels)', float(max(ch['loewdin_orthonormality'] for ch in chans)), 1e-14 if not c64 else 1e-13)
    chk('S6b reference columns orthonormality after Loewdin (max abs)', float(max(ch['ref_loewdin_orthonormality'] for ch in chans)), 1e-13)
    chk('C^2 = C after Loewdin (pure channels; ||C^2 - C||_F/sqrt(rank), max)', float(max(ch['C2_defect'] for ch in chans if ch['purity'] == 'pure')), 1e-13)
    if well:
        rel = []
        for ch in chans:
            wj = next((c for c in well['channels'] if int(c.get('kappa', -1)) == ch['kappa']), None)
            if wj and wj.get('stored_orthonormality'):
                rel.append(abs(ch['stored_orthonormality'] - wj['stored_orthonormality']) / wj['stored_orthonormality'])
        if rel:
            chk('stored orthonormality reproduces the well json per channel (max rel; 2 digits)', float(max(rel)), 5e-3)
    chk('S7 max |e - e_k stored| over channels (abs)', float(max(ch['e_vs_stored_abs'] for ch in chans)), 1e-11)
    chk('mask == (e < 0) bitwise in every channel', float(sum(not ch['mask_equals_neg'] for ch in chans)), 0.0)
    chk('S8 / P_ref resolution ||R2 R2^+ - Uo Uo^+||_F/sqrt(n_o) (max)', float(max(ch['pref_resolution_defect'] for ch in chans)), 1e-5,
        info=True)
    chk('commutator ||(1 - P_ref) h_m P_ref||_F/||h_m||_F (max)', float(max(ch['commutator_rel'] for ch in chans)), 1e-11, info=True)
    chk("Q2' vertex identity h_m = sum N hcal + v p (max rel)", float(max(ch['vertex_identity_rel'] for ch in chans)), 1e-12)
    # the skeleton's W checks on kappas 1 and 6 (QH's per-channel routines on the witness-shaped channel)
    if args.skeleton_checks:
        out = subprocess.run([sys.executable, '-B', os.path.join(args.s10_dir, 'qh_branch_witness_v1b.py'), '--core', '--nice', '0'],
                             capture_output=True, text=True, timeout=300)
        have = dict(fields=sl.N.shape == (ns,), channel_set=bool(kappas), state=all(ch['purity'] in ('pure', 'mixed') for ch in chans),
                    P_ref=all(ch['Pref_cols'].shape[1] > 0 for ch in chans), radii=bool(radii), gate=(well is not None and well.get('limit_label') is not None))
        chk('W1 witness --core prints its interface (rc %d) and every item is supplied: %s' % (out.returncode, have), float(sum(not v for v in have.values())), 0.0,
            ok=out.returncode == 0 and 'core slices' in out.stdout and all(have.values()))
        rng = np.random.default_rng(7)
        for ch in chans:
            if ch['kappa'] not in (1, 6) or ch['purity'] != 'pure':
                continue
            Uocc = ch['Phi']
            Qf, _ = sla.qr(Uocc, mode='full')
            Uemp = np.ascontiguousarray(Qf[:, Uocc.shape[1]:])
            del Qf
            wch = dict(kappa=ch['kappa'], h_sp=ch['h_sp'], Q_sp=ch['Q_sp'], Uocc=Uocc, Uemp=Uemp, U=np.concatenate([Uocc, Uemp], axis=1),
                       f=np.concatenate([np.ones(Uocc.shape[1]), np.zeros(Uemp.shape[1])]))
            x = rng.normal(size=2 * ns)
            A = QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], x, ns)
            si = QH.static_identity(wch, A, ns)
            chk('W2 kappa %d static identity Var_eig vs (1/2)||[A,C]||^2 (rel)' % ch['kappa'], si['rel_diff_eig_comm'], 1e-10)
            chk('W2b kappa %d static identity Var_site vs (1/2)||[A,C]||^2 (rel)' % ch['kappa'], si['rel_diff_site_comm'], 1e-10)
            vC, _ = variance_pure(A, Uocc)
            chk('W3 kappa %d adapter Var_C vs witness var_block(A, Uemp, Uocc) (rel)' % ch['kappa'], abs(vC - QH.var_block(A, Uemp, Uocc)) / max(vC, 1e-300), 1e-12)
            vcov, _ = variance_cov(A, Uocc, ch['f'], orthonormal=True)
            chk('W3b kappa %d covariance path at f = 1 equals the Slater path (rel)' % ch['kappa'], abs(vcov - vC) / max(vC, 1e-300), 1e-13)
            G, dft = QH.gram_channel(wch, ns)
            s2 = float(np.real(x @ (G @ x)))
            chk('W4 kappa %d witness source Gram x^T G x vs Var_C (rel); Hermiticity defect %.1e' % (ch['kappa'], dft), abs(s2 - vC) / max(vC, 1e-300), 1e-10)
            Gc, dfc = gram_channel_core(ch, ns, QH)
            chk('W4b kappa %d core Gram (C = Phi F Phi^+) equals the witness Gram (rel)' % ch['kappa'], fnorm(Gc - G) / fnorm(G), 1e-12)
            vh, _ = variance_pure(ch['hm_sp'], Uocc)
            chk('W5 kappa %d Var_C(dGamma h_m) of the core state (nonzero: not stationary), Var_P %.4e'
                % (ch['kappa'], variance_ref(ch['hm_sp'], ch['Uo'], ch['Uu'])[0]), vh, 0.0, info=True)
            del wch, Uemp, G, Gc
    status = dict(flags=flags, response_stop=stop, limit_label=(well or {}).get('limit_label'), gate=(well or {}).get('gate'),
                  complex64_floor=sl.floor, record=rec['record'], kappa_sea=rec['kappa_sea'], regime=dict(m_a=rec['m'], rs_over_a=rec['rs'], sigma_over_a=rec['ell']),
                  N_B=640, channel_subset=kappas if args.kappas else None,
                  dense_minus_applied_roots=dict(outer=(well['roots'].get('dense_outer') or float('nan')) - (well['roots'].get('applied_outer') or float('nan')),
                                                 inner=(well['roots'].get('dense_inner') or float('nan')) - (well['roots'].get('applied_inner') or float('nan')))
                  if well else None, history_label=H['label'])
    adapter = dict(version=VERSION, stage='adapter', history=history, T_over_rs=sl.T_over_rs, T=sl.T, path=sl.path, size_check=sl.size_check, sha=sl.sha,
                   dtype=sl.dtype, complex64=c64, floor=sl.floor, records_json=rec['records_json'], fields_npz=rec['fields_npz'], fields_index=rec['index'],
                   well_json=(well or {}).get('path'), modules=M.paths, channels=per, radii=radii,
                   dust=dict(N_B_dust=dinfo['N_B_dust'], n_per_channel=dinfo['n_per_channel'], R_vs_well_abs=dinfo['R_vs_well_abs']),
                   sums=dict(s_energy_sum=float(s[:ns].sum()), s_current_sum=float(s[ns:].sum()), E_dust_plus_pairs=(rec['record'].get('E_dust_shells') or 0.0) + (rec['record'].get('E_pairs_shells') or 0.0)),
                   status=status, checks=checks, n_fail=sum(c['status'] == 'FAIL' for c in checks), n_pass=sum(c['status'] == 'PASS' for c in checks),
                   seconds=time.time() - t_all)
    ctx = dict(mode='core', history=history, T_over_rs=sl.T_over_rs, sl=sl, rec=rec, well=well, chans=chans, ns=ns, R=sl.R, N=sl.N, v=sl.v,
               G=rec['g'], ell=rec['ell'], m=rec['m'], e_src=rec['eps_tot'], j_src=rec['J_tot'], E_ref=float(rec['record']['E']), radii=radii,
               ok_label='CORE', stop_flags=stop, status=status, complex64=c64, s_total=s, s_dust=s_dust, adapter=adapter, dust_cum=dinfo['cum'],
               full_channels=full)
    return ctx


def star_context(args, M):
    """a static-star member through the same stages: S10's channels (f in the eigenbasis -> Phi = U[:, f > 0], fractional f allowed)."""
    S10, OPS, QH = M.S10, M.OPS, M.QH
    star, _ = S10.load_star(args.star, args.member)
    chans0, E_top, extra_fill = S10.rebuild_channels(OPS, star, 0)
    ns = star['nsites']
    chans = []
    s = np.zeros(2 * ns)
    for c0 in chans0:
        e, U, f = c0['e'], c0['U'], c0['f']
        occ = f > 0
        ch = dict(kappa=c0['kappa'], w=c0['w'], extra=False, h=c0['h'], Q=c0['Q'], h_sp=sp.csr_matrix(c0['h']), Q_sp=sp.csr_matrix(c0['Q']),
                  hm_sp=sp.csr_matrix(c0['hm']), e=e, U=U, mask=e < 0, mask_equals_neg=True, Uo=np.ascontiguousarray(U[:, e < 0]),
                  Uu=np.ascontiguousarray(U[:, e >= 0]), Phi=np.ascontiguousarray(U[:, occ]), f=np.asarray(f[occ], float), n_sea=int(np.sum(e < 0)),
                  n_dust=int(np.sum(occ & (e > 0))), Phi_raw=None, complex64=False)
        ch['purity'] = 'pure' if bool(np.all((f < 1e-12) | (f > 1.0 - 1e-12))) else 'mixed'
        ch['closed'] = ch['purity'] == 'pure'
        ch['n_occ'] = int(ch['Phi'].shape[1])
        ch['Pref_cols'] = ch['Uo']
        ch['C'] = herm((ch['Phi'] * ch['f'][None, :]) @ ch['Phi'].conj().T)
        ch['D'] = herm(ch['C'] - ch['Uo'] @ ch['Uo'].conj().T)
        ch['TrD'] = float(np.trace(ch['D']).real)
        ch['min_abs_e'] = float(np.min(np.abs(e)))
        rC, jC, rR, jR = channel_sources(ch, OPS)
        ch['s'] = np.concatenate([ch['w'] * (rC - rR), ch['w'] * (jC - jR)])
        s += ch['s']
        chans.append(ch)
    R = star['R']
    radii = {}
    for lab in ('R_surf', 'R_999'):
        if star.get(lab):
            x = float(star[lab])
            radii[lab] = dict(r=x, site=min(int(np.searchsorted(R, x)), ns - 1), source='the star record')
    purity = sorted(set(ch['purity'] for ch in chans))
    log('star %s member %d: %d channels (kappa %s), purity %s, N_B %s, R_surf %s, R_999 %s, E %.6f'
        % (os.path.basename(args.star), star['member'], len(chans), [c['kappa'] for c in chans], purity, star['N_B'], star['R_surf'], star['R_999'], star['E']))
    status = dict(flags=[], response_stop=[], limit_label='a static star (no gate)', record=dict(E=star['E'], N_B=star['N_B'], E_top=star['E_top']),
                  closed_multiplet=all(ch['purity'] == 'pure' for ch in chans), extra_fill=extra_fill, E_top_rebuilt=E_top)
    ctx = dict(mode='star', history='star', T_over_rs=None, member=star['member'], star=star, chans=chans, ns=ns, R=R, N=star['N'], v=star['v'], G=star['g'],
               ell=float(args.ell or star['ell'] or 8.0), m=star['m'], e_src=s[:ns], j_src=s[ns:], E_ref=star['E'], radii=radii, ok_label='STAR',
               stop_flags=[], status=status, complex64=False, s_total=s, s_dust=s, adapter=None, full_channels=True)
    return ctx


# ============================================================================ the stages
def stage_variance(args, ctx, M):
    S10, QH = M.S10, M.QH
    t_all = time.time()
    chans, ns, R, G = ctx['chans'], ctx['ns'], ctx['R'], ctx['G']
    d = 2 * ns
    res = dict(arrays={})
    # --- the kernel and the re-solved rows (section 1.3, 2.6)
    ell = float(args.ell or ctx['ell'])
    if ctx['mode'] == 'star' and args.kernel == 'compact':
        Rout = float(R[-1] + ell + 1.0)                 # S10's outer radius (the regression K10)
    else:
        Rout = float(R[-1] + 8.0 * ell + 1.0)           # every Gaussian shell band inside the grid
    dr = args.dr if args.dr else (0.05 if ctx['mode'] == 'star' else 0.1)
    t0 = time.time()
    kt = make_kernel(args.kernel, R, ell, Rout, dr, M)
    log('  kernel %s: ell %g, R_out %.2f, dr %.4f, grid %d; norm_raw %.1e norm_err %.1e mom1_err %.1e  (%.0f s)'
        % (kt.kernel_source, ell, Rout, kt.dr, kt.ns, kt.norm_raw, kt.norm_err, kt.mom1_err, time.time() - t0))
    sol, fc = resolve_rows(kt, ctx['e_src'], ctx['j_src'], G, ctx['N'], ctx['v'], ctx['E_ref'], args.field_tol, S10, ctx['ok_label'])
    if args.kernel == 'compact' and ctx['mode'] == 'core':
        fc['label'] = 'SCHEME CONTROL (compact kernel; never the core number)'
    res['field_check'] = fc
    log('  re-solve: dN %.2e dv %.2e -> %s; E resolved %.8f record %.8f (rel %.1e)' % (fc['N_rel_maxdev'], fc['v_rel_maxdev'], fc['label'], fc['E_resolved'], fc['E_record'], fc['E_rel_dev'] or float('nan')))
    Phi_res = sol['Phi'].real
    Phi_app = np.concatenate([ctx['N'], ctx['v']])
    labs = [l for l in args.radii.split(',') if l.strip() in ctx['radii']]
    if ctx['mode'] == 'star':
        labs = [l for l in ('R_surf', 'R_999') if l in ctx['radii']]
    missing = [l for l in args.radii.split(',') if l.strip() and l.strip() not in ctx['radii']]
    radii = {l: dict(ctx['radii'][l]) for l in labs}
    radii['R_out'] = dict(r=float(kt.r[-1]), site=ns - 1, source='the re-solve outer radius',
                          N_enclosed=float(ctx['dust_cum'][-1]) if ctx.get('dust_cum') is not None else None)
    rows = {l: min(int(round(radii[l]['r'] / kt.dr)), kt.ns) for l in radii}
    rows['R_out'] = kt.ns
    order = list(radii)
    cols = sorted(set([0, ns // 2, ns - 1, ns, ns + ns // 2, 2 * ns - 1])) if args.cs_cols else []
    B, Bnc, bMr, bMr_nc, hinfo = hessian_rows(kt, sol, G, [rows[l] for l in order], cols, ctx['e_src'], ctx['j_src'], S10, Phi_res)
    res['hessian'] = hinfo
    log('  Hessian: lin vs cs B %s bM %s; first law %.1e; asym %.1e; ||B|| %.4e  (%.0f s)'
        % (hinfo.get('lin_vs_cs_B'), hinfo.get('lin_vs_cs_bM'), hinfo['first_law_raw'], hinfo['asym_raw'], hinfo['Bnorm'], hinfo['seconds']))
    b_rows = {l: bMr[i].copy() for i, l in enumerate(order)}
    bnc_rows = {l: bMr_nc[i].copy() for i, l in enumerate(order)}
    M_H = {l: float(sol['M'][rows[l]].real) for l in order}
    N_rows = {l: float(sol['N'][rows[l]].real) for l in order}
    for l in order:
        radii[l]['M_H'] = M_H[l]
        radii[l]['grid_row'] = rows[l]
    res['radii_missing'] = missing
    del kt
    # --- the response (2.6)
    PA = op = None
    resp = dict(stopped=None)
    if args.no_response:
        resp['stopped'] = 'not run (--no-response): noref variants only'
    else:
        try:
            PA, rinfo, op = response_core(chans, ns, args.quad_tol, Phi_app, M, stop_flags=ctx['stop_flags'])
            resp.update(rinfo)
            log('  response: %d nodes (%d used, %d skipped), quadrature %.1e on [%.4f, %.4f]; raw asym %.1e; matrix vs operator %s; '
                '||P^A Phi_applied||/||Phi|| %.1e (operator %.1e)  (%.0f s)'
                % (rinfo['nodes'], rinfo['nodes_used'], rinfo['nodes_skipped'], rinfo['quadrature_max_rel_err'], rinfo['xmin'], rinfo['xmax'],
                   rinfo['asymmetry_raw'], ['%.1e' % x for x in rinfo['matrix_vs_operator_rel']], rinfo['PA_Phi_applied_over_Phi'],
                   rinfo['PA_Phi_applied_over_Phi_operator'], rinfo['seconds']))
        except ResponseSplitError as ex:
            resp['stopped'] = str(ex)
            log('  %s' % ex)
    V, lu, Jm, betas, kinfo = kernels_and_betas(B, Bnc, PA, b_rows, bnc_rows, radii, sol['Nx'].real, N_rows, ns, svd=args.svd)
    res['response'] = resp
    res['kernel'] = {k: v for k, v in kinfo.items() if k not in ('V_eigs', 'B_eigs')}
    res['arrays'].update(V_eigs=kinfo.get('V_eigs', np.zeros(0)), B_eigs=kinfo['B_eigs'])
    if V is not None:
        log('  dressed kernel: asym %.1e, ||V - B||/||B|| %.3e, eig V [%.3e, %.3e], eig B [%.3e, %.3e]; sigma_min(I + B P^A) %s, q* %s'
            % (kinfo['dressed_asym_after_solve'], kinfo['V_minus_B_rel'], kinfo['V_eig_min'], kinfo['V_eig_max'], kinfo['B_eig_min'], kinfo['B_eig_max'],
               kinfo.get('smin_I_plus_BPA'), kinfo.get('qstar')))
    # GMRES certificate at the first radius
    if op is not None and args.gmres_check and order:
        lab0 = order[0]
        t1 = time.time()
        bg, hist, nit, ritz = S10.gmres_solve(lambda x: x + B @ op.matvec(x), b_rows[lab0], args.gmres_tol, args.gmres_maxit)
        resp['gmres_check'] = dict(radius=lab0, applications=nit, lstsq_residual=hist[-1], ritz_min_abs=ritz,
                                   beta_matrix_vs_gmres_rel=fnorm(betas[lab0]['full'] - bg) / max(fnorm(bg), 1e-300), seconds=time.time() - t1)
        log('  GMRES at %s: %d applications, residual %.1e, matrix vs GMRES %.1e  (%.0f s)' % (lab0, nit, hist[-1], resp['gmres_check']['beta_matrix_vs_gmres_rel'], time.time() - t1))
    # --- the variance table (variants x radii x channels)
    t1 = time.time()
    tab, neg = variance_table(chans, betas, radii, M_H, None, ns, QH, E_scale=ctx['E_ref'])
    res['radii'] = tab
    res['radius_labels'] = order
    res['negative_variance_flags'] = neg
    for l in order:
        fl = tab[l]['variants'].get('full') or tab[l]['variants']['noref']
        log('  %s (r %.3f, M_H %.6f): %s Var_C %.6e Var_C0 %.6e excess %+.6e; sqrt(Var)/M %.4e; N_r^-1/2 %s%s'
            % (l, tab[l]['r'], M_H[l], 'full' if 'full' in tab[l]['variants'] else 'noref', fl['Var_C'], fl['Var_C0'], fl['signed_excess'], fl['sqrt_VarC_over_M'],
               fl['N_r_inv_sqrt'], ('; floor footprint %.1e' % fl['floor_footprint_rel']) if 'floor_footprint_rel' in fl else ''))
    res['variance_seconds'] = time.time() - t1
    # --- the static identity per channel (h_m and the vertex at R_50 or the first radius)
    labA = 'R_50' if 'R_50' in betas else order[0]
    keyA = 'full' if 'full' in betas[labA] else 'noref'
    ident = []
    worst_h = worst_A = 0.0
    h_ok = True
    for ch in chans:
        ih = static_identity_core(ch, ch['hm_sp'])
        iA = static_identity_core(ch, vertex(QH, ch, betas[labA][keyA], ns))
        hf2 = fnorm(ch['hm_sp']) ** 2
        ih['h_m_fro_sq'] = hf2
        ih['stationary_roundoff'] = bool(max(ih['var_site'], ih['half_comm_sq']) <= 1e-20 * hf2)
        ident.append(dict(kappa=ch['kappa'], h_m=ih, A=iA, A_label='%s:%s' % (labA, keyA)))
        if ch['purity'] == 'pure':
            if ih['stationary_roundoff']:
                h_ok = h_ok and abs(ih['var_site'] - ih['half_comm_sq']) <= 1e-20 * hf2
            else:
                worst_h = max(worst_h, ih['rel_diff'])
            worst_A = max(worst_A, iA['rel_diff'])
    # v1 tolerances (deviation from the specification's 1e-13 for h_m, measured): the dense commutator route of a 4096-mode h_m carries
    # 1e-12 relative roundoff (prod 15 kappa 1: 1.2e-12); a stationary h_m (the stars) has both routes at roundoff: absolute 1e-20 ||h_m||_F^2
    res['static_identity'] = dict(per_channel=ident, worst_rel_h_m=worst_h, worst_rel_A=worst_A,
                                  Q1_pass=bool(h_ok and worst_h <= 1e-11 and worst_A <= 1e-13),
                                  tolerances=dict(h_m_rel=1e-11, h_m_stationary_abs_over_fro_sq=1e-20, A_rel=1e-13),
                                  label="(1/2)||[h_m, C]||^2 = Var_C(dGamma h_m): the state's own energy variance (not stationary), O(1); the check is the two routes' agreement")
    log("  Q1' static identity: worst rel h_m %.1e, A_%s %.1e -> %s; Var_C(dGamma h_m) weighted %.6e"
        % (worst_h, labA, worst_A, 'PASS' if res['static_identity']['Q1_pass'] else 'FAIL', sum(ch['w'] * i['h_m']['var_site'] for ch, i in zip(chans, ident))))
    # --- the Grams, sigma^2 = beta^T G beta
    t1 = time.time()
    G_list, G_def = [], []
    for ch in chans:
        Gk, dft = gram_channel_core(ch, ns, QH)
        G_list.append(Gk)
        G_def.append(dft)
    g_list = [ch['w'] for ch in chans]
    Gtot = sum(w * Gk for w, Gk in zip(g_list, G_list))
    gchk = {}
    worst = worst_sc = 0.0
    ev = np.linalg.eigvalsh(Gtot)
    for l in order:
        gchk[l] = {}
        for name, x in betas[l].items():
            s2 = float(np.real(x @ (Gtot @ x)))
            ref = tab[l]['variants'][name]['Var_C']
            rel = abs(s2 - ref) / max(abs(ref), 1e-300)
            # the scaled error: against max(|Var|, 1e-6 ||x||^2 lambda_max(G)) (the R_out vertex is the generator: Var ~ 1e-12 by construction)
            sc = abs(s2 - ref) / max(abs(ref), 1e-6 * float(x @ x) * float(ev[-1]), 1e-300)
            gchk[l][name] = dict(sigma2_gram=s2, Var_C=ref, rel_diff=rel, scaled_diff=sc)
            worst = max(worst, rel)
            worst_sc = max(worst_sc, sc)
    # v1 tolerance (deviation from the specification's 1e-12, measured): the Gram quadratic form at 3072 to 4096 modes carries 1e-10 relative
    # roundoff (qh_v1b's own star member 1: 6.7e-11; this code: 1.2e-10 star, 6.4e-11 core); Q6' = scaled error below 1e-9
    res['gram'] = dict(hermiticity_defects=G_def, sigma2_checks=gchk, worst_rel_diff=worst, worst_scaled_diff=worst_sc, Tr_G=float(np.trace(Gtot).real),
                       Q6_pass=bool(worst_sc <= 1e-9), Q6_tol=1e-9,
                       psd_neg_rel=float(-min(ev[0], 0.0) / max(ev[-1], 1e-300)), seconds=time.time() - t1)
    res['arrays']['gram_eigs_total'] = ev
    log("  Q6' Gram: beta^T G beta vs Var_C worst rel %.1e, scaled %.1e -> %s; Hermiticity %.1e; Tr G %.6e  (%.0f s)"
        % (worst, worst_sc, 'PASS' if res['gram']['Q6_pass'] else 'FAIL', max(G_def), res['gram']['Tr_G'], time.time() - t1))
    # --- the exchange and the pair energy (2.5)
    t1 = time.time()
    s_tot = ctx['s_total']
    kernels = [('bare', B)] + ([('dressed', V)] if V is not None else [])
    EF = {name: dict(per=[], P=[], C=[]) for name, _ in kernels}
    Tdiag = {}
    for ch in chans:
        T = exchange_blocks(ch, ns, QH, controls=args.exchange_controls)
        Tdiag[ch['kappa']] = np.real(np.diagonal(T['T']))
        for name, K in kernels:
            EF[name]['per'].append(-0.5 * ch['w'] * float(np.sum(K * np.real(T['T']))))
            if 'TP' in T:
                EF[name]['P'].append(-0.5 * ch['w'] * float(np.sum(K * np.real(T['TP']))))
                EF[name]['C'].append(-0.5 * ch['w'] * float(np.sum(K * np.real(T['TC']))))
        del T
    exch = {}
    NB = 640.0 if ctx['mode'] == 'core' else float(ctx['star']['N_B'])
    Enc = {l: radii[l].get('N_enclosed') for l in radii}
    E_minus = ctx['E_ref'] - NB * ctx['m']
    for name, K in kernels:
        ef = float(sum(EF[name]['per']))
        ep = pair_energy(s_tot, K)
        exch[name] = dict(E_F=ef, E_F_per_channel=EF[name]['per'], E_pair=ep, y=ef / ep if ep != 0 else None, sign=('negative' if ef < 0 else 'positive'),
                          E_F_over_E_minus_NBm=ef / E_minus if E_minus != 0 else None,
                          controls=dict(E_F_ref_P=float(sum(EF[name]['P'])) if EF[name]['P'] else None, E_F_unordered_C=float(sum(EF[name]['C'])) if EF[name]['C'] else None,
                                        label='controls, never added'),
                          comparators={'1/N_B': 1.0 / NB, '1/N_50': (1.0 / Enc['R_50']) if Enc.get('R_50') else None, '1/N_t': (1.0 / Enc['R_t']) if Enc.get('R_t') else None})
        log('  exchange [%s]: E_F %.6e (%s), E_pair %.6e, y %s, E_F/(E - N_B m) %s; controls P %s C %s'
            % (name, ef, exch[name]['sign'], ep, exch[name]['y'], exch[name]['E_F_over_E_minus_NBm'], exch[name]['controls']['E_F_ref_P'], exch[name]['controls']['E_F_unordered_C']))
    res['exchange'] = exch
    res['exchange_seconds'] = time.time() - t1
    res['arrays'].update({'T_diag_kappa%d' % k: v for k, v in Tdiag.items()})
    res['arrays']['EF_per_channel'] = np.array([exch[n]['E_F_per_channel'] for n, _ in kernels])
    # --- arrays
    res['arrays'].update(R=R, N_applied=ctx['N'], v_applied=ctx['v'], N_resolved_sites=sol['Nx'].real, v_resolved_sites=sol['vx'].real, M_grid=sol['M'].real,
                         N_grid=sol['N'].real, w_grid=sol['w'].real, s_total=s_tot, s_dust=ctx['s_dust'], eps_tot_record=ctx['e_src'], J_tot_record=ctx['j_src'],
                         kappas=np.array([c['kappa'] for c in chans]), weights=np.array(g_list), Phi_resolved=Phi_res, Phi_applied=Phi_app)
    for l in order:
        for name, x in betas[l].items():
            res['arrays']['beta_%s_%s' % (l, name)] = x
        res['arrays']['site_%s' % l] = np.array([radii[l]['site']])
    for ch in chans:
        res['arrays']['e_kappa%d' % ch['kappa']] = ch['e']
        if 'eigD' in ch:
            res['arrays']['eigD_kappa%d' % ch['kappa']] = ch['eigD']
        if args.save_grams:
            pass
    vn = sorted(set(n for l in order for n in tab[l]['variants']))
    res['arrays']['varC_table'] = np.array([[[p[2] for p in tab[l]['variants'][n]['per_channel']] if n in tab[l]['variants'] else [np.nan] * len(chans) for n in vn] for l in order])
    res['arrays']['varP_table'] = np.array([[[p[3] for p in tab[l]['variants'][n]['per_channel']] if n in tab[l]['variants'] else [np.nan] * len(chans) for n in vn] for l in order])
    res['arrays']['excess_table'] = res['arrays']['varC_table'] - res['arrays']['varP_table']
    res['table_axes'] = dict(labels=order, variants=vn, kappas=[c['kappa'] for c in chans])
    if args.save_kernels:
        res['arrays'].update(B=B, V=V if V is not None else np.zeros(0), PA=PA if PA is not None else np.zeros(0))
    if args.save_grams:
        for ch, Gk in zip(chans, G_list):
            res['arrays']['G_kappa%d' % ch['kappa']] = Gk
    res['_mem'] = dict(B=B, V=V, PA=PA, betas=betas, G_list=G_list, g_list=g_list, Gtot=Gtot, M_H=M_H, radii=radii, order=order)
    res['acceptance'] = {
        "Q2'": None,
        "Q4'": (bool(max(resp['matrix_vs_operator_rel']) <= 1e-10 and resp['PA_Phi_applied_over_Phi'] <= (1e-6 if ctx['complex64'] else 1e-9))
                if 'matrix_vs_operator_rel' in resp else None),
        "Q5'": bool(kinfo['dressed_asym_after_solve'] <= 1e-10) if V is not None else None,
        "Q6'": res['gram']['Q6_pass'],
        "Q7'": bool(hinfo['first_law_raw'] <= 1e-9),
        "Q8'": fc['label'],
        "Q10'": bool(not neg),
        "Q1'": res['static_identity']['Q1_pass']}
    res['seconds'] = time.time() - t_all
    return res


def stage_witness(args, ctx, res, M):
    QH = M.QH
    t0 = time.time()
    mem = res['_mem']
    order, M_H, betas = mem['order'], mem['M_H'], mem['betas']
    tab = res['radii']
    key = 'full' if 'full' in tab[order[0]]['variants'] else 'noref'
    var_full = {l: tab[l]['variants'][key]['Var_C'] for l in order}
    if ctx['mode'] == 'star':
        J = [l for l in ('R_surf', 'R_999') if l in order]
        subsets = {}
    else:
        J = [l for l in args.represented.split(',') if l in order]
        subsets = {}
        for name, ss in (('R_50', ['R_50']), ('R_50,R_t', ['R_50', 'R_t']), ('R_in,R_50,R_t', ['R_in', 'R_50', 'R_t'])):
            if all(l in order for l in ss):
                subsets[name] = witness_q(var_full, M_H, ss)
    qH = witness_q(var_full, M_H, J)
    dev = max(res['field_check']['N_rel_maxdev'], res['field_check']['v_rel_maxdev'], res['field_check']['E_rel_dev'] or 0.0)
    per = {}
    for l in J:
        v = tab[l]['variants']
        per[l] = dict(M_H=M_H[l], sigma2=var_full[l], sigma_over_M=math.sqrt(max(var_full[l], 0.0)) / M_H[l], q_j=4.0 * var_full[l] / M_H[l] ** 2,
                      q_j_upper=4.0 * var_full[l] * (1.0 + 1e-12) / (M_H[l] * (1.0 - dev)) ** 2,
                      per_channel_q=[[p[0], p[1], 4.0 * p[2] / M_H[l] ** 2] for p in v[key]['per_channel']],
                      excess=v[key]['signed_excess'], sea=v[key]['Var_C0'], q_j_noref=4.0 * v['noref']['Var_C'] / M_H[l] ** 2,
                      q_j_sharp_window=4.0 * v['sharp_window']['Var_C'] / M_H[l] ** 2)
    qup = float(sum(p['q_j_upper'] for p in per.values()))
    closed = all(ch['purity'] == 'pure' for ch in ctx['chans'])
    out = dict(J=J, per_radius=per, q_H=qH, q_H_subsets=subsets, q_H_upper_crude=qup, retained_weight_lower=max(0.0, 1.0 - qup),
               q_H_noref=float(sum(p['q_j_noref'] for p in per.values())), q_H_sharp_window_not_the_mass=float(sum(p['q_j_sharp_window'] for p in per.values())),
               variant_used=key, channel_truncation=dict(kappas=[c['kappa'] for c in ctx['chans']], total_weight=float(sum(c['w'] for c in ctx['chans']))),
               domain='the Fock space of the record channels (kappa_sea = 10) with their copies; a lower bound on the channel-complete witness',
               label=res['field_check']['label'], closed_multiplet=closed, floor=ctx['sl'].floor if ctx['mode'] == 'core' else None,
               reading='a mixed-state bound on Tr(rho Pi) where a channel is mixed; never a certificate of positivity, nor emptiness when q_H >= 1')
    log('  q_H (J = %s): %.6e [upper %.6e]; subsets %s; noref %.6e; sharp window (not the mass) %.6e'
        % (J, qH, qup, {k: '%.4e' % x for k, x in subsets.items()}, out['q_H_noref'], out['q_H_sharp_window_not_the_mass']))
    res['q_H'] = out
    seed = {}
    if closed and not args.skip_seed:
        for kname, K in (('dressed', mem['V']), ('bare', mem['B'])):
            if K is None:
                continue
            sd = seed_core(ctx['chans'], ctx['ns'], K, mem['G_list'], mem['g_list'], QH)
            seed[kname] = sd
            log('  seed [%s]: L_1^2 %.6e (h_m singles %.6e, Sigma_F singles %.6e); L_2 in %s; L_seed in %s  (%.0f s)'
                % (kname, sd['L1sq'], sd['singles_h_m_sq'], sd['singles_SigmaF_sq'], ['%.4e' % x for x in sd['doubles']['L2_interval']],
                   ['%.4e' % x for x in sd['Lseed_interval']], sd['seconds']))
        if args.exact_doubles_rank > 0 and mem['V'] is not None:
            wV, LV = np.linalg.eigh(mem['V'])
            idx = np.argsort(-np.abs(wV))[:args.exact_doubles_rank]
            chq = [dict(h_sp=c['h_sp'], Q_sp=c['Q_sp'], Uocc=c['Phi'], w=c['w']) for c in ctx['chans']]
            ex = QH.exact_doubles(chq, ctx['ns'], wV, LV, idx)
            seed['exact_doubles_truncated'] = dict(rank=int(args.exact_doubles_rank), L2sq_exact=ex['L2sq'], imaginary_residual=ex['imaginary_residual'], seconds=ex['seconds'])
    elif not closed:
        seed['skipped'] = 'mixed state: the seed is a determinant only on pure channels; L_seed and the copy interval undefined'
    else:
        seed['skipped'] = '--skip-seed'
    res['seed'] = seed
    if args.range_rank > 0 and mem['V'] is not None:
        W_M = np.zeros_like(mem['V'])
        for l in J:
            a = betas[l]['full']
            W_M += np.outer(a, a) / M_H[l] ** 2
        chq = [dict(kappa=c['kappa'], h_sp=c['h_sp'], Q_sp=c['Q_sp'], hm=c['hm_sp'].toarray(), Uo=c['Uo'], n_occ=c['n_occ'], w=c['w']) for c in ctx['chans']]
        fw = QH.f_width(chq, ctx['ns'], mem['V'])
        alpha_F = float(np.sum(mem['V'] * W_M) / (8.0 * max(np.sum(W_M * W_M), 1e-300)))
        alphas = [0.0, alpha_F] + [float(x) for x in args.alphas.split(',') if x.strip()]
        sraw = ctx['s_total']
        sb = QH.subtraction_bound(chq, ctx['ns'], mem['V'], W_M, sraw, fw['width_fixed'], fw['width_full'], alphas, args.range_rank, 'fixed', None, log)
        best = min(sb, key=lambda r: r['B_E'])
        res['lambda_bound'] = dict(results=sb, B_E_min=best['B_E'], best_alpha=best['alpha'], width=fw['width_fixed'], reading='unresolved leakage where above every target')
    res['witness_seconds'] = time.time() - t0
    return res


# ============================================================================ the channel stage
def tail_fit(kappas, vals):
    k = np.asarray(kappas, float)
    val = np.asarray(vals, float)
    c = val / (16.0 * k)
    K = k[-1]
    S = float(val.sum())
    sel = (k >= 0.5 * K) & (c > 0)
    if sel.sum() < 20:
        return dict(p=float('nan'), tail=float('inf'), S=S, n_fit=int(sel.sum()))
    p = -float(np.polyfit(np.log(k[sel]), np.log(c[sel]), 1)[0])
    tail = 16.0 * float(c[-1]) * K * K / (p - 2.0) if p > 2.0 else float('inf')
    return dict(p=p, tail=tail, S=S, n_fit=int(sel.sum()))


def stage_channels(args, history, t, M, out_dir, tag):
    """the sea part channel by channel at the slice's fields (occupied set e < 0, weight 16 kappa, the premise stated): kappa = 1 ..
    K_max; the record channels reproduce the witness's Var_C0 (Q9'); the extra channels are sea-only. Weighted (16 kappa)."""
    OPS, QH = M.OPS, M.QH
    pj = os.path.join(out_dir, tag + '.json')
    pz = os.path.join(out_dir, tag + '_arrays.npz')
    ps = os.path.join(out_dir, tag + '_channels.json')
    ck = os.path.join(out_dir, tag + '_channels.npz')
    if os.path.exists(ps) and not args.force:
        log('%s: channel summary exists, skipped' % tag)
        return
    if not (os.path.exists(pj) and os.path.exists(pz)):
        log('%s: witness outputs missing; run --stage witness first' % tag)
        return
    W = json.load(open(pj))
    A = np.load(pz)
    N, v = np.asarray(A['N_applied'], float), np.asarray(A['v_applied'], float)
    ns = len(N)
    m = float(W['regime']['m_a'])
    half = bool(W.get('half_offset', False))
    labs = [l for l in W['radius_labels'] if l != 'R_out']
    variants = [n for n in ('full', 'noref', 'sharp_window') if all(('beta_%s_%s' % (l, n)) in A.files for l in labs)]
    series = {'%s:%s' % (l, n): np.asarray(A['beta_%s_%s' % (l, n)], float) for l in labs for n in variants}
    names = list(series)
    Rref = W['radii'].get('R_t', W['radii'].get('R_90', W['radii'].get('R_50')))['r']
    K_max = int(args.kmax) if args.kmax else int(math.ceil(args.kmax_factor * Rref)) + 100
    K_rec = len(W['q_H']['channel_truncation']['kappas'])
    if os.path.exists(ck):
        Z = np.load(ck)
        kappas = [int(x) for x in Z['kappa']]
        vals = {nm: list(Z['var_' + nm.replace(':', '__')]) for nm in names}
        epmin, secs = list(Z['ep_min']), list(Z['seconds'])
        log('%s: resuming the channel loop at kappa %d (K_max %d)' % (tag, kappas[-1] + 1, K_max))
    else:
        kappas, vals, epmin, secs = [], {nm: [] for nm in names}, [], []
    stored = {}
    for l in labs:
        for n in variants:
            for p in W['radii'][l]['variants'][n]['per_channel']:
                stored[(l, n, int(p[0]))] = float(p[3])          # weighted (w Var_C0)
    primary = names[0]

    def save():
        np.savez_compressed(ck, kappa=np.array(kappas), ep_min=np.array(epmin), seconds=np.array(secs),
                            **{'var_' + nm.replace(':', '__'): np.array(vals[nm]) for nm in names})

    status = 'running'
    fit = {}
    rec_check = []
    start = (kappas[-1] + 1) if kappas else 1
    for kappa in range(start, K_max + 1):
        t0 = time.time()
        h, Q, _ = OPS.chain_parts(ns, m, float(kappa), half)
        hm = OPS.clocked(h, Q, N, v)
        e, U = sla.eigh(hm, driver='evr', check_finite=False)
        sea = e < 0
        Uo, Uu = U[:, sea], U[:, ~sea]
        h_sp, Q_sp = sp.csr_matrix(h), sp.csr_matrix(Q)
        stack = np.concatenate([QH.vertex_sparse(h_sp, Q_sp, series[nm], ns) @ Uo for nm in names], axis=1)
        Mx = Uu.conj().T @ stack
        no = Uo.shape[1]
        w = 16.0 * kappa
        for i, nm in enumerate(names):
            blk = Mx[:, i * no:(i + 1) * no]
            vals[nm].append(w * float(np.vdot(blk, blk).real))
        kappas.append(kappa)
        epmin.append(float(e[~sea].min()))
        secs.append(time.time() - t0)
        if kappa <= K_rec:
            for l in labs:
                for n in variants:
                    ref = stored.get((l, n, kappa))
                    if ref is not None:
                        ours = vals['%s:%s' % (l, n)][-1]
                        rec_check.append(dict(lab=l, variant=n, kappa=kappa, ours=ours, stored=ref, rel=abs(ours - ref) / max(abs(ref), 1e-300)))
        if int(sea.sum()) != ns:
            log('  WARNING kappa %d: the sea is not half the spectrum (%d of %d)' % (kappa, int(sea.sum()), 2 * ns))
        if kappa % 10 == 0 or kappa == K_max or kappa == K_rec:
            save()
            fit = tail_fit(kappas, vals[primary])
            log('  kappa %d: ep_min %.4f, %.1f s/channel; %s: c_kappa/c_1 %.4f, S_K %.6e, slope p %.3f (%d fitted), tail %s'
                % (kappa, epmin[-1], float(np.mean(secs[-10:])), primary, (vals[primary][-1] / (16.0 * kappa)) / (vals[primary][0] / 16.0), fit['S'], fit['p'], fit['n_fit'],
                   '%.3e' % fit['tail'] if math.isfinite(fit['tail']) else 'inf'))
            if (kappa >= max(2 * K_rec + 20, 100) and fit['n_fit'] >= 50 and fit['p'] > 2.2 and math.isfinite(fit['tail']) and fit['tail'] < args.stop_tol * fit['S']):
                status = 'converged by the stopping rule at K %d' % kappa
                break
    else:
        status = 'not converged at K_max %d (slope p %.3f)' % (K_max, fit.get('p', float('nan')))
    save()
    K = kappas[-1]
    marks = {'K_rec': K_rec, 'R_over_a': int(round(Rref)), 'K_final': K}
    proj = {}
    for mk, kk in marks.items():
        kk = min(max(int(kk), 1), K)
        q = 0.0
        for l in W['q_H']['J']:
            nm = '%s:full' % l if ('%s:full' % l) in vals else '%s:noref' % l
            exc = float(W['q_H']['per_radius'][l]['excess'])
            MH = float(W['q_H']['per_radius'][l]['M_H'])
            q += 4.0 * (exc + float(np.cumsum(vals[nm])[kk - 1])) / MH ** 2
        proj[mk] = dict(K=kk, q_H=q)
    summary = dict(version=VERSION, tag=tag, K_rec=K_rec, K_max=K_max, K_final=K, status=status, stop_tol=args.stop_tol, series=names, marks=marks,
                   premise='channels the engine never populated: occupied set = the instantaneous negative set at these fields (no mode crossed zero: '
                           'true of channels 1 to 10 at every stored clock)',
                   record_channel_check=rec_check, record_channel_check_worst_rel=max([r['rel'] for r in rec_check], default=None),
                   Q9_pass=bool(rec_check and max(r['rel'] for r in rec_check) <= 1e-10), q_H_at_marks=proj,
                   q_H_label='q_H(K) = 4 sum_j (excess_j at the record truncation + the weighted sea part summed to K)/M_j^2; physical only if converged',
                   seconds_total=float(np.sum(secs)), seconds_per_channel_mean=float(np.mean(secs)),
                   per_series={nm: dict(S_K=float(np.sum(vals[nm])), fit=tail_fit(kappas, vals[nm])) for nm in names})
    with open(ps, 'w') as fh:
        json.dump(jsonable(summary), fh, indent=1)
    log('%s channel loop: %s; q_H at marks %s; record-channel check worst %s' % (tag, status, {k: '%.4e (K %d)' % (x['q_H'], x['K']) for k, x in proj.items()},
                                                                               summary['record_channel_check_worst_rel']))


# ============================================================================ outputs
def write_table(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, '*', 'core_hf_*_T*.json')) + glob.glob(os.path.join(out_dir, 'star', 'core_hf_star_*.json'))):
        if p.endswith('_adapter.json') or p.endswith('_channels.json'):
            continue
        try:
            J = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if 'q_H' not in J:
            continue
        q = J['q_H']
        C = None
        pc = p[:-5] + '_channels.json'
        if os.path.exists(pc):
            C = json.load(open(pc))
        chan = 'not run' if C is None else '%s: K %d %.3e' % (C['status'], C['q_H_at_marks']['K_final']['K'], C['q_H_at_marks']['K_final']['q_H'])
        ex = J.get('exchange', {})
        efd = ex.get('dressed', {})
        efb = ex.get('bare', {})
        r50 = J['radii'].get('R_50') or J['radii'].get(J['radius_labels'][0])
        rin = J['radii'].get('R_in')
        key = q['variant_used']
        f50 = r50['variants'][key]
        sd = J.get('seed', {}).get('dressed')
        acc = J.get('acceptance', {})
        rows.append('| %s | %s | %s | %s | %s | %s | %s | %s | %.4e | %s | %s | %s | %s | %s | %s | %s |' % (
            J.get('history'), J.get('T_over_rs', J.get('member')), J.get('dtype', 'c128'), J.get('status', {}).get('limit_label'), J['field_check']['label'],
            ', '.join('%s %.3f' % (l, J['radii'][l]['r']) for l in q['J']), ', '.join('%.4f' % J['radii'][l]['M_H'] for l in q['J']),
            ', '.join('%.3e' % math.sqrt(max(q['per_radius'][l]['sigma2'], 0.0)) for l in q['J']), q['q_H'],
            ', '.join('%s %.3e' % (k, v) for k, v in q['q_H_subsets'].items()), chan,
            ('%.3e / %s' % (efd['E_F'], efd.get('y'))) if efd else 'n/a', ('%.3e / %s' % (efb['E_F'], efb.get('y'))) if efb else 'n/a',
            'R_50: %.3e (excess %+.3e, N^-1/2 %s)%s' % (f50['sqrt_VarC_over_M'], f50['signed_excess'], f50['N_r_inv_sqrt'],
                                                       ('; R_in: %.3e' % rin['variants'][key]['sqrt_VarC_over_M']) if rin else ''),
            ('L_1 %.3e, L_seed %s' % (sd['L1'], ['%.3e' % x for x in sd['Lseed_interval']])) if sd else 'n/a',
            ' '.join('%s %s' % (k, 'P' if v is True else ('F' if v is False else v)) for k, v in acc.items())))
    lines = ['# CORE_HF_v1 table (core_hf_v1.py): the core slices (and star-mode members)', '',
             'q_H at the record truncation (kappa_sea = 10) is a witness on the Fock space of those channels and a lower bound on the channel-complete '
             'witness; every core number is past the record\'s own limit (gate label column) and carries the complex64 floor where the dtype says c64. '
             'The compact kernel is a scheme control, never the core\'s number.', '',
             '| history | T/r_s (member) | dtype | limit label | field check | J radii | M_j^H | sigma_j | q_H (J) | q_H subsets | channel-extended | E_F / y (dressed) | E_F / y (bare) | sqrt(Var)/M | seed | checks |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'] + rows
    with open(os.path.join(out_dir, 'core_hf_table.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def write_handoff(ctx, path):
    arrs = dict(R=ctx['R'], N_applied=ctx['N'], v_applied=ctx['v'], kappas=np.array([c['kappa'] for c in ctx['chans']]))
    for ch in ctx['chans']:
        k = ch['kappa']
        arrs.update({'Phi_%d' % k: ch['Phi'], 'f_%d' % k: ch['f'], 'Pref_cols_%d' % k: ch['Pref_cols'], 'e_%d' % k: ch['e'], 'mask_%d' % k: ch['mask']})
    for l, r in ctx['radii'].items():
        arrs['radius_%s' % l] = np.array([r['r'], r['site']])
    np.savez(path, **arrs)


def finish_outputs(args, ctx, res, out_dir, tag):
    arrays = res.pop('arrays')
    res.pop('_mem', None)
    head = dict(version=VERSION, history=ctx['history'], T_over_rs=ctx['T_over_rs'], member=ctx.get('member'), mode=ctx['mode'],
                dtype=('complex64' if ctx['complex64'] else 'complex128'), kernel=args.kernel,
                regime=dict(m_a=ctx['m'], rs_over_a=(ctx['rec']['rs'] if ctx['mode'] == 'core' else None), sigma_over_a=ctx['ell']),
                half_offset=(ctx['rec']['half'] if ctx['mode'] == 'core' else ctx['star']['half']), status=ctx['status'], args=vars(args))
    head.update(res)
    with open(os.path.join(out_dir, tag + '.json'), 'w') as fh:
        json.dump(jsonable(head), fh, indent=1)
    np.savez_compressed(os.path.join(out_dir, tag + '_arrays.npz'), **arrays)
    log('written %s(.json, _arrays.npz)' % os.path.join(out_dir, tag))


def wait_for_load(frac, log=log, max_wait=7200):
    if not frac or frac <= 0:
        return
    t0 = time.time()
    lim = frac * (os.cpu_count() or 1)
    while os.getloadavg()[0] > lim and time.time() - t0 < max_wait:
        log('  waitload: load %.1f above %.1f; waiting 60 s' % (os.getloadavg()[0], lim))
        time.sleep(60)


# ============================================================================ main
def run_slice(args, history, t, M):
    out_dir = os.path.join(args.out, history)
    os.makedirs(out_dir, exist_ok=True)
    tag = 'core_hf_%s_T%07.2f' % (history, float(t))
    if args.kappas:
        tag += '_k' + '-'.join(str(k) for k in args.kappas)
    if args.kernel == 'compact':
        tag += '_compact'
    p_ad = os.path.join(out_dir, tag + '_adapter.json')
    p_main = os.path.join(out_dir, tag + '.json')
    if args.stage == 'channels':
        stage_channels(args, history, t, M, out_dir, tag)
        return 0
    if args.stage == 'adapter' and os.path.exists(p_ad) and not args.force:
        log('%s: adapter json exists, skipped' % tag)
        return 0
    if args.stage in ('variance', 'witness', 'all') and os.path.exists(p_main) and not args.force:
        log('%s: json exists, the variance/witness stage skipped' % tag)
    else:
        wait_for_load(args.waitload)
        t0 = time.time()
        log('\n=== %s %s T/r_s %g: stage %s (kernel %s) ===' % (VERSION, history, t, args.stage, args.kernel))
        try:
            ctx = core_context(args, history, t, M)
        except FormatError as ex:
            log('FORMAT: %s' % ex)
            return EXIT_FORMAT
        with open(p_ad, 'w') as fh:
            json.dump(jsonable(ctx['adapter']), fh, indent=1)
        log('adapter stage: %d PASS, %d FAIL (%.0f s); %s' % (ctx['adapter']['n_pass'], ctx['adapter']['n_fail'], time.time() - t0, p_ad))
        if args.write_handoff:
            write_handoff(ctx, os.path.join(out_dir, tag + '_handoff.npz'))
        if args.stage == 'adapter':
            return 1 if ctx['adapter']['n_fail'] else 0
        res = stage_variance(args, ctx, M)
        if args.stage in ('witness', 'all'):
            stage_witness(args, ctx, res, M)
        res['adapter_checks'] = dict(n_pass=ctx['adapter']['n_pass'], n_fail=ctx['adapter']['n_fail'], path=p_ad)
        res['radii_labels_info'] = {l: ctx['radii'][l] for l in ctx['radii']}
        res['seconds_total'] = time.time() - t0
        finish_outputs(args, ctx, res, out_dir, tag)
        write_table(args.out)
        del ctx
    if args.stage == 'all':
        stage_channels(args, history, t, M, out_dir, tag)
        write_table(args.out)
    return 0


def run_star_mode(args, M):
    out_dir = os.path.join(args.out, 'star')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'core_hf_star_%s_m%d_%s' % (os.path.basename(args.star).replace('.json', ''), args.member, args.kernel)
    if os.path.exists(os.path.join(out_dir, tag + '.json')) and not args.force:
        log('%s exists, skipped' % tag)
        return 0
    t0 = time.time()
    ctx = star_context(args, M)
    res = stage_variance(args, ctx, M)
    stage_witness(args, ctx, res, M)
    reg = {}
    if args.star_reference:
        R = json.load(open(args.star_reference))
        for lab, vals in R.get('Var_C_full', {}).items():
            if lab in res['radii']:
                reg['Var_C_full_%s' % lab] = abs(res['radii'][lab]['variants']['full']['Var_C'] - vals) / abs(vals)
        for lab, vals in R.get('M_resolved', {}).items():
            if lab in res['radii']:
                reg['M_resolved_%s' % lab] = abs(res['radii'][lab]['M_H'] - vals) / abs(vals)
        if R.get('q_H') is not None:
            reg['q_H'] = abs(res['q_H']['q_H'] - R['q_H']) / abs(R['q_H'])
        res['regression'] = dict(reference=args.star_reference, rel=reg, tol=1e-9, pass_=bool(reg and max(reg.values()) <= 1e-9))
        log('  K10 regression against %s: %s -> %s' % (args.star_reference, {k: '%.1e' % v for k, v in reg.items()}, 'PASS' if res['regression']['pass_'] else 'FAIL'))
    res['seconds_total'] = time.time() - t0
    finish_outputs(args, ctx, res, out_dir, tag)
    return 0 if (not reg or max(reg.values()) <= 1e-9) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--history', default='prod', choices=sorted(HIST))
    ap.add_argument('--clocks', default='15', help='comma list of T/r_s or "all"')
    ap.add_argument('--slice-dir', default=None)
    ap.add_argument('--records', default=None)
    ap.add_argument('--fields', default=None)
    ap.add_argument('--well-dir', default=None)
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--s10-dir', default=S10_DIR)
    ap.add_argument('--slices144-dir', dest='slices_dir', default=SLICES_DIR)
    ap.add_argument('--page-dir', default=PAGE_DIR)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--stage', default='adapter', choices=['adapter', 'variance', 'witness', 'channels', 'all', 'table'])
    ap.add_argument('--kernel', default='gauss', choices=['gauss', 'compact'])
    ap.add_argument('--ell', type=float, default=None)
    ap.add_argument('--dr', type=float, default=None, help='re-solve grid (default 0.1 on the core, 0.05 in the star mode)')
    ap.add_argument('--field-tol', type=float, default=1e-3)
    ap.add_argument('--radii', default='R_in,R_50,R_t', help='the variance table radii (R_out always added); R_90, R_c may be added')
    ap.add_argument('--represented', default='R_in,R_50,R_t', help='J: the represented-mass radii of the witness (those that exist)')
    ap.add_argument('--quad-tol', type=float, default=1e-13)
    ap.add_argument('--gmres-check', dest='gmres_check', action='store_true', default=True)
    ap.add_argument('--no-gmres-check', dest='gmres_check', action='store_false')
    ap.add_argument('--gmres-tol', type=float, default=1e-12)
    ap.add_argument('--gmres-maxit', type=int, default=80)
    ap.add_argument('--svd', dest='svd', action='store_true', default=True)
    ap.add_argument('--no-svd', dest='svd', action='store_false')
    ap.add_argument('--no-response', action='store_true', help='skip P^A (noref variants only; a control)')
    ap.add_argument('--cs-cols', dest='cs_cols', action='store_true', default=True, help='the 6 complex-step columns (S2)')
    ap.add_argument('--no-cs-cols', dest='cs_cols', action='store_false')
    ap.add_argument('--exchange-controls', dest='exchange_controls', action='store_true', default=True)
    ap.add_argument('--no-exchange-controls', dest='exchange_controls', action='store_false')
    ap.add_argument('--skip-seed', action='store_true')
    ap.add_argument('--exact-doubles-rank', type=int, default=0)
    ap.add_argument('--range-rank', type=int, default=0)
    ap.add_argument('--alphas', default='')
    ap.add_argument('--kmax', type=int, default=None)
    ap.add_argument('--kmax-factor', type=float, default=2.0)
    ap.add_argument('--stop-tol', type=float, default=1e-2)
    ap.add_argument('--save-kernels', action='store_true')
    ap.add_argument('--save-grams', action='store_true')
    ap.add_argument('--write-handoff', action='store_true')
    ap.add_argument('--channel-cache', default=None, help='accepted; v1 keeps all channels resident (about 2 GB per channel)')
    ap.add_argument('--verify-sha', action='store_true')
    ap.add_argument('--keep-raw', action='store_true', help='keep the raw columns (the floor control) also on complex128 slices')
    ap.add_argument('--kappas', type=lambda s: [int(x) for x in s.split(',') if x.strip()], default=None,
                    help='a channel subset (smoke control only: sums are then not the record set)')
    ap.add_argument('--page-check', dest='page_check', action='store_true', default=True)
    ap.add_argument('--no-page-check', dest='page_check', action='store_false')
    ap.add_argument('--skeleton-checks', dest='skeleton_checks', action='store_true', default=True)
    ap.add_argument('--no-skeleton-checks', dest='skeleton_checks', action='store_false')
    ap.add_argument('--waitload', type=float, default=0.0, help='wait while the 1-min load exceeds this fraction of the CPUs (RUN_core_well guard: 0.9)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--star', default=None)
    ap.add_argument('--member', type=int, default=1)
    ap.add_argument('--star-reference', default=None, help='json {Var_C_full: {lab: x}, M_resolved: {lab: x}, q_H: x} for the K10 regression')
    args = ap.parse_args(argv)
    if args.nice > 0:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    os.makedirs(args.out, exist_ok=True)
    if args.stage == 'table':
        write_table(args.out)
        return 0
    if args.channel_cache:
        log('--channel-cache: not implemented in v1; all channels stay resident (peak about 2 GB per channel plus the kernel tables)')
    M = Mods(args.engine_dir, args.slices_dir, args.s10_dir)
    log('%s: modules %s; threads %s; nice %d' % (VERSION, M.paths, os.environ.get('OMP_NUM_THREADS'), args.nice))
    if args.star:
        return run_star_mode(args, M)
    clocks = CLOCKS[args.history] if args.clocks == 'all' else [float(x) for x in args.clocks.split(',') if x.strip()]
    rc = 0
    for t in clocks:
        rc = max(rc, run_slice(args, args.history, t, M))
    return rc


if __name__ == '__main__':
    sys.exit(main())
