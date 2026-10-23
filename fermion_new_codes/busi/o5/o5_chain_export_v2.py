#!/usr/bin/env python
"""o5_chain_export_v2.py -- O5 (GAP10_T7_final_v1.1 section 9): the chain side of the same-quantity witness exchange on the solve code's own
operator and HORIZON's own launch column (v1 reconstructed both from HORIZON v2's written definition).

What changes against o5_chain_export_v1.py (imported read-only for everything else: the record loader and the table roots, the detectors, the
local dispersion and its outgoing root, the shell rules, the Killing bins, the lattice-scale share, the exact integrated current, the Simpson
integral, the half-weight clock, the json/npz keys):
  * the chain operator is page_ops_v6.clocked_csr(page_ops_v6.chain_sparse(r, m, kappa), N, v) on the record's own grid r = 1 .. n a (the matrix
    HORIZON propagates on; page_ops_v6 and engine_single_object_B_v1 are imported from --solve_path, nothing copied or edited), made dense for
    the exact propagation (the dense eigendecomposition with its stated rounding bound, as v1); the fixed-cut current operator is
    page_ops_v6.number_current_csr(h, r, r_h) (v1's i[h, P_>] is formed beside it and the max entry difference printed);
  * the launch column is engine_single_object_B_v1.launch_columns' 'annulus middle' member at R_c (the roots argument (R_c, R_c) places it there;
    the depth is T7's, HORIZON's own depth rule on this record is printed beside): HORIZON's envelope exp(-(r - R_c)^2/(2 sigma^2)) (sigma the
    amplitude's standard deviation; the density's is sigma/sqrt 2; v1's sigma was the density's), the carrier exp(i k_0 r), the particle spinor
    of the local symbol by the engine's eigh, unit norm on the chain grid, no cut. When (|a k_0|, sigma) is not one of HORIZON's own pairs
    (2 pi/64 with 16, pi/4 with 8, pi/2 with 4) the engine module's KS tuple is set to the one requested pair for the call and restored after
    (the engine's file is untouched; printed and recorded as column.ks_override). The column's radial profile is declared in the json
    (column.profile) and re-evaluated on the chain grid against the engine's column at every run (the max difference printed; above 1e-12 the
    run stops): o5_lift_v2.py evaluates the declared profile on the cube's site radii, so the lifted column is this column;
  * --column_source lane rebuilds v1's column on the solve operator (the lane's Gaussian of density width sigma with the cut; --spinor rest for
    section 9's fallback shell);
  * no vacuum exterior is appended: chain_sparse takes the record's grid as it is; the unit-speed chain-end reflection clock (the front from
    R_c to r_n and back to the outermost readout radius, HORIZON's rule) is printed and every readout time at or after it is flagged in the
    json. HORIZON appends the exterior to 128 r_s for its 300 r_s propagation; this 10 to 16 r_s window on the record's 1024 sites does not reach
    the chain's end (the flag count is printed).
Outputs: <out>/o5_chain_Rc<Rc>_k<kappa><tag>.npz and .json with v1's keys plus solve_code, chain.operator, column.profile/source/ks_override,
readout.reflection and a profile per detector member. Resumable; no plots; CPU, 1 to 3 minutes per column at n 1024. Nothing here is a result
of the object: every readout is on 1 held record at 1 averaging width.
"""
import argparse
import importlib
import json
import math
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import o5_chain_export_v1 as V1  # noqa: E402  (read-only import of the lane's own v1: its readouts; its RadialChain is not used here)

SX, SY, SZ = V1.SX, V1.SY, V1.SZ
md5 = V1.md5
_json_default = V1._json_default
ROUNDING_TOL = 1e-12

SPINOR_CONVENTION = (V1.SPINOR_CONVENTION + " v2: the matrix is page_ops_v6.clocked_csr(page_ops_v6.chain_sparse(r, m, kappa), N, v) itself (flattened index "
                     "2 j + a, site j and component a, the lane's layout; on-site m sigma_z + (kappa/R) sigma_y, the sine bond -(i/2) sigma_x from j to j + 1, the "
                     "two-step current -(i/4) 1 from j to j + 2, h = (1/2){N, h_0} + (1/2){v, Q}), the same matrix as v1's reconstruction to rounding "
                     "(o5_operator_check_v1.py prints the difference on the record).")


# ------------------------------------------------------------------ the solve code, imported read-only
def load_solve_modules(path):
    """Import page_ops_v6 and engine_single_object_B_v1 from the directory `path` (read-only; nothing copied or edited). engine_single_object_B_v1
    imports page_history_v6b at module level and uses it only inside its propagation functions, none of which run here: when that module is not
    importable from `path`, an empty placeholder module is registered under its name (recorded) and the engine's import is retried."""
    path = os.path.realpath(os.path.expanduser(path))
    need = ('page_ops_v6.py', 'engine_single_object_B_v1.py')
    missing = [f for f in need if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f"--solve_path {path}: {missing} not found there (the solve code is imported read-only from that directory)")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        OPS = importlib.import_module('page_ops_v6')
    except ModuleNotFoundError as e:
        if e.name == 'torch':
            raise ImportError("page_ops_v6 imports torch at module level: install torch in this environment (the box's gap10 environment has it)") from e
        raise
    placeholder = False
    try:
        ESB = importlib.import_module('engine_single_object_B_v1')
    except ModuleNotFoundError as e:
        if e.name != 'page_history_v6b':
            raise
        stub = types.ModuleType('page_history_v6b')
        stub.__doc__ = ("placeholder registered by o5_chain_export_v2.load_solve_modules: page_history_v6b was not importable; engine_single_object_B_v1 "
                        "imports it at module level and uses it only in its propagation functions, none of which run in O5")
        sys.modules['page_history_v6b'] = stub
        placeholder = True
        ESB = importlib.import_module('engine_single_object_B_v1')
    for mod in (OPS, ESB):
        if os.path.realpath(os.path.dirname(os.path.abspath(mod.__file__))) != path:
            raise ImportError(f"{mod.__name__} resolved to {mod.__file__}, outside --solve_path {path} (another copy is earlier on sys.path): stop")
    info = dict(path=path, page_ops_v6=dict(file=OPS.__file__, md5=md5(OPS.__file__)),
                engine_single_object_B_v1=dict(file=ESB.__file__, md5=md5(ESB.__file__)),
                page_history_v6b_placeholder=placeholder, KS=[[float(k), float(s), str(nm)] for k, s, nm in ESB.KS])
    return OPS, ESB, info


# ------------------------------------------------------------------ the declared radial profiles
def profile_horizon(Rc, sigma, k0, Z):
    """HORIZON's launch column's radial profile: Z exp(-(r - R_c)^2/(2 sigma^2)) exp(i k_0 r), truncated to the chain grid, no cut."""
    return dict(form='gaussian', Rc=float(Rc), sigma=float(sigma), envelope_denominator=2.0, sigma_meaning="amplitude standard deviation (the density's is sigma/sqrt 2)",
                k0=float(k0), carrier_reference=0.0, Z=float(Z), cut=None, grid="the chain grid r = 1 .. n a (the envelope is truncated there, not cut)",
                source='engine_single_object_B_v1.launch_columns')


def profile_lane(Rc, sigma, k0, Z, cut):
    """v1's column and this lane's detectors: Z exp(-(r - R_c)^2/(4 sigma^2)) exp(i k_0 (r - R_c)) on |r - R_c| <= cut sigma (sigma the density's standard deviation)."""
    return dict(form='gaussian', Rc=float(Rc), sigma=float(sigma), envelope_denominator=4.0, sigma_meaning="density standard deviation (GAP10's packet_mode convention)",
                k0=float(k0), carrier_reference=float(Rc), Z=float(Z), cut=float(cut), grid="the chain grid r = 1 .. n a", source='o5_chain_export_v1.gaussian_column')


def radial_profile(r, prof, apply_cut=True):
    """f(r) = Z exp(-(r - R_c)^2/(den sigma^2)) exp(i k_0 (r - r_ref)) [1_{|r - R_c| <= cut sigma}] for a profile dict (form 'gaussian')."""
    if prof.get('form', 'gaussian') != 'gaussian':
        raise ValueError(f"unknown profile form {prof.get('form')}")
    r = np.asarray(r, dtype=float)
    Rc, sigma, den = float(prof['Rc']), float(prof['sigma']), float(prof['envelope_denominator'])
    k0, r_ref = float(prof['k0']), float(prof['carrier_reference'])
    Zp = prof['Z']
    Z = complex(Zp[0], Zp[1]) if isinstance(Zp, (list, tuple)) else float(Zp)
    d = r - Rc
    env = np.exp(-d * d / (den * sigma * sigma))
    cut = prof.get('cut')
    if apply_cut and cut is not None:
        env = env * (np.abs(d) <= float(cut) * sigma)
    return Z * env * np.exp(1.0j * k0 * (r - r_ref))


def density_std(prof):
    """The density's standard deviation of the profile's Gaussian: sigma sqrt(den)/2 (sigma/sqrt 2 for den 2, sigma for den 4)."""
    return float(prof['sigma']) * math.sqrt(float(prof['envelope_denominator'])) / 2.0


def envelope_weights(prof, r):
    """The continuous Gaussian density's weight outside the cut, below the grid (r < r_1 - 1/2) and beyond it (r > r_n + 1/2)."""
    s = density_std(prof)
    Rc = float(prof['Rc'])
    Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    cut = prof.get('cut')
    outside = 2.0 * (1.0 - Phi(float(cut) * float(prof['sigma']) / s)) if cut is not None else 0.0
    return dict(density_std=s, gaussian_weight_outside_cut=outside, gaussian_weight_below_grid=Phi((float(r[0]) - 0.5 - Rc) / s),
                gaussian_weight_beyond_grid=1.0 - Phi((float(r[-1]) + 0.5 - Rc) / s))


def reflection_clock(r_last, Rc, R_far):
    """HORIZON's rule: a unit-speed front from R_c reaches the chain's end r_last and returns to R_far at (r_last - R_c) + (r_last - R_far) after launch."""
    return (float(r_last) - float(Rc)) + (float(r_last) - float(R_far))


# ------------------------------------------------------------------ the chain operator (the solve code's)
class SolveChain(V1.RadialChain):
    """v1's interface (diagonalize, prop_error_bound, expectation, cut_operator, propagate, the 2 defects) on the solve code's own matrix:
    h = page_ops_v6.clocked_csr(page_ops_v6.chain_sparse(r, m, kappa), N, v), dense; the flattened index 2 j + a is the lane's layout."""

    def __init__(self, r, N, v, m, kappa, OPS):
        r = np.asarray(r, dtype=float)
        N = np.asarray(N, dtype=float)
        v = np.asarray(v, dtype=float)
        n = r.size
        self.r, self.N, self.v, self.m, self.kappa, self.n, self.dim = r, N, v, float(m), float(kappa), n, 2 * n
        self.OPS = OPS
        self.h0_csr, self.Q_csr = OPS.chain_sparse(r, self.m, self.kappa)
        self.h_csr = OPS.clocked_csr(self.h0_csr, self.Q_csr, N, v)
        self.H = np.ascontiguousarray(np.asarray(self.h_csr.toarray(), dtype=np.complex128))
        self.gershgorin = float(OPS.gershgorin(self.h_csr))
        self.w = self.V = None
        self.eps_res = self.eps_orth = self.wmax = None

    def number_current(self, R_c):
        """page_ops_v6.number_current_csr(h, r, R_c) = i[h, Pi_{r > R_c}] dense (outward positive), and the max entry difference to v1's i[h, P_>]."""
        Qs = np.asarray(self.OPS.number_current_csr(self.h_csr, self.r, float(R_c)).toarray(), dtype=np.complex128)
        Qv = self.cut_operator(self.r > float(R_c))
        return Qs, float(np.max(np.abs(Qs - Qv)))


# ------------------------------------------------------------------ the launch column (HORIZON's)
def horizon_column(ESB, r, m, kappa, Rc, ak0, sigma, N, v, tol=ROUNDING_TOL):
    """HORIZON's launch column at R_c: engine_single_object_B_v1.launch_columns(R, m, kappa, (R_c, R_c), N, v) builds its launch set with the
    'annulus middle' at R_c; the member with k_0 = ak0 (signed) and width sigma is taken. When (|ak0|, sigma) is not one of HORIZON's own (k_0, sigma)
    pairs, the module's KS tuple is set to that single pair for the call and restored (the engine's file is untouched; recorded as ks_override).
    The declared profile (profile_horizon) is evaluated on the chain grid and compared with the engine's column; a difference above tol stops."""
    r = np.asarray(r, dtype=float)
    k_abs = abs(float(ak0))
    sgn = 1 if float(ak0) >= 0.0 else -1
    if k_abs == 0.0:
        raise ValueError("HORIZON's launch set has no k_0 = 0 member: the fallback shell is the lane's column (--column_source lane --spinor rest --ak0 0)")
    ks_orig = ESB.KS
    match = [p for p in ks_orig if abs(float(p[0]) - k_abs) < 1e-9 and abs(float(p[1]) - float(sigma)) < 1e-9]
    override = len(match) == 0
    try:
        if override:
            ESB.KS = ((k_abs, float(sigma), f"{k_abs:.10g} (O5 exchange member, width {float(sigma):g} a; not one of HORIZON's own pairs)"),)
        X, info = ESB.launch_columns(r, float(m), int(kappa), (float(Rc), float(Rc)), np.asarray(N, dtype=float), np.asarray(v, dtype=float))
    finally:
        ESB.KS = ks_orig
    sel = [i for i, q in enumerate(info) if q['depth'] == 'annulus middle' and abs(float(q['k0']) - sgn * k_abs) < 1e-12 and abs(float(q['sigma']) - float(sigma)) < 1e-9]
    if len(sel) != 1:
        raise RuntimeError(f"HORIZON's launch set holds {len(sel)} 'annulus middle' members at k_0 {sgn * k_abs:+.6f}, sigma {float(sigma):g}: "
                           f"{[(q['depth'], round(q['k0'], 6), q['sigma']) for q in info]}")
    i = sel[0]
    x = np.asarray(X[:, i], dtype=np.complex128)
    psi0 = np.ascontiguousarray(x.reshape(r.size, 2))
    q = info[i]
    u = np.array([complex(q['spinor'][0], q['spinor'][1]), complex(q['spinor'][2], q['spinor'][3])], dtype=np.complex128)
    R_sp = max(float(Rc), 1.0)
    E_sym = math.sqrt(m * m + math.sin(sgn * k_abs) ** 2 + (kappa / R_sp) ** 2)
    D = m * SZ + math.sin(sgn * k_abs) * SX + (kappa / R_sp) * SY
    eig_res = float(np.linalg.norm(D @ u - E_sym * u))
    d = r - float(Rc)
    env = np.exp(-d * d / (2.0 * float(sigma) * float(sigma)))
    Z = 1.0 / math.sqrt(float(np.sum(env * env)))
    prof = profile_horizon(Rc, sigma, sgn * k_abs, Z)
    decl = radial_profile(r, prof)[:, None] * u[None, :]
    diff = float(np.max(np.abs(decl - psi0)))
    if diff > tol:
        raise RuntimeError(f"the declared profile (envelope exp(-(r - R_c)^2/(2 sigma^2)), carrier exp(i k_0 r), unit norm on the chain grid, no cut) does not reproduce the "
                           f"engine's column: max difference {diff:.2e} (tolerance {tol:g}); engine_single_object_B_v1.launch_columns changed: stop")
    return dict(psi0=psi0, spinor=u, E_symbol=E_sym, spinor_eigen_residual=eig_res, Z=Z, info=q, column_index=i, n_columns_built=int(X.shape[1]),
                members=[dict(depth=qq['depth'], R_c=float(qq['R_c']), k0=float(qq['k0']), sigma=float(qq['sigma'])) for qq in info],
                ks_override=override, ks_original=[[float(a), float(b), str(c)] for a, b, c in ks_orig], profile=prof, declared_profile_max_diff=diff,
                norm=float(np.vdot(x, x).real))


def _fmt(x, f='%.4f'):
    return (f % x) if isinstance(x, (int, float)) and x is not None else '-'


# ------------------------------------------------------------------ the run
def build_parser():
    ap = argparse.ArgumentParser(description="O5 chain side (v2): HORIZON's launch column on the solve code's own chain operator, on the T7 slice's held fields, exported per section 9 (a) to (e)")
    ap.add_argument('--record', required=True, help="record_slice.npz (keys r, N, v)")
    ap.add_argument('--record_json', default='', help="record_slice.json (provenance; its roots and md5 are cross-checked when given)")
    ap.add_argument('--solve_path', default=os.environ.get('O5_SOLVE_PATH', ''), help="the directory holding page_ops_v6.py and engine_single_object_B_v1.py (imported read-only; or the environment's O5_SOLVE_PATH)")
    ap.add_argument('--column_source', choices=['horizon', 'lane'], default='horizon', help="horizon: engine_single_object_B_v1.launch_columns' member (the exchange); lane: v1's Gaussian (the fallback shell with --spinor rest --ak0 0 --sigma 6)")
    ap.add_argument('--Rc', type=float, required=True, help="the column's centre in a (20.0 the T7 d 2 radius; 14.0 the d 8 radius)")
    ap.add_argument('--kappa', type=int, default=1, help="the angular channel (HORIZON's kappa; 1 the lowest, 2 the second member)")
    ap.add_argument('--ma', type=float, default=0.2)
    ap.add_argument('--ak0', type=float, default=2.0 * math.pi / 64.0, help="the signed carrier a k_0 (2 pi/64 HORIZON's long-wavelength member; pi/4 its threshold member; pi/2 its lattice-scale member; the sign is HORIZON's sgn)")
    ap.add_argument('--sigma', type=float, default=8.0, help="horizon: the amplitude's standard deviation in a (HORIZON's convention; its own pairs 16, 8, 4); lane: the density's standard deviation (v1's convention)")
    ap.add_argument('--cut', type=float, default=5.0, help="lane only: the profile is set to 0 beyond cut sigma from R_c (HORIZON's column has no cut)")
    ap.add_argument('--spinor', choices=['particle', 'rest'], default='particle', help="particle: the symbol's positive-energy spinor (the engine's eigh for horizon, v1's closed form for lane); rest: (1, 0), the fallback shell's (lane only)")
    ap.add_argument('--rs', type=float, default=29.0, help="r_s in a: the shells and the sweep (T7's 29 a)")
    ap.add_argument('--tau_rs', type=float, default=1.0, help="sweep length in units of r_s")
    ap.add_argument('--nsweeps', type=int, default=10, help="10 for the exchange window; 16 for the collar leg")
    ap.add_argument('--dt_read', type=float, default=0.5, help="readout step within a sweep (the cube driver's dt_read)")
    ap.add_argument('--radii_rs', type=float, nargs='*', default=[1.25, 1.5, 2.0, 3.0], help="shells beyond f r_s read beside r_h")
    ap.add_argument('--rh', type=float, default=0.0, help="the cut radius; 0 = the table's outer root of N - |v|")
    ap.add_argument('--det_frac', type=float, default=2.0, help="detector radius in r_s")
    ap.add_argument('--sigma_det', type=float, default=6.0, help="detector width in a (the density's standard deviation; GAP10's receiver packets' 6 a)")
    ap.add_argument('--det_cut', type=float, default=5.0)
    ap.add_argument('--kdet_kmax', type=float, default=math.pi / 2.0, help="the outgoing root is searched on (0, kdet_kmax)")
    ap.add_argument('--kdet_nscan', type=int, default=20001)
    ap.add_argument('--edge_window', type=float, default=0.005, help="half-width of the threshold columns about +-m and +-b (assigned to no bin)")
    ap.add_argument('--fine_sub', type=int, default=8, help="the Simpson integral of the current runs on dt_read/fine_sub (even)")
    ap.add_argument('--exact_integral', choices=['sweeps', 'all'], default='sweeps', help="the exact spectral integral at the sweep ends, or at every readout time")
    ap.add_argument('--out', default='o5_chain')
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    return ap


def run(args):
    rs, m = float(args.rs), float(args.ma)
    stem = f"o5_chain_Rc{args.Rc:g}_k{args.kappa}{args.tag}"
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, stem)
    if (not args.force) and os.path.exists(base + '.json') and os.path.exists(base + '.npz'):
        print(f"[skip] {base}.npz and .json exist (--force to redo)", flush=True)
        return 0
    if not args.solve_path:
        raise SystemExit("--solve_path (or the environment's O5_SOLVE_PATH) is required: the directory holding page_ops_v6.py and engine_single_object_B_v1.py")
    if args.column_source == 'horizon' and args.spinor != 'particle':
        raise SystemExit("HORIZON's column carries the particle spinor of the local symbol; the fallback shell (--spinor rest) is the lane's column: give --column_source lane")
    tw = time.time()
    OPS, ESB, solve_info = load_solve_modules(args.solve_path)
    print(f"[solve code] {solve_info['path']}: page_ops_v6.py md5 {solve_info['page_ops_v6']['md5']}; engine_single_object_B_v1.py md5 {solve_info['engine_single_object_B_v1']['md5']}; "
          f"HORIZON's own (k_0, sigma) pairs {[(round(k, 6), s) for k, s, _ in solve_info['KS']]}; page_history_v6b placeholder registered: {solve_info['page_history_v6b_placeholder']} "
          f"(the engine imports it at module level and uses it only in its propagation functions, none of which run here)", flush=True)
    r, N, v = V1.load_record(args.record)
    n = r.size
    rec_md5 = md5(args.record)
    print(f"[record] {args.record} (md5 {rec_md5}): n={n}, r={r[0]:g}..{r[-1]:g} a; N in [{N.min():.6f}, {N.max():.10f}]; v in [{v.min():.6f}, {v.max():.6f}] (<= 0 inward)", flush=True)
    roots = V1.table_roots(r, N, v)
    if args.rh > 0.0:
        r_h, slope_h = float(args.rh), float('nan')
    elif roots:
        r_h, slope_h = roots[-1]
    else:
        raise ValueError("no root of N - |v| on the table: give --rh")
    kappa_h = abs(slope_h) if np.isfinite(slope_h) else float('nan')
    N_inf = float(N[-1])
    b1, b3 = math.sqrt(m * m + 1.0), math.sqrt(m * m + 3.0)
    print(f"[roots] N - |v| = 0 on the table at {[round(x[0], 4) for x in roots]} (slopes {[round(x[1], 6) for x in roots]}); the cut r_h = {r_h:.6f} a, "
          f"kappa_h = |slope| = {kappa_h:.6f}; N_inf = {N_inf:.10f}; b1 = {b1:.4f} (chain), b3 = {b3:.4f} (cube); edges scaled by N_inf", flush=True)
    rec_json_info = None
    if args.record_json:
        with open(args.record_json) as fh:
            RJ = json.load(fh)
        rj = RJ.get('roots_on_table', {}).get('roots', [])
        diffs = [abs(a_[0] - float(b_['r'])) for a_, b_ in zip(roots, rj)]
        rec_json_info = dict(path=args.record_json, md5=md5(args.record_json), roots_json=rj, root_differences=diffs,
                             two_gE_tail_last=RJ.get('slice', {}).get('two_gE_tail_last'), N_max_json=RJ.get('fields_written', {}).get('N_max'),
                             md5_written_npz=RJ.get('md5_written_npz'), md5_matches=(RJ.get('md5_written_npz') == rec_md5),
                             axes_r_h={k: RJ.get('axes', {}).get(k, {}).get('r_h') for k in ('100', '111')})
        print(f"[record json] roots on table {[round(float(x['r']), 4) for x in rj]} against this script's: differences {['%.1e' % d for d in diffs]}; "
              f"2GE tail {rec_json_info['two_gE_tail_last']}; the json's md5 of the written npz matches the file read: {rec_json_info['md5_matches']}; "
              f"the cube's allocated r_h [100] {rec_json_info['axes_r_h']['100']}, [111] {rec_json_info['axes_r_h']['111']} (the cube reads the chain's cut with --o5_rh)", flush=True)

    # ---- the chain (the solve code's operator)
    t0 = time.time()
    chain = SolveChain(r, N, v, m, args.kappa, OPS)
    herm = chain.hermiticity_defect()
    ph = chain.particle_hole_defect()
    chain.diagonalize()
    t_eig = time.time() - t0
    print(f"[chain] page_ops_v6.clocked_csr(chain_sparse(r, m={m:g}, kappa={args.kappa}), N, v) on the record's grid, dense {chain.dim} x {chain.dim} ({chain.h_csr.nnz} nonzeros; "
          f"Gershgorin bound {chain.gershgorin:.4f}): Hermiticity defect {herm:.1e}; particle-hole anticommutation defect {ph:.1e} (with (-1)^j sigma_y K); eigendecomposition {t_eig:.1f}s: "
          f"spectrum [{chain.w[0]:+.6f}, {chain.w[-1]:+.6f}]; ||hV - V Lambda||_F {chain.eps_res:.2e}; ||V^dag V - I||_F {chain.eps_orth:.2e}", flush=True)

    # ---- the column
    if args.column_source == 'horizon':
        col = horizon_column(ESB, r, m, args.kappa, args.Rc, args.ak0, args.sigma, N, v)
        psi0, u, E_sym, prof, Z = col['psi0'], col['spinor'], col['E_symbol'], col['profile'], col['Z']
        spinor_mode = 'particle'
        print(f"[horizon column] engine_single_object_B_v1.launch_columns(R, m, kappa, (R_c, R_c), N, v): member {col['column_index']} of {col['n_columns_built']} built "
              f"('annulus middle', k_0 {col['info']['k0']:+.6f}, sigma {col['info']['sigma']:g} a amplitude width, {col['info']['ak0_name']}); KS override for this call: {col['ks_override']} "
              f"(HORIZON's own pairs {[(round(k, 6), s) for k, s, _ in col['ks_original']]}); the declared profile (envelope exp(-(r - R_c)^2/(2 sigma^2)), carrier exp(i k_0 r), unit norm on the "
              f"chain grid, no cut) reproduces the engine's column to {col['declared_profile_max_diff']:.1e}; the spinor's eigen-residual on the local symbol {col['spinor_eigen_residual']:.1e}; "
              f"N, v at the launch site (the engine's info) {col['info']['N_at']:.6f}, {col['info']['v_at']:+.6f}", flush=True)
    else:
        if args.spinor == 'particle':
            u, E_sym = V1.particle_spinor(args.ak0, args.Rc, m, args.kappa)
        else:
            u, E_sym = np.array([1.0, 0.0], dtype=np.complex128), m
        psi0, cinfo0 = V1.gaussian_column(r, args.Rc, args.sigma, args.ak0, args.cut, u)
        prof = profile_lane(args.Rc, args.sigma, args.ak0, cinfo0['Z'], args.cut)
        Z, col, spinor_mode = cinfo0['Z'], None, args.spinor
        print(f"[lane column] o5_chain_export_v1.gaussian_column: the lane's Gaussian of density width {args.sigma:g} a with the {args.cut:g} sigma cut, spinor {args.spinor} "
              f"(section 9's fallback shell when --spinor rest --ak0 0 --sigma 6)", flush=True)
    cinfo = envelope_weights(prof, r)
    cinfo.update(Z=float(Z), n_support=int(np.count_nonzero(np.sum(np.abs(psi0) ** 2, 1))))
    inner = [x for x in roots if x[1] < 0.0]
    outer = [x for x in roots if x[1] > 0.0]
    depth_rule = dict(annulus_middle=(0.5 * (inner[-1][0] + outer[-1][0]) if (inner and outer) else None),
                      two_sigma_inside_outer_root=((outer[-1][0] - 2.0 * args.sigma) if outer else None),
                      note="HORIZON's own depths (engine_single_object_B_v1.launch_columns: (r_in + R_t)/2 and R_t - 2 sigma) on this record's table roots; the exchange keeps T7's R_c")
    print(f"[depth] R_c = {args.Rc:g} a (T7's); HORIZON's own depth rule on this record would give the annulus middle {_fmt(depth_rule['annulus_middle'])} a and "
          f"R_t - 2 sigma = {_fmt(depth_rule['two_sigma_inside_outer_root'])} a", flush=True)
    psi0f = psi0.reshape(-1)
    E_mean, E2 = chain.expectation(psi0f)
    E_rms = math.sqrt(max(E2 - E_mean * E_mean, 0.0))
    c0 = chain.V.conj().T @ psi0f
    wts = np.abs(c0) ** 2
    edges = dict(m=N_inf * m, b1=N_inf * b1, b3=N_inf * b3)
    kb1 = V1.killing_bins(chain.w, wts, edges['m'], edges['b1'], args.edge_window)
    kb3 = V1.killing_bins(chain.w, wts, edges['m'], edges['b3'], args.edge_window)
    k3 = V1.killing_three(chain.w, wts, edges['m'], args.edge_window)
    ls = V1.lattice_scale_weight(psi0)
    soft = bool(abs(E_mean) < edges['m'])
    print(f"[column] R_c={args.Rc:g} a, sigma={args.sigma:g} a ({prof['sigma_meaning']}; density std {cinfo['density_std']:.4f} a), a k_0={args.ak0:+.6f} "
          f"(wavelength {(2 * math.pi / abs(args.ak0)) if args.ak0 else float('inf'):.1f} a), cut {prof.get('cut')}, spinor {spinor_mode} u=({u[0]:.5f}, {u[1]:.5f}) (E of the symbol {E_sym:.5f}); "
          f"support {cinfo['n_support']} sites, Z={Z:.6e}; the Gaussian's weight outside the cut {cinfo['gaussian_weight_outside_cut']:.1e}, below the grid (r < 1/2) {cinfo['gaussian_weight_below_grid']:.3e}, "
          f"beyond it {cinfo['gaussian_weight_beyond_grid']:.1e}; norm {float(np.vdot(psi0f, psi0f).real):.12f}; <h> = {E_mean:.6f}, rms {E_rms:.6f} "
          f"({'soft: |<h>| < N_inf m, exit to infinity barred by the mass threshold (HORIZON B.4)' if soft else 'hard: |<h>| > N_inf m'}); unsigned windowed share at folded |q| > pi/4: {ls['lattice_scale']:.4f}", flush=True)
    for nm, kb in (('b1', kb1), ('b3', kb3)):
        bb, th = kb['bins'], kb['threshold_columns']
        print(f"[killing {nm}={kb['edges']['b']:.4f}] below -b {bb['below_minus_b']:.3e}; (-b, -m) {bb['minus_b_to_minus_m']:.4f}; gap (-m, m) {bb['gap']:.4f}; (m, b) {bb['m_to_b']:.4f}; "
              f"above b {bb['above_b']:.3e}; threshold columns (within {args.edge_window:g} of an edge, assigned to neither): -b {th['-b']:.1e}, -m {th['-m']:.1e}, +m {th['+m']:.1e}, +b {th['+b']:.1e}; "
              f"total {kb['total']:.10f} (exact: the dense eigendecomposition, no KPM resolution)", flush=True)

    # ---- the readout times, the reflection clock and the states
    tau = args.tau_rs * rs
    nread = int(round(tau / args.dt_read))
    if abs(nread * args.dt_read - tau) > 1e-9:
        raise ValueError("tau must be a multiple of dt_read")
    S = int(args.nsweeps)
    times = args.dt_read * np.arange(0, nread * S + 1)
    sweep_idx = nread * np.arange(0, S + 1)
    sweep_t = times[sweep_idx]
    r_det = args.det_frac * rs
    rad_names = ['r_h'] + [f'{f:g}rs' for f in args.radii_rs]
    radii = np.array([r_h] + [f * rs for f in args.radii_rs], dtype=float)
    R_far = max(float(np.max(radii)), float(r_det))
    t_refl = reflection_clock(r[-1], args.Rc, R_far)
    refl_flags = times >= t_refl - 1e-9
    n_flag = int(np.count_nonzero(refl_flags))
    print(f"[reflection] a unit-speed front from R_c {args.Rc:g} a reaches the chain's end r_n = {r[-1]:g} a and returns to the outermost readout radius {R_far:g} a at t = {t_refl:g} a "
          f"({t_refl / rs:.1f} r_s) after launch (HORIZON's rule); readout times at or after it: {n_flag} of {times.size}"
          + (" -- WARNING: those readouts include the chain-end reflection; shorten the window (HORIZON appends the vacuum exterior to 128 r_s for its 300 r_s propagation)" if n_flag else
             "; no vacuum exterior appended (the window does not reach the chain's end)"), flush=True)
    t0 = time.time()
    states, _ = chain.propagate(psi0f, times)
    flat = states.reshape(len(times), chain.dim)
    norms = np.sum(np.abs(flat) ** 2, 1)
    energies = np.real(np.sum(np.conj(flat) * (flat @ chain.H.T), 1))
    dens = np.sum(np.abs(states) ** 2, 2)
    p_mid = V1.beyond_weights(r, dens, radii, 'midpoint')
    p_strict = V1.beyond_weights(r, dens, radii, 'strict')
    prop_err = chain.prop_error_bound(times)
    print(f"[propagation] {len(times)} readout times to t={times[-1]:g} a ({times[-1] / rs:g} r_s), {S} sweeps of tau={tau:g} a, dt_read={args.dt_read:g} ({time.time() - t0:.1f}s); "
          f"norm drift max {np.max(np.abs(norms - 1.0)):.1e}; energy drift max {np.max(np.abs(energies - energies[0])):.1e}; "
          f"the stated propagation error at the last sweep end {prop_err[-1]:.2e} (eps_res t + 2 eps_orth (1 + |Lambda|_max t))", flush=True)

    # ---- the fixed-cut current through r_h (the solve code's number-current operator)
    mask = r > r_h
    Q, q_diff = chain.number_current(r_h)
    Jc = np.sum(np.conj(flat) * (flat @ Q.T), 1)
    J = np.real(Jc)
    J_imag_max = float(np.max(np.abs(np.imag(Jc))))
    t0 = time.time()
    Jint_sweeps = V1.integrated_current_exact(chain.w, chain.V, Q, c0, sweep_t)
    Jint_all = V1.integrated_current_exact(chain.w, chain.V, Q, c0, times) if args.exact_integral == 'all' else None
    sub = int(args.fine_sub)
    t_fine = (args.dt_read / sub) * np.arange(0, nread * S * sub + 1)
    states_f, _ = chain.propagate(psi0f, t_fine)
    flat_f = states_f.reshape(len(t_fine), chain.dim)
    J_fine = np.real(np.sum(np.conj(flat_f) * (flat_f @ Q.T), 1))
    del states_f, flat_f
    Jint_simpson = V1.cumulative_simpson(J_fine, args.dt_read / sub, sub)
    dp_strict = p_strict[:, 0] - p_strict[0, 0]
    ident_exact = Jint_sweeps - dp_strict[sweep_idx]
    ident_simpson = Jint_simpson - dp_strict
    print(f"[current] page_ops_v6.number_current_csr(h, r, r_h) against v1's i[h, P_>]: max entry difference {q_diff:.1e}; J_>(r_h) at t 0 {J[0]:+.4e}; the fixed-cut identity "
          f"(strict weight change minus the integrated current): exact spectral form at the sweep ends max {np.max(np.abs(ident_exact)):.1e} (rounding), Simpson on dt_read/{sub} at every "
          f"readout time max {np.max(np.abs(ident_simpson)):.1e} (quadrature); Im J max {J_imag_max:.1e} ({time.time() - t0:.1f}s)", flush=True)

    # ---- the detectors at 2 r_s, both carrier signs (the lane's detectors, as v1)
    N_det, v_det = float(np.interp(r_det, r, N)), float(np.interp(r_det, r, v))
    k_root, roots_k = V1.outgoing_root_chain(E_mean, N_det, v_det, m, args.kappa, r_det, args.kdet_kmax, args.kdet_nscan)
    if k_root is None:
        k_det, matched = abs(args.ak0), False
        note = ("no outgoing root of the chain's local dispersion omega(k) = N E(k) + (v/2) sin 2k at r_det at the column's mean Killing energy (the energy lies below "
                "the outgoing branch's minimum there, or above the band): the detector carries the column's own carrier |a k_0|, not energy matched")
    else:
        k_det, matched = k_root, True
        note = "k_det the smallest root with positive group velocity of omega(k) = N E(k) + (v/2) sin 2k at r_det at the column's mean Killing energy <h>"
    det = V1.detector_packets(r, r_det, args.sigma_det, args.det_cut, k_det, m, args.kappa, spinor_mode)
    for d in det:
        d['profile'] = profile_lane(r_det, args.sigma_det, d['k'], d['Z'], args.det_cut)
    det_vecs = np.stack([d['vec'] for d in det], 0)
    amp = np.stack([np.einsum('js,tjs->t', np.conj(d['vec']), states) for d in det], 1)
    print(f"[detector] r_det={r_det:g} a (N {N_det:.5f}, v {v_det:+.5f}); roots of the local dispersion at <h>={E_mean:.5f}: {[(round(k, 5), round(vg, 4)) for k, vg in roots_k]}; "
          f"k_det={k_det:+.5f} energy matched: {matched}; width {args.sigma_det:g} a (density std), cut {args.det_cut:g}; "
          + '; '.join(f"{d['label']}: |A| peak {np.max(np.abs(amp[:, i])):.4e} at t={times[int(np.argmax(np.abs(amp[:, i])))]:g} a ({times[int(np.argmax(np.abs(amp[:, i])))] / rs:.2f} r_s)" for i, d in enumerate(det)), flush=True)

    # ---- the sweep-end table
    delta_s = prop_err[sweep_idx]
    band_mid = 2.0 * np.sqrt(np.clip(p_mid[sweep_idx], 0.0, None)) * delta_s[:, None] + delta_s[:, None] ** 2
    band_strict = 2.0 * np.sqrt(np.clip(p_strict[sweep_idx], 0.0, None)) * delta_s[:, None] + delta_s[:, None] ** 2
    sweeps = []
    print(f"[sweeps] t/rs | " + ' | '.join(f"p>{nm} (cell; strict)" for nm in rad_names) + " | J_>(r_h) | int J (exact) | dp_strict(r_h) | prop. error | reflection", flush=True)
    for s in range(S + 1):
        i = sweep_idx[s]
        row = dict(s=s, t=float(times[i]), t_rs=float(times[i] / rs), p_mid={nm: float(p_mid[i, k]) for k, nm in enumerate(rad_names)},
                   p_strict={nm: float(p_strict[i, k]) for k, nm in enumerate(rad_names)}, band_mid={nm: float(band_mid[s, k]) for k, nm in enumerate(rad_names)},
                   band_strict={nm: float(band_strict[s, k]) for k, nm in enumerate(rad_names)}, prop_err=float(delta_s[s]), J=float(J[i]), Jint_exact=float(Jint_sweeps[s]),
                   Jint_simpson=float(Jint_simpson[i]), dp_strict_rh=float(dp_strict[i]), identity_defect_exact=float(ident_exact[s]), identity_defect_simpson=float(ident_simpson[i]),
                   norm_minus_1=float(norms[i] - 1.0), energy_drift=float(energies[i] - energies[0]), amp_abs=[float(abs(amp[i, kd])) for kd in range(len(det))],
                   includes_chain_end_reflection=bool(refl_flags[i]))
        sweeps.append(row)
        print(f"[sweeps] {row['t_rs']:4.1f} | " + ' | '.join(f"{row['p_mid'][nm]:.5f}; {row['p_strict'][nm]:.5f}" for nm in rad_names)
              + f" | {row['J']:+.4e} | {row['Jint_exact']:+.6f} | {row['dp_strict_rh']:+.6f} | {row['prop_err']:.1e} | {'yes' if row['includes_chain_end_reflection'] else 'no'}", flush=True)
    i15 = rad_names.index('1.5rs') if '1.5rs' in rad_names else 1
    clk = dict(half_weight_1p5rs_sweeps=V1.half_weight_clock(sweep_t, p_mid[sweep_idx, i15]), half_weight_1p5rs_dense=V1.half_weight_clock(times, p_mid[:, i15]))
    clk.update(half_weight_1p5rs_sweeps_rs=clk['half_weight_1p5rs_sweeps'] / rs, half_weight_1p5rs_dense_rs=clk['half_weight_1p5rs_dense'] / rs)
    print(f"[clocks] half-weight crossing of p beyond 1.5 r_s (half its value at the last sweep end): on the sweep ends {clk['half_weight_1p5rs_sweeps_rs']:.2f} r_s, "
          f"on the dense readout {clk['half_weight_1p5rs_dense_rs']:.2f} r_s", flush=True)

    # ---- write
    arrays = dict(r=r, N=N, v=v, psi0=psi0, times=times, sweep_t=sweep_t, sweep_idx=sweep_idx, states=states, eigvals=chain.w, c0=c0, killing_weights=wts,
                  radii=radii, p_mid=p_mid, p_strict=p_strict, J=J, Jint_simpson=Jint_simpson, Jint_exact_sweeps=Jint_sweeps, amp=amp, det_vecs=det_vecs,
                  prop_err=prop_err, norms=norms, energies=energies, fft_q=ls['q'], fft_power=ls['power'], mask_rh=mask, t_fine=t_fine, J_fine=J_fine,
                  reflection_flags=refl_flags)
    if Jint_all is not None:
        arrays['Jint_exact_all'] = Jint_all
    np.savez_compressed(base + '.npz', **arrays)
    operator = dict(source='page_ops_v6.clocked_csr(page_ops_v6.chain_sparse(r, m, kappa), N, v), dense (scipy csr toarray)', solve_code=True, path=solve_info['path'],
                    page_ops_v6_md5=solve_info['page_ops_v6']['md5'], engine_md5=solve_info['engine_single_object_B_v1']['md5'], nnz=int(chain.h_csr.nnz),
                    gershgorin_bound=chain.gershgorin, number_current='page_ops_v6.number_current_csr(h, r, r_h)', number_current_vs_v1_max_diff=q_diff,
                    page_history_v6b_placeholder=solve_info['page_history_v6b_placeholder'], exterior_appended=False)
    meta = dict(script='o5_chain_export_v2.py', args=vars(args), solve_code=solve_info,
                record=dict(path=args.record, md5=rec_md5, n=int(n), r_first=float(r[0]), r_last=float(r[-1]), N_inf=N_inf, N_min=float(N.min()),
                            roots_on_table=[[a_, b_] for a_, b_ in roots], r_h=float(r_h), kappa_h=float(kappa_h), record_json=rec_json_info),
                spinor_convention=SPINOR_CONVENTION,
                chain=dict(kappa=int(args.kappa), m=m, b1=b1, b3=b3, N_inf=N_inf, edges=edges, dim=int(chain.dim), hermiticity_defect=herm, particle_hole_defect=ph,
                           eigen=dict(eps_res=chain.eps_res, eps_orth=chain.eps_orth, abs_max=chain.wmax, min=float(chain.w[0]), max=float(chain.w[-1]), wall=t_eig),
                           operator=operator),
                column=dict(Rc=float(args.Rc), sigma=float(args.sigma), sigma_density=cinfo['density_std'], ak0=float(args.ak0), cut=prof.get('cut'), spinor_mode=spinor_mode,
                            spinor=[[u[0].real, u[0].imag], [u[1].real, u[1].imag]], E_symbol=float(E_sym), Z=float(Z), n_support=cinfo['n_support'],
                            gaussian_weight_outside_cut=cinfo['gaussian_weight_outside_cut'], gaussian_weight_below_grid=cinfo['gaussian_weight_below_grid'],
                            gaussian_weight_beyond_grid=cinfo['gaussian_weight_beyond_grid'], E_mean=E_mean, E_rms=E_rms, norm=float(np.vdot(psi0f, psi0f).real), soft=soft,
                            source=('engine_single_object_B_v1.launch_columns' if col is not None else 'o5_chain_export_v1.gaussian_column'), profile=prof,
                            ks_override=(col['ks_override'] if col is not None else None), ks_original=(col['ks_original'] if col is not None else None),
                            horizon_info=(col['info'] if col is not None else None), horizon_members=(col['members'] if col is not None else None),
                            column_index=(col['column_index'] if col is not None else None), n_columns_built=(col['n_columns_built'] if col is not None else None),
                            declared_profile_max_diff=(col['declared_profile_max_diff'] if col is not None else 0.0),
                            spinor_eigen_residual=(col['spinor_eigen_residual'] if col is not None else None), depth_rule_horizon=depth_rule),
                killing=dict(b1=kb1, b3=kb3, three=k3, resolution='exact (dense eigendecomposition)'),
                lattice_scale_weight=ls['lattice_scale'],
                readout=dict(rs=rs, tau=tau, nsweeps=S, dt_read=float(args.dt_read), nread=nread, radii_names=rad_names, radii=[float(x) for x in radii],
                             primary_cut_rule='midpoint', two_gE_tail_last=(rec_json_info or {}).get('two_gE_tail_last'),
                             reflection=dict(t=t_refl, t_rs=t_refl / rs, R_far=R_far, r_last=float(r[-1]), n_readout_times_flagged=n_flag,
                                             sweep_ends_flagged=[bool(refl_flags[i]) for i in sweep_idx],
                                             rule="a unit-speed front from R_c to the chain's end and back to the outermost readout radius (HORIZON's rule); no vacuum exterior appended")),
                sweeps=sweeps,
                detector=dict(r_det=float(r_det), sigma_det=float(args.sigma_det), cut=float(args.det_cut), omega_target=E_mean, k_det=float(k_det), matched=bool(matched), note=note,
                              roots=[[k, vg] for k, vg in roots_k], N_det=N_det, v_det=v_det, profile_convention=profile_lane(r_det, args.sigma_det, k_det, det[0]['Z'], args.det_cut)['sigma_meaning'],
                              members=[dict(sign=d['sign'], k=d['k'], spinor=d['spinor'], E_symbol=d['E_symbol'], Z=d['Z'], label=d['label'], profile=d['profile'],
                                            peak_abs=float(np.max(np.abs(amp[:, i]))), peak_t=float(times[int(np.argmax(np.abs(amp[:, i])))]),
                                            peak_t_rs=float(times[int(np.argmax(np.abs(amp[:, i])))] / rs)) for i, d in enumerate(det)]),
                clocks=clk,
                propagation_error=dict(eps_res=chain.eps_res, eps_orth=chain.eps_orth, abs_max=chain.wmax, formula='eps_res t + 2 eps_orth (1 + |Lambda|_max t)',
                                       at_sweep_ends=[float(x) for x in delta_s]),
                checks=dict(norm_drift_max=float(np.max(np.abs(norms - 1.0))), energy_drift_max=float(np.max(np.abs(energies - energies[0]))),
                            identity_defect_exact_max=float(np.max(np.abs(ident_exact))), identity_defect_simpson_max=float(np.max(np.abs(ident_simpson))), J_imag_max=J_imag_max,
                            number_current_vs_v1_max_diff=q_diff, declared_profile_max_diff=(col['declared_profile_max_diff'] if col is not None else 0.0)),
                files=dict(npz=base + '.npz', json=base + '.json'), wall=time.time() - tw)
    with open(base + '.json', 'w') as fh:
        json.dump(meta, fh, indent=1, default=_json_default)
    print(f"[done] {base}.npz and .json ({time.time() - tw:.0f}s)", flush=True)
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
