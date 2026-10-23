#!/usr/bin/env python
"""dust_readouts_v1.py (busi lane, reviewer A; WEB40G icloud item 7; spec busi/specs/ICLOUD7_dust_readouts_v1.md): an independent recomputation
of the solve lane's DUST_root_history readouts (engine_dust_root_history_v1 / v1b) on the stored history, by default the 2 records T/r_s
12.125 kappa 1 and 15.0 kappa 1 (--all-channels adds kappa 2).

Read from the solve lane, read-only and by path: the HORIZON B.5 v2 dust columns npz (the columns X, 16384 rows = 8192 sites x 2 components
in site-major order, their occupations f, the launch clock per tag), the solved history's record json and fields npz (the production's to
T/r_s 40, this lane's verified snapshot of the continuation after it) through the solve lane's history reader
(engine_single_object_B_v2.CombinedHistory: the sites R, the clocks T, the records with their trapping zeros, the applied site fields per
record), and the radial operator through its operator builders (OPS.chain_sparse and OPS.clocked_csr: h = (1/2){N, h_0} + (1/2){v, Q} at the
record's applied fields). Nothing of the producer's propagation or readouts is imported; what follows is this script's own:
- the tagging at the launch record by the dense outer root (the record's largest trapping zero, which the producer checked equals the dense
  root at these records; --dense-root-from-engine runs the solve lane's dense field solve instead; --dense-root-override types a value) and
  by the applied outer root (this script's largest positive-slope zero of N + v on the engine's sites, linear between sites):
  X_in = Pi_{R < R_o} X, X_out = X - X_in;
- the propagation record by record, state(t_k) = exp(-i h(k-1) tau_k) state(t_{k-1}) with the piecewise-constant applied fields of the record
  before each step (the producer's convention), of X, X_in(dense) and X_in(applied) together, X_out by linearity; the exponential action is
  this script's Chebyshev expansion with Gershgorin bounds (coefficients 2 (-i)^k J_k(z), cut below --cheb-tol); --propagator taylor uses
  scipy's expm_multiply (Al-Mohy and Higham) as the method control;
- per record and column: the weight beyond 16 and 24 r_s of X, X_in, X_out for both cuts with the cross term as the remainder and explicitly
  as 2 Re <U X_in| Pi |U X_out>, the weight beyond the record's applied outer root R_t, the moving-cut sweep term and the fixed-cut transport
  over the step, the fixed-cut number currents F_c = Tr[f X^dag i[h, Pi_{R > c}] X] on the record's own fields at R_t, at the inner root and
  at 16 r_s (outward positive; i[h, Pi_c] built from h and the projector), the column norms (unitarity), this script's applied roots from the
  fields beside the record's zeros;
- the Killing bins at the launch and end records by a dense eigendecomposition of h (scipy eigh, --eig-driver): the 5 bins (-inf,-b], (-b,-m],
  (-m,m], (m,b], (b,inf), b = sqrt(m^2 + 1/a^2), the mean Killing energy per column; the spectral-projector jump ledger
  Delta p_I = Tr[(1_I(h_k) - 1_I(h_{k-1})) C_d(t_k)] at the records k0 + j s (s = --jump-every-rs r_s in records; the state at t_k in h_k's
  and in h_{k-1}'s basis), above m and in (m,b] and (b,inf) separately with the 5 bins before and after; within a record 1_I(h) is conserved,
  so the change of the weight above m from the launch to the end bins is the sum over every record jump, the rows a sample of it;
- the outward crossing clock per column (the first record with more than half the column's weight beyond R_t) and the first closing after the
  launch (the first record without a trapping zero; 0, NaN, None read as absent; the producer's literal test printed when it differs);
- a checkpoint of the propagated columns and of every per-record array every --checkpoint-every-rs (4 r_s = 32 records, the jump samples'
  records, so no eigendecomposition is redone on resume); a key whose per-key json and npz exist is skipped (--redo recomputes);
- the producer's numbers beside: --producer-json (repeatable; launches matched by key) and --producer-npz (repeatable; per-record arrays), the
  differences printed against --tol (1e-6, the acceptance), the currents also relative to the producer's largest value.
Outputs under --out-dir: <tag>_<key>.json and <tag>_<key>_arrays.npz per key; <tag>.json, <tag>.npz, <tag>.md combined. --compare-only
recomposes the combined outputs from finished per-key outputs (no engine needed). --derive prints the coordinate-rest arithmetic of the spec.
CPU only (numpy, scipy); niceness raised to at least 10; at most 16 threads (--threads, capped); no bytecode written into the solve lane's dirs."""
import os
import sys

sys.dont_write_bytecode = True   # the solve lane's code dirs are read-only to this lane: never write __pycache__ there


def _argv_value(flag, default):
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


try:
    _T = max(1, min(int(_argv_value("--threads", "16")), 16))
except ValueError:
    _T = 16
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_T)
os.environ.setdefault("PAGE_V6_PATH", _argv_value("--page-v6-path", "<internal>/v6"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.special import jv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR_DEFAULT = "<internal>/code"
ENGINE_DIR_FALLBACK = "<internal>/engine"
ENGINE_MODULE = "engine_single_object_B_v2"
TOL_DEFAULT = 1e-6
BINS = ["(-inf, -b]", "(-b, -m]", "(-m, m]", "(m, b]", "(b, inf)"]
SPLITS = ("all", "in_dense", "out_dense", "in_applied", "out_applied")
CUTS = ("dense", "applied")
COL_NAMES = (["b16_%s" % nm for nm in SPLITS] + ["b24_%s" % nm for nm in SPLITS]
             + ["cross_b16_dense", "cross_b24_dense", "cross_b16_applied", "cross_b24_applied", "beyond_Rt", "norm2",
                "cur_outer_col", "cur_inner_col", "cur_r16_col"])
REC_NAMES = ["Rt", "Ri", "Rt_fields", "Ri_fields", "cur_outer", "cur_inner", "cur_r16", "mov", "transport_step", "chebK", "step_seconds"]
SPECTRA = ("bins_launch", "mean_launch", "e_launch", "bins_end", "mean_end", "e_end")
NPZ_MAP = ([("b16_%s" % nm, "%s_b16" % nm) for nm in SPLITS] + [("b24_%s" % nm, "%s_b24" % nm) for nm in SPLITS]
           + [("beyond_Rt", "beyond_Rt"), ("T", "T"), ("Rt", "Rt"), ("Ri", "Ri"), ("bins_launch", "bins_launch"), ("bins_end", "bins_end"),
              ("f", "f"), ("cur_outer", "current_outer"), ("cur_inner", "current_inner"), ("cur_r16", "current_r16"), ("mov", "moving_cut_step")])
CURRENT_ARRAYS = ("cur_outer", "cur_inner", "cur_r16")
NOT_COUNTS = ("T", "Rt", "Ri", "f")


class Interrupted(Exception):
    """raised by run_key right after a checkpoint when interrupt_at is set (the selftest's resume check only)."""


# ------------------------------------------------------------------------------------------------------------------- small helpers
def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg), flush=True)


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_nice(target=10):
    """the box rule: niceness at least 10 (os.nice adds; nothing happens when the shell already niced the process)."""
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
    except (OSError, AttributeError):
        pass


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fmt(x, f="%.6e"):
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "nan"
    return f % xf


def atomic_savez(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def nanmax_or_none(x):
    x = np.asarray(x, float)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return None
    return float(np.nanmax(x))


def to_csr(A):
    """a scipy csr from whatever the operator builders return (scipy sparse, numpy dense, torch sparse or dense)."""
    if sp.issparse(A):
        return A.tocsr()
    if hasattr(A, "detach"):
        A = A.detach().cpu()
        if str(getattr(A, "layout", "torch.strided")) != "torch.strided":
            A = A.to_dense()
        return sp.csr_matrix(A.numpy())
    return sp.csr_matrix(np.asarray(A))


def site_weights(X, n):
    """site weights (sum of the 2 components' |X|^2), sites x columns; rows are site-major (2 s, 2 s + 1)."""
    return (np.abs(X) ** 2).reshape(n, 2, -1).sum(axis=1)


# ------------------------------------------------------------------------------------------------------------------- the propagation (own code)
def gershgorin_bounds(h):
    """an enclosing spectral interval of the Hermitian h from the Gershgorin discs."""
    h = h.tocsr()
    diag = h.diagonal()
    radius = np.asarray(abs(h).sum(axis=1)).ravel() - np.abs(diag)
    d = np.real(diag)
    return float(np.min(d - radius)), float(np.max(d + radius))


def cheb_expm_apply(h, X, tau, tol=1e-15, bounds=None):
    """exp(-i tau h) X by the Chebyshev expansion exp(-i z y) = J_0(z) + 2 sum_k (-i)^k J_k(z) T_k(y) with y = (h - c)/d on [-1, 1] (c the centre
    and d the half-width of the Gershgorin interval, z = |tau| d), the series cut at the first k > z with |2 J_k(z)| < tol; the three-term
    recurrence of the T_k(y) X. Returns (Y, K), K the number of matrix-vector products."""
    emin, emax = bounds if bounds is not None else gershgorin_bounds(h)
    c = 0.5 * (emax + emin)
    d = 0.5 * (emax - emin)
    if d <= 0.0:
        return X * np.exp(-1j * tau * c), 0
    z = abs(tau) * d
    Kmax = int(math.ceil(z)) + 60
    while True:
        J = jv(np.arange(Kmax + 1), z)
        small = np.nonzero(np.abs(2.0 * J) < tol)[0]
        small = small[small > z]
        if small.size:
            K = int(small[0])
            break
        Kmax *= 2
        if Kmax > 200000:
            raise RuntimeError("the Chebyshev series did not reach tol %g at z %g" % (tol, z))
    unit = -1j if tau >= 0 else 1j            # exp(-i tau d y) = exp(-i z y) for tau > 0 and exp(+i z y) for tau < 0
    coef = 2.0 * J[: K + 1] * np.array([unit ** k for k in range(K + 1)])
    coef[0] = J[0]
    T0 = X
    Y = coef[0] * T0
    if K >= 1:
        T1 = (h @ X - c * X) / d
        Y = Y + coef[1] * T1
        for k in range(2, K + 1):
            T2 = 2.0 * ((h @ T1 - c * T1) / d) - T0
            Y = Y + coef[k] * T2
            T0, T1 = T1, T2
    return Y * np.exp(-1j * tau * c), K


def taylor_expm_apply(h, X, tau):
    """the method control: scipy's expm_multiply (Al-Mohy and Higham's scaled Taylor action)."""
    return spla.expm_multiply((-1j * tau) * h.tocsc(), X)


def propagate(h, X, tau, method="chebyshev", tol=1e-15, bounds=None):
    if method == "taylor":
        return taylor_expm_apply(h, X, tau), -1
    return cheb_expm_apply(h, X, tau, tol, bounds)


# ------------------------------------------------------------------------------------------------------------------- the readouts (own code)
def current_operator(h, beyond):
    """J_c = i[h, Pi_c], Pi_c the projector on the 2 components of the sites beyond the cut (beyond: a boolean per site), so that
    d/dt Tr[Pi_c C] = Tr[J_c C] with the outward direction positive; only the cross-cut block of h survives (the same-side products cancel exactly)."""
    pi = sp.diags(np.repeat(np.asarray(beyond, bool).astype(float), 2))
    J = (1j * (h @ pi - pi @ h)).tocsr()
    J.eliminate_zeros()
    return J


def current_value(J, X, f):
    """the occupation-weighted current and the per-column values, Re[x^dag J x]."""
    per = np.real(np.einsum("ij,ij->j", X.conj(), J @ X))
    return float(np.dot(f, per)), per


def killing_bins(h, X, m, b, driver="evr"):
    """the dust's Killing occupation per column in the 5 bins (lo, hi] at -b, -m, m, b by a dense eigendecomposition of the record's h
    (the matrix in Fortran order so that LAPACK takes it in place), the mean Killing energy per column and the spectrum. Returns
    (bins 5 x columns, mean, eigenvalues, seconds)."""
    t0 = time.time()
    hd = h.toarray(order="F")
    e, U = sla.eigh(hd, lower=True, overwrite_a=True, check_finite=False, driver=driver)
    del hd
    c = (X.conj().T @ U).conj().T
    del U
    w = np.abs(c) ** 2
    edges = [-np.inf, -b, -m, m, b, np.inf]
    bins = np.array([w[(e > edges[i]) & (e <= edges[i + 1])].sum(axis=0) for i in range(5)])
    mean = (e[:, None] * w).sum(axis=0)
    return bins, mean, np.asarray(e, float), time.time() - t0


def zeros_of_sum(N, v, R, ne=None):
    """this script's zeros of f = N + v on the engine's sites, linear between sites: the outer applied root (the largest zero with positive slope,
    f < 0 before, f >= 0 after) and the inner one (the smallest zero with negative slope); None where absent."""
    ne = len(R) if ne is None else int(min(ne, len(R)))
    f = np.asarray(N, float)[:ne] + np.asarray(v, float)[:ne]
    Rr = np.asarray(R, float)[:ne]
    up = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    dn = np.nonzero((f[:-1] >= 0) & (f[1:] < 0))[0]

    def interp(i):
        return float(Rr[i] - f[i] * (Rr[i + 1] - Rr[i]) / (f[i + 1] - f[i]))

    return (interp(up[-1]) if up.size else None), (interp(dn[0]) if dn.size else None)


def valid_pos(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if (math.isfinite(x) and x > 0.0) else None


def record_roots(rec):
    """the record's own trapping zeros: the largest as the outer root, the smallest as the inner one when there are 2 or more (the producer's
    rule); the keys R_t and r_in as the fallback; 0, NaN and None read as absent (P10 claim 10)."""
    z = [valid_pos(x) for x in (rec.get("trapping_zeros") or [])]
    z = [x for x in z if x is not None]
    if z:
        return max(z), (min(z) if len(z) > 1 else None)
    return valid_pos(rec.get("R_t")), valid_pos(rec.get("r_in"))


def producer_has_root(rec):
    """the producer's literal test for an open trapping zero (a NaN R_t counts as open there)."""
    return bool(rec.get("trapping_zeros") or rec.get("R_t"))


def coordinate_rest_killing(ma, s):
    """the spec's section 2: on the chain's upper band omega(q) = sqrt(ma^2 + sin^2 q) - s sin(2q)/2 (N = 1, v = -s) the coordinate-rest wavenumber
    solves sin q cos q / sqrt(ma^2 + sin^2 q) = s cos 2q (bisection on (0, pi/4), the left side rising from 0, the right side falling to 0);
    returns q, eps, Q, omega, omega/m, the continuum comparator sqrt(1 - s^2) and the comoving one (omega = N m)."""
    def g(q):
        return math.sin(q) * math.cos(q) / math.sqrt(ma * ma + math.sin(q) ** 2) - s * math.cos(2.0 * q)
    lo, hi = 0.0, math.pi / 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    eps = math.sqrt(ma * ma + math.sin(q) ** 2)
    Qq = 0.5 * math.sin(2.0 * q)
    omega = eps - s * Qq
    return {"ma": ma, "s": s, "q": q, "eps": eps, "Q": Qq, "omega": omega, "omega_over_m": omega / ma,
            "continuum_omega_over_m": math.sqrt(max(0.0, 1.0 - s * s)), "comoving_omega_over_m": 1.0, "group_velocity_residual": g(q)}


# ------------------------------------------------------------------------------------------------------------------- the solve lane's inputs (read-only)
def import_engine(engine_dir, log=log):
    """engine_single_object_B_v2 (the solve lane's history reader and operator builders), imported read-only by path: the given dir, then its
    parent, then the producer's own dir (BOX_PATHS); no bytecode is written."""
    cands = []
    for d in (engine_dir, os.path.dirname(engine_dir.rstrip("/")) if engine_dir else "", ENGINE_DIR_FALLBACK):
        if d and d not in cands:
            cands.append(d)
    for d in cands:
        if os.path.exists(os.path.join(d, ENGINE_MODULE + ".py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            mod = importlib.import_module(ENGINE_MODULE)
            log("engine module %s imported read-only from %s" % (ENGINE_MODULE, d))
            return mod, d
    raise FileNotFoundError("%s.py not found in %s" % (ENGINE_MODULE, cands))


class StoredHistory:
    """the solved history read through the solve lane's reader, exposing what this script needs: R, T, rs, tau, ne, rec, fields(k)."""

    def __init__(self, BV2, record, fields, nsites, record2, fields2, switch_T_rs):
        self.H = BV2.CombinedHistory(record, fields, nsites, record2, fields2, switch_T_rs if record2 else None)
        self.R = np.asarray(self.H.R, float)
        self.T = np.asarray(self.H.T, float)
        self.rs = float(self.H.rs_nom)
        self.tau = float(getattr(self.H, "tau", np.median(np.diff(self.T))))
        self.ne = int(getattr(self.H, "ne", len(self.R)))
        self.rec = self.H.rec
        self.files = getattr(self.H, "files", None)

    def fields(self, k):
        N, v = self.H.fields(k)
        return np.asarray(N, float), np.asarray(v, float)


class EngineOps:
    """the solve lane's operator builders (read-only): the chain (h_0, Q) per channel and the clocked operator at the fields."""

    def __init__(self, BV2):
        self.OPS = BV2.OPS

    def chain(self, R, m, kappa):
        return self.OPS.chain_sparse(R, m, float(kappa))

    def clocked(self, h0, Q, N, v):
        return to_csr(self.OPS.clocked_csr(h0, Q, N, v))


# ------------------------------------------------------------------------------------------------------------------- one record and channel
def run_key(hist, ops, key, tag, kap, X0, f, T_launch, m, a, out_dir, dense_override=None, BV2=None, log=log, interrupt_at=None):
    """the propagation of one record's columns of one channel to T_end with every readout, checkpointed and resumable; returns (result, arrays)."""
    final_json = os.path.join(out_dir, "%s_%s.json" % (tag, key))
    final_npz = os.path.join(out_dir, "%s_%s_arrays.npz" % (tag, key))
    if os.path.exists(final_json) and os.path.exists(final_npz) and not a.redo:
        log("%s: finished (%s and %s present), skipped" % (key, os.path.basename(final_json), os.path.basename(final_npz)))
        with np.load(final_npz) as Z:
            arrs = {k_: Z[k_] for k_ in Z.files}
        return json.load(open(final_json)), arrs
    t_key = time.time()
    R = hist.R
    n = len(R)
    rs = hist.rs
    T = hist.T
    TT = T / rs
    k0 = int(np.argmin(np.abs(TT - T_launch)))
    kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
    if kend <= k0:
        raise ValueError("%s: T_end %g does not lie after the launch record %d (T/r_s %g)" % (key, a.T_end_rs, k0, TT[k0]))
    nr = kend - k0 + 1
    nc = X0.shape[1]
    f = np.asarray(f, float)
    b = math.sqrt(m * m + 1.0)
    step = max(1, int(round(a.jump_every_rs * rs / hist.tau)))
    cstep = max(1, int(round(a.checkpoint_every_rs * rs / hist.tau)))
    samples = [] if a.no_jumps else list(range(step, nr - 1, step))
    n_eig_expected = 2 + 2 * len(samples)
    h0, Q = ops.chain(R, m, kap)
    beyond16 = R > 16.0 * rs
    beyond24 = R > 24.0 * rs
    ck = os.path.join(out_dir, "ckpt_%s_%s.npz" % (tag, key))
    A = {}
    if os.path.exists(ck) and not a.redo:
        with np.load(ck) as Z:
            meta = json.loads(Z["meta"].item())
            for nm in COL_NAMES + REC_NAMES + ["T"] + list(SPECTRA):
                if nm in Z.files:
                    A[nm] = Z[nm]
            S = {nm: Z["S_" + nm] for nm in ("all", "in_dense", "in_applied")}
            Wall_prev = Z["Wall_prev"] if "Wall_prev" in Z.files else None
        kstart = int(meta["kk_done"]) + 1
        log("%s: resuming after record %d (T/r_s %.3f), %d eigendecompositions already done" % (key, meta["kk_done"], TT[meta["kk_done"]], len(meta["eig_seconds"])))
    else:
        N0, v0 = hist.fields(k0)
        Rt0_rec, Ri0_rec = record_roots(hist.rec[k0])
        Ro_app, Ri_app = zeros_of_sum(N0, v0, R, hist.ne)
        Ro_dense, Ri_dense = Rt0_rec, Ri0_rec
        src = "the record's trapping zeros at the launch record (the producer's check: equal to the dense roots at T/r_s 12.125 and 15.0)"
        if dense_override is not None:
            Ro_dense, src = float(dense_override), "typed (--dense-root-override)"
        elif a.dense_root_from_engine and BV2 is not None and hasattr(hist, "H"):
            try:
                rd = BV2.roots_at(hist.H, k0, a.fields_module)
                Ro_d, Ri_d, _ = BV2.outer_inner_of(rd, hist.rec[k0])
                Ro_dense, Ri_dense, src = Ro_d, Ri_d, "the solve lane's dense field solve (BV2.roots_at on the record's sources; its code, read-only)"
            except Exception as ex:   # noqa: BLE001
                log("%s: the engine's dense root solve failed (%s); the record's trapping zero is used" % (key, ex))
        if Ro_dense is None:
            log("%s: no dense outer root at the launch record; the inside-the-dense-root tag is empty" % key)
        if Ro_app is None:
            log("%s: no applied outer root of N + v on the sites at the launch record; the inside-the-applied-root tag is empty" % key)
        mask_d = np.repeat(R < (Ro_dense if Ro_dense is not None else -np.inf), 2)
        mask_a = np.repeat(R < (Ro_app if Ro_app is not None else -np.inf), 2)
        S = {"all": np.array(X0, dtype=np.complex128, copy=True), "in_dense": X0 * mask_d[:, None], "in_applied": X0 * mask_a[:, None]}
        inside0 = {cut: float(np.dot(f, site_weights(S["in_" + cut], n).sum(axis=0))) for cut in CUTS}
        A = {nm: np.full((nr, nc), np.nan) for nm in COL_NAMES}
        A.update({nm: np.full(nr, np.nan) for nm in REC_NAMES})
        A["T"] = TT[k0: kend + 1].copy()
        meta = {"key": key, "tag": tag, "kappa": int(kap), "k0": k0, "k_end": kend, "kk_done": k0 - 1, "jumps": [], "eig_seconds": [], "K_max": 0,
                "prop_seconds": 0.0, "roots": {"dense_outer": Ro_dense, "dense_inner": Ri_dense, "dense_source": src, "applied_outer": Ro_app,
                                                "applied_inner": Ri_app, "record_outer_zero": Rt0_rec, "record_inner_zero": Ri0_rec,
                                                "sites_inside_dense": int(mask_d.sum() // 2), "sites_inside_applied": int(mask_a.sum() // 2)},
                "inside_root_at_launch": inside0, "step_records": step, "checkpoint_records": cstep, "samples_j": samples, "utc_start": utc()}
        Wall_prev = None
        kstart = k0
        log("%s: launch record %d (T/r_s %.3f) to %d (%.3f), %d records, %d columns, f %s; roots dense %s applied %s (inner %s / %s); "
            "inside at launch dense %.6f applied %.6f; %d eigendecompositions expected" % (
                key, k0, TT[k0], kend, TT[kend], nr, nc, np.round(f, 4).tolist(), fmt(Ro_dense, "%.6f"), fmt(Ro_app, "%.6f"), fmt(Ri_dense, "%.6f"),
                fmt(Ri_app, "%.6f"), inside0["dense"], inside0["applied"], n_eig_expected))

    def save_ckpt():
        arrays = dict(A)
        for nm in ("all", "in_dense", "in_applied"):
            arrays["S_" + nm] = S[nm]
        if Wall_prev is not None:
            arrays["Wall_prev"] = Wall_prev
        arrays["meta"] = np.array(json.dumps(meta, default=jdefault))
        atomic_savez(ck, **arrays)

    for kk in range(kstart, kend + 1):
        j = kk - k0
        t_rec = time.time()
        hprev = None
        if kk > k0:
            Np, vp = hist.fields(kk - 1)
            hprev = ops.clocked(h0, Q, Np, vp)
            tau = float(T[kk] - T[kk - 1])
            bnd = gershgorin_bounds(hprev) if a.propagator == "chebyshev" else None
            t_p = time.time()
            Y, K = propagate(hprev, np.concatenate([S["all"], S["in_dense"], S["in_applied"]], axis=1), tau, a.propagator, a.cheb_tol, bnd)
            S["all"], S["in_dense"], S["in_applied"] = Y[:, :nc], Y[:, nc: 2 * nc], Y[:, 2 * nc:]
            meta["prop_seconds"] += time.time() - t_p
            A["chebK"][j] = K
            meta["K_max"] = max(meta["K_max"], int(K))
        Nk, vk = hist.fields(kk)
        hk = ops.clocked(h0, Q, Nk, vk)
        Xall = S["all"]
        if kk == k0:
            bins0, mean0, e0, te = killing_bins(hk, Xall, m, b, a.eig_driver)
            meta["eig_seconds"].append(te)
            A["bins_launch"], A["mean_launch"], A["e_launch"] = bins0, mean0, e0
            log("%s: launch Killing bins in %.0f s (spectrum %.4f to %.4f); weighted bins %s; %d more eigendecompositions expected, about %.0f s" % (
                key, te, e0.min(), e0.max(), np.round(bins0 @ f, 6).tolist(), n_eig_expected - 1, te * (n_eig_expected - 1)))
        if (not a.no_jumps) and kk > k0 and kk < kend and j % step == 0:
            ba, ma_, ea, tea = killing_bins(hprev, Xall, m, b, a.eig_driver)
            bb, mb_, eb, teb = killing_bins(hk, Xall, m, b, a.eig_driver)
            meta["eig_seconds"] += [tea, teb]
            dj = bb - ba
            row = {"t_over_rs": float(TT[kk]), "k": int(kk), "above_m_jump_weighted": float(np.dot(f, dj[3] + dj[4])),
                   "above_m_jump_per_column": (dj[3] + dj[4]).tolist(), "jump_m_to_b_per_column": dj[3].tolist(), "jump_above_b_per_column": dj[4].tolist(),
                   "jump_per_bin_weighted": (dj @ f).tolist(), "bins_before": ba.T.tolist(), "bins_after": bb.T.tolist(),
                   "mean_before": ma_.tolist(), "mean_after": mb_.tolist(), "spectrum_before": [float(ea.min()), float(ea.max())],
                   "spectrum_after": [float(eb.min()), float(eb.max())], "eig_seconds": [tea, teb]}
            meta["jumps"].append(row)
            log("%s: jump at T/r_s %.3f: Delta p above m %+.6e (per column %s); (m,b] %+.6e, (b,inf) %+.6e; %.0f + %.0f s" % (
                key, TT[kk], row["above_m_jump_weighted"], np.round(row["above_m_jump_per_column"], 8).tolist(),
                row["jump_per_bin_weighted"][3], row["jump_per_bin_weighted"][4], tea, teb))
        if kk == kend:
            binsE, meanE, eE, te = killing_bins(hk, Xall, m, b, a.eig_driver)
            meta["eig_seconds"].append(te)
            A["bins_end"], A["mean_end"], A["e_end"] = binsE, meanE, eE
            log("%s: end Killing bins in %.0f s; weighted bins %s" % (key, te, np.round(binsE @ f, 6).tolist()))
        Wall = site_weights(Xall, n)
        A["norm2"][j] = Wall.sum(axis=0)
        A["b16_all"][j] = Wall[beyond16].sum(axis=0)
        A["b24_all"][j] = Wall[beyond24].sum(axis=0)
        for cut in CUTS:
            Xin = S["in_" + cut]
            Xout = Xall - Xin
            Win = site_weights(Xin, n)
            Wout = site_weights(Xout, n)
            cross_site = (2.0 * np.real(Xin.conj() * Xout)).reshape(n, 2, -1).sum(axis=1)
            for cname, msk in (("b16", beyond16), ("b24", beyond24)):
                A["%s_in_%s" % (cname, cut)][j] = Win[msk].sum(axis=0)
                A["%s_out_%s" % (cname, cut)][j] = Wout[msk].sum(axis=0)
                A["cross_%s_%s" % (cname, cut)][j] = cross_site[msk].sum(axis=0)
        Rt, Ri = record_roots(hist.rec[kk])
        A["Rt"][j] = Rt if Rt is not None else np.nan
        A["Ri"][j] = Ri if Ri is not None else np.nan
        Rtf, Rif = zeros_of_sum(Nk, vk, R, hist.ne)
        A["Rt_fields"][j] = Rtf if Rtf is not None else np.nan
        A["Ri_fields"][j] = Rif if Rif is not None else np.nan
        if Rt is not None:
            A["beyond_Rt"][j] = Wall[R > Rt].sum(axis=0)
            if j > 0 and Wall_prev is not None and np.isfinite(A["Rt"][j - 1]):
                Rt_prev = A["Rt"][j - 1]
                A["mov"][j] = float(np.dot(f, Wall[R > Rt].sum(axis=0) - Wall[R > Rt_prev].sum(axis=0)))
                A["transport_step"][j] = float(np.dot(f, Wall[R > Rt_prev].sum(axis=0) - Wall_prev[R > Rt_prev].sum(axis=0)))
        for cname, cutR, colname in (("cur_outer", Rt, "cur_outer_col"), ("cur_inner", Ri, "cur_inner_col"), ("cur_r16", 16.0 * rs, "cur_r16_col")):
            if cutR is not None:
                tot, per = current_value(current_operator(hk, R > cutR), Xall, f)
                A[cname][j] = tot
                A[colname][j] = per
        Wall_prev = Wall
        A["step_seconds"][j] = time.time() - t_rec
        meta["kk_done"] = int(kk)
        if j % cstep == 0 or kk == kend:
            save_ckpt()
            log("%s: record %d (T/r_s %.3f): beyond 16 r_s %.6e, beyond R_t %s (R_t %s, fields %s), current r16 %s outer %s inner %s, K %s; checkpoint" % (
                key, kk, TT[kk], float(np.dot(f, A["b16_all"][j])), fmt(float(np.dot(f, A["beyond_Rt"][j])) if np.isfinite(A["beyond_Rt"][j]).all() else None),
                fmt(A["Rt"][j], "%.4f"), fmt(A["Rt_fields"][j], "%.4f"), fmt(A["cur_r16"][j], "%+.4e"), fmt(A["cur_outer"][j], "%+.4e"),
                fmt(A["cur_inner"][j], "%+.4e"), fmt(A["chebK"][j], "%.0f")))
            if interrupt_at is not None and kk == interrupt_at:
                raise Interrupted("interrupted after the checkpoint at record %d" % kk)

    # -------------------------------------------------------------------------------------------------------------- the summary
    tt = A["T"]
    bins0, mean0, binsE, meanE = A["bins_launch"], A["mean_launch"], A["bins_end"], A["mean_end"]

    def fw(v):
        return float(np.dot(f, np.asarray(v, float)))

    cross_clock = [next((float(tt[i]) for i in range(nr) if np.isfinite(A["beyond_Rt"][i, c]) and A["beyond_Rt"][i, c] > 0.5), None) for c in range(nc)]
    kc = next((k for k in range(k0 + 1, len(TT)) if record_roots(hist.rec[k])[0] is None), None)
    kc_p = next((k for k in range(k0 + 1, len(TT)) if not producer_has_root(hist.rec[k])), None)
    end = {nm: {"b16": fw(A["b16_%s" % nm][-1]), "b24": fw(A["b24_%s" % nm][-1])} for nm in SPLITS}
    cross_rem = {c: {cut: fw(A["%s_all" % c][-1] - A["%s_in_%s" % (c, cut)][-1] - A["%s_out_%s" % (c, cut)][-1]) for cut in CUTS} for c in ("b16", "b24")}
    cross_exp = {c: {cut: fw(A["cross_%s_%s" % (c, cut)][-1]) for cut in CUTS} for c in ("b16", "b24")}
    rem_minus_exp = 0.0
    for c in ("b16", "b24"):
        for cut in CUTS:
            r_ = A["%s_all" % c] - A["%s_in_%s" % (c, cut)] - A["%s_out_%s" % (c, cut)] - A["cross_%s_%s" % (c, cut)]
            rem_minus_exp = max(rem_minus_exp, float(np.nanmax(np.abs(r_))))
    above0 = fw(bins0[3] + bins0[4])
    aboveE = fw(binsE[3] + binsE[4])
    sampled = float(sum(r["above_m_jump_weighted"] for r in meta["jumps"]))
    both = np.isfinite(A["Rt_fields"]) & np.isfinite(A["Rt"])
    both_i = np.isfinite(A["Ri_fields"]) & np.isfinite(A["Ri"])
    res = {"key": key, "tag": tag, "kappa": int(kap), "T_launch_over_rs": float(TT[k0]), "k0": k0, "k_end": kend, "T_end_over_rs": float(TT[kend]),
           "n_records": nr, "n_columns": nc, "f": f.tolist(), "occupation_sum": float(f.sum()), "m": m, "b": b, "rs_nominal": rs,
           "record_spacing_lattice_time": hist.tau, "sites": {"n": n, "ne": hist.ne, "R_first": float(R[0]), "R_last": float(R[-1]),
                                                                "first_site_beyond_16rs": (int(np.argmax(beyond16)) if beyond16.any() else None)},
           "roots_launch": meta["roots"], "inside_root_at_launch": meta["inside_root_at_launch"],
           "first_closing_after_launch_over_rs": (float(TT[kc]) if kc is not None else None),
           "first_closing_producer_rule_over_rs": (float(TT[kc_p]) if kc_p is not None else None),
           "end": end, "cross_term_b16_end": cross_rem["b16"], "cross_term_b24_end": cross_rem["b24"],
           "cross_explicit_b16_end": cross_exp["b16"], "cross_explicit_b24_end": cross_exp["b24"],
           "killing_launch_per_column": bins0.T.tolist(), "killing_end_per_column": binsE.T.tolist(),
           "killing_launch_weighted": (bins0 @ f).tolist(), "killing_end_weighted": (binsE @ f).tolist(),
           "mean_killing_launch": mean0.tolist(), "mean_killing_end": meanE.tolist(),
           "above_m_launch_weighted": above0, "above_m_end_weighted": aboveE, "above_m_change_total_weighted": aboveE - above0,
           "above_m_change_sampled_sum_weighted": sampled, "n_jumps_total": nr - 1, "n_jumps_sampled": len(meta["jumps"]),
           "outward_crossing_clock_per_column": cross_clock,
           "jump_ledger_sampled": {"every_rs": a.jump_every_rs, "every_records": step,
                                   "label": "sampled every %g r_s, not summed over every jump" % a.jump_every_rs, "rows": meta["jumps"]},
           "checks": {"norm2_max_drift": float(np.nanmax(np.abs(A["norm2"] - A["norm2"][0][None, :]))),
                      "cross_remainder_minus_explicit_max": rem_minus_exp,
                      "Rt_fields_minus_record_max": (float(np.max(np.abs(A["Rt_fields"][both] - A["Rt"][both]))) if both.any() else None),
                      "Ri_fields_minus_record_max": (float(np.max(np.abs(A["Ri_fields"][both_i] - A["Ri"][both_i]))) if both_i.any() else None),
                      "records_with_record_root_but_no_field_root": int(np.sum(np.isfinite(A["Rt"]) & ~np.isfinite(A["Rt_fields"]))),
                      "records_with_field_root_but_no_record_root": int(np.sum(~np.isfinite(A["Rt"]) & np.isfinite(A["Rt_fields"]))),
                      "cheb_K_max": meta["K_max"], "propagator": a.propagator, "cheb_tol": a.cheb_tol, "eig_driver": a.eig_driver,
                      "spectrum_launch": [float(A["e_launch"].min()), float(A["e_launch"].max())],
                      "spectrum_end": [float(A["e_end"].min()), float(A["e_end"].max())],
                      "eig_seconds": meta["eig_seconds"], "propagate_seconds": meta["prop_seconds"],
                      "current_sign_at_outer_root": {"records_with_outer_root": int(np.sum(np.isfinite(A["cur_outer"]))),
                                                     "records_outward_positive": int(np.sum(A["cur_outer"] > 0)),
                                                     "max_outward": nanmax_or_none(A["cur_outer"]),
                                                     "min_inward": (float(np.nanmin(A["cur_outer"])) if np.any(np.isfinite(A["cur_outer"])) else None)}},
           "seconds": time.time() - t_key, "utc_start": meta.get("utc_start"), "utc_end": utc()}
    arrays = dict(A)
    arrays["f"] = f
    for nm in ("all", "in_dense", "in_applied"):
        arrays["X_end_" + nm] = S[nm]
    atomic_savez(final_npz, **arrays)
    atomic_write_text(final_json, json.dumps(res, default=jdefault, indent=1))
    if os.path.exists(ck):
        os.remove(ck)
    log("%s: f %s; roots dense %s applied %s; the dust beyond 16 r_s at %.3f: all %.4e, from inside the dense root %.4e, outside %.4e, cross %+.2e; "
        "from inside the applied root %.4e, outside %.4e, cross %+.2e; above m launch %.6f end %.6f (total change %+.6f; sampled rows %+.6f over %d of %d jumps); "
        "crossing clocks %s; first closing %s; norm drift %.1e; cross remainder - explicit %.1e; K max %d; %.0f s (eig %.0f s, propagation %.0f s)" % (
            key, np.round(f, 4).tolist(), fmt(meta["roots"]["dense_outer"], "%.6f"), fmt(meta["roots"]["applied_outer"], "%.6f"), TT[kend], end["all"]["b16"],
            end["in_dense"]["b16"], end["out_dense"]["b16"], cross_rem["b16"]["dense"], end["in_applied"]["b16"], end["out_applied"]["b16"],
            cross_rem["b16"]["applied"], above0, aboveE, aboveE - above0, sampled, len(meta["jumps"]), nr - 1, cross_clock, res["first_closing_after_launch_over_rs"],
            res["checks"]["norm2_max_drift"], rem_minus_exp, meta["K_max"], res["seconds"], sum(meta["eig_seconds"]), meta["prop_seconds"]))
    return res, arrays


# ------------------------------------------------------------------------------------------------------------------- the producer beside
def load_producer_jsons(paths):
    out = []
    for p in paths:
        try:
            P = json.load(open(p))
        except Exception as ex:   # noqa: BLE001
            out.append((os.path.basename(p), p, None, "unreadable: %s" % ex))
            continue
        out.append((P.get("script", os.path.basename(p)), p, P, None))
    return out


def load_producer_npzs(paths):
    out = []
    for p in paths:
        try:
            Z = np.load(p)
            out.append((os.path.basename(p), p, {k_: Z[k_] for k_ in Z.files}, None))
            Z.close()
        except Exception as ex:   # noqa: BLE001
            out.append((os.path.basename(p), p, None, "unreadable: %s" % ex))
    return out


def array_diff(x, y):
    """nan-aware comparison of 2 arrays: the largest |x - y| where both are finite, the number of places where only one is finite, and the
    producer's largest magnitude."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        return {"shape_this": list(x.shape), "shape_producer": list(y.shape), "max_abs_diff": None, "nan_mismatch": None, "max_abs_producer": None, "n_both_finite": 0}
    fx, fy = np.isfinite(x), np.isfinite(y)
    both = fx & fy
    return {"shape_this": list(x.shape), "shape_producer": list(y.shape), "n_both_finite": int(both.sum()),
            "max_abs_diff": (float(np.max(np.abs(x[both] - y[both]))) if both.any() else None),
            "nan_mismatch": int(np.sum(fx != fy)), "max_abs_producer": (float(np.max(np.abs(y[fy]))) if fy.any() else None)}


def compare_key(res, A, pjsons, pnpzs, tol):
    key = res["key"]
    out = {"key": key, "json": [], "npz": [], "max_abs_diff_counts": None, "max_abs_diff_currents": None, "clocks_equal": None, "within_tol": None}
    counts_max, cur_max, clocks_eq = [], [], []
    for lab, p, P, err in pjsons:
        c = {"label": lab, "path": p, "found": False, "error": err, "scalars": {}, "roots": {}, "clocks": {}, "killing": {}, "jumps": None, "max_abs_diff_counts": None}
        if P is None:
            out["json"].append(c)
            continue
        L = next((x for x in P.get("launches", []) if x.get("key") == key), None)
        if L is None:
            out["json"].append(c)
            continue
        c["found"] = True
        mx = []

        def put(name, x, y, count=True):
            d = None if (x is None or y is None) else float(x) - float(y)
            c["scalars"][name] = {"this": x, "producer": y, "diff": d}
            if count and d is not None and math.isfinite(d):
                mx.append(abs(d))

        for nm in SPLITS:
            for cc in ("b16", "b24"):
                put("end_%s_%s" % (nm, cc), res["end"][nm][cc], ((L.get("end") or {}).get(nm) or {}).get(cc))
        for cut in CUTS:
            put("cross_term_b16_end_%s" % cut, res["cross_term_b16_end"][cut], (L.get("cross_term_b16_end") or {}).get(cut))
            put("inside_root_at_launch_%s" % cut, res["inside_root_at_launch"][cut], (L.get("inside_root_at_launch") or {}).get(cut))
        for rn in ("dense_outer", "dense_inner", "applied_outer", "applied_inner"):
            x, y = res["roots_launch"].get(rn), (L.get("roots_launch") or {}).get(rn)
            c["roots"][rn] = {"this": x, "producer": y, "diff": (None if (x is None or y is None) else float(x) - float(y))}
        c["scalars"]["cheb_K_max"] = {"this": res["checks"]["cheb_K_max"], "producer": L.get("cheb_K_max"), "diff": None}
        c["scalars"]["occupation_sum"] = {"this": res["occupation_sum"], "producer": L.get("occupation_sum"), "diff": None}
        fc_p = L.get("first_closing_after_launch_over_rs")
        fc_eq = (fc_p is None and res["first_closing_after_launch_over_rs"] is None) or (
            fc_p is not None and res["first_closing_after_launch_over_rs"] is not None and abs(float(fc_p) - float(res["first_closing_after_launch_over_rs"])) < 1e-9)
        cr_p = L.get("outward_crossing_clock_per_column")
        cr_eq = None
        if cr_p is not None:
            cr_eq = len(cr_p) == len(res["outward_crossing_clock_per_column"]) and all(
                (x is None and y is None) or (x is not None and y is not None and abs(float(x) - float(y)) < 1e-9)
                for x, y in zip(res["outward_crossing_clock_per_column"], cr_p))
        c["clocks"] = {"first_closing": {"this": res["first_closing_after_launch_over_rs"], "this_producer_rule": res["first_closing_producer_rule_over_rs"],
                                         "producer": fc_p, "equal": bool(fc_eq)},
                       "crossing": {"this": res["outward_crossing_clock_per_column"], "producer": cr_p, "equal": cr_eq},
                       "k0": {"this": res["k0"], "producer": L.get("k0"), "equal": (L.get("k0") == res["k0"])},
                       "k_end": {"this": res["k_end"], "producer": L.get("k_end"), "equal": (L.get("k_end") == res["k_end"])}}
        clocks_eq.append(bool(fc_eq) and (cr_eq is not False) and (L.get("k0") in (None, res["k0"])) and (L.get("k_end") in (None, res["k_end"])))
        for nm in ("killing_launch_per_column", "killing_end_per_column", "mean_killing_launch", "mean_killing_end"):
            if L.get(nm) is not None:
                d = array_diff(np.asarray(res[nm], float), np.asarray(L[nm], float))
                c["killing"][nm] = d
                if d["max_abs_diff"] is not None:
                    mx.append(d["max_abs_diff"])
        J = (L.get("jump_ledger_sampled") or {}).get("rows")
        if J is not None:
            rows = []
            for r in res["jump_ledger_sampled"]["rows"]:
                pr = next((q for q in J if abs(float(q.get("t_over_rs", -1)) - r["t_over_rs"]) < 1e-9), None)
                e = {"t_over_rs": r["t_over_rs"], "this_weighted": r["above_m_jump_weighted"], "producer_weighted": None, "diff_weighted": None,
                     "max_abs_diff_per_column": None, "max_abs_diff_bins": None}
                if pr is not None:
                    e["producer_weighted"] = pr.get("above_m_jump_weighted")
                    if e["producer_weighted"] is not None:
                        e["diff_weighted"] = r["above_m_jump_weighted"] - float(e["producer_weighted"])
                        mx.append(abs(e["diff_weighted"]))
                    if pr.get("above_m_jump_per_column") is not None:
                        e["max_abs_diff_per_column"] = array_diff(r["above_m_jump_per_column"], pr["above_m_jump_per_column"])["max_abs_diff"]
                        if e["max_abs_diff_per_column"] is not None:
                            mx.append(e["max_abs_diff_per_column"])
                    bd = []
                    for nmb in ("bins_before", "bins_after"):
                        if pr.get(nmb) is not None:
                            dd = array_diff(r[nmb], pr[nmb])["max_abs_diff"]
                            if dd is not None:
                                bd.append(dd)
                    if bd:
                        e["max_abs_diff_bins"] = max(bd)
                        mx.append(e["max_abs_diff_bins"])
                rows.append(e)
            c["jumps"] = {"producer_rows": len(J), "this_rows": len(res["jump_ledger_sampled"]["rows"]), "matched": sum(1 for e in rows if e["producer_weighted"] is not None),
                          "producer_sum_weighted": float(sum(float(q.get("above_m_jump_weighted", 0.0)) for q in J)), "rows": rows}
        c["max_abs_diff_counts"] = max(mx) if mx else None
        if mx:
            counts_max.append(max(mx))
        out["json"].append(c)
    for lab, p, Z, err in pnpzs:
        c = {"label": lab, "path": p, "found": False, "error": err, "arrays": {}, "max_abs_diff_counts": None, "max_abs_diff_currents": None, "max_rel_diff_currents": None}
        if Z is None:
            out["npz"].append(c)
            continue
        cm, cu, rel = [], [], []
        for mine, theirs in NPZ_MAP:
            pk = "%s_%s" % (key, theirs)
            if pk not in Z or mine not in A:
                continue
            c["found"] = True
            d = array_diff(A[mine], Z[pk])
            c["arrays"][mine] = d
            if d["max_abs_diff"] is None:
                continue
            if mine in CURRENT_ARRAYS:
                cu.append(d["max_abs_diff"])
                if d["max_abs_producer"]:
                    rel.append(d["max_abs_diff"] / d["max_abs_producer"])
            elif mine not in NOT_COUNTS:
                cm.append(d["max_abs_diff"])
        c["max_abs_diff_counts"] = max(cm) if cm else None
        c["max_abs_diff_currents"] = max(cu) if cu else None
        c["max_rel_diff_currents"] = max(rel) if rel else None
        if cm:
            counts_max.append(max(cm))
        if cu:
            cur_max.append(max(cu))
        out["npz"].append(c)
    out["max_abs_diff_counts"] = max(counts_max) if counts_max else None
    out["max_abs_diff_currents"] = max(cur_max) if cur_max else None
    out["clocks_equal"] = (all(clocks_eq) if clocks_eq else None)
    comp = [x for x in (out["max_abs_diff_counts"], out["max_abs_diff_currents"]) if x is not None]
    if comp or clocks_eq:
        out["within_tol"] = bool(all(x <= tol for x in comp) and (out["clocks_equal"] is not False))
    return out


def verdict_line(cmp_, tol):
    return "VERDICT %s: max |this - producer| counts %s, currents %s, clocks equal %s -> within %g: %s" % (
        cmp_["key"], fmt(cmp_["max_abs_diff_counts"], "%.2e"), fmt(cmp_["max_abs_diff_currents"], "%.2e"), cmp_["clocks_equal"], tol,
        "PASS" if cmp_["within_tol"] else ("FAIL" if cmp_["within_tol"] is False else "no producer file given"))


# ------------------------------------------------------------------------------------------------------------------- the markdown
def compose_md(prov, results, comps, tol):
    L = []
    L.append("# dust_readouts_v1: the dust readouts recomputed on the stored history (busi lane, reviewer A; WEB40G icloud item 7)")
    L.append("")
    L.append("Composed %s by dust_readouts_v1.py (md5 %s); every number printed by the script; the producer's numbers (engine_dust_root_history_v1 / v1b "
             "json and npz at the paths named in the provenance) stand beside this script's and are never averaged with them. Definitions in "
             "busi/specs/ICLOUD7_dust_readouts_v1.md section 3: the dust = the stored dust columns with their occupations f; the tagging at the launch record by the "
             "dense outer root (the record's trapping zero unless the provenance says otherwise) and by the applied outer root (this script's zero of N + v on "
             "the sites); the propagation by exp(-i h(k-1) tau) on the piecewise-constant applied fields (this script's %s, control: scipy's Taylor); the weight "
             "beyond 16 and 24 r_s of X, X_in, X_out with the cross term; the fixed-cut number currents Tr[f X^dag i[h, Pi_{R > c}] X] at the record's outer "
             "root, inner root and 16 r_s on the record's own fields, outward positive, in fermions per unit lattice time; the Killing bins (-inf,-b], (-b,-m], "
             "(-m,m], (m,b], (b,inf) at m = %g, b = %.6f by a dense eigendecomposition; the jump ledger Delta p_I = Tr[(1_I(h_k) - 1_I(h_(k-1))) C_d(t_k)] sampled "
             "every %g r_s; counts in fermions; the acceptance tolerance %g absolute." % (
                 utc(), prov.get("script_md5"), prov.get("propagator"), prov.get("m"), prov.get("b"), prov.get("jump_every_rs"), tol))
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("| key | max abs diff, counts | max abs diff, currents | clocks equal | within %g |" % tol)
    L.append("|---|---|---|---|---|")
    for key in results:
        c = comps.get(key)
        if c is None:
            L.append("| %s | - | - | - | no producer file given |" % key)
        else:
            L.append("| %s | %s | %s | %s | %s |" % (key, fmt(c["max_abs_diff_counts"], "%.2e"), fmt(c["max_abs_diff_currents"], "%.2e"), c["clocks_equal"],
                                                   "PASS" if c["within_tol"] else ("FAIL" if c["within_tol"] is False else "no producer file given")))
    for key, res in results.items():
        c = comps.get(key) or {"json": [], "npz": []}
        js = [x for x in c["json"] if x.get("found")]
        L.append("")
        L.append("## %s: launch T/r_s %.3f (record %d), end %.3f (record %d), %d records, %d columns, f %s, sum %g" % (
            key, res["T_launch_over_rs"], res["k0"], res["T_end_over_rs"], res["k_end"], res["n_records"], res["n_columns"], res["f"], res["occupation_sum"]))
        L.append("")
        rl = res["roots_launch"]
        L.append("Roots at launch (units a): dense outer %s (%s), dense inner %s; applied outer %s (this script's zero of N + v on the sites), applied inner %s; "
                 "the record's zeros %s / %s; sites inside the dense root %d, inside the applied root %d." % (
                     fmt(rl.get("dense_outer"), "%.9f"), rl.get("dense_source"), fmt(rl.get("dense_inner"), "%.9f"), fmt(rl.get("applied_outer"), "%.9f"),
                     fmt(rl.get("applied_inner"), "%.9f"), fmt(rl.get("record_outer_zero"), "%.9f"), fmt(rl.get("record_inner_zero"), "%.9f"),
                     rl.get("sites_inside_dense", 0), rl.get("sites_inside_applied", 0)))
        for jj in js:
            L.append("Producer %s: roots %s." % (jj["label"], "; ".join("%s %s (this - producer %s)" % (rn, fmt(d["producer"], "%.9f"), fmt(d["diff"], "%.1e")) for rn, d in jj["roots"].items())))
        L.append("")
        hdr = "| quantity | this |" + "".join(" %s | this - producer |" % jj["label"] for jj in js)
        L.append(hdr)
        L.append("|---|---|" + "---|---|" * len(js))

        def row(name, label):
            cells = [label, fmt(res_lookup(res, name))]
            for jj in js:
                s = jj["scalars"].get(name) or {}
                cells += [fmt(s.get("producer")), fmt(s.get("diff"), "%.2e")]
            L.append("| " + " | ".join(cells) + " |")

        for cut in CUTS:
            row("inside_root_at_launch_%s" % cut, "inside the %s root at launch" % cut)
        for nm in SPLITS:
            for cc in ("b16", "b24"):
                row("end_%s_%s" % (nm, cc), "at the end, %s, beyond %s r_s" % (nm, cc[1:]))
        for cut in CUTS:
            row("cross_term_b16_end_%s" % cut, "cross term beyond 16 r_s at the end, %s cut (remainder)" % cut)
        L.append("| cross term beyond 24 r_s at the end (remainder), dense; applied | %s; %s |%s" % (
            fmt(res["cross_term_b24_end"]["dense"]), fmt(res["cross_term_b24_end"]["applied"]), " not printed by the producer | - |" * len(js)))
        L.append("| cross term explicit 2 Re <U X_in|Pi|U X_out> beyond 16 r_s, dense; applied | %s; %s |%s" % (
            fmt(res["cross_explicit_b16_end"]["dense"]), fmt(res["cross_explicit_b16_end"]["applied"]), " - | - |" * len(js)))
        L.append("| weight above m, weighted: launch; end; total change (= the sum over every record jump) | %s; %s; %s |%s" % (
            fmt(res["above_m_launch_weighted"]), fmt(res["above_m_end_weighted"]), fmt(res["above_m_change_total_weighted"], "%+.6e"), " - | - |" * len(js)))
        L.append("| the sampled rows' sum above m (%d of %d jumps) | %s |%s" % (
            res["n_jumps_sampled"], res["n_jumps_total"], fmt(res["above_m_change_sampled_sum_weighted"], "%+.6e"),
            "".join(" %s | - |" % (fmt((jj.get("jumps") or {}).get("producer_sum_weighted"), "%+.6e") if jj.get("jumps") else "no rows in this json") for jj in js)))
        L.append("| first closing after launch (T/r_s): this rule; the producer's literal rule | %s; %s |%s" % (
            fmt(res["first_closing_after_launch_over_rs"], "%.3f"), fmt(res["first_closing_producer_rule_over_rs"], "%.3f"),
            "".join(" %s | equal %s |" % (fmt(jj["clocks"]["first_closing"]["producer"], "%.3f"), jj["clocks"]["first_closing"]["equal"]) for jj in js)))
        L.append("| outward crossing clocks per column (T/r_s) | %s |%s" % (
            res["outward_crossing_clock_per_column"], "".join(" %s | equal %s |" % (jj["clocks"]["crossing"]["producer"], jj["clocks"]["crossing"]["equal"]) for jj in js)))
        L.append("| Chebyshev order K max | %d |%s" % (res["checks"]["cheb_K_max"], "".join(" %s | - |" % fmt(jj["scalars"].get("cheb_K_max", {}).get("producer"), "%.0f") for jj in js)))
        L.append("")
        L.append("Killing bins per column (weighted), launch: %s; end: %s; mean Killing energy per column launch %s, end %s." % (
            [fmt(x, "%.6e") for x in res["killing_launch_weighted"]], [fmt(x, "%.6e") for x in res["killing_end_weighted"]],
            [fmt(x, "%.6f") for x in res["mean_killing_launch"]], [fmt(x, "%.6f") for x in res["mean_killing_end"]]))
        for jj in js:
            if jj["killing"]:
                L.append("Producer %s: max abs diff per column, %s." % (jj["label"], "; ".join("%s %s" % (nm, fmt(d["max_abs_diff"], "%.2e")) for nm, d in jj["killing"].items())))
        L.append("")
        L.append("### %s: the jump ledger, sampled every %g r_s (%s)" % (key, res["jump_ledger_sampled"]["every_rs"], res["jump_ledger_sampled"]["label"]))
        L.append("")
        L.append("| T/r_s | record | Delta p above m, weighted | per column | (m, b]; (b, inf) weighted | 5 bins before (weighted) | 5 bins after (weighted) |" + "".join(" %s weighted | diff | per column max diff | bins max diff |" % jj["label"] for jj in js if jj.get("jumps")))
        L.append("|---|---|---|---|---|---|---|" + "---|---|---|---|" * sum(1 for jj in js if jj.get("jumps")))
        for r in res["jump_ledger_sampled"]["rows"]:
            fb = np.asarray(r["bins_before"], float).T @ np.asarray(res["f"], float)
            fa = np.asarray(r["bins_after"], float).T @ np.asarray(res["f"], float)
            cells = ["%.3f" % r["t_over_rs"], "%d" % r["k"], fmt(r["above_m_jump_weighted"], "%+.6e"), str([fmt(x, "%+.3e") for x in r["above_m_jump_per_column"]]),
                     "%s; %s" % (fmt(r["jump_per_bin_weighted"][3], "%+.3e"), fmt(r["jump_per_bin_weighted"][4], "%+.3e")),
                     str([fmt(x, "%.6e") for x in fb]), str([fmt(x, "%.6e") for x in fa])]
            for jj in js:
                if not jj.get("jumps"):
                    continue
                e = next((q for q in jj["jumps"]["rows"] if abs(q["t_over_rs"] - r["t_over_rs"]) < 1e-9), None) or {}
                cells += [fmt(e.get("producer_weighted"), "%+.6e"), fmt(e.get("diff_weighted"), "%.2e"), fmt(e.get("max_abs_diff_per_column"), "%.2e"), fmt(e.get("max_abs_diff_bins"), "%.2e")]
            L.append("| " + " | ".join(cells) + " |")
        if not res["jump_ledger_sampled"]["rows"]:
            L.append("| no rows (--no-jumps) | | | | | | |")
        L.append("")
        ck = res["checks"]
        cs = ck.get("current_sign_at_outer_root") or {}
        L.append("### %s: checks" % key)
        L.append("")
        L.append("- unitarity: max drift of the column norms %s; the split identity: max |all - in - out - cross_explicit| over records and columns %s." % (
            fmt(ck["norm2_max_drift"], "%.1e"), fmt(ck["cross_remainder_minus_explicit_max"], "%.1e")))
        L.append("- this script's applied roots from the fields against the record's zeros: outer max |diff| %s, inner %s; records with a record root but no field root %d, "
                 "with a field root but no record root %d." % (fmt(ck["Rt_fields_minus_record_max"], "%.3e"), fmt(ck["Ri_fields_minus_record_max"], "%.3e"),
                                                               ck["records_with_record_root_but_no_field_root"], ck["records_with_field_root_but_no_record_root"]))
        L.append("- the fixed-cut current at the outer root while a root exists: %s records, outward (positive) at %s of them, largest outward %s, most inward %s "
                 "(the continuum theorem: never outward through a persisting root; the lattice: finite-spacing transport)." % (
                     cs.get("records_with_outer_root"), cs.get("records_outward_positive"), fmt(cs.get("max_outward"), "%+.3e"), fmt(cs.get("min_inward"), "%+.3e")))
        L.append("- the spectrum of h at launch %s, at the end %s; propagator %s (tol %g, K max %d); eigendecompositions %d, %.0f s each on average; propagation %.0f s; "
                 "the key %.0f s." % (["%.4f" % x for x in ck["spectrum_launch"]], ["%.4f" % x for x in ck["spectrum_end"]], ck["propagator"], ck["cheb_tol"],
                                      ck["cheb_K_max"], len(ck["eig_seconds"]), (np.mean(ck["eig_seconds"]) if ck["eig_seconds"] else 0.0), ck["propagate_seconds"], res["seconds"]))
        for nz in c["npz"]:
            if not nz.get("found"):
                L.append("- producer npz %s: no arrays of this key found%s." % (nz["label"], (" (%s)" % nz["error"]) if nz.get("error") else ""))
                continue
            L.append("")
            L.append("### %s: per-record arrays against the producer's npz %s (counts max abs diff %s; currents max abs diff %s, relative to the producer's largest %s)" % (
                key, nz["label"], fmt(nz["max_abs_diff_counts"], "%.2e"), fmt(nz["max_abs_diff_currents"], "%.2e"), fmt(nz["max_rel_diff_currents"], "%.2e")))
            L.append("")
            L.append("| array | shape | both finite | max abs diff | nan pattern mismatches | producer's largest magnitude |")
            L.append("|---|---|---|---|---|---|")
            for nm, d in nz["arrays"].items():
                L.append("| %s | %s / %s | %s | %s | %s | %s |" % (nm, d["shape_this"], d["shape_producer"], d["n_both_finite"], fmt(d["max_abs_diff"], "%.2e"),
                                                                 d["nan_mismatch"], fmt(d["max_abs_producer"], "%.3e")))
    L.append("")
    L.append("## Provenance")
    L.append("")
    for k_, v_ in prov.items():
        L.append("- %s: %s" % (k_, v_))
    return "\n".join(L) + "\n"


def res_lookup(res, name):
    if name.startswith("end_"):
        rest = name[4:]
        nm, cc = rest.rsplit("_", 1)
        return res["end"][nm][cc]
    if name.startswith("cross_term_b16_end_"):
        return res["cross_term_b16_end"][name.rsplit("_", 1)[1]]
    if name.startswith("inside_root_at_launch_"):
        return res["inside_root_at_launch"][name.rsplit("_", 1)[1]]
    return None


# ------------------------------------------------------------------------------------------------------------------- main
def default_args(**kw):
    a = argparse.Namespace(engine_dir=ENGINE_DIR_DEFAULT, page_v6_path=os.environ.get("PAGE_V6_PATH"), columns_npz="", record="", fields="", record2=None,
                           fields2=None, switch_T_rs=40.0, fields_module="engine_fields_v14", nsites=8192, m=0.2, tags="T12p125,T15p000", kappas="1",
                           all_channels=False, T_end_rs=56.0, jump_every_rs=4.0, checkpoint_every_rs=4.0, no_jumps=False, propagator="chebyshev",
                           cheb_tol=1e-15, eig_driver="evr", dense_root_from_engine=False, dense_root_override=[], producer_json=[], producer_npz=[],
                           out_dir="", tag="dust_readouts_v1", threads=_T, overwrite=False, redo=False, compare_only=False, tol=TOL_DEFAULT, derive=False,
                           ma=0.2, R0=256.0, two_GE=59.1)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine-dir", default=ENGINE_DIR_DEFAULT, help="the solve lane's code dir (read-only); its parent and %s are tried after it" % ENGINE_DIR_FALLBACK)
    ap.add_argument("--page-v6-path", default=os.environ.get("PAGE_V6_PATH"), help="PAGE_V6_PATH for the engine's import (set before numpy loads)")
    ap.add_argument("--columns-npz", default=""); ap.add_argument("--record", default=""); ap.add_argument("--fields", default="")
    ap.add_argument("--record2", default=None); ap.add_argument("--fields2", default=None); ap.add_argument("--switch-T-rs", type=float, default=40.0)
    ap.add_argument("--fields-module", default="engine_fields_v14", help="used only with --dense-root-from-engine")
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--tags", default="T12p125,T15p000"); ap.add_argument("--kappas", default="1"); ap.add_argument("--all-channels", action="store_true", help="kappa 1 and 2")
    ap.add_argument("--T-end-rs", type=float, default=56.0); ap.add_argument("--jump-every-rs", type=float, default=4.0)
    ap.add_argument("--checkpoint-every-rs", type=float, default=4.0); ap.add_argument("--no-jumps", action="store_true", help="skip the jump ledger (2 eigendecompositions only)")
    ap.add_argument("--propagator", default="chebyshev", choices=["chebyshev", "taylor"]); ap.add_argument("--cheb-tol", type=float, default=1e-15)
    ap.add_argument("--eig-driver", default="evr", choices=["evr", "evd"])
    ap.add_argument("--dense-root-from-engine", action="store_true", help="the dense root by the solve lane's dense field solve (BV2.roots_at) instead of the record's zero")
    ap.add_argument("--dense-root-override", action="append", default=[], help="KEY=VALUE, the dense outer root typed for a key (repeatable)")
    ap.add_argument("--producer-json", action="append", default=[]); ap.add_argument("--producer-npz", action="append", default=[])
    ap.add_argument("--out-dir", default=""); ap.add_argument("--tag", default="dust_readouts_v1"); ap.add_argument("--threads", type=int, default=_T)
    ap.add_argument("--overwrite", action="store_true", help="rewrite existing combined outputs"); ap.add_argument("--redo", action="store_true", help="recompute finished keys")
    ap.add_argument("--compare-only", action="store_true", help="recompose from finished per-key outputs; no engine, no columns")
    ap.add_argument("--tol", type=float, default=TOL_DEFAULT)
    ap.add_argument("--derive", action="store_true", help="print the coordinate-rest Killing-energy arithmetic (spec section 2) and exit")
    ap.add_argument("--ma", type=float, default=0.2); ap.add_argument("--R0", type=float, default=256.0); ap.add_argument("--two-GE", type=float, default=59.1)
    return ap.parse_args()


def derive(a):
    s = math.sqrt(a.two_GE / a.R0)
    d = coordinate_rest_killing(a.ma, s)
    print("coordinate rest on the chain's upper band, ma %g, R_0 %g a, 2GE %g a: s = |v| = sqrt(2GE/R_0) = %.9f" % (a.ma, a.R0, a.two_GE, s))
    print("  stationarity sin q cos q / sqrt(ma^2 + sin^2 q) = s cos 2q at q = %.9f (residual %.1e); eps = %.9f, Q = sin(2q)/2 = %.9f" % (d["q"], d["group_velocity_residual"], d["eps"], d["Q"]))
    print("  omega = eps - s Q = %.9f (1/a); omega/m = %.9f; continuum comparator sqrt(1 - s^2) = %.9f; comoving release (q = 0): omega/m = %.1f, dR/dt = v = %.6f" % (
        d["omega"], d["omega_over_m"], d["continuum_omega_over_m"], d["comoving_omega_over_m"], -s))
    print("  the current matrix at an applied root in the continuum, N(sigma_x - I): eigenvalues %s (times N)" % np.linalg.eigvalsh(np.array([[0, 1], [1, 0]], float) - np.eye(2)).tolist())
    for q in (0.1708, math.pi / 4):
        print("  the lattice velocity matrix at a root for q = %.4f: larger eigenvalue N(cos q - cos 2q) = %.5f N; the upper band's group velocity %.5f N" % (
            q, math.cos(q) - math.cos(2 * q), math.sin(q) * math.cos(q) / math.sqrt(a.ma ** 2 + math.sin(q) ** 2) - math.cos(2 * q)))


def main():
    a = parse_args()
    if a.derive:
        derive(a)
        return
    ensure_nice(10)
    if not a.out_dir:
        sys.exit("--out-dir is required")
    os.makedirs(a.out_dir, exist_ok=True)
    combined = {ext: os.path.join(a.out_dir, "%s.%s" % (a.tag, ext)) for ext in ("json", "npz", "md")}
    if any(os.path.exists(p) for p in combined.values()) and not (a.overwrite or a.compare_only):
        sys.exit("refusing to overwrite %s (use --overwrite, or --compare-only to recompose)" % [p for p in combined.values() if os.path.exists(p)])
    prov = {"script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)), "args": " ".join(sys.argv[1:]), "host": socket.gethostname(),
            "utc_start": utc(), "threads": _T, "nice": os.nice(0), "numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0],
            "m": a.m, "b": math.sqrt(a.m * a.m + 1.0), "T_end_rs": a.T_end_rs, "jump_every_rs": a.jump_every_rs, "checkpoint_every_rs": a.checkpoint_every_rs,
            "propagator": a.propagator, "cheb_tol": a.cheb_tol, "eig_driver": a.eig_driver, "tol": a.tol, "compare_only": a.compare_only,
            "PAGE_V6_PATH": os.environ.get("PAGE_V6_PATH")}
    kappas = [1, 2] if a.all_channels else [int(x) for x in a.kappas.split(",") if x]
    tags = [x for x in a.tags.split(",") if x]
    overrides = {}
    for kv in a.dense_root_override:
        k_, v_ = kv.split("=", 1)
        overrides[k_] = float(v_)
    results, arrays_all = {}, {}
    if a.compare_only:
        for tg in tags:
            for kap in kappas:
                key = "%s_k%d" % (tg, kap)
                fj = os.path.join(a.out_dir, "%s_%s.json" % (a.tag, key))
                fz = os.path.join(a.out_dir, "%s_%s_arrays.npz" % (a.tag, key))
                if not (os.path.exists(fj) and os.path.exists(fz)):
                    log("%s: not finished (%s or %s missing), skipped" % (key, os.path.basename(fj), os.path.basename(fz)))
                    continue
                results[key] = json.load(open(fj))
                with np.load(fz) as Z:
                    arrays_all[key] = {k_: Z[k_] for k_ in Z.files}
    else:
        if not (a.columns_npz and a.record and a.fields):
            sys.exit("--columns-npz, --record and --fields are required (or --compare-only)")
        BV2, edir = import_engine(a.engine_dir)
        prov["engine_dir"] = edir
        prov["engine_module"] = getattr(BV2, "__file__", None)
        prov["engine_module_md5"] = md5(BV2.__file__) if getattr(BV2, "__file__", None) and os.path.exists(BV2.__file__) else None
        ops_file = getattr(getattr(BV2, "OPS", None), "__file__", None)
        prov["ops_module"] = ops_file
        prov["ops_module_md5"] = md5(ops_file) if ops_file and os.path.exists(ops_file) else None
        hist = StoredHistory(BV2, a.record, a.fields, a.nsites, a.record2, a.fields2, a.switch_T_rs)
        ops = EngineOps(BV2)
        prov["history_files"] = hist.files
        prov["record_files"] = [{"record": a.record, "record_md5": md5(a.record), "fields": a.fields, "fields_md5": md5(a.fields)}]
        if a.record2:
            prov["record_files"].append({"record": a.record2, "record_md5": md5(a.record2), "fields": a.fields2, "fields_md5": md5(a.fields2), "switch_T_over_rs": a.switch_T_rs})
        prov["columns_npz"] = a.columns_npz
        prov["columns_npz_md5"] = md5(a.columns_npz)
        prov["sites"] = {"n": int(len(hist.R)), "ne": hist.ne, "rs_nominal": hist.rs, "record_spacing_lattice_time": hist.tau, "n_records": int(len(hist.T))}
        log("history: %d sites (engine %d), r_s %g a, %d records, spacing %g (lattice time) = %g r_s; the dense root from %s" % (
            len(hist.R), hist.ne, hist.rs, len(hist.T), hist.tau, hist.tau / hist.rs, "the engine's field solve" if a.dense_root_from_engine else "the record's trapping zero"))
        O = np.load(a.columns_npz)
        n = len(hist.R)
        for tg in tags:
            if ("T_%s" % tg) not in O.files:
                log("tag %s: no T_%s in the columns npz, skipped" % (tg, tg))
                continue
            T_rs = float(O["T_%s" % tg])
            for kap in kappas:
                key = "%s_k%d" % (tg, kap)
                kd = "cols_dust_%s_k%d" % (tg, kap)
                if kd not in O.files:
                    log("%s: %s not in the columns npz, skipped" % (key, kd))
                    continue
                A_ = np.asarray(O[kd], np.complex128)
                f = np.asarray(O["f_dust_%s_k%d" % (tg, kap)], float)
                X0 = np.zeros((2 * n, A_.shape[1]), np.complex128)
                X0[: A_.shape[0]] = A_
                res, arrs = run_key(hist, ops, key, a.tag, kap, X0, f, T_rs, a.m, a, a.out_dir, overrides.get(key), BV2)
                results[key] = res
                arrays_all[key] = arrs
        O.close()
    pjs = load_producer_jsons(a.producer_json)
    pnz = load_producer_npzs(a.producer_npz)
    comps = {}
    for key, res in results.items():
        comps[key] = compare_key(res, arrays_all[key], pjs, pnz, a.tol)
        log(verdict_line(comps[key], a.tol))
    prov["producer_json"] = [{"path": p, "script": lab, "error": err} for lab, p, _, err in pjs]
    prov["producer_npz"] = [{"path": p, "error": err} for lab, p, _, err in pnz]
    prov["utc_end"] = utc()
    out = {"script": "dust_readouts_v1.py", "provenance": prov, "bins": BINS, "keys": results, "comparisons": comps,
           "verdict": {k_: {"within_tol": c_["within_tol"], "max_abs_diff_counts": c_["max_abs_diff_counts"], "max_abs_diff_currents": c_["max_abs_diff_currents"],
                            "clocks_equal": c_["clocks_equal"]} for k_, c_ in comps.items()}}
    atomic_write_text(combined["json"], json.dumps(out, default=jdefault, indent=1))
    flat = {}
    for key, arrs in arrays_all.items():
        for nm, arr in arrs.items():
            flat["%s_%s" % (key, nm)] = arr
    atomic_savez(combined["npz"], **flat)
    atomic_write_text(combined["md"], compose_md(prov, results, comps, a.tol))
    log("wrote %s, %s and %s" % (combined["json"], combined["npz"], combined["md"]))


if __name__ == "__main__":
    main()
