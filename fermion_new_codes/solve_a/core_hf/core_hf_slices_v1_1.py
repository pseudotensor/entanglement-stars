#!/usr/bin/env python3
"""
core_hf_slices_v1.py (solve lane; CORE_HF, SPEC_core_hf_v1.md beside this file): the core's Hartree-Fock level on the collapse's stored
core slices: (1) the state per channel from the slice, (2) the record's Gauss rows re-solved with the native Gaussian kernel in
hf_star_variance_v3's scheme (B = dPhi/ds, b(R_j) = dM(R_j)/ds, the field check), (3) the sea's response P^A as an explicit matrix by the
exponential sum, (4) the differentiated-vertex variance Var_C(dGamma(A_beta)) = sum_k 16 kappa beta^T G_k beta with the Slater Grams
G_k = Re Tr(C A_mu (1 - C) A_nu), the sea part with P_ref, the signed excess, (5) the Fock term E_F = -(1/2) sum_k 16 kappa <V, T_k>,
T_k = Tr(A_mu D_k A_nu D_k), with the bare (B) and dressed ((I + B P^A)^-1 B) native kernels, the Hartree pair energy (1/2) s^T V s and the
small-cluster correlation bound quoted beside, (6) both evolution rows' residuals at the core nodes (engine_rows_v2 on the history's own
field module and the handoff's stresses) and the rows' defects W_0, W_r by the spherical Bianchi identities.

Imports by path, edits nothing: hf_star_variance_v3 (radial_solve, linearized, complex_step_columns, gmres_solve, ResponseOperator,
accumulate_response, simpson_weights, ShellKernel, load_star, rebuild_channels, variance_of_vertex), engine_ops_v1 (chain_parts, clocked,
site_density, sources_dynamic), engine_fields_v2 (kernels; the formula re-typed as the fallback and tested against the import),
engine_rows_v2 + engine_fields_v14 | v16 + engine_stress_v1 (the evolution stage), slices144_io_v1 (the self-test's reader cross-check only).
The slices are read by a built-in zip-offset memory-mapped reader (the same layout as slices144_io_v1.SliceReader).

Data only: every number to <out>/<history>/core_hf_<history>_T<TTTT.TT>.json, every array to _arrays.npz; no plots (core_hf_table_v1.py
reads them). --device cpu (numpy, scipy; at most 16 threads; renices itself) or cuda (torch; the same arithmetic in complex128).
"""
import os
import sys


def _threads():
    t = '16'
    for i, x in enumerate(sys.argv):
        if x == '--threads' and i + 1 < len(sys.argv):
            t = sys.argv[i + 1]
        elif x.startswith('--threads='):
            t = x.split('=', 1)[1]
    t = str(max(1, min(int(t), 16)))
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = t


_threads()
import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from numpy.lib import format as npf  # noqa: E402

VERSION = 'core_hf_slices_v1_1'   # v1.1 (20 Oct, #167): additive controls only; every v1 number is computed by the v1 code path unchanged
FOURPI = 4.0 * math.pi
HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

# ============================================================================ locations (the box; every path overridable)
BOX_D = '<internal>/solve_a_bulk'
BOX_E = '<internal>/engine'
ENGINE_CODE = BOX_E + '/code'
SLICES144_DIR = BOX_E + '/slices144'
S10_DIR = '<internal>/hf'
OUT_DIR = BOX_D + '/core_hf_v1'
HIST = {
    'prod': dict(tag='c19_rs64_A_R2_tau8_ell4_avg_r11', label='the production (r_s/a 64, tau 8, sigma/a 4)', base=BOX_E + '/H100b/C3prod19',
                 fields_module='engine_fields_v14', well=BOX_D + '/core_well_v1/well', clocks=[12, 15, 16, 20, 24, 25, 28, 30, 35, 40],
                 slice_dirs=[BOX_D + '/engine/H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11', BOX_D + '/pod_sync/pod1_engine/states/c19_rs64_A_R2_tau8_ell4_avg_r11',
                             BOX_E + '/H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11']),
    's8': dict(tag='c27_rs64_A_R2_tau8_ell8_prep4_avg', label='the sigma/a 8 member (r_s/a 64, tau 8, ell 8, prep 4)', base=BOX_E + '/H100b/C3lad27',
               fields_module='engine_fields_v16', well=BOX_D + '/core_well_v1/well', clocks=[12, 13, 14, 16, 20, 25, 30, 40],
               slice_dirs=[BOX_D + '/engine/H100b/states/c27_rs64_A_R2_tau8_ell8_prep4_avg', BOX_E + '/H100b/states/c27_rs64_A_R2_tau8_ell8_prep4_avg']),
}
SIZES = {('c128', 2048): 2685802106, ('c64', 2048): 1343100290}
C64_FLOOR = dict(source_rel_range=[6.2e-6, 8.7e-6], orthonormality_defect=6.5e-8, complex128_reference=3.9e-14,
                 source='results/engine/C3ext19/slice_restart_check_v1.md (slices144_io_v1.C64_FLOOR)')
CLUSTER_CORRELATION = dict(rel=2.7e-4, source='05_star.tex: unsmoothed monopole radial-chain controls, Hartree-Fock trial energies exceed each '
                                               "sector's exact ground-state energy by at most 2.7e-4 |E_H| (largest gap at 12 shells); a cluster "
                                               "control, not the core's own")


# ============================================================================ helpers
def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:8]


CODE_MD5_AT_START = md5(os.path.abspath(__file__))   # 20 Oct: stamped when the module loads, so an edit during a run cannot relabel it


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


def fnorm(X):
    if sp.issparse(X):
        return float(np.sqrt(np.sum(np.abs(X.data) ** 2)))
    return float(np.linalg.norm(np.asarray(X).ravel()))


def import_by_path(name, dirs):
    for d in dirs:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    return importlib.import_module(name)


class Mods:
    """the modules imported by name from the given directories; S10 and OPS required, the rest optional."""

    def __init__(self, s10_dir, engine_dir, slices_dir=None):
        self.S10 = import_by_path('hf_star_variance_v3', [s10_dir])
        self.OPS = import_by_path('engine_ops_v1', [engine_dir])
        self.OPS.GPU['dev'] = None
        self.EF2 = self.RW = self.STR = self.IO = None
        self.err = {}
        for key, name, dirs in (('EF2', 'engine_fields_v2', [engine_dir]), ('RW', 'engine_rows_v2', [engine_dir]),
                                ('STR', 'engine_stress_v1', [engine_dir]), ('IO', 'slices144_io_v1', [slices_dir])):
            try:
                setattr(self, key, import_by_path(name, dirs))
            except Exception as ex:  # noqa: BLE001
                self.err[name] = '%s: %s' % (type(ex).__name__, ex)
        self.engine_dir = engine_dir
        self.paths = {k: getattr(getattr(self, k), '__file__', None) for k in ('S10', 'OPS', 'EF2', 'RW', 'STR', 'IO') if getattr(self, k) is not None}
        self.md5 = {k: md5(p) for k, p in self.paths.items() if p and os.path.isfile(p)}

    def fields_module(self, name):
        return import_by_path(name, [self.engine_dir])


# ============================================================================ the backend (numpy or torch)
class Backend:
    """dense linear algebra on the CPU (numpy; h, Q, h_m sparse CSR) or on a CUDA device (torch; everything dense complex128)."""

    def __init__(self, device='cpu'):
        self.device = device
        self.gpu = device != 'cpu'
        self.torch = None
        if self.gpu:
            import torch
            self.torch = torch
            self.dev = torch.device(device)

    # ---- transfer
    def to(self, a, dtype=np.complex128):
        if self.gpu:
            if isinstance(a, self.torch.Tensor):
                return a
            arr = np.ascontiguousarray(np.asarray(a, dtype))
            if not arr.flags.writeable:
                arr = arr.copy()                       # a memory-mapped (read-only) input: torch.as_tensor would warn
            return self.torch.as_tensor(arr, device=self.dev)
        return np.asarray(a, dtype)

    def host(self, a):
        if self.gpu and isinstance(a, self.torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    # ---- operators (h, Q, h_m, the vertices): banded 2x2-block operators on either device (BandOp)
    def op(self, M):
        return BandOp.from_dense(M, self)

    def apply(self, A, X):
        if isinstance(A, BandOp):
            return A.left(X)
        return A @ X

    def zeros_like(self, X):
        if self.gpu:
            return self.torch.zeros_like(X)
        return np.zeros_like(X)

    def stack(self, parts, axis):
        if self.gpu:
            return self.torch.stack(parts, dim=axis)
        return np.stack(parts, axis=axis)

    def adjT3(self, B):
        """the per-block conjugate transpose of a (n, 2, 2) stack."""
        if self.gpu:
            return B.conj().transpose(1, 2)
        return B.conj().transpose(0, 2, 1)

    def mm(self, X, Y):
        return X @ Y

    def adj(self, X):
        if self.gpu:
            return X.conj().transpose(0, 1)
        return X.conj().T

    def tr(self, X):
        if self.gpu:
            return X.transpose(0, 1)
        return X.T

    def eye(self, n):
        if self.gpu:
            return self.torch.eye(n, dtype=self.torch.complex128, device=self.dev)
        return np.eye(n, dtype=np.complex128)

    def fro2(self, X):
        if self.gpu:
            return float(self.torch.sum(X.real ** 2 + X.imag ** 2).item()) if X.is_complex() else float(self.torch.sum(X ** 2).item())
        return float(np.vdot(X, X).real)

    def herm(self, X):
        return 0.5 * (X + self.adj(X))

    def eigh(self, H, driver='evr'):
        """(e ascending as numpy float64, U on the device) of the Hermitian H (numpy or device). CPU: scipy evr for h_m (S10's choice),
        evd (divide and conquer; orthogonal eigenvectors on clustered spectra) for the Loewdin Grams."""
        if self.gpu:
            Hd = self.to(H)
            e, U = self.torch.linalg.eigh(0.5 * (Hd + Hd.conj().transpose(0, 1)))
            return self.host(e).astype(float), U
        Hn = np.asarray(H, np.complex128)
        e, U = sla.eigh(0.5 * (Hn + Hn.conj().T), driver=driver, check_finite=False)
        return np.asarray(e, float), np.asarray(U, np.complex128)

    def eigvalsh(self, H):
        if self.gpu:
            return self.host(self.torch.linalg.eigvalsh(self.to(H))).astype(float)
        return np.asarray(sla.eigvalsh(np.asarray(H, np.complex128), driver='evr', check_finite=False), float)

    def scale_cols(self, X, s):
        """X diag(s) (s real)."""
        if self.gpu:
            return X * self.torch.as_tensor(np.asarray(s, float), device=self.dev).to(X.dtype)[None, :]
        return X * np.asarray(s, float)[None, :]

    def vertex(self, ch, x, ns):
        """A_x = (1/2){D_e, h} + (1/2){D_p, Q}, D_e = diag(x[:ns] per component), D_p = diag(x[ns:] per component) (S10's convention):
        the block at (n, n + k) of h scaled by (x_n + x_{n+k})/2, likewise Q with x_p."""
        x = np.asarray(x, float)
        return ch['h_op'].scaled(x[:ns]).add(ch['Q_op'].scaled(x[ns:]))

    def block_sum_conj(self, X, Y, ns):
        """sum_{i in n, j in n'} X_ij conj(Y_ij) (2 components per site), on the device: S(X, Z) with Z^T = conj(Y), contiguous."""
        Z = X * Y.conj()
        if self.gpu:
            return Z.reshape(ns, 2, ns, 2).sum(dim=(1, 3))
        return Z.reshape(ns, 2, ns, 2).sum(axis=(1, 3))

    def block_sum(self, X, Y, ns):
        """S(X, Y)[n, n'] = sum_{i in n, j in n'} X_ij Y_ji (2 components per site), on the device (the transposed form; the self-test's reference)."""
        if self.gpu:
            Z = X * Y.transpose(0, 1)
            return Z.reshape(ns, 2, ns, 2).sum(dim=(1, 3))
        Z = np.multiply(X, Y.T)
        return Z.reshape(ns, 2, ns, 2).sum(axis=(1, 3))

    def site_density(self, A, Phi, f=None):
        """sum_k f_k Re(Phi_k^+ Pi_x A Phi_k) per site (engine_ops_v1.site_density's arithmetic)."""
        AP = self.apply(A, Phi)
        if f is not None:
            AP = self.scale_cols(AP, f)
        if self.gpu:
            d = self.torch.sum((Phi.conj() * AP).real, dim=1)
            return self.host(d).reshape(-1, 2).sum(axis=1)
        d = np.real(np.sum(np.conj(Phi) * AP, axis=1))
        return d.reshape(-1, 2).sum(axis=1)

    def loewdin(self, X):
        """X (X^+ X)^(-1/2); returns (Xo, the input's defect max|G - I|, the output's defect)."""
        X = self.to(X)
        n = X.shape[1]
        if n == 0:
            return X, 0.0, 0.0
        G = self.herm(self.mm(self.adj(X), X))
        d0 = float(self.host(abs(G - self.eye(n))).max())
        w, V = self.eigh(G, driver='evd')
        w = np.clip(w, 1e-300, None)
        Xo = self.mm(X, self.mm(self.scale_cols(V, 1.0 / np.sqrt(w)), self.adj(V)))
        G1 = self.mm(self.adj(Xo), Xo)
        d1 = float(self.host(abs(G1 - self.eye(n))).max())
        if d1 > 4e-15:
            w, V = self.eigh(G1, driver='evd')
            w = np.clip(w, 1e-300, None)
            Xo = self.mm(Xo, self.mm(self.scale_cols(V, 1.0 / np.sqrt(w)), self.adj(V)))
            G1 = self.mm(self.adj(Xo), Xo)
            d1 = float(self.host(abs(G1 - self.eye(n))).max())
        return Xo, d0, d1


# ============================================================================ banded chain operators
class BandOp:
    """a Hermitian operator on the 2-component chain as 2x2 site blocks per offset k = 0, 1, 2: M = B_0 (site diagonal, Hermitian) +
    sum_{k>0} [B_k at (n, n + k)] + h.c.; applied on the left (M X) and on the right (X M) by broadcast block products (no dense or
    sparse matrix product; the same code on numpy and torch). engine_ops_v1's h (offsets 0, 1), Q (offset 2), h_m (0, 1, 2) and every
    site vertex (1/2){Pi_n, h_t} and their combinations are of this form."""
    KMAX = 2

    def __init__(self, ns, blocks, LA):
        self.ns, self.LA = ns, LA
        self.blocks = blocks          # {k: (ns - k, 2, 2) complex on the device}

    @classmethod
    def from_dense(cls, M, LA, kmax=KMAX):
        M = np.asarray(M, np.complex128)
        ns = M.shape[0] // 2
        M4 = M.reshape(ns, 2, ns, 2)
        blocks = {}
        mass = 0.0
        asym = 0.0
        for k in range(kmax + 1):
            idx = np.arange(ns - k)
            Bk = np.ascontiguousarray(M4[idx, :, idx + k, :])
            if k == 0:
                asym = max(asym, float(np.max(np.abs(Bk - Bk.conj().transpose(0, 2, 1)))) if ns else 0.0)
            else:
                Lk = M4[idx + k, :, idx, :]
                asym = max(asym, float(np.max(np.abs(Lk - Bk.conj().transpose(0, 2, 1)))) if ns - k else 0.0)
            mass += (1.0 if k == 0 else 2.0) * float(np.sum(np.abs(Bk) ** 2))
            blocks[k] = LA.to(Bk)
        tot = float(np.sum(np.abs(M) ** 2))
        if abs(tot - mass) > 1e-12 * max(tot, 1e-300) or asym > 1e-13 * max(np.sqrt(tot), 1e-300):
            raise RuntimeError('BandOp.from_dense: the matrix is not a Hermitian band of block width %d (mass defect %.1e, asym %.1e)' % (kmax, abs(tot - mass), asym))
        return cls(ns, blocks, LA)

    def _bl(self, B, X0, X1):
        """Y[n, a, :] = sum_b B[n, a, b] X[n, b, :] for X given as its two component rows."""
        return (B[:, 0, 0, None] * X0 + B[:, 0, 1, None] * X1, B[:, 1, 0, None] * X0 + B[:, 1, 1, None] * X1)

    def _br(self, X0, X1, B):
        """Z[:, n, b] = sum_a X[:, n, a] B[n, a, b] for X given as its two component columns."""
        return (X0 * B[None, :, 0, 0] + X1 * B[None, :, 1, 0], X0 * B[None, :, 0, 1] + X1 * B[None, :, 1, 1])

    def left(self, X):
        """M X."""
        LA, ns = self.LA, self.ns
        m = X.shape[1]
        X3 = X.reshape(ns, 2, m)
        X0, X1 = X3[:, 0, :], X3[:, 1, :]
        Y0, Y1 = LA.zeros_like(X0), LA.zeros_like(X1)
        for k, B in self.blocks.items():
            if k == 0:
                a, b = self._bl(B, X0, X1)
                Y0 = Y0 + a
                Y1 = Y1 + b
            else:
                a, b = self._bl(B, X0[k:], X1[k:])
                Y0[:ns - k] += a
                Y1[:ns - k] += b
                a, b = self._bl(LA.adjT3(B), X0[:ns - k], X1[:ns - k])
                Y0[k:] += a
                Y1[k:] += b
        return LA.stack([Y0, Y1], 1).reshape(2 * ns, m)

    def right(self, X):
        """X M."""
        LA, ns = self.LA, self.ns
        m = X.shape[0]
        X3 = X.reshape(m, ns, 2)
        X0, X1 = X3[:, :, 0], X3[:, :, 1]
        Z0, Z1 = LA.zeros_like(X0), LA.zeros_like(X1)
        for k, B in self.blocks.items():
            if k == 0:
                a, b = self._br(X0, X1, B)
                Z0 = Z0 + a
                Z1 = Z1 + b
            else:
                a, b = self._br(X0[:, :ns - k], X1[:, :ns - k], B)
                Z0[:, k:] += a
                Z1[:, k:] += b
                a, b = self._br(X0[:, k:], X1[:, k:], LA.adjT3(B))
                Z0[:, :ns - k] += a
                Z1[:, :ns - k] += b
        return LA.stack([Z0, Z1], 2).reshape(m, 2 * ns)

    def scaled(self, c):
        """(1/2){diag(c per component), M}: the block at (n, n + k) times (c_n + c_{n+k})/2 (c real per site)."""
        c = np.asarray(c, float)
        out = {}
        for k, B in self.blocks.items():
            w = 0.5 * (c[:self.ns - k] + c[k:]) if k else c
            out[k] = B * self.LA.to(w)[:, None, None]
        return BandOp(self.ns, out, self.LA)

    def add(self, other, sign=1.0):
        out = {}
        for k in set(self.blocks) | set(other.blocks):
            if k in self.blocks and k in other.blocks:
                out[k] = self.blocks[k] + sign * other.blocks[k]
            elif k in self.blocks:
                out[k] = self.blocks[k]
            else:
                out[k] = sign * other.blocks[k]
        return BandOp(self.ns, out, self.LA)

    def fro2(self):
        return float(sum((1.0 if k == 0 else 2.0) * self.LA.fro2(B) for k, B in self.blocks.items()))

    def to_dense(self):
        ns = self.ns
        M = np.zeros((2 * ns, 2 * ns), np.complex128)
        M4 = M.reshape(ns, 2, ns, 2)
        for k, B in self.blocks.items():
            Bh = self.LA.host(B)
            idx = np.arange(ns - k)
            M4[idx, :, idx + k, :] += Bh
            if k:
                M4[idx + k, :, idx, :] += Bh.conj().transpose(0, 2, 1)
        return M


# ============================================================================ the slice reader (the engine's npz layout)
_LFH = struct.Struct('<4s2B4HL2L2H')


def zip_member_offsets(path):
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, 'rb') as f:
        for info in zf.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError('%s: member %s is compressed; the slices are stored uncompressed' % (path, info.filename))
            f.seek(info.header_offset)
            hdr = _LFH.unpack(f.read(_LFH.size))
            if hdr[0] != b'PK\x03\x04':
                raise ValueError('%s: no local file header for %s' % (path, info.filename))
            f.seek(info.header_offset + _LFH.size + hdr[10] + hdr[11])
            version = npf.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = npf.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = npf.read_array_header_2_0(f)
            else:
                shape, fortran, dtype = npf._read_array_header(f, version)
            key = info.filename[:-4] if info.filename.endswith('.npy') else info.filename
            out[key] = dict(offset=int(f.tell()), shape=tuple(int(s) for s in shape), dtype=np.dtype(dtype), fortran=bool(fortran))
    return out


class SliceFile:
    """one stored slice: the scalars and site arrays loaded, the column arrays memory-mapped at their zip offsets."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.members = zip_member_offsets(self.path)
        with np.load(self.path, allow_pickle=False) as Z:
            self.T = float(Z['T'])
            self.rs = float(Z['rs'])
            self.R = np.asarray(Z['R'], float).reshape(-1)
            self.N = np.asarray(Z['N_applied'], float).reshape(-1)
            self.v = np.asarray(Z['v_applied'], float).reshape(-1)
            kap = np.atleast_1d(Z['kappas'])
            self.kappas = [int(k) for k in kap]
            self.weights = {int(k): float(w) for k, w in zip(kap, np.atleast_1d(Z['weights']))}
            self.dtype_str = str(Z['dtype']) if 'dtype' in self.members else None
            self.f = {k: np.asarray(Z['f_%d' % k], float).reshape(-1) for k in self.kappas}
            self.e_stored = {k: (np.asarray(Z['e_%d' % k], float).reshape(-1) if ('e_%d' % k) in self.members else None) for k in self.kappas}
            self.mask_stored = {}
            for k in self.kappas:
                if ('mask_%d' % k) in self.members:
                    self.mask_stored[k] = np.unpackbits(np.asarray(Z['mask_%d' % k], np.uint8))[: 2 * len(self.R)].astype(bool)
                else:
                    self.mask_stored[k] = None
        self.T_over_rs = self.T / self.rs
        self.nsites = len(self.R)
        self.n2 = 2 * self.nsites
        self.n_sea = {k: int(self.members['sea_%d' % k]['shape'][1]) for k in self.kappas}
        self.n_dust = {k: (int(self.members['dust_%d' % k]['shape'][1]) if ('dust_%d' % k) in self.members else 0) for k in self.kappas}
        self.column_dtype = self.members['sea_%d' % self.kappas[0]]['dtype']
        self.complex64 = self.column_dtype == np.dtype(np.complex64)
        self.size = os.path.getsize(self.path)
        self.floor = dict(C64_FLOOR, applies=bool(self.complex64))

    def mmap(self, key):
        mb = self.members[key]
        return np.memmap(self.path, dtype=mb['dtype'], mode='r', offset=mb['offset'], shape=mb['shape'], order='F' if mb['fortran'] else 'C')

    def columns(self, k):
        """Phi_raw = [sea_k | dust_k] in complex128 and f = [1, ..., 1, f_k]."""
        n_s, n_d = self.n_sea[k], self.n_dust[k]
        Phi = np.empty((self.n2, n_s + n_d), np.complex128)
        Phi[:, :n_s] = self.mmap('sea_%d' % k)
        if n_d:
            Phi[:, n_s:] = self.mmap('dust_%d' % k)
        return Phi, np.concatenate([np.ones(n_s), self.f[k]])

    def dust(self, k):
        if self.n_dust[k] == 0:
            return np.zeros((self.n2, 0), np.complex128), self.f[k]
        return np.asarray(self.mmap('dust_%d' % k), np.complex128), self.f[k]

    def R2(self, k):
        return np.asarray(self.mmap('R2_%d' % k), np.complex128)


def find_slice(history, t, slice_dir=None):
    H = HIST[history]
    name = 'orbitals_%s_T%07.2f.npz' % (H['tag'], float(t))
    cands = ([os.path.join(slice_dir, name)] if slice_dir else []) + [os.path.join(d, name) for d in H['slice_dirs']]
    path = next((p for p in cands if os.path.isfile(p)), None)
    return path, name, cands


def load_record(records_json, fields_npz, t):
    J = json.load(open(records_json))
    A = J['args']
    rs = float(A['rs'])
    recs = [r for r in J['records'] if r.get('E') is not None]
    rec = None
    for r in recs:
        tr = float(r['T_over_rs']) if r.get('T_over_rs') is not None else float(r['T']) / rs
        if abs(tr - float(t)) < 1e-6:
            rec = r
    if rec is None:
        raise RuntimeError('%s: no record at T/r_s %g' % (records_json, t))
    keys = ('T', 'T_over_rs', 'E', 'E_over_E0', 'E_dust_shells', 'E_pairs_shells', 'trapping_zeros', 'r_in', 'R_t', 'R_c', 'misner_sharp_at_Rt',
            'converged', 'n_pocket', 'inversions_total', 'intermediate_weights_total', 'mask_equals_negative_set', 'min_occupied_weight', 'N_min',
            'continuation_label', 'commutator_h_R2_F')
    out = dict(records_json=records_json, fields_npz=fields_npz, m=float(A['m']), g=float(A['g']), ell=float(A['ell']), rs=rs, tau=float(A.get('tau', 8.0)),
               half=bool(A.get('half_offset', 0)), kappa_sea=A.get('kappa_sea'), collar_policy=A.get('collar_policy', 'flag'),
               support_rel=float(A.get('support_rel', 1e-14)), script=J.get('script', ''), record={k: rec.get(k) for k in keys})
    out['record']['gauge_label'] = (rec.get('gauge') or {}).get('label')
    sv = rec.get('solver') or {}
    out['record']['solver_residual_W'] = sv.get('residual_W')
    out['record']['solver_residual_invN'] = sv.get('residual_invN')
    Z = np.load(fields_npz, allow_pickle=True)
    Tz = np.asarray(Z['T'], float)
    i = int(np.argmin(np.abs(Tz / rs - float(t))))
    if abs(Tz[i] / rs - float(t)) > 1e-6:
        raise RuntimeError('%s: no fields row at T/r_s %g' % (fields_npz, t))
    out['index'] = i
    out['T_fields'] = Tz
    for k in ('eps_tot', 'J_tot', 'eps_pairs', 'J_pairs', 'N_applied', 'v_applied', 'N', 'v'):
        if k in Z.files:
            out[k] = np.asarray(Z[k][i], float)
    out['fields_ell'] = float(Z['ell']) if 'ell' in Z.files else None
    return out


def load_well(well_dir, history, t):
    p = os.path.join(well_dir, 'well_%s_T%07.2f.json' % (history, float(t)))
    if not os.path.isfile(p):
        return None
    W = json.load(open(p))
    out = dict(path=p, roots=W.get('roots', {}), dust=W.get('dust', {}), record=W.get('record', {}), gate=W.get('gate'), limit_label=W.get('limit_label'),
               channels=W.get('channels', []), arrays={})
    pz = p[:-5] + '.npz'
    if os.path.isfile(pz):
        Z = np.load(pz)
        for k in ('dust_enclosed', 'M_w_dense', 'M_w_applied', 'N_dense_sites', 'v_dense_sites', 'rho_dust_density'):
            if k in Z.files:
                out['arrays'][k] = np.asarray(Z[k], float)
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


def frac_radius(R, cum, q):
    i = int(np.searchsorted(cum, q))
    if i <= 0:
        return float(R[0])
    if i >= len(R):
        return float(R[-1])
    c0, c1 = cum[i - 1], cum[i]
    return float(R[i - 1] + (q - c0) / (c1 - c0) * (R[i] - R[i - 1])) if c1 > c0 else float(R[i])


# ============================================================================ the native kernel
def gauss_kernels_formula(r, R, ell):
    """engine_fields_v2.kernels re-typed: K_0, K_1 of the 3D Gaussian of standard deviation ell per axis (the l = 0 and l = 1 shell averages)."""
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
    """hf_star_variance_v3.ShellKernel's interface (r, rf, W, Wf, ns, dr, K0, K0f, K1, K1f, norm_raw, norm_err, mom1_err) with the engine's
    Gaussian kernels: the l = 0 tables normalized column by column under the two Simpson quadratures, K1 unnormalized."""

    def __init__(self, Rs, ell, Rout, dr, S10, EF2=None, chunk=2_000_000):
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
        step = max(1, chunk // max(nR, 1))
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
        return GaussShellKernel(Rs, ell, Rout, dr, M.S10, M.EF2)
    kt = M.S10.ShellKernel(Rs, ell, Rout, dr)
    kt.kernel_source = "hf_star_variance_v3.ShellKernel (GAP06's compact kernel: a scheme control, never the core's number)"
    return kt


# ============================================================================ the bilinear block form and the exponential sum
def bilinear(LA, hs, X, Y, ns):
    """T[mu, nu] = Tr(A_mu X A_nu Y) for the site vertices A_mu = (1/2){Pi_n, h_t} (t = 0: h, t = 1: Q), X, Y Hermitian on the device;
    = (1/4)[S(h_t X, h_t' Y) + S(h_t X h_t', Y) + S(X, h_t' Y h_t) + S(X h_t', Y h_t)]; returns a host complex128 (2ns x 2ns), energy block first."""
    d = 2 * ns
    hX = [h.left(X) for h in hs]
    hY = [h.left(Y) for h in hs]
    T = np.empty((d, d), np.complex128)
    for t1 in range(2):
        for t2 in range(2):
            hp = hs[t2]
            # for Hermitian X, Y, h, h': S(hX, h'Y) = <hX, Yh'>, S(hXh', Y) = <hXh', Y>, S(X, h'Yh) = <X, hYh'>, S(X h', Y h) = <Xh', hY>,
            # <A, B>[n, n'] = sum_{i in n, j in n'} A_ij conj(B_ij) (every operand contiguous; no transposes)
            S = (LA.block_sum_conj(hX[t1], hp.right(Y), ns) + LA.block_sum_conj(hp.right(hX[t1]), Y, ns)
                 + LA.block_sum_conj(X, hp.right(hY[t1]), ns) + LA.block_sum_conj(hp.right(X), hY[t1], ns))
            T[t1 * ns:(t1 + 1) * ns, t2 * ns:(t2 + 1) * ns] = 0.25 * LA.host(S)
            del S
    return T


def exp_sum_nodes(xmin, xmax, tol=1e-13, h0=0.3, max_halvings=8):
    """1/x = sum_k c_k exp(-x t_k) on [xmin, xmax] by the double-exponential trapezoid rule in u, t = exp(u - exp(-u)); the step halved
    until the measured maximal relative error on a geometric grid of 4000 points is below tol. Returns (t, c, error, step)."""
    if not (0.0 < xmin <= xmax):
        raise ValueError('exp_sum_nodes needs 0 < xmin <= xmax')
    xs = np.geomspace(xmin, xmax, 4000)
    umax = 1.0
    while math.exp(-xmin * math.exp(umax - math.exp(-umax))) * math.exp(umax - math.exp(-umax)) * (1.0 + math.exp(-umax)) * xmax > 1e-3 * tol:
        umax += 0.25
    umin = -2.0
    while math.exp(umin - math.exp(-umin)) * (1.0 + math.exp(-umin)) * xmax > 1e-3 * tol:
        umin -= 0.25
    h = h0
    best = None
    for _ in range(max_halvings + 1):
        u = np.arange(umin, umax + 0.5 * h, h)
        t = np.exp(u - np.exp(-u))
        c = h * t * (1.0 + np.exp(-u))
        err = float(np.max(np.abs((np.exp(-np.outer(xs, t)) @ c) * xs - 1.0)))
        best = (t, c, err, h)
        if err < tol:
            break
        h *= 0.5
    return best


def response_matrix(LA, chans, ns, tol=1e-13, log_fn=log):
    """P^A[mu, nu] = sum_k w_k Tr(A_mu Pi_k(A_nu)), Pi the spectral derivative of the sea projector (the occupied-empty block
    1/(e_o - e_u)): = -2 sum_nodes c sum_k w Re Tr(A_mu E_o(t) A_nu E_u(t)), E_o = U_o e^{t e_o} U_o^+, E_u = U_u e^{-t e_u} U_u^+."""
    gaps = {ch['kappa']: float(ch['e'][ch['e'] > 0].min() - ch['e'][ch['e'] < 0].max()) for ch in chans}
    xmin = min(gaps.values())
    xmax = max(float(ch['e'].max() - ch['e'].min()) for ch in chans)
    t, c, err, h = exp_sum_nodes(xmin, xmax, tol)
    d = 2 * ns
    PA = np.zeros((d, d))
    per = {}
    t0 = time.time()
    for ch in chans:
        eo = ch['e'][ch['e'] < 0]
        eu = ch['e'][ch['e'] > 0]
        Uo, Uu = ch['Uo'], ch['Uu']
        hs = (ch['h_op'], ch['Q_op'])
        used = 0
        t1 = time.time()
        for tk, ck in zip(t, c):
            if ck * math.exp(-tk * gaps[ch['kappa']]) < 1e-18:
                continue
            Eo = LA.mm(LA.scale_cols(Uo, np.exp(tk * eo)), LA.adj(Uo))
            Eu = LA.mm(LA.scale_cols(Uu, np.exp(-tk * eu)), LA.adj(Uu))
            T = bilinear(LA, hs, Eo, Eu, ns)
            PA -= 2.0 * ck * ch['w'] * T.real
            used += 1
            del Eo, Eu, T
        per[ch['kappa']] = dict(nodes_used=used, gap=gaps[ch['kappa']], seconds=time.time() - t1)
        log_fn('    response: kappa %d, %d of %d nodes, gap %.4f, %.0f s' % (ch['kappa'], used, len(t), gaps[ch['kappa']], time.time() - t1))
    asym = fnorm(PA - PA.T) / max(fnorm(PA), 1e-300)
    PA = 0.5 * (PA + PA.T)
    info = dict(nodes=int(len(t)), step=h, quadrature_max_rel_err=err, xmin=xmin, xmax=xmax, per_channel=per, asymmetry_raw=asym, seconds=time.time() - t0,
                PAnorm=fnorm(PA))
    return PA, info


def response_matvec(LA, chans, ns, x):
    """(P^A x) by the operator route (S10's ResponseOperator arithmetic on the device): sum_k w [Tr(A_mu Pi_k(A_x))]_mu."""
    x = np.asarray(x, float).ravel()
    y = np.zeros(2 * ns)
    for ch in chans:
        A = LA.vertex(ch, x, ns)
        AU = LA.apply(A, ch['Uu'])
        At = LA.mm(LA.adj(ch['Uo']), AU)
        X = LA.mm(LA.mm(ch['Uo'], At * ch['W_ou']), LA.adj(ch['Uu']))
        for t, op in ((0, ch['h_op']), (1, ch['Q_op'])):
            hX = op.left(X)
            hXd = op.left(LA.adj(X))
            if LA.gpu:
                dv = LA.host((LA.torch.diagonal(hX) + LA.torch.diagonal(hXd)).real)
            else:
                dv = np.real(np.diagonal(hX) + np.diagonal(hXd))
            y[t * ns:(t + 1) * ns] += ch['w'] * dv.reshape(-1, 2).sum(axis=1)
            del hX, hXd
        del A, AU, At, X
    return y


# ============================================================================ the channel
def chain_ops(OPS, ns, m, kappa, half, R):
    h, Q, Rc = OPS.chain_parts(ns, m, float(kappa), half)
    if np.max(np.abs(Rc - R)) > 1e-12:
        raise RuntimeError('engine_ops_v1 chain radii differ from the slice R (half_offset %s)' % half)
    return np.asarray(h), np.asarray(Q)


def build_channel(LA, M, sl, kappa, m, half, controls=True, floor_control=None, eig_d=False):
    """everything of SPEC section 2 for one channel: the operators, eigh(h_m), the Loewdin state, P_ref, D, the sources, the Grams
    G_C = Re Tr(C A (1 - C) A), G_P (with P_ref), the exchange block T = Re Tr(A D A D), the checks."""
    t0 = time.time()
    OPS = M.OPS
    ns, w = sl.nsites, sl.weights[kappa]
    h, Q = chain_ops(OPS, ns, m, kappa, half, sl.R)
    hm = OPS.clocked(h, Q, sl.N, sl.v)
    ch = dict(kappa=int(kappa), w=float(w), n_sea=sl.n_sea[kappa], n_dust=sl.n_dust[kappa])
    ch['h_op'], ch['Q_op'], ch['hm_op'] = LA.op(h), LA.op(Q), LA.op(hm)
    t1 = time.time()
    e, U = LA.eigh(hm)
    ch['eigh_seconds'] = time.time() - t1
    del hm, h, Q
    ch['e'] = e
    mask = e < 0
    ch['mask'] = mask
    ch['Uo'] = LA.to(U[:, mask]) if not LA.gpu else U[:, LA.torch.as_tensor(mask, device=LA.dev)].contiguous()
    ch['Uu'] = LA.to(U[:, ~mask]) if not LA.gpu else U[:, LA.torch.as_tensor(~mask, device=LA.dev)].contiguous()
    del U
    eo, eu = e[mask], e[~mask]
    ch['W_ou'] = LA.to(1.0 / (eo[:, None] - eu[None, :]))
    ch['gap'] = float(eu.min() - eo.max())
    ch['n_neg'] = int(mask.sum())
    es, ms = sl.e_stored[kappa], sl.mask_stored[kappa]
    ch['e_vs_stored_abs'] = float(np.max(np.abs(es - e))) if es is not None else None
    ch['mask_equals_neg'] = bool(ms is not None and np.array_equal(ms, mask))
    ch['stored_mask_equals_stored_neg'] = bool(ms is not None and es is not None and np.array_equal(ms, es < 0))
    # the state
    Phi_raw, f = sl.columns(kappa)
    ch['f_min'], ch['f_max'] = float(f.min()), float(f.max())
    ch['pure'] = bool(np.all((f < 1e-12) | (f > 1.0 - 1e-12)))
    Phi, d0, d1 = LA.loewdin(Phi_raw)
    ch['stored_orthonormality'], ch['loewdin_orthonormality'] = d0, d1
    R2, r0, r1 = LA.loewdin(sl.R2(kappa))
    ch['ref_stored_orthonormality'], ch['ref_loewdin_orthonormality'] = r0, r1
    ch['Phi'], ch['Pref'] = Phi, R2
    ch['n_occ'] = int(Phi.shape[1])
    # the sources (Loewdin state and the raw columns), the dust part, the dust count per site
    rhoC = LA.site_density(ch['h_op'], Phi)
    jC = LA.site_density(ch['Q_op'], Phi)
    rhoR = LA.site_density(ch['h_op'], R2)
    jR = LA.site_density(ch['Q_op'], R2)
    ch['s'] = np.concatenate([w * (rhoC - rhoR), w * (jC - jR)])
    Praw = LA.to(Phi_raw)
    R2raw = LA.to(sl.R2(kappa))
    ch['s_raw'] = np.concatenate([w * (LA.site_density(ch['h_op'], Praw, f) - LA.site_density(ch['h_op'], R2raw)),
                                  w * (LA.site_density(ch['Q_op'], Praw, f) - LA.site_density(ch['Q_op'], R2raw))])
    dcols, fd = sl.dust(kappa)
    if dcols.shape[1]:
        Dc = LA.to(dcols)
        ch['s_dust_raw'] = np.concatenate([w * LA.site_density(ch['h_op'], Dc, fd), w * LA.site_density(ch['Q_op'], Dc, fd)])
        ch['dust_site'] = w * (np.abs(dcols) ** 2 @ fd).reshape(-1, 2).sum(axis=1)
        del Dc
    else:
        ch['s_dust_raw'] = np.zeros(2 * ns)
        ch['dust_site'] = np.zeros(ns)
    del dcols, R2raw
    # the reference's resolution, the commutator
    X = LA.mm(LA.adj(ch['Uu']), R2)
    leak = LA.fro2(X)
    del X
    ch['pref_resolution_defect'] = math.sqrt(2.0 * max(leak, 0.0)) / math.sqrt(max(R2.shape[1], 1))
    HP = LA.apply(ch['hm_op'], R2)
    res = HP - LA.mm(R2, LA.mm(LA.adj(R2), HP))
    hm_norm = math.sqrt(ch['hm_op'].fro2())
    ch['commutator_rel'] = math.sqrt(LA.fro2(res)) / hm_norm
    del HP, res
    # the vertex identity
    A = LA.vertex(ch, np.concatenate([sl.N, sl.v]), ns)
    ch['vertex_identity_rel'] = math.sqrt(A.add(ch['hm_op'], -1.0).fro2()) / hm_norm
    del A
    # C, P, D and the Grams
    t1 = time.time()
    C = LA.herm(LA.mm(Phi, LA.adj(Phi)))
    P = LA.herm(LA.mm(R2, LA.adj(R2)))
    D = C - P
    ch['TrD'] = float(LA.host(D).trace().real) if not LA.gpu else float(LA.torch.trace(D).real.item())
    ch['TrD2'] = LA.fro2(D)
    if eig_d:
        ev = LA.eigvalsh(D)
        ch['eigD'] = ev
        ch['n_eigD_above_1e-6'] = int(np.sum(np.abs(ev) > 1e-6))
    hs = (ch['h_op'], ch['Q_op'])
    I = LA.eye(2 * ns)
    GC = bilinear(LA, hs, I - C, C, ns)
    GP = bilinear(LA, hs, I - P, P, ns)
    TD = bilinear(LA, hs, D, D, ns)
    ch['gram_defects'] = dict(G_C=fnorm(GC - GC.conj().T) / max(fnorm(GC), 1e-300), G_P=fnorm(GP - GP.conj().T) / max(fnorm(GP), 1e-300),
                              T=fnorm(TD - TD.conj().T) / max(fnorm(TD), 1e-300))
    ch['G_C'], ch['G_P'], ch['T'] = np.ascontiguousarray(GC.real), np.ascontiguousarray(GP.real), np.ascontiguousarray(TD.real)
    del GC, GP, TD
    if controls:
        ch['T_P'] = np.ascontiguousarray(bilinear(LA, hs, P, P, ns).real)
        ch['T_C'] = np.ascontiguousarray(bilinear(LA, hs, C, C, ns).real)
    if floor_control is None:
        floor_control = sl.complex64
    if floor_control:
        Craw = LA.herm(LA.mm(Praw, LA.adj(Praw)))
        ch['G_C_raw'] = np.ascontiguousarray(bilinear(LA, hs, I - Craw, Craw, ns).real)
        del Craw
    del Praw
    # the static identity for h_m: Var_C(h_m) (Slater, residual form) against (1/2)||[h_m, C]||_F^2; the eigen-sea's Var_P(h_m) (= 0)
    HPhi = LA.apply(ch['hm_op'], Phi)
    rs_ = HPhi - LA.mm(Phi, LA.mm(LA.adj(Phi), HPhi))
    var_hm = LA.fro2(rs_)
    del HPhi, rs_
    AC = LA.apply(ch['hm_op'], C)
    comm = AC - LA.adj(AC)
    half_comm = 0.5 * LA.fro2(comm)
    del AC, comm
    ch['static_identity_hm'] = dict(var_C=var_hm, half_comm_sq=half_comm, rel_diff=abs(var_hm - half_comm) / max(var_hm, half_comm, 1e-300))
    Xs = LA.mm(LA.adj(ch['Uu']), LA.apply(ch['hm_op'], ch['Uo']))
    ch['var_sea_hm'] = LA.fro2(Xs)
    del Xs, C, P, D, I
    ch['gram_seconds'] = time.time() - t1
    ch['seconds'] = time.time() - t0
    return ch


def direct_variances(LA, ch, beta, ns):
    """the direct Slater forms for one coefficient vector: Var_C = ||(1 - C) A Phi||^2, Var_P = ||(1 - P) A R2||^2 (the Gram route's check)."""
    A = LA.vertex(ch, beta, ns)
    out = {}
    for key, cols in (('var_C', ch['Phi']), ('var_P', ch['Pref'])):
        AX = LA.apply(A, cols)
        res = AX - LA.mm(cols, LA.mm(LA.adj(cols), AX))
        out[key] = LA.fro2(res)
        out['scale_' + key] = LA.fro2(AX)          # ||A Phi||^2: the Gram route's roundoff is a fraction of this, not of the (cancelling) variance
        del AX, res
    return out


def gram_vs_direct(ch, beta, dv):
    """|Gram - direct| over max(direct, 1e-6 ||A Phi||^2) for Var_C and Var_P (the generator's Var_P is zero up to roundoff)."""
    rc = abs(dv['var_C'] - quad(ch['G_C'], beta)) / max(dv['var_C'], 1e-6 * dv['scale_var_C'], 1e-300)
    rp = abs(dv['var_P'] - quad(ch['G_P'], beta)) / max(dv['var_P'], 1e-6 * dv['scale_var_P'], 1e-300)
    return max(rc, rp)


# ============================================================================ the rows (S10's scheme with the native kernel)
def rows_stage(M, args, R, ell, e_src, j_src, G, N_app, v_app, E_ref, radii, ok_label='CORE'):
    S10 = M.S10
    ns = len(R)
    t0 = time.time()
    if args.kernel == 'compact' and args.star:
        Rout = float(R[-1] + ell + 1.0)
    else:
        Rout = float(R[-1] + 8.0 * ell + 1.0)
    dr = args.dr if args.dr else (0.05 if args.star else 0.1)
    kt = make_kernel(args.kernel, R, ell, Rout, dr, M)
    out = dict(kernel=kt.kernel_source, ell=ell, R_out=Rout, dr=kt.dr, grid=kt.ns, kernel_norm_raw=kt.norm_raw, kernel_norm_err=kt.norm_err,
               kernel_mom1_err=kt.mom1_err, kernel_seconds=time.time() - t0)
    log('  kernel %s: ell %g, R_out %.2f, dr %.4f, grid %d; norm_raw %.1e norm_err %.1e mom1_err %.1e (%.0f s)'
        % (kt.kernel_source, ell, Rout, kt.dr, kt.ns, kt.norm_raw, kt.norm_err, kt.mom1_err, out['kernel_seconds']))
    sol = S10.radial_solve(kt, np.asarray(e_src, float), np.asarray(j_src, float), G)
    Phi_res = sol['Phi'].real
    Phi_app = np.concatenate([N_app, v_app])
    dN = float(np.max(np.abs(sol['Nx'].real - N_app)) / np.max(np.abs(N_app)))
    dv = float(np.max(np.abs(sol['vx'].real - v_app)) / max(np.max(np.abs(v_app)), 1e-300))
    E_res = float(sol['E'].real)
    label = ok_label if max(dN, dv) < args.field_tol else 'SCHEME CONTROL'
    if args.kernel == 'compact' and not args.star:
        label = "SCHEME CONTROL (compact kernel; never the core's number)"
    out['field_check'] = dict(N_rel_maxdev=dN, v_rel_maxdev=dv, E_resolved=E_res, E_record=E_ref, E_rel_dev=(abs(E_res - E_ref) / abs(E_ref)) if E_ref else None,
                              label=label, field_tol=args.field_tol, N_center=float(sol['N'][0].real), wmax=float(np.max(np.abs(sol['w'].real))),
                              Phi_resolved_minus_applied_rel=fnorm(Phi_res - Phi_app) / fnorm(Phi_app))
    log('  re-solve: max rel dev N %.2e v %.2e -> %s; E resolved %.8f record %.8f (rel %s); N(0) %.5f' % (dN, dv, label, E_res, E_ref, out['field_check']['E_rel_dev'], sol['N'][0].real))
    order = list(radii)
    rows = {}
    for lab in order:
        rows[lab] = min(int(round(radii[lab]['r'] / kt.dr)), kt.ns)
    rows['R_out'] = kt.ns
    radii['R_out'] = dict(r=float(kt.r[-1]), site=ns - 1, source='the re-solve outer radius')
    order.append('R_out')
    cols = sorted(set([0, ns // 2, ns - 1, ns, ns + ns // 2, 2 * ns - 1])) if args.cs_cols else []
    t1 = time.time()
    B, bMr, bMc = S10.linearized(kt, sol, G, True, rows=[rows[l] for l in order], cols=cols)
    Bnc, bMr_nc, _ = S10.linearized(kt, sol, G, False, rows=[rows[l] for l in order], cols=[])
    hess = dict(seconds_linearized=time.time() - t1)
    if cols:
        Bcs, bMcs = S10.complex_step_columns(kt, np.asarray(e_src, float), np.asarray(j_src, float), G, cols)
        hess['lin_vs_cs_B'] = max(fnorm(B[:, nu] - Bcs[nu]) / max(fnorm(Bcs[nu]), 1e-300) for nu in cols)
        hess['lin_vs_cs_bM'] = max(fnorm(bMc[:, k] - bMcs[nu]) / max(fnorm(bMcs[nu]), 1e-300) for k, nu in enumerate(cols))
    hess['first_law_raw'] = fnorm(bMr[-1] - Phi_res) / fnorm(Phi_res)
    hess['asym_raw'] = fnorm(B - B.T) / max(fnorm(B), 1e-300)
    B = 0.5 * (B + B.T)
    Bnc = 0.5 * (Bnc + Bnc.T)
    hess['nocur_current_columns_zero'] = bool(fnorm(Bnc[:, ns:]) == 0.0)
    hess['Bnorm'] = fnorm(B)
    hess['seconds'] = time.time() - t1
    out['hessian'] = hess
    log('  Hessian: lin vs complex step B %s bM %s; first law %.1e; asym %.1e; ||B|| %.4e (%.0f s)'
        % (hess.get('lin_vs_cs_B'), hess.get('lin_vs_cs_bM'), hess['first_law_raw'], hess['asym_raw'], hess['Bnorm'], hess['seconds']))
    M_H = {l: float(sol['M'][rows[l]].real) for l in order}
    N_rows = {l: float(sol['N'][rows[l]].real) for l in order}
    for l in order:
        radii[l]['M_H'] = M_H[l]
        radii[l]['N_H'] = N_rows[l]
        radii[l]['grid_row'] = rows[l]
    arrays = dict(r_grid=kt.r, M_grid=sol['M'].real, N_grid=sol['N'].real, w_grid=sol['w'].real, rho_grid=sol['rho'].real, J_grid=sol['J'].real,
                  N_resolved_sites=sol['Nx'].real, v_resolved_sites=sol['vx'].real, Phi_resolved=Phi_res, Phi_applied=Phi_app)
    del kt
    return dict(info=out, B=B, Bnc=Bnc, b={l: bMr[i].copy() for i, l in enumerate(order)}, bnc={l: bMr_nc[i].copy() for i, l in enumerate(order)},
                M_H=M_H, N_rows=N_rows, Nx=sol['Nx'].real, order=order, arrays=arrays, radii=radii, Phi_app=Phi_app)


def betas_and_kernel(rows, PA, ns, svd=True):
    B, Bnc = rows['B'], rows['Bnc']
    d = 2 * ns
    info = {}
    t0 = time.time()
    V = lu = lunc = Jm = None
    if PA is not None:
        Jm = np.eye(d) + B @ PA
        lu = sla.lu_factor(Jm)
        V = sla.lu_solve(lu, B)
        info['dressed_asym_after_solve'] = fnorm(V - V.T) / max(fnorm(V), 1e-300)
        V = 0.5 * (V + V.T)
        lunc = sla.lu_factor(np.eye(d) + Bnc @ PA)
        if svd:
            info['smin_I_plus_BPA'] = float(np.linalg.svd(Jm, compute_uv=False).min())
            info['qstar_norm_PAB'] = float(np.linalg.norm(PA @ B, 2))
        info['V_minus_B_rel'] = fnorm(V - B) / fnorm(B)
        ev = np.linalg.eigvalsh(V)
        info['V_eig_min'], info['V_eig_max'] = float(ev[0]), float(ev[-1])
        info['V_eigs'] = ev
    evB = np.linalg.eigvalsh(B)
    info['B_eig_min'], info['B_eig_max'], info['B_eigs'] = float(evB[0]), float(evB[-1]), evB
    betas = {}
    for lab in rows['order']:
        b, bnc = rows['b'][lab], rows['bnc'][lab]
        ic = rows['radii'][lab]['site']
        ent = dict(noref=b.copy(), nocur=bnc.copy())
        if lu is not None:
            ent['full'] = sla.lu_solve(lu, b)
            ent['nocur_dressed'] = sla.lu_solve(lunc, bnc)
        lapse = np.zeros(d)
        lapse[:ic + 1] = rows['Nx'][:ic + 1] / rows['N_rows'][lab]
        ent['lapse_window'] = lapse
        win = np.zeros(d)
        win[:ic + 1] = 1.0
        ent['sharp_window'] = win
        betas[lab] = ent
    if Jm is not None:
        lab0 = rows['order'][0]
        info['Jm_true_residual'] = fnorm(Jm @ betas[lab0]['full'] - rows['b'][lab0]) / max(fnorm(rows['b'][lab0]), 1e-300)
    info['seconds'] = time.time() - t0
    return V, betas, info


# ============================================================================ the variances and the exchange
def quad(G, x):
    return float(x @ (G @ x))


def variance_table(chans, betas, radii, M_H, order, dust_cum, R, E_scale):
    tab = {}
    neg = []
    for lab in order:
        r = radii[lab]['r']
        Nr = float(np.interp(r, R, dust_cum)) if dust_cum is not None else None
        row = dict(r=r, site=radii[lab]['site'], M_H=M_H[lab], N_enclosed=Nr, source=radii[lab].get('source'), variants={})
        for k in ('M_w_dense', 'M_w_applied'):
            if k in radii[lab]:
                row[k] = radii[lab][k]
        for name, beta in betas[lab].items():
            per = []
            vC = vP = vR = 0.0
            have_raw = False
            for ch in chans:
                c = quad(ch['G_C'], beta)
                p = quad(ch['G_P'], beta)
                vC += ch['w'] * c
                vP += ch['w'] * p
                if 'G_C_raw' in ch:
                    vR += ch['w'] * quad(ch['G_C_raw'], beta)
                    have_raw = True
                per.append([ch['kappa'], ch['w'], ch['w'] * c, ch['w'] * p])
            if vC < -1e-9 * max(abs(vC), abs(vP)) or vP < -1e-9 * max(abs(vC), abs(vP)):   # a negative beyond the Gram's roundoff (the reference's exact zero at R_out is at 1e-10 of Var_C)
                neg.append((lab, name, vC, vP))
            Mref = M_H[lab] if M_H[lab] > 0 else (E_scale or 1.0)
            wtot = float(sum(p[1] for p in per))
            ent = dict(Var_C=vC, Var_C0=vP, signed_excess=vC - vP, per_channel=per, total_weight=wtot, sea_variance_per_unit_weight=vP / wtot,
                       sqrt_VarC_over_M=math.sqrt(max(vC, 0.0)) / Mref, sqrt_VarC0_over_M=math.sqrt(max(vP, 0.0)) / Mref,
                       signed_sqrt_excess_over_M=math.copysign(math.sqrt(abs(vC - vP)), vC - vP) / Mref,
                       sqrt_VarC_over_E=math.sqrt(max(vC, 0.0)) / E_scale if E_scale else None,
                       N_r_inv_sqrt=(Nr ** -0.5) if (Nr and Nr > 0) else None,
                       beta_energy_max=float(np.max(np.abs(beta[:len(beta) // 2]))), beta_current_max=float(np.max(np.abs(beta[len(beta) // 2:]))))
            if have_raw:
                ent['Var_C_raw_columns'] = vR
                ent['floor_footprint_rel'] = abs(vR - vC) / max(abs(vC), 1e-300)
            row['variants'][name] = ent
        tab[lab] = row
    return tab, neg


def exchange_stage(chans, s_tot, kernels, NB, m, E_ref, N_enc):
    T_tot = sum(ch['w'] * ch['T'] for ch in chans)
    out = {}
    for name, V in kernels:
        per = [-0.5 * ch['w'] * float(np.sum(V * ch['T'])) for ch in chans]
        EF = float(sum(per))
        Ep = 0.5 * float(s_tot @ (V @ s_tot))
        ent = dict(E_F=EF, E_F_per_channel=per, E_pair=Ep, y=(EF / Ep) if Ep != 0 else None, E_F_over_abs_E_pair=(EF / abs(Ep)) if Ep != 0 else None,
                   sign='negative' if EF < 0 else 'positive', E_F_over_E_minus_NBm=(EF / (E_ref - NB * m)) if (E_ref - NB * m) != 0 else None,
                   comparators={'1/N_B': 1.0 / NB, '5/(4 N_B) monopole bound': 5.0 / (4.0 * NB),
                                '1/N_50': (1.0 / N_enc['R_50']) if N_enc.get('R_50') else None, '1/N_t': (1.0 / N_enc['R_t']) if N_enc.get('R_t') else None},
                   cluster_correlation=dict(CLUSTER_CORRELATION, comparison_abs=CLUSTER_CORRELATION['rel'] * abs(Ep),
                                            label="the HF level E_pair + E_F; beside it the static star's monopole cluster control rel x |E_pair|, "
                                                  "a comparison and no bound on the core's correlation"))
        # v1.1: the energy, mixed and current blocks (energy index first, as bilinear() lays T out)
        n_ = V.shape[0] // 2
        blk = lambda X, a, b: X[a * n_:(a + 1) * n_, b * n_:(b + 1) * n_]  # noqa: E731
        parts = (('energy', [(0, 0)]), ('mixed', [(0, 1), (1, 0)]), ('current', [(1, 1)]))
        bF = {nm: float(sum(-0.5 * ch['w'] * sum(float(np.sum(blk(V, a, b) * blk(ch['T'], a, b))) for a, b in pr) for ch in chans)) for nm, pr in parts}
        bFk = {nm: [float(-0.5 * ch['w'] * sum(float(np.sum(blk(V, a, b) * blk(ch['T'], a, b))) for a, b in pr)) for ch in chans] for nm, pr in parts}
        bP = {nm: float(sum(0.5 * float(s_tot[a * n_:(a + 1) * n_] @ (blk(V, a, b) @ s_tot[b * n_:(b + 1) * n_])) for a, b in pr)) for nm, pr in parts}
        ent['blocks'] = dict(E_F=bF, E_F_per_channel=bFk, E_pair=bP, E_F_blocks_sum_minus_E_F=sum(bF.values()) - EF, E_pair_blocks_sum_minus_E_pair=sum(bP.values()) - Ep,
                             cancellation_E_F=abs(EF) / max(sum(abs(x) for x in bF.values()), 1e-300),
                             cancellation_E_pair=abs(Ep) / max(sum(abs(x) for x in bP.values()), 1e-300),
                             label='energy = (0,0) block, mixed = (0,1) + (1,0), current = (1,1) of V and T; cancellation = |total| / sum |blocks|')
        if all('T_P' in ch for ch in chans):
            ent['controls'] = dict(E_F_removed_sea_term_P=float(sum(-0.5 * ch['w'] * float(np.sum(V * ch['T_P'])) for ch in chans)),
                                   E_F_unordered_C=float(sum(-0.5 * ch['w'] * float(np.sum(V * ch['T_C'])) for ch in chans)), label='controls, never added')
        out[name] = ent
        log('  exchange [%s]: E_F %.6e (%s), E_pair %.6e, y = E_F/E_pair %s, E_F/(E - N_B m) %s; comparators 1/N_B %.3e, 1/N_50 %s%s'
            % (name, EF, ent['sign'], Ep, ent['y'], ent['E_F_over_E_minus_NBm'], 1.0 / NB, ent['comparators']['1/N_50'],
               ('; controls P %.3e C %.3e' % (ent['controls']['E_F_removed_sea_term_P'], ent['controls']['E_F_unordered_C'])) if 'controls' in ent else ''))
    return out, T_tot


# ============================================================================ the evolution rows at the core nodes
def shell_stresses(H, k, part, R):
    F = np.asarray(H['F_face_%s' % part][k], float)
    fc = np.asarray(H['fc_%s' % part][k], float)
    Rf = np.concatenate([R[:-1] + 0.5 * np.diff(R), [R[-1] + 0.5]])
    Srf = F / (FOURPI * Rf ** 2)
    Sr = 0.5 * (np.concatenate([[Srf[0]], Srf[:-1]]) + Srf)
    St = fc / (8.0 * math.pi * R)
    return Sr, St


def evolution_stage(M, args, history, rec, radii, core_radius, R, ell, g, tau, handoff_npz):
    """both evolution rows' residuals at the core nodes r_i = i ell (<= core_radius) and at the core radii; d_t k centred over the
    neighbouring records; the stresses from the handoff (dust + pairs) smoothed by engine_stress_v1; W_0, W_r by the Bianchi identities."""
    RW, STR = M.RW, M.STR
    FM = M.fields_module(HIST[history]['fields_module'])
    t0 = time.time()
    Z = np.load(rec['fields_npz'], allow_pickle=True)
    Tz = np.asarray(Z['T'], float)
    H = np.load(handoff_npz, allow_pickle=True)
    TH = np.asarray(H['T'], float)
    T = float(rec['record']['T'])
    out = dict(convention="engine_rows_v2: res_rr = d_t k_r - rhs_rr, res_tt = d_t k_t - rhs_tt (k_r = v'/N, k_t = v/(rN); the matter term "
                          "4 pi g N (rho + S_r - 2 S_t)_ell and 4 pi g N (rho - S_r)_ell); R_r = res_rr/N, R_t = res_tt/N in the Bianchi identities "
                          "4 pi G W_0/N = B R_r + (A + B) R_t, 4 pi G W_r = (N R_t)' + N (R_t - R_r)/r, A = v'/N, B = v/(rN)",
               fields_module=FM.__name__, stress_module=STR.__name__, rows_module=RW.__name__, handoff=handoff_npz, tau=tau, h_frac=args.h_frac,
               core_radius=core_radius)
    nodes = np.arange(ell, core_radius + 1e-9, ell)
    extra = [(lab, radii[lab]['r']) for lab in ('R_in', 'R_50', 'R_90', 'R_t') if lab in radii]
    rr = np.concatenate([nodes, np.array([x for _, x in extra], float)])
    rfine = np.arange(ell / 8.0, core_radius + 2.0 * ell + 1e-9, ell / 8.0)
    h = ell * args.h_frac
    times = {}
    for name, Tk in (('minus', T - tau), ('zero', T), ('plus', T + tau)):
        kz = int(np.argmin(np.abs(Tz - Tk)))
        kh = int(np.argmin(np.abs(TH - Tk)))
        if abs(Tz[kz] - Tk) > 1e-6 or abs(TH[kh] - Tk) > 1e-6:
            times[name] = dict(missing=True, T=Tk)
            continue
        e = np.asarray(Z['eps_tot'][kz], float)
        Jt = np.asarray(Z['J_tot'][kz], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, g, R, ell, collar_policy=rec['collar_policy'], support_rel=rec['support_rel'])
        if info.get('failed'):
            times[name] = dict(failed=info['failed'], T=Tk)
            continue
        Srd, Std = shell_stresses(H, kh, 'dust', R)
        Srp, Stp = shell_stresses(H, kh, 'pairs', R)
        Sr_l, St_l, tr_err = STR.smooth_stress(Srd + Srp, Std + Stp, R, ell, rr)
        Sr_f, St_f, _ = STR.smooth_stress(Srd + Srp, Std + Stp, R, ell, rfine)
        rho = np.asarray(info['rho_ell'](rr), float)
        rho_f = np.asarray(info['rho_ell'](rfine), float)
        b = RW.brackets(info['N_dense'], info['v_dense'], rho, Sr_l, St_l, g, rr, h)
        bf = RW.brackets(info['N_dense'], info['v_dense'], rho_f, Sr_f, St_f, g, rfine, h)
        ent = dict(T=Tk, T_over_rs=Tk / rec['rs'], b=b, bf=bf, rho=rho, rho_f=rho_f, Sr=Sr_l, St=St_l, trace_err=tr_err, E_collar=info.get('E_collar'),
                   r_collar=info.get('r_collar'), gauge=(info.get('gauge') or {}).get('label'))
        # the handoff's own smoothed stresses at the nodes (an INFO comparison)
        rn = np.asarray(H['r_nodes'], float)
        sel = (rn > 0) & (rn <= core_radius + 1e-9)
        if sel.any():
            Srn, Stn, _ = STR.smooth_stress(Srd + Srp, Std + Stp, R, ell, rn[sel])
            ent['stress_vs_handoff_nodes'] = dict(S_r_max_abs_diff=float(np.max(np.abs(Srn - np.asarray(H['S_r_ell'][kh], float)[sel]))),
                                                  S_t_max_abs_diff=float(np.max(np.abs(Stn - np.asarray(H['S_t_ell'][kh], float)[sel]))),
                                                  S_r_scale=float(np.max(np.abs(np.asarray(H['S_r_ell'][kh], float)[sel]))),
                                                  S_t_scale=float(np.max(np.abs(np.asarray(H['S_t_ell'][kh], float)[sel]))))
        times[name] = ent
    out['times'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('b', 'bf', 'rho', 'rho_f', 'Sr', 'St')} for k, v in times.items()}
    arrays = dict(ev_r=rr, ev_n_nodes=np.array([len(nodes)]), ev_rfine=rfine)
    if any(('b' not in times.get(k, {})) for k in ('minus', 'zero', 'plus')):
        out['status'] = 'incomplete: a neighbouring record or its solve is missing (%s)' % {k: v.get('failed', v.get('missing')) for k, v in times.items() if 'b' not in v}
        log('  evolution rows: %s' % out['status'])
        return out, arrays
    b0, bm, bp = times['zero']['b'], times['minus']['b'], times['plus']['b']
    dkr = (bp['k_r'] - bm['k_r']) / (2.0 * tau)
    dkt = (bp['k_t'] - bm['k_t']) / (2.0 * tau)
    sfr = (bp['k_r'] - 2.0 * b0['k_r'] + bm['k_r']) / tau
    sft = (bp['k_t'] - 2.0 * b0['k_t'] + bm['k_t']) / tau
    res_rr = dkr - b0['rhs_rr']
    res_tt = dkt - b0['rhs_tt']
    sc_rr = np.maximum(b0['terms_rr'], np.abs(dkr))
    sc_tt = np.maximum(b0['terms_tt'], np.abs(dkt))
    rho = times['zero']['rho']
    sup = np.abs(rho) > 1e-6 * max(float(np.max(np.abs(rho))), 1e-300)
    n = len(nodes)
    # the fine grid: W_0, W_r by the Bianchi identities (engine_rows_v2's residual convention)
    f0, fm, fp = times['zero']['bf'], times['minus']['bf'], times['plus']['bf']
    dkr_f = (fp['k_r'] - fm['k_r']) / (2.0 * tau)
    dkt_f = (fp['k_t'] - fm['k_t']) / (2.0 * tau)
    Rr = (dkr_f - f0['rhs_rr']) / f0['N']
    Rt = (dkt_f - f0['rhs_tt']) / f0['N']
    Nf, vf = f0['N'], f0['v']
    A_ = f0['k_r']
    B_ = f0['k_t']
    W0 = Nf * (B_ * Rr + (A_ + B_) * Rt) / (FOURPI * g)
    dNRt = np.gradient(Nf * Rt, rfine)
    Wr = (dNRt + Nf * (Rt - Rr) / rfine) / (FOURPI * g)
    arrays.update(ev_res_rr=res_rr, ev_res_tt=res_tt, ev_dkr=dkr, ev_dkt=dkt, ev_rhs_rr=b0['rhs_rr'], ev_rhs_tt=b0['rhs_tt'], ev_terms_rr=b0['terms_rr'],
                  ev_terms_tt=b0['terms_tt'], ev_floor_rr=sfr, ev_floor_tt=sft, ev_rho=rho, ev_Sr=times['zero']['Sr'], ev_St=times['zero']['St'],
                  ev_N=b0['N'], ev_v=b0['v'], ev_k_r=b0['k_r'], ev_k_t=b0['k_t'], ev_support=sup,
                  ev_fine_res_rr=dkr_f - f0['rhs_rr'], ev_fine_res_tt=dkt_f - f0['rhs_tt'], ev_fine_N=Nf, ev_fine_v=vf, ev_fine_rho=times['zero']['rho_f'],
                  ev_fine_W0=W0, ev_fine_Wr=Wr, ev_fine_Rr=Rr, ev_fine_Rt=Rt)
    rel_rr = np.abs(res_rr) / sc_rr
    rel_tt = np.abs(res_tt) / sc_tt
    nodes_tab = [dict(r=float(rr[i]), res_rr=float(res_rr[i]), res_tt=float(res_tt[i]), rel_rr=float(rel_rr[i]), rel_tt=float(rel_tt[i]),
                      floor_rr=float(abs(sfr[i]) / sc_rr[i]), floor_tt=float(abs(sft[i]) / sc_tt[i]), rho=float(rho[i]), support=bool(sup[i]),
                      d_t_k_r=float(dkr[i]), d_t_k_t=float(dkt[i]), rhs_rr=float(b0['rhs_rr'][i]), rhs_tt=float(b0['rhs_tt'][i])) for i in range(n)]
    at_radii = {lab: dict(r=float(rr[n + j]), res_rr=float(res_rr[n + j]), res_tt=float(res_tt[n + j]), rel_rr=float(rel_rr[n + j]), rel_tt=float(rel_tt[n + j]),
                          floor_rr=float(abs(sfr[n + j]) / sc_rr[n + j]), floor_tt=float(abs(sft[n + j]) / sc_tt[n + j]),
                          W0=float(np.interp(rr[n + j], rfine, W0)), Wr=float(np.interp(rr[n + j], rfine, Wr))) for j, (lab, _) in enumerate(extra)}
    supn = sup[:n]
    summ = dict(n_nodes=n, nodes_r=[float(x) for x in nodes], core_radius=core_radius,
                res_rr_support_max_rel=float(np.max(rel_rr[:n][supn])) if supn.any() else None,
                res_tt_support_max_rel=float(np.max(rel_tt[:n][supn])) if supn.any() else None,
                stepping_floor_rr=float(np.max((np.abs(sfr) / sc_rr)[:n][supn])) if supn.any() else None,
                stepping_floor_tt=float(np.max((np.abs(sft) / sc_tt)[:n][supn])) if supn.any() else None,
                W0_max_abs_inside=float(np.max(np.abs(W0[rfine <= core_radius]))), Wr_max_abs_inside=float(np.max(np.abs(Wr[rfine <= core_radius]))),
                rho_scale=float(np.max(np.abs(rho))), trace_identity_err=times['zero']['trace_err'], d_t='centred (records T - tau, T + tau)',
                stress_vs_handoff_nodes=times['zero'].get('stress_vs_handoff_nodes'), E_collar=times['zero']['E_collar'], gauge=times['zero']['gauge'])
    out.update(summary=summ, nodes=nodes_tab, at_radii=at_radii, seconds=time.time() - t0, status='done')
    log('  evolution rows at %d core nodes (<= %.2f): max |res|/max term over the support rr %.3e, tt %.3e (stepping floors %.1e, %.1e); at radii %s (%.0f s)'
        % (n, core_radius, summ['res_rr_support_max_rel'] or float('nan'), summ['res_tt_support_max_rel'] or float('nan'),
           summ['stepping_floor_rr'] or float('nan'), summ['stepping_floor_tt'] or float('nan'),
           {k: ('%.2e, %.2e' % (v['rel_rr'], v['rel_tt'])) for k, v in at_radii.items()}, summ['seconds'] if 'seconds' in summ else time.time() - t0))
    return out, arrays


# ============================================================================ the slice run
def chk_factory(checks):
    def chk(name, val, tol, ok=None, info=False):
        ok = (val <= tol) if ok is None else bool(ok)
        tag = 'INFO' if info else ('PASS' if ok else 'FAIL')
        checks.append(dict(name=name, value=val, tol=tol, status=tag))
        log('  [%s] %s: %s (tol %s)' % (tag, name, val, tol))
    return chk


def run_slice(args, M, LA, history, t):
    H = HIST[history]
    out_dir = os.path.join(args.out, history)
    os.makedirs(out_dir, exist_ok=True)
    tag = 'core_hf_%s_T%07.2f' % (history, float(t))
    if args.kappas:
        tag += '_k' + '-'.join(str(k) for k in args.kappas)
    if args.kernel == 'compact':
        tag += '_compact'
    pj = os.path.join(out_dir, tag + '.json')
    if os.path.exists(pj) and not args.force:
        log('%s exists, skipped (--force recomputes)' % pj)
        return 0
    t_all = time.time()
    checks = []
    chk = chk_factory(checks)
    path, name, cands = find_slice(history, t, args.slice_dir)
    if path is None:
        log('MISSING slice %s (searched %s)' % (name, cands))
        return 2
    records = args.records or os.path.join(H['base'], 'collapse_%s.json' % H['tag'])
    fields = args.fields or os.path.join(H['base'], 'collapse_%s_fields.npz' % H['tag'])
    handoff = args.handoff or os.path.join(H['base'], 'collapse_%s_handoff.npz' % H['tag'])
    rec = load_record(records, fields, t)
    sl = SliceFile(path)
    exp = SIZES.get(('c64' if sl.complex64 else 'c128', sl.nsites))
    log('\n=== %s %s T/r_s %g (%s): %s ===' % (VERSION, history, t, H['label'], path))
    log('  %d sites, %s, kappas %s, dust %s, size %d (expected %s), m %g g %g ell %g tau %g; complex64 floor %s'
        % (sl.nsites, sl.column_dtype, sl.kappas, sl.n_dust, sl.size, exp, rec['m'], rec['g'], rec['ell'], rec['tau'], sl.complex64))
    if exp is not None and sl.size != exp:
        log('  SIZE MISMATCH: %d against the inventory\'s %d: stopped' % (sl.size, exp))
        return 2
    well = load_well(args.well_dir or H['well'], history, t)
    flags, stop = status_flags(rec)
    ns, R = sl.nsites, sl.R
    m, g, ell = rec['m'], rec['g'], float(rec['ell'])
    chk('S1 slice applied fields equal the fields-npz applied row (max abs)', float(max(np.max(np.abs(rec['N_applied'] - sl.N)), np.max(np.abs(rec['v_applied'] - sl.v)))), 0.0)
    chk('S1b slice radii R = 1..ns (max abs)', float(np.max(np.abs(R - np.arange(1, ns + 1)))), 1e-12)
    chk('S1c weights = 16 kappa (max abs)', float(max(abs(sl.weights[k] - 16.0 * k) for k in sl.kappas)), 0.0)
    # ---- radii (the well json and the record)
    radii = {}
    r = rec['record']
    tz = list(r.get('trapping_zeros') or [])
    if len(tz) >= 2:
        radii['R_in'] = dict(r=float(tz[0]), source="the record's dense inner root")
    if len(tz) >= 1:
        radii['R_t'] = dict(r=float(tz[-1]), source="the record's dense outer root")
    # ---- channels
    kappas = [k for k in sl.kappas if (not args.kappas or k in args.kappas)]
    chans = []
    s_tot = np.zeros(2 * ns)
    s_raw = np.zeros(2 * ns)
    s_dust = np.zeros(2 * ns)
    dust_site = np.zeros(ns)
    per = []
    for k in kappas:
        ch = build_channel(LA, M, sl, k, m, rec['half'], controls=args.exchange_controls, floor_control=(sl.complex64 or args.floor_control), eig_d=args.eig_d)
        s_tot += ch['s']
        s_raw += ch['s_raw']
        s_dust += ch['s_dust_raw']
        dust_site += ch['dust_site']
        row = {kk: ch[kk] for kk in ('kappa', 'w', 'n_sea', 'n_dust', 'n_occ', 'pure', 'f_min', 'f_max', 'stored_orthonormality', 'loewdin_orthonormality',
                                     'ref_stored_orthonormality', 'ref_loewdin_orthonormality', 'e_vs_stored_abs', 'mask_equals_neg', 'stored_mask_equals_stored_neg',
                                     'n_neg', 'gap', 'pref_resolution_defect', 'commutator_rel', 'vertex_identity_rel', 'TrD', 'TrD2', 'gram_defects',
                                     'static_identity_hm', 'var_sea_hm', 'eigh_seconds', 'gram_seconds', 'seconds')}
        row['n_eigD_above_1e-6'] = ch.get('n_eigD_above_1e-6')
        row['record_commutator_h_R2_F'] = (r.get('commutator_h_R2_F') or {}).get(str(k))
        per.append(row)
        log('  kappa %d: n_occ %d (sea %d, dust %d) %s; stored orth %.2e Loewdin %.1e; e vs e_k %.1e; mask==neg %s; gap %.4f; P_ref res %.1e; comm %.1e; '
            'vertex id %.1e; Tr D %.6f; Var_C(h_m) %.4e vs (1/2)||[h_m,C]||^2 rel %.1e; Var_sea(h_m) %.1e; eigh %.0f s, grams %.0f s, total %.0f s'
            % (k, ch['n_occ'], ch['n_sea'], ch['n_dust'], 'pure' if ch['pure'] else 'MIXED', ch['stored_orthonormality'], ch['loewdin_orthonormality'],
               ch['e_vs_stored_abs'] if ch['e_vs_stored_abs'] is not None else float('nan'), ch['mask_equals_neg'], ch['gap'], ch['pref_resolution_defect'],
               ch['commutator_rel'], ch['vertex_identity_rel'], ch['TrD'], ch['static_identity_hm']['var_C'], ch['static_identity_hm']['rel_diff'],
               ch['var_sea_hm'], ch['eigh_seconds'], ch['gram_seconds'], ch['seconds']))
        chans.append(ch)
    c64 = sl.complex64
    nrm = lambda x: float(np.max(np.abs(x)))  # noqa: E731
    full = not args.kappas
    if full:
        chk("S3 sources from the stored columns, sum 16k Tr[(C - P_ref) hcal_n], vs the record's eps_tot (max rel)", nrm(s_raw[:ns] - rec['eps_tot']) / nrm(rec['eps_tot']), 1e-5 if c64 else 1e-12)
        chk('S3b the same for the current p_n vs J_tot (max rel)', nrm(s_raw[ns:] - rec['J_tot']) / nrm(rec['J_tot']), 1e-5 if c64 else 1e-12)
        chk('S3c the Loewdin state: eps vs eps_tot (max rel)', nrm(s_tot[:ns] - rec['eps_tot']) / nrm(rec['eps_tot']), 1e-4 if c64 else 1e-9)
        chk('S3d the Loewdin state: J vs J_tot (max rel)', nrm(s_tot[ns:] - rec['J_tot']) / nrm(rec['J_tot']), 1e-4 if c64 else 1e-9)
        if 'eps_pairs' in rec:
            ref = rec['eps_tot'] - rec['eps_pairs']
            chk('S4 the dust part vs eps_tot - eps_pairs (max rel)', nrm(s_dust[:ns] - ref) / nrm(ref), 1e-4 if c64 else 1e-10)
        Etot = (r.get('E_dust_shells') or 0.0) + (r.get('E_pairs_shells') or 0.0)
        chk("S4b sum of the energy sources vs the record's E_dust_shells + E_pairs_shells (rel)", abs(float(s_tot[:ns].sum()) - Etot) / abs(Etot), 1e-4 if c64 else 1e-9)
        NB = float(sum(ch['w'] * ch['TrD'] for ch in chans))
        chk('S5 N_B = sum 16 kappa Tr D = 640 (abs)', abs(NB - 640.0), 1e-5 if c64 else 1e-9)
        chk('S5b Tr D_kappa = n_dust per channel (max abs)', float(max(abs(ch['TrD'] - ch['n_dust']) for ch in chans)), 1e-5 if c64 else 1e-9)
    else:
        log('  CHANNEL SUBSET %s: the sums below are over these channels only (a smoke control, not the record set)' % kappas)
    chk('S6 occupied columns orthonormal after Loewdin (max over channels)', float(max(ch['loewdin_orthonormality'] for ch in chans)), 1e-13)
    chk('S6b reference columns orthonormal after Loewdin (max)', float(max(ch['ref_loewdin_orthonormality'] for ch in chans)), 1e-13)
    chk('S7 max |e - e_k stored| (abs)', float(max(ch['e_vs_stored_abs'] or 0.0 for ch in chans)), 1e-11)
    chk('S7b mask == (e < 0) bitwise in every channel', float(sum(not ch['mask_equals_neg'] for ch in chans)), 0.0)
    focc = [max(abs(ch['f_min'] - 1.0), abs(ch['f_max'] - 1.0)) for ch in chans]
    chk('U1 (v1.1) unit occupation: max |f - 1| over every stored column of every channel (the Slater forms are exact only at f = 1)', float(max(focc)), 1e-12)
    chk('S8 P_ref resolution ||R2 R2^+ - Uo Uo^+||_F/sqrt(n_o) (max)', float(max(ch['pref_resolution_defect'] for ch in chans)), 1e-5, info=True)
    chk('S8b ||(1 - P_ref) h_m P_ref||_F/||h_m||_F (max)', float(max(ch['commutator_rel'] for ch in chans)), 1e-10, info=True)
    chk("Q2' vertex identity h_m = sum N hcal + v p (max rel)", float(max(ch['vertex_identity_rel'] for ch in chans)), 1e-12)
    chk("Q1' static identity Var_C(h_m) = (1/2)||[h_m, C]||^2 (max rel over channels; both O(1))", float(max(ch['static_identity_hm']['rel_diff'] for ch in chans)), 1e-12)
    chk('S9 Var_P(h_m) of the eigen-sea = 0 (max over channels, relative to Var_C(h_m))', float(max(ch['var_sea_hm'] / max(ch['static_identity_hm']['var_C'], 1e-300) for ch in chans)), 1e-20)
    chk('S9b Gram Hermiticity defects (max over channels and G_C, G_P, T)', float(max(max(ch['gram_defects'].values()) for ch in chans)), 1e-11)
    # the dust count and the radii
    dust_cum = np.cumsum(dust_site)
    tot = float(dust_cum[-1]) if len(dust_cum) else 0.0
    if tot > 0:
        for lab, q in (('R_50', 0.5), ('R_90', 0.9)):
            x = frac_radius(R, dust_cum / tot, q)
            radii[lab] = dict(r=float(x), source='recomputed from the slice dust columns (frac_radius, linear between sites)')
            if well and full and well['dust'].get(lab) is not None:
                chk('S10 %s recomputed vs the well json (abs)' % lab, abs(float(x) - float(well['dust'][lab])), 1e-9)
    if r.get('R_c') is not None:
        radii['R_c'] = dict(r=float(r['R_c']), source="the record's R_c (a window edge only)")
    if well:
        for lab in ('applied_inner', 'applied_outer'):
            if well['roots'].get(lab) is not None:
                radii[lab] = dict(r=float(well['roots'][lab]), source='CORE_well applied root (the radius systematic)')
    A_ = (well or {}).get('arrays', {})
    for lab in radii:
        x = radii[lab]['r']
        radii[lab]['site'] = min(int(np.searchsorted(R, x)), ns - 1)
        for key in ('M_w_dense', 'M_w_applied'):
            if key in A_:
                radii[lab][key] = float(np.interp(x, R, A_[key]))
    core_radius = radii['R_t']['r'] if 'R_t' in radii else (radii['R_c']['r'] if 'R_c' in radii else float(R[-1]))
    order_core = [l for l in ('R_in', 'R_50', 'R_90', 'R_t', 'R_c', 'applied_inner', 'applied_outer') if l in radii]
    log('  radii: %s; core radius %.3f (%s)' % ({l: '%.3f' % radii[l]['r'] for l in order_core}, core_radius, 'R_t' if 'R_t' in radii else 'R_c (no root)'))
    # ---- the rows
    rows = rows_stage(M, args, R, ell, rec['eps_tot'], rec['J_tot'], g, sl.N, sl.v, float(r['E']), {l: dict(radii[l]) for l in order_core}, ok_label='CORE')
    radii = rows['radii']
    order = rows['order']
    chk("Q7' first law b(R_out) = Phi_resolved (rel)", rows['info']['hessian']['first_law_raw'], 1e-9)
    if 'lin_vs_cs_B' in rows['info']['hessian']:
        chk('S2 linearized vs complex-step Hessian columns (rel)', rows['info']['hessian']['lin_vs_cs_B'], 1e-8)
    chk("Q8' field check: re-solved site fields vs the applied (max rel N, v)", max(rows['info']['field_check']['N_rel_maxdev'], rows['info']['field_check']['v_rel_maxdev']), args.field_tol,
        info=True)
    # ---- the response
    PA = None
    resp = dict(stopped=None)
    if args.no_response:
        resp['stopped'] = 'not run (--no-response): the noref and nocur variants only'
    elif stop or any(not ch['mask_equals_neg'] for ch in chans):
        resp['stopped'] = 'stopped: the sign split needs mask == (e < 0) and no inversions (%s)' % (stop or 'mask != neg')
        log('  response %s' % resp['stopped'])
    else:
        PA, rinfo = response_matrix(LA, chans, ns, args.quad_tol)
        resp.update(rinfo)
        rng = np.random.default_rng(11)
        rel = []
        for _ in range(2):
            xr = rng.normal(size=2 * ns)
            yo = response_matvec(LA, chans, ns, xr)
            rel.append(fnorm(PA @ xr - yo) / max(fnorm(yo), 1e-300))
        resp['matrix_vs_operator_rel'] = rel
        resp['PA_Phi_applied_over_Phi'] = fnorm(PA @ rows['Phi_app']) / fnorm(rows['Phi_app'])
        resp['PA_Phi_applied_over_PAnorm_Phi'] = fnorm(PA @ rows['Phi_app']) / max(fnorm(PA) * fnorm(rows['Phi_app']), 1e-300)
        chk("Q4' P^A matrix (exponential sum) vs the operator route on random vectors (rel)", float(max(rel)), max(1e-10, 100.0 * args.quad_tol))
        chk("Q4'b ||P^A Phi_applied||/||Phi|| (the sea does not respond to its own generator)", resp['PA_Phi_applied_over_Phi'], 1e-6 if c64 else 1e-9)
        log('  response: %d nodes, quadrature %.1e on [%.4f, %.4f]; raw asym %.1e; matrix vs operator %s; ||P^A Phi_applied||/||Phi|| %.1e (%.0f s)'
            % (rinfo['nodes'], rinfo['quadrature_max_rel_err'], rinfo['xmin'], rinfo['xmax'], rinfo['asymmetry_raw'], ['%.1e' % x for x in rel], resp['PA_Phi_applied_over_Phi'], rinfo['seconds']))
    V, betas, kinfo = betas_and_kernel(rows, PA, ns, svd=args.svd)
    if V is not None:
        chk("Q5' dressed kernel V symmetric (asym after the solve)", kinfo['dressed_asym_after_solve'], 1e-10)
        log('  dressed kernel: ||V - B||/||B|| %.3e, eig V [%.3e, %.3e], eig B [%.3e, %.3e]; sigma_min(I + B P^A) %s, ||P^A B|| %s'
            % (kinfo['V_minus_B_rel'], kinfo['V_eig_min'], kinfo['V_eig_max'], kinfo['B_eig_min'], kinfo['B_eig_max'], kinfo.get('smin_I_plus_BPA'), kinfo.get('qstar_norm_PAB')))
        if args.gmres_check:
            lab0 = order[0]
            t1 = time.time()
            bg, hist, nit, ritz = M.S10.gmres_solve(lambda x: x + rows['B'] @ response_matvec(LA, chans, ns, x), rows['b'][lab0], args.gmres_tol, args.gmres_maxit)
            resp['gmres_check'] = dict(radius=lab0, applications=nit, lstsq_residual=hist[-1], ritz_min_abs=ritz,
                                       beta_matrix_vs_gmres_rel=fnorm(betas[lab0]['full'] - bg) / max(fnorm(bg), 1e-300), seconds=time.time() - t1)
            chk("Q4'c beta_full by the explicit matrix vs GMRES on the operator at %s (rel)" % lab0, resp['gmres_check']['beta_matrix_vs_gmres_rel'], 1e-8)
    # ---- the variances
    tab, neg = variance_table(chans, betas, radii, rows['M_H'], order, dust_cum if tot > 0 else None, R, float(r['E']))
    chk("Q10' no negative variance", float(len(neg)), 0.0)
    rng = np.random.default_rng(3)
    worst = 0.0
    labA = 'R_50' if 'R_50' in betas else order[0]
    keyA = 'full' if 'full' in betas[labA] else 'noref'
    for ch in chans:
        for beta in (betas[labA][keyA], rng.normal(size=2 * ns)):
            worst = max(worst, gram_vs_direct(ch, beta, direct_variances(LA, ch, beta, ns)))
    chk("Q6' Gram route beta^T G beta vs the direct Slater variance (worst over channels, A_%s and a random vertex; over max(Var, 1e-6 ||A Phi||^2))" % labA, worst, 1e-9)
    for l in order:
        fl = tab[l]['variants'].get('full') or tab[l]['variants']['noref']
        log('  %s (r %.3f, M_H %.6f, N_< %s): %s Var_C %.6e Var_C0 %.6e excess %+.6e; sqrt(Var_C)/M_H %.4e; N_<^-1/2 %s%s'
            % (l, tab[l]['r'], rows['M_H'][l], tab[l]['N_enclosed'], 'full' if 'full' in tab[l]['variants'] else 'noref', fl['Var_C'], fl['Var_C0'], fl['signed_excess'],
               fl['sqrt_VarC_over_M'], fl['N_r_inv_sqrt'], ('; floor footprint %.1e' % fl['floor_footprint_rel']) if 'floor_footprint_rel' in fl else ''))
    # ---- v1.1 extras (additive): external betas, Q6 localized, R_out in residual form, the radial-grid refinement
    v11 = dict(code='core_hf_slices_v1_1', unit_occupation_max_abs_f_minus_1={ch['kappa']: x for ch, x in zip(chans, focc)})
    if args.ref_arrays:
        Zr = np.load(args.ref_arrays)
        got = []
        for l in order:
            for vn_ in ('full', 'nocur_dressed'):
                k_ = 'beta_%s_%s' % (l, vn_)
                if k_ in Zr.files and vn_ not in betas[l] and np.asarray(Zr[k_]).shape == (2 * ns,):
                    betas[l][vn_ + '_ext'] = np.asarray(Zr[k_], float)
                    got.append(k_)
        v11['ref_arrays'] = dict(path=args.ref_arrays, md5=md5(args.ref_arrays), betas_loaded=got,
                                 note="the reference run's dressed betas applied to this run's Grams (the response itself not recomputed)")
        lx = [l for l in order if any(n.endswith('_ext') for n in betas[l])]
        if lx:
            tx, nx = variance_table(chans, {l: {n: b for n, b in betas[l].items() if n.endswith('_ext')} for l in lx}, radii, rows['M_H'], lx,
                                    dust_cum if tot > 0 else None, R, float(r['E']))
            for l in lx:
                tab[l]['variants'].update(tx[l]['variants'])
            v11['negative_variance_flags_ext'] = nx
    rng = np.random.default_rng(3)
    keyL = 'full_ext' if 'full_ext' in betas[labA] else keyA
    q6 = []
    for ch in chans:
        for vtag, beta in (('A_%s:%s' % (labA, keyL), betas[labA][keyL]), ('random (seed 3, as Q6)', rng.normal(size=2 * ns))):
            dv = direct_variances(LA, ch, beta, ns)
            gC, gP = quad(ch['G_C'], beta), quad(ch['G_P'], beta)
            dC = max(dv['var_C'], 1e-6 * dv['scale_var_C'], 1e-300)
            dP = max(dv['var_P'], 1e-6 * dv['scale_var_P'], 1e-300)
            q6.append(dict(kappa=ch['kappa'], vertex=vtag, var_C_direct=dv['var_C'], var_C_gram=gC, abs_C=abs(dv['var_C'] - gC), den_C=dC, rel_C=abs(dv['var_C'] - gC) / dC,
                           den_C_is_floor=bool(dC > dv['var_C']), var_P_direct=dv['var_P'], var_P_gram=gP, abs_P=abs(dv['var_P'] - gP), den_P=dP,
                           rel_P=abs(dv['var_P'] - gP) / dP, den_P_is_floor=bool(dP > dv['var_P']), scale_C=dv['scale_var_C'], scale_P=dv['scale_var_P']))
    wq = max(q6, key=lambda e: max(e['rel_C'], e['rel_P']))
    v11['Q6_localized'] = dict(entries=q6, worst=dict(wq, contraction='C (physical)' if wq['rel_C'] >= wq['rel_P'] else 'P (reference)', rel=max(wq['rel_C'], wq['rel_P'])))
    log('  v1.1 Q6 localized: worst %.3e at kappa %d, %s, %s (abs %.3e, den %.3e)' % (v11['Q6_localized']['worst']['rel'], wq['kappa'], wq['vertex'],
        v11['Q6_localized']['worst']['contraction'], wq['abs_C'] if wq['rel_C'] >= wq['rel_P'] else wq['abs_P'], wq['den_C'] if wq['rel_C'] >= wq['rel_P'] else wq['den_P']))
    ro = {}
    for vn_ in [n for n in ('noref', 'full', 'full_ext') if n in betas['R_out']]:
        beta = betas['R_out'][vn_]
        per_ = []
        for ch in chans:
            dv = direct_variances(LA, ch, beta, ns)
            per_.append(dict(kappa=ch['kappa'], w=ch['w'], var_P_residual=dv['var_P'], var_P_gram=quad(ch['G_P'], beta), var_C_residual=dv['var_C'],
                             var_C_gram=quad(ch['G_C'], beta), scale_P=dv['scale_var_P']))
        sP = sum(e['w'] * e['var_P_residual'] for e in per_)
        sC = sum(e['w'] * e['var_C_residual'] for e in per_)
        ro[vn_] = dict(Var_C0_residual_form=sP, Var_C0_gram=sum(e['w'] * e['var_P_gram'] for e in per_), Var_C_residual_form=sC,
                       Var_C_gram=sum(e['w'] * e['var_C_gram'] for e in per_), Var_C0_over_Var_C_residual=sP / max(sC, 1e-300),
                       weighted_scale_P=sum(e['w'] * e['scale_P'] for e in per_), per_channel=per_)
        log('  v1.1 R_out %s: Var_C0 residual form %.3e (Gram %.3e), Var_C %.6e; ratio %.2e' % (vn_, sP, ro[vn_]['Var_C0_gram'], sC, ro[vn_]['Var_C0_over_Var_C_residual']))
    v11['R_out_residual_form'] = ro
    if args.dr_refine:
        import copy
        base = dict(dr=rows['info']['dr'], labels={})
        for l in order_core:
            base['labels'][l] = dict(r=radii[l]['r'], grid_row=radii[l]['grid_row'], r_eval=radii[l]['grid_row'] * rows['info']['dr'], M_H=rows['M_H'][l],
                                     M_interp=float(np.interp(radii[l]['r'], rows['arrays']['r_grid'], rows['arrays']['M_grid'])),
                                     Var_C_noref=tab[l]['variants']['noref']['Var_C'], Var_C0_noref=tab[l]['variants']['noref']['Var_C0'])
        refs = [base]
        for d2 in args.dr_refine:
            try:
                a2 = copy.copy(args)
                a2.dr = d2
                a2.cs_cols = False
                t2 = time.time()
                rows2 = rows_stage(M, a2, R, ell, rec['eps_tot'], rec['J_tot'], g, sl.N, sl.v, float(r['E']), {l: dict(r=radii[l]['r'], site=radii[l]['site'], source=radii[l].get('source')) for l in order_core}, ok_label='CORE')
                ent = dict(dr=d2, field_check=rows2['info']['field_check'], first_law_raw=rows2['info']['hessian']['first_law_raw'], labels={})
                for l in order_core:
                    b2 = rows2['b'][l]
                    gr = rows2['radii'][l]['grid_row']
                    ent['labels'][l] = dict(r=radii[l]['r'], grid_row=gr, r_eval=gr * d2, M_H=rows2['M_H'][l],
                                            M_interp=float(np.interp(radii[l]['r'], rows2['arrays']['r_grid'], rows2['arrays']['M_grid'])),
                                            Var_C_noref=float(sum(ch['w'] * quad(ch['G_C'], b2) for ch in chans)),
                                            Var_C0_noref=float(sum(ch['w'] * quad(ch['G_P'], b2) for ch in chans)))
                ent['seconds'] = time.time() - t2
                refs.append(ent)
                log('  v1.1 dr %.4f: %s (%.0f s)' % (d2, {l: '%.4f/%.5e' % (e['M_H'], e['Var_C_noref']) for l, e in ent['labels'].items()}, ent['seconds']))
                del rows2
            except Exception as ex:  # noqa: BLE001
                refs.append(dict(dr=d2, failed='%s: %s' % (type(ex).__name__, ex)))
                log('  v1.1 dr %.4f failed: %s' % (d2, ex))
        v11['dr_refine'] = refs
    # ---- the exchange
    kernels = [('bare', rows['B'])] + ([('dressed', V)] if V is not None else [])
    N_enc = {l: tab[l]['N_enclosed'] for l in order}
    exch, T_tot = exchange_stage(chans, s_tot, kernels, 640.0 if full else float(sum(ch['w'] * ch['TrD'] for ch in chans)), m, float(r['E']), N_enc)
    # ---- the evolution rows
    evo, ev_arrays = None, {}
    if args.evolution and M.RW is not None and M.STR is not None and os.path.isfile(handoff):
        try:
            evo, ev_arrays = evolution_stage(M, args, history, rec, radii, core_radius, R, ell, g, rec['tau'], handoff)
        except Exception as ex:  # noqa: BLE001
            evo = dict(status='failed: %s: %s' % (type(ex).__name__, ex))
            log('  evolution rows failed: %s' % evo['status'])
    elif args.evolution:
        evo = dict(status='not run: engine_rows_v2/engine_stress_v1 not importable (%s) or the handoff is absent (%s)' % (M.err, handoff))
        log('  evolution rows: %s' % evo['status'])
    # ---- outputs
    status = dict(flags=flags, response_stop=stop, limit_label=(well or {}).get('limit_label'), gate=(well or {}).get('gate'), complex64=c64, floor=sl.floor,
                  record=r, kappa_sea=rec['kappa_sea'], regime=dict(m_a=m, rs_over_a=rec['rs'], sigma_over_a=ell), N_B=640, channel_subset=kappas if args.kappas else None,
                  history_label=H['label'], core_radius=core_radius, core_definition='the matter inside the dense outer root R_t (R_c where the record has no root)')
    vn = sorted(set(n for l in order for n in tab[l]['variants']))
    arrays = dict(R=R, N_applied=sl.N, v_applied=sl.v, s_total=s_tot, s_raw=s_raw, s_dust=s_dust, dust_site=dust_site, dust_cum=dust_cum,
                  eps_tot_record=rec['eps_tot'], J_tot_record=rec['J_tot'], kappas=np.array([c['kappa'] for c in chans]), weights=np.array([c['w'] for c in chans]),
                  T_tot_diag=np.diagonal(T_tot).copy(), B_eigs=kinfo['B_eigs'], V_eigs=kinfo.get('V_eigs', np.zeros(0)),
                  varC_table=np.array([[[p[2] for p in tab[l]['variants'][n]['per_channel']] if n in tab[l]['variants'] else [np.nan] * len(chans) for n in vn] for l in order]),
                  varP_table=np.array([[[p[3] for p in tab[l]['variants'][n]['per_channel']] if n in tab[l]['variants'] else [np.nan] * len(chans) for n in vn] for l in order]),
                  EF_per_channel=np.array([exch[n]['E_F_per_channel'] for n, _ in kernels]))
    arrays['excess_table'] = arrays['varC_table'] - arrays['varP_table']
    arrays.update(rows['arrays'])
    for l in order:
        for n, x in betas[l].items():
            arrays['beta_%s_%s' % (l, n)] = x
        arrays['site_%s' % l] = np.array([radii[l]['site']])
    for ch in chans:
        arrays['e_kappa%d' % ch['kappa']] = ch['e']
        arrays['T_diag_kappa%d' % ch['kappa']] = np.diagonal(ch['T']).copy()
        if 'eigD' in ch:
            arrays['eigD_kappa%d' % ch['kappa']] = ch['eigD']
    if args.save_kernels:
        arrays.update(B=rows['B'], V=V if V is not None else np.zeros(0), PA=PA if PA is not None else np.zeros(0))
    if args.save_grams:
        for ch in chans:
            arrays['G_C_kappa%d' % ch['kappa']] = ch['G_C']
            arrays['G_P_kappa%d' % ch['kappa']] = ch['G_P']
            arrays['T_kappa%d' % ch['kappa']] = ch['T']
    arrays.update(ev_arrays)
    head = dict(version=VERSION, code_md5=CODE_MD5_AT_START, modules=M.paths, modules_md5=M.md5, device=LA.device, history=history, T_over_rs=sl.T_over_rs, T=sl.T,
                slice=path, size=sl.size, dtype=str(sl.column_dtype), complex64=c64, records_json=records, fields_npz=fields, handoff_npz=handoff if os.path.isfile(handoff) else None,
                well_json=(well or {}).get('path'), fields_index=rec['index'], kernel=args.kernel, args=vars(args), status=status, channels=per,
                radii={l: {k: v for k, v in radii[l].items()} for l in order}, radius_labels=order, table_axes=dict(labels=order, variants=vn, kappas=[c['kappa'] for c in chans]),
                rows=rows['info'], response=resp, kernel_info={k: v for k, v in kinfo.items() if k not in ('V_eigs', 'B_eigs')}, variance=tab, negative_variance_flags=neg,
                exchange=exch, evolution_rows=evo, v1_1=v11, checks=checks, n_fail=sum(c['status'] == 'FAIL' for c in checks), n_pass=sum(c['status'] == 'PASS' for c in checks),
                seconds=time.time() - t_all)
    with open(pj, 'w') as fh:
        json.dump(jsonable(head), fh, indent=1)
    np.savez_compressed(os.path.join(out_dir, tag + '_arrays.npz'), **{k: np.asarray(v) for k, v in arrays.items()})
    log('written %s (.json, _arrays.npz): %d PASS, %d FAIL, %.0f s' % (os.path.join(out_dir, tag), head['n_pass'], head['n_fail'], head['seconds']))
    return 1 if head['n_fail'] else 0


# ============================================================================ the star mode (the S10 regression and the native-kernel star)
class StarSlice:
    """a static-star member in the slice's representation: Phi = U[:, f > 0] (closed multiplets), R2 = U[:, e < 0]."""


def star_channels(LA, M, star, chans0, controls=False):
    chans = []
    ns = star['nsites']
    for c0 in chans0:
        e, U, f = c0['e'], np.asarray(c0['U']), c0['f']
        if not bool(np.all((f < 1e-12) | (f > 1.0 - 1e-12))):
            raise RuntimeError('the star member has a fractional multiplet: the pure-state path does not apply')
        ch = dict(kappa=c0['kappa'], w=c0['w'], n_sea=int(np.sum(e < 0)), n_dust=int(np.sum((f > 0) & (e > 0))))
        ch['h_op'], ch['Q_op'], ch['hm_op'] = LA.op(c0['h']), LA.op(c0['Q']), LA.op(c0['hm'])
        ch['e'] = e
        mask = e < 0
        ch['mask'] = mask
        ch['mask_equals_neg'] = True
        ch['Uo'], ch['Uu'] = LA.to(U[:, mask]), LA.to(U[:, ~mask])
        ch['W_ou'] = LA.to(1.0 / (e[mask][:, None] - e[~mask][None, :]))
        ch['Phi'] = LA.to(U[:, f > 0])
        ch['Pref'] = ch['Uo']
        ch['n_occ'] = int(ch['Phi'].shape[1])
        rhoC = LA.site_density(ch['h_op'], ch['Phi'])
        jC = LA.site_density(ch['Q_op'], ch['Phi'])
        rhoR = LA.site_density(ch['h_op'], ch['Pref'])
        jR = LA.site_density(ch['Q_op'], ch['Pref'])
        ch['s'] = np.concatenate([ch['w'] * (rhoC - rhoR), ch['w'] * (jC - jR)])
        C = LA.herm(LA.mm(ch['Phi'], LA.adj(ch['Phi'])))
        P = LA.herm(LA.mm(ch['Pref'], LA.adj(ch['Pref'])))
        D = C - P
        ch['TrD'] = float(LA.host(D).trace().real) if not LA.gpu else float(LA.torch.trace(D).real.item())
        hs = (ch['h_op'], ch['Q_op'])
        I = LA.eye(2 * ns)
        ch['G_C'] = np.ascontiguousarray(bilinear(LA, hs, I - C, C, ns).real)
        ch['G_P'] = np.ascontiguousarray(bilinear(LA, hs, I - P, P, ns).real)
        ch['T'] = np.ascontiguousarray(bilinear(LA, hs, D, D, ns).real)
        if controls:
            ch['T_P'] = np.ascontiguousarray(bilinear(LA, hs, P, P, ns).real)
            ch['T_C'] = np.ascontiguousarray(bilinear(LA, hs, C, C, ns).real)
        del C, P, D, I
        chans.append(ch)
    return chans


def run_star(args, M, LA):
    S10 = M.S10
    star, _ = S10.load_star(args.star, args.member)
    t0 = time.time()
    chans0, E_top, extra_fill = S10.rebuild_channels(M.OPS, star, 0)
    log('star %s member %d: %d channels (kappa %s), N_B %s, E %.6f, R_surf %s, R_999 %s (%.0f s)'
        % (os.path.basename(args.star), star['member'], len(chans0), [c['kappa'] for c in chans0], star['N_B'], star['E'], star['R_surf'], star['R_999'], time.time() - t0))
    ns, R = star['nsites'], star['R']
    chans = star_channels(LA, M, star, chans0, controls=args.exchange_controls)
    s_tot = sum(ch['s'] for ch in chans)
    radii = {}
    for lab in ('R_surf', 'R_999'):
        if star.get(lab):
            x = float(star[lab])
            radii[lab] = dict(r=x, site=min(int(np.searchsorted(R, x)), ns - 1), source='the star record')
    ell = float(args.ell or star['ell'] or 8.0)
    rows = rows_stage(M, args, R, ell, s_tot[:ns], s_tot[ns:], star['g'], star['N'], star['v'], star['E'], radii, ok_label='STAR')
    radii, order = rows['radii'], rows['order']
    PA, rinfo = response_matrix(LA, chans, ns, args.quad_tol)
    rng = np.random.default_rng(11)
    xr = rng.normal(size=2 * ns)
    rel = fnorm(PA @ xr - response_matvec(LA, chans, ns, xr)) / fnorm(response_matvec(LA, chans, ns, xr))
    op = S10.ResponseOperator([dict(e=c0['e'], U=np.asarray(c0['U']), h=c0['h'], Q=c0['Q'], w=c0['w']) for c0 in chans0], ns)
    rel_s10 = fnorm(PA @ xr - op.matvec(xr)) / fnorm(op.matvec(xr))
    V, betas, kinfo = betas_and_kernel(rows, PA, ns, svd=args.svd)
    tab, neg = variance_table(chans, betas, radii, rows['M_H'], order, None, R, star['E'])
    # S10's own eigenbasis variance on the same betas (in-process)
    s10 = {}
    for lab in order:
        s10[lab] = {}
        for name, beta in betas[lab].items():
            vC = vS = 0.0
            for c0 in chans0:
                vv = S10.variance_of_vertex(c0['h'], c0['Q'], c0['e'], np.asarray(c0['U']), c0['f'], c0['fsea'], beta)
                vC += c0['w'] * vv['var_C']
                vS += c0['w'] * vv['var_sea']
            s10[lab][name] = dict(Var_C=vC, Var_C0=vS, rel_C=abs(vC - tab[lab]['variants'][name]['Var_C']) / max(abs(vC), 1e-300),
                                  rel_C0=abs(vS - tab[lab]['variants'][name]['Var_C0']) / max(abs(vS), 1e-300))
    worst = max(max(x['rel_C'], x['rel_C0']) for lab in s10 for x in s10[lab].values())
    kernels = [('bare', rows['B']), ('dressed', V)]
    exch, _ = exchange_stage(chans, s_tot, kernels, float(star['N_B']), star['m'], star['E'], {})
    reg = {}
    if args.star_reference:
        Rf = json.load(open(args.star_reference))
        for lab, val in Rf.get('Var_C_full', {}).items():
            if lab in tab:
                reg['Var_C_full_%s' % lab] = abs(tab[lab]['variants']['full']['Var_C'] - val) / abs(val)
        for lab, val in Rf.get('M_resolved', {}).items():
            if lab in rows['M_H']:
                reg['M_resolved_%s' % lab] = abs(rows['M_H'][lab] - val) / abs(val)
        for lab, val in Rf.get('Var_C_sharp_window', {}).items():
            if lab in tab:
                reg['Var_C_sharp_window_%s' % lab] = abs(tab[lab]['variants']['sharp_window']['Var_C'] - val) / abs(val)
    out = dict(version=VERSION, mode='star', star=args.star, member=star['member'], kernel=args.kernel, ell=ell, E=star['E'], N_B=star['N_B'], rows=rows['info'],
               response=dict(rinfo, matrix_vs_operator_rel=rel, matrix_vs_S10_operator_rel=rel_s10, PA_Phi_record_over_Phi=fnorm(PA @ rows['Phi_app']) / fnorm(rows['Phi_app'])),
               kernel_info={k: v for k, v in kinfo.items() if k not in ('V_eigs', 'B_eigs')}, variance=tab, S10_in_process=s10, worst_rel_vs_S10_variance_of_vertex=worst,
               exchange=exch, regression=dict(reference=args.star_reference, rel=reg, pass_=bool(reg and max(reg.values()) <= 1e-9), tol=1e-9), seconds=time.time() - t0)
    os.makedirs(os.path.join(args.out, 'star'), exist_ok=True)
    tag = 'core_hf_star_%s_m%d_%s' % (os.path.basename(args.star).replace('.json', ''), star['member'], args.kernel)
    with open(os.path.join(args.out, 'star', tag + '.json'), 'w') as fh:
        json.dump(jsonable(out), fh, indent=1)
    arrays = dict(rows['arrays'])
    for l in order:
        for n, x in betas[l].items():
            arrays['beta_%s_%s' % (l, n)] = x
    np.savez_compressed(os.path.join(args.out, 'star', tag + '_arrays.npz'), **arrays)
    log('star mode: P^A vs operator %.1e (vs S10 operator %.1e); worst rel vs S10.variance_of_vertex %.1e; regression %s -> %s; written %s'
        % (rel, rel_s10, worst, {k: '%.1e' % v for k, v in reg.items()}, 'PASS' if out['regression']['pass_'] else ('FAIL' if reg else 'no reference'), tag))
    return out


# ============================================================================ the self-test
def selftest(args, M, LA):
    S10, OPS = M.S10, M.OPS
    fails = 0
    res = []

    def report(name, val, tol, ok=None):
        nonlocal fails
        ok = (val <= tol) if ok is None else bool(ok)
        fails += 0 if ok else 1
        res.append((name, val, tol, ok))
        log('%s  %-96s %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol))

    rng = np.random.default_rng(5)
    log('%s self-test (device %s)' % (VERSION, LA.device))
    # S1 the Gaussian kernel and S10's S2/S3 checks with it
    ns = 12
    R = np.arange(ns) + 1.0
    ell, G = 3.0, 1.0 / 64.0
    kt = GaussShellKernel(R, ell, R[-1] + 8.0 * ell + 1.0, 0.02, S10, M.EF2)
    report('S1 Gaussian kernel l=0 normalization (exact discrete)', kt.norm_err, 1e-14)
    report('S1 Gaussian kernel l=0 raw normalization (quadrature)', kt.norm_raw, 1e-8)
    report('S1 Gaussian kernel l=1 first moment = shell radius', kt.mom1_err, 1e-8)
    if M.EF2 is not None:
        a0, a1 = M.EF2.kernels(kt.rf[:500], R, ell)
        b0, b1 = gauss_kernels_formula(kt.rf[:500], R, ell)
        report('S1 re-typed formula equals engine_fields_v2.kernels (max abs rel)', float(max(np.max(np.abs(a0 - b0)) / np.max(np.abs(a0)), np.max(np.abs(a1 - b1)) / np.max(np.abs(a1)))), 1e-15)
    e = 0.5 * np.exp(-((R - 4.0) / 3.0) ** 2) * R ** 2
    j = -0.05 * e
    sol = S10.radial_solve(kt, e, j, G)
    cols = [0, 5, 11, 12, 17, 23]
    rws = [kt.ns // 2, kt.ns]
    B, bMr, bMc = S10.linearized(kt, sol, G, True, rows=rws, cols=cols)
    Bcs, bMcs = S10.complex_step_columns(kt, e, j, G, cols)
    report('S1 linearized vs complex-step Hessian B (Gaussian kernel)', max(fnorm(B[:, nu] - Bcs[nu]) / fnorm(Bcs[nu]) for nu in cols), 1e-8)
    report('S1 linearized vs complex-step dM/ds', max(fnorm(bMc[:, k] - bMcs[nu]) / fnorm(bMcs[nu]) for k, nu in enumerate(cols)), 1e-8)
    report('S1 first law dM(R_out)/ds = Phi', fnorm(bMr[1] - sol['Phi'].real) / fnorm(sol['Phi'].real), 1e-9)
    report('S1 Hessian symmetry', fnorm(B - B.T) / fnorm(B), 1e-8)
    Bnc, _, _ = S10.linearized(kt, sol, G, False, rows=rws, cols=[])
    report('S1 no-current variant: current columns zero', fnorm(Bnc[:, ns:]), 0.0)
    # S2 the block-sum bilinear form against brute force
    ns4 = 4
    h4, Q4, _ = OPS.chain_parts(ns4, 0.2, 1.0, False)
    X4 = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    X4 = 0.5 * (X4 + X4.conj().T)
    Y4 = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    Y4 = 0.5 * (Y4 + Y4.conj().T)
    ch4 = dict(h_op=LA.op(h4), Q_op=LA.op(Q4))
    Tb = bilinear(LA, (ch4['h_op'], ch4['Q_op']), LA.to(X4), LA.to(Y4), ns4)
    Tref = np.empty((8, 8), complex)
    Av = []
    for t, hh in ((0, h4), (1, Q4)):
        for n in range(ns4):
            Pn = np.zeros((8, 8))
            Pn[2 * n, 2 * n] = Pn[2 * n + 1, 2 * n + 1] = 1.0
            Av.append(0.5 * (Pn @ hh + hh @ Pn))
    for mu in range(8):
        for nu in range(8):
            Tref[mu, nu] = np.trace(Av[mu] @ X4 @ Av[nu] @ Y4)
    report('S2 block-sum bilinear form Tr(A_mu X A_nu Y) vs brute force (max abs rel)', float(np.max(np.abs(Tb - Tref)) / np.max(np.abs(Tref))), 1e-13)
    xv = rng.normal(size=8)
    Ax = LA.vertex(ch4, xv, ns4).to_dense()
    report('S2 vertex A_x = sum x_mu A_mu (banded)', float(np.max(np.abs(Ax - sum(xv[k] * Av[k] for k in range(8))))), 1e-14)
    Xr = rng.normal(size=(8, 5)) + 1j * rng.normal(size=(8, 5))
    hm4 = OPS.clocked(h4, Q4, 0.9 + 0.05 * np.arange(ns4), -0.2 * np.ones(ns4))
    bop = LA.op(hm4)
    report('S2 banded left and right application equal the dense products (h_m, random X)',
           float(max(np.max(np.abs(LA.host(bop.left(LA.to(Xr))) - hm4 @ Xr)), np.max(np.abs(LA.host(bop.right(LA.to(Xr.T.copy()))) - Xr.T @ hm4)))), 1e-14)
    report('S2 banded reconstruction equals the dense operator', float(np.max(np.abs(bop.to_dense() - hm4))), 0.0)
    # S3 the fixture channel: the Gram-route variances against S10.variance_of_vertex
    ns = 12
    R = np.arange(ns) + 1.0
    Nf = 0.75 + 0.2 * np.cos(R / 5.0)
    vf = -0.4 * np.exp(-((R - 6.0) / 4.0) ** 2)
    fix = []
    for kappa in (1, 2):
        h, Q, _ = OPS.chain_parts(ns, 0.2, float(kappa), False)
        hm = OPS.clocked(h, Q, Nf, vf)
        e, U = S10.spectrum(hm) if hasattr(S10, 'spectrum') else OPS.spectrum(hm)
        e, U = np.asarray(e), np.asarray(U)
        nd = 2 if kappa == 1 else 1
        pos = np.where(e > 0)[0][:nd]
        f = np.where(e < 0, 1.0, 0.0)
        f[pos] = 1.0
        fix.append(dict(kappa=kappa, w=16.0 * kappa, h=h, Q=Q, hm=hm, e=e, U=U, f=f, fsea=np.where(e < 0, 1.0, 0.0), pos=pos))

    def unitary(n):
        Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Qm, Rm = np.linalg.qr(Z)
        return Qm * (np.diagonal(Rm) / np.abs(np.diagonal(Rm)))[None, :]

    def fixture_channel(c, rotate):
        ch = dict(kappa=c['kappa'], w=c['w'], e=c['e'], mask=c['e'] < 0, mask_equals_neg=True)
        ch['h_op'], ch['Q_op'], ch['hm_op'] = LA.op(c['h']), LA.op(c['Q']), LA.op(c['hm'])
        Uo, Uu = c['U'][:, c['e'] < 0], c['U'][:, c['e'] > 0]
        ch['Uo'], ch['Uu'] = LA.to(Uo), LA.to(Uu)
        ch['W_ou'] = LA.to(1.0 / (c['e'][c['e'] < 0][:, None] - c['e'][c['e'] > 0][None, :]))
        Phi = np.concatenate([Uo, c['U'][:, c['pos']]], axis=1)
        R2 = Uo.copy()
        if rotate:
            Phi = Phi @ unitary(Phi.shape[1])
            R2 = R2 @ unitary(R2.shape[1])
        ch['Phi'], ch['Pref'] = LA.to(Phi), LA.to(R2)
        C = LA.herm(LA.mm(ch['Phi'], LA.adj(ch['Phi'])))
        P = LA.herm(LA.mm(ch['Pref'], LA.adj(ch['Pref'])))
        D = C - P
        I = LA.eye(2 * ns)
        hs = (ch['h_op'], ch['Q_op'])
        ch['G_C'] = bilinear(LA, hs, I - C, C, ns).real
        ch['G_P'] = bilinear(LA, hs, I - P, P, ns).real
        ch['T'] = bilinear(LA, hs, D, D, ns).real
        ch['C'], ch['P'] = C, P
        ch['s'] = np.concatenate([c['w'] * (LA.site_density(ch['h_op'], ch['Phi']) - LA.site_density(ch['h_op'], ch['Pref'])),
                                  c['w'] * (LA.site_density(ch['Q_op'], ch['Phi']) - LA.site_density(ch['Q_op'], ch['Pref']))])
        return ch

    chA = [fixture_channel(c, False) for c in fix]
    chB = [fixture_channel(c, True) for c in fix]
    worst = dict(gram_vs_s10=0.0, direct_vs_s10=0.0, sea_vs_s10=0.0, rot=0.0, static=0.0, mean=0.0)
    for _ in range(20):
        beta = rng.normal(size=2 * ns)
        for c, a, b in zip(fix, chA, chB):
            vv = S10.variance_of_vertex(c['h'], c['Q'], c['e'], c['U'], c['f'], c['fsea'], beta)
            gC, gP = quad(a['G_C'], beta), quad(a['G_P'], beta)
            dv = direct_variances(LA, a, beta, ns)
            worst['gram_vs_s10'] = max(worst['gram_vs_s10'], abs(gC - vv['var_C']) / vv['var_C'])
            worst['direct_vs_s10'] = max(worst['direct_vs_s10'], abs(dv['var_C'] - vv['var_C']) / vv['var_C'])
            worst['sea_vs_s10'] = max(worst['sea_vs_s10'], abs(gP - vv['var_sea']) / vv['var_sea'], abs(dv['var_P'] - vv['var_sea']) / vv['var_sea'])
            worst['rot'] = max(worst['rot'], abs(quad(b['G_C'], beta) - gC) / gC, abs(quad(b['G_P'], beta) - gP) / gP, fnorm(b['s'] - a['s']) / fnorm(a['s']), fnorm(b['T'] - a['T']) / fnorm(a['T']))
            A = LA.vertex(b, beta, ns)
            AC = LA.apply(A, b['C'])
            comm = AC - LA.adj(AC)
            worst['static'] = max(worst['static'], abs(0.5 * LA.fro2(comm) - gC) / gC)
    report('S3 Gram-route Var_C vs hf_star_variance_v3.variance_of_vertex (eigen-slice, 20 vertices)', worst['gram_vs_s10'], 1e-13)
    report('S3 direct Slater Var_C vs S10', worst['direct_vs_s10'], 1e-13)
    report('S3 sea part (Gram and direct) vs S10 var_sea', worst['sea_vs_s10'], 1e-13)
    report('S3 invariance under unitary rotations of the occupied and reference columns (Var, s, T)', worst['rot'], 1e-12)
    report('S3 static identity Var_C = (1/2)||[A, C]||_F^2 on the rotated state', worst['static'], 1e-13)
    GCt = sum(a['w'] * a['G_C'] for a in chA)
    evG = np.linalg.eigvalsh(0.5 * (GCt + GCt.T))
    report('S3 total Gram positive semidefinite (most negative over largest)', float(max(-evG[0], 0.0) / evG[-1]), 1e-12)
    # S4 the response
    PA, rinfo = response_matrix(LA, chB, ns, 1e-13, log_fn=lambda *a: None)
    PAref = np.zeros((2 * ns, 2 * ns))
    for c in fix:
        S10.accumulate_response(c['h'], c['Q'], c['e'], c['U'], c['fsea'], c['w'], PAref)
    PAref = 0.5 * (PAref + PAref.T)
    report('S4 exponential-sum P^A vs S10.accumulate_response (explicit; %d nodes, quadrature %.1e)' % (rinfo['nodes'], rinfo['quadrature_max_rel_err']), fnorm(PA - PAref) / fnorm(PAref), 1e-11)
    x9 = rng.normal(size=2 * ns)
    op9 = S10.ResponseOperator([dict(e=c['e'], U=c['U'], h=c['h'], Q=c['Q'], w=c['w']) for c in fix], ns)
    report('S4 P^A x vs S10.ResponseOperator.matvec (random vector)', fnorm(PA @ x9 - op9.matvec(x9)) / fnorm(op9.matvec(x9)), 1e-11)
    report('S4 the device operator route vs S10.ResponseOperator.matvec', fnorm(response_matvec(LA, chB, ns, x9) - op9.matvec(x9)) / fnorm(op9.matvec(x9)), 1e-12)
    xg = np.concatenate([Nf, vf])
    report('S4 P^A on the fixture generator (N, v) vanishes (over ||P^A|| ||x||)', fnorm(PA @ xg) / (fnorm(PA) * fnorm(xg)), 1e-12)
    report('S4 P^A symmetric (raw asymmetry)', rinfo['asymmetry_raw'], 1e-12)
    # S5 the exchange: the rank-one identity, the one-body limit
    Vr = rng.normal(size=(2 * ns, 2 * ns))
    Vr = 0.5 * (Vr + Vr.T)
    dvec = rng.normal(size=2 * ns) + 1j * rng.normal(size=2 * ns)
    dvec /= np.linalg.norm(dvec)
    Dd = LA.to(np.outer(dvec, dvec.conj()))
    T1 = bilinear(LA, (chA[0]['h_op'], chA[0]['Q_op']), Dd, Dd, ns).real
    Ad = []
    for t, hh in ((0, fix[0]['h']), (1, fix[0]['Q'])):
        for n in range(ns):
            Pn = np.zeros((2 * ns, 2 * ns))
            Pn[2 * n, 2 * n] = Pn[2 * n + 1, 2 * n + 1] = 1.0
            Ad.append(0.5 * (Pn @ hh + hh @ Pn))
    sd = np.array([np.real(np.vdot(dvec, A_ @ dvec)) for A_ in Ad])
    EF1 = -0.5 * float(np.sum(Vr * T1))
    report('S5 one particle: E_F = -(1/2) s_d^T V s_d = -E_pair(s_d)', abs(EF1 + 0.5 * float(sd @ Vr @ sd)) / abs(0.5 * float(sd @ Vr @ sd)), 1e-13)
    Tsum = sum(a['w'] * a['T'] for a in chA)
    report('S5 exchange block Hermitian/symmetric (weighted real part)', fnorm(Tsum - Tsum.T) / fnorm(Tsum), 1e-12)
    # S6 GMRES vs LU for (I + B P^A) x = b
    B9 = Vr * (0.3 / float(np.linalg.norm(Vr @ PA, 2)))
    b9 = rng.normal(size=2 * ns)
    xg9, hist9, nit9, _ = S10.gmres_solve(lambda v: v + B9 @ response_matvec(LA, chB, ns, v), b9, 1e-13, 80)
    xlu = np.linalg.solve(np.eye(2 * ns) + B9 @ PA, b9)
    report('S6 GMRES on the operator route equals the LU solve with the explicit P^A (%d applications)' % nit9, fnorm(xg9 - xlu) / fnorm(xlu), 1e-10)
    # S7 the reader on a synthetic slice in the engine's layout; the sources against engine_ops_v1
    import tempfile
    tmp = tempfile.mkdtemp(prefix='core_hf_selftest_')
    arrs = dict(T=12.0 * 64.0, rs=64.0, N_applied=Nf, v_applied=vf, R=R, kappas=np.array([1, 2]), weights=np.array([16.0, 32.0]), dtype='complex128')
    for c in fix:
        Uo = c['U'][:, c['e'] < 0] @ unitary(ns)
        dust = c['U'][:, c['pos']] @ unitary(len(c['pos']))
        arrs['sea_%d' % c['kappa']] = Uo.astype(np.complex128)
        arrs['dust_%d' % c['kappa']] = dust.astype(np.complex128)
        arrs['f_%d' % c['kappa']] = np.ones(len(c['pos']))
        arrs['R2_%d' % c['kappa']] = (c['U'][:, c['e'] < 0] @ unitary(ns)).astype(np.complex128)
        arrs['e_%d' % c['kappa']] = c['e']
        arrs['mask_%d' % c['kappa']] = np.packbits(c['e'] < 0)
    path = os.path.join(tmp, 'orbitals_selftest_T0012.00.npz')
    np.savez(path, **arrs)
    sl = SliceFile(path)
    Zr = np.load(path)
    d_read = max(float(np.max(np.abs(np.asarray(sl.mmap('sea_%d' % k)) - Zr['sea_%d' % k]))) for k in (1, 2))
    d_read = max(d_read, float(np.max(np.abs(sl.N - Nf))), float(np.max(np.abs(sl.v - vf))), float(np.max(np.abs(sl.R - R))))
    report('S7 built-in reader reproduces np.load (max abs)', d_read, 0.0)
    report('S7 reader: kappas, weights, dust counts, mask unpacking', 0.0, 0.0, ok=(sl.kappas == [1, 2] and sl.weights == {1: 16.0, 2: 32.0} and sl.n_dust == {1: 2, 2: 1}
                                                                                 and all(np.array_equal(sl.mask_stored[k], fix[k - 1]['e'] < 0) for k in (1, 2))))
    if M.IO is not None:
        rd = M.IO.SliceReader(path, log=lambda s: None)
        c1 = rd.channel(1, orthonormalize=False)
        P1, f1 = sl.columns(1)
        report('S7 reader vs slices144_io_v1.SliceReader (Phi, f of channel 1; max abs)', float(max(np.max(np.abs(c1['Phi'] - P1)), np.max(np.abs(c1['f'] - f1)))), 0.0)
    srcs = np.zeros(2 * ns)
    sref = np.zeros(2 * ns)
    worst_vid = 0.0
    for c in fix:
        ch = build_channel(LA, M, sl, c['kappa'], 0.2, False, controls=False, floor_control=False)
        srcs += ch['s']
        Pr, fr = sl.columns(c['kappa'])
        er, Jr = OPS.sources_dynamic(c['h'], c['Q'], Pr, fr, sl.R2(c['kappa']), c['w'])
        sref += np.concatenate([er, Jr])
        worst_vid = max(worst_vid, ch['vertex_identity_rel'], ch['loewdin_orthonormality'], abs(ch['TrD'] - len(c['pos'])))
        report('S7 channel %d: mask == (e < 0), e vs stored, P_ref resolution, commutator (all at roundoff)' % c['kappa'],
               max(ch['e_vs_stored_abs'], ch['pref_resolution_defect'], ch['commutator_rel']), 1e-11, ok=(ch['mask_equals_neg'] and ch['e_vs_stored_abs'] < 1e-11
                                                                                                          and ch['pref_resolution_defect'] < 1e-11 and ch['commutator_rel'] < 1e-11))
    report('S7 sources from the slice = engine_ops_v1.sources_dynamic (max abs rel)', float(np.max(np.abs(srcs - sref)) / np.max(np.abs(sref))), 1e-12)
    report('S7 vertex identity, Loewdin identity on orthonormal input, Tr D = n_dust', worst_vid, 1e-12)
    # S8 engine_rows_v2 on the Painleve-Gullstrand vacuum
    if M.RW is not None:
        Mv = 10.0
        rr = np.linspace(2.0 * Mv + 1.0, 200.0, 200)
        one = lambda x: np.ones_like(np.asarray(x, float))  # noqa: E731
        vpg = lambda x: -np.sqrt(2.0 * Mv / np.asarray(x, float))  # noqa: E731
        z = np.zeros_like(rr)
        bb = M.RW.brackets(one, vpg, z, z, z, 0.25, rr, 3.0 / 32.0)
        report('S8 engine_rows_v2 brackets vanish on the Painleve-Gullstrand vacuum (rr, tt over the largest term)',
               float(max(np.max(np.abs(bb['rhs_rr']) / bb['terms_rr']), np.max(np.abs(bb['rhs_tt']) / bb['terms_tt']))), 1e-7)
    else:
        log('S8 skipped: engine_rows_v2 not importable (%s)' % M.err.get('engine_rows_v2'))
    log('SELFTEST %s (%d failures of %d)' % ('PASS' if fails == 0 else 'FAIL', fails, len(res)))
    return fails


def selftest_slice(args, M, LA):
    """the elementary limits on a real slice: the sources identity over all channels (no eigh), then the channel subset's identities."""
    fails = 0
    path, name, cands = find_slice(args.history, float(args.selftest_slice), args.slice_dir)
    if path is None:
        log('selftest-slice: no copy of %s (searched %s)' % (name, cands))
        return 1
    H = HIST[args.history]
    rec = load_record(args.records or os.path.join(H['base'], 'collapse_%s.json' % H['tag']), args.fields or os.path.join(H['base'], 'collapse_%s_fields.npz' % H['tag']), float(args.selftest_slice))
    sl = SliceFile(path)
    ns = sl.nsites
    log('selftest-slice %s: %d sites, %s, kappas %s' % (path, ns, sl.column_dtype, sl.kappas))
    checks = []
    chk = chk_factory(checks)
    t0 = time.time()
    s_raw = np.zeros(2 * ns)
    for k in sl.kappas:
        h, Q = chain_ops(M.OPS, ns, rec['m'], k, rec['half'], sl.R)
        Pr, f = sl.columns(k)
        er, Jr = M.OPS.sources_dynamic(sp.csr_matrix(h), sp.csr_matrix(Q), Pr, f, sl.R2(k), sl.weights[k])
        s_raw += np.concatenate([er, Jr])
        del h, Q, Pr
    tol = 1e-5 if sl.complex64 else 1e-12
    chk('L1 the stored columns of all %d channels reproduce the record eps_tot (max rel; %.0f s)' % (len(sl.kappas), time.time() - t0), float(np.max(np.abs(s_raw[:ns] - rec['eps_tot'])) / np.max(np.abs(rec['eps_tot']))), tol)
    chk('L1b the same for J_tot (max rel)', float(np.max(np.abs(s_raw[ns:] - rec['J_tot'])) / np.max(np.abs(rec['J_tot']))), tol)
    kappas = args.kappas or [1]
    chans = []
    for k in kappas:
        ch = build_channel(LA, M, sl, k, rec['m'], rec['half'], controls=False, floor_control=False)
        chans.append(ch)
        chk('L2 kappa %d vertex identity (rel)' % k, ch['vertex_identity_rel'], 1e-12)
        chk('L3 kappa %d static identity Var_C(h_m) = (1/2)||[h_m, C]||^2 (rel; both O(1): %.4e)' % (k, ch['static_identity_hm']['var_C']), ch['static_identity_hm']['rel_diff'], 1e-12)
        chk('L4 kappa %d Var_P(h_m) of the eigen-sea = 0 (relative to Var_C(h_m))' % k, ch['var_sea_hm'] / max(ch['static_identity_hm']['var_C'], 1e-300), 1e-20)
        chk('L5 kappa %d mask == (e < 0), e vs stored e_k (abs)' % k, ch['e_vs_stored_abs'], 1e-11, ok=(ch['mask_equals_neg'] and ch['e_vs_stored_abs'] < 1e-11))
        chk('L6 kappa %d P_ref resolution defect (expected 1e-12 complex128, 1e-7 complex64)' % k, ch['pref_resolution_defect'], 1e-5, info=True)
        rng = np.random.default_rng(k)
        worst = 0.0
        for beta in (np.concatenate([sl.N, sl.v]), rng.normal(size=2 * ns)):
            worst = max(worst, gram_vs_direct(ch, beta, direct_variances(LA, ch, beta, ns)))
        chk('L7 kappa %d Gram route vs direct Slater variance (the generator and a random vertex; over max(Var, 1e-6 ||A Phi||^2))' % k, worst, 1e-9)
    if not args.no_response:
        PA, rinfo = response_matrix(LA, chans, ns, args.quad_tol)
        phi = np.concatenate([sl.N, sl.v])
        chk('L8 P^A (channels %s) on Phi_applied over ||P^A|| ||Phi|| (%d nodes, %.0f s)' % (kappas, rinfo['nodes'], rinfo['seconds']), fnorm(PA @ phi) / (fnorm(PA) * fnorm(phi)), 1e-6 if sl.complex64 else 1e-9)
        xr = np.random.default_rng(7).normal(size=2 * ns)
        chk('L8b P^A matrix vs the operator route (random vector, rel)', fnorm(PA @ xr - response_matvec(LA, chans, ns, xr)) / fnorm(response_matvec(LA, chans, ns, xr)), max(1e-10, 100.0 * args.quad_tol))
    else:
        xr = np.random.default_rng(7).normal(size=2 * ns)
        phi = np.concatenate([sl.N, sl.v])
        y = response_matvec(LA, chans, ns, phi)
        chk('L8 (operator route only, --no-response) P^A Phi_applied over ||P^A x_random|| ||Phi||/||x||', fnorm(y) / fnorm(phi) / (fnorm(response_matvec(LA, chans, ns, xr)) / fnorm(xr)), 1e-6 if sl.complex64 else 1e-9)
    radii = {'R_c': dict(r=float(rec['record']['R_c']), site=min(int(np.searchsorted(sl.R, rec['record']['R_c'])), ns - 1), source='R_c')}
    rows = rows_stage(M, args, sl.R, float(rec['ell']), rec['eps_tot'], rec['J_tot'], rec['g'], sl.N, sl.v, float(rec['record']['E']), radii)
    chk('L9 native-kernel re-solve: max rel dev of the back-averaged N, v from the applied (label %s)' % rows['info']['field_check']['label'],
        max(rows['info']['field_check']['N_rel_maxdev'], rows['info']['field_check']['v_rel_maxdev']), args.field_tol)
    chk('L9b first law b(R_out) = Phi_resolved (rel)', rows['info']['hessian']['first_law_raw'], 1e-9)
    chk('L9c E resolved vs the record (rel)', rows['info']['field_check']['E_rel_dev'], 1e-5, info=True)
    fails = sum(c['status'] == 'FAIL' for c in checks)
    log('SELFTEST-SLICE %s (%d failures of %d)' % ('PASS' if fails == 0 else 'FAIL', fails, len(checks)))
    return fails


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--history', default='prod', choices=sorted(HIST))
    ap.add_argument('--clocks', default='16', help='comma list of T/r_s or all')
    ap.add_argument('--slice-dir', default=None)
    ap.add_argument('--records', default=None)
    ap.add_argument('--fields', default=None)
    ap.add_argument('--handoff', default=None)
    ap.add_argument('--well-dir', default=None)
    ap.add_argument('--engine-dir', default=ENGINE_CODE)
    ap.add_argument('--s10-dir', default=S10_DIR)
    ap.add_argument('--slices144-dir', default=SLICES144_DIR)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--device', default='cpu', help='cpu or cuda[:n]')
    ap.add_argument('--kernel', default='gauss', choices=['gauss', 'compact'])
    ap.add_argument('--ell', type=float, default=None)
    ap.add_argument('--dr', type=float, default=None, help='the re-solve grid (0.1 on the core, 0.05 in the star mode)')
    ap.add_argument('--field-tol', type=float, default=1e-3)
    ap.add_argument('--quad-tol', type=float, default=1e-13)
    ap.add_argument('--no-response', action='store_true')
    ap.add_argument('--gmres-check', dest='gmres_check', action='store_true', default=True)
    ap.add_argument('--no-gmres-check', dest='gmres_check', action='store_false')
    ap.add_argument('--gmres-tol', type=float, default=1e-12)
    ap.add_argument('--gmres-maxit', type=int, default=80)
    ap.add_argument('--svd', dest='svd', action='store_true', default=True)
    ap.add_argument('--no-svd', dest='svd', action='store_false')
    ap.add_argument('--cs-cols', dest='cs_cols', action='store_true', default=True)
    ap.add_argument('--no-cs-cols', dest='cs_cols', action='store_false')
    ap.add_argument('--exchange-controls', dest='exchange_controls', action='store_true', default=True)
    ap.add_argument('--no-exchange-controls', dest='exchange_controls', action='store_false')
    ap.add_argument('--floor-control', action='store_true', help='the raw-column variance control also on complex128 slices')
    ap.add_argument('--eig-d', action='store_true', help='the spectrum of D per channel (one 4096 eigvalsh per channel)')
    ap.add_argument('--evolution', dest='evolution', action='store_true', default=True)
    ap.add_argument('--no-evolution', dest='evolution', action='store_false')
    ap.add_argument('--h-frac', type=float, default=1.0 / 32.0, help='the dense derivative step ell x h_frac (engine_rows_v2)')
    ap.add_argument('--ref-arrays', default=None, help='v1.1: a previous run of the same slice (_arrays.npz): its dressed betas on this run\'s Grams')
    ap.add_argument('--dr-refine', type=lambda s: [float(x) for x in s.split(',') if x.strip()], default=None, help='v1.1: re-solve grids for M_H and the noref vertex')
    ap.add_argument('--kappas', type=lambda s: [int(x) for x in s.split(',') if x.strip()], default=None, help='a channel subset (smoke control)')
    ap.add_argument('--save-kernels', action='store_true')
    ap.add_argument('--save-grams', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--selftest-slice', default=None, help='T/r_s of a real slice for the elementary limits (with --history, --kappas)')
    ap.add_argument('--star', default=None, help='the star mode: a static-star family json')
    ap.add_argument('--member', type=int, default=1)
    ap.add_argument('--star-reference', default=None, help='json {Var_C_full: {lab: x}, M_resolved: {lab: x}, Var_C_sharp_window: {lab: x}} (the S10 regression)')
    args = ap.parse_args(argv)
    if args.nice > 0:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    M = Mods(args.s10_dir, args.engine_dir, args.slices144_dir)
    LA = Backend(args.device)
    log('%s: modules %s md5 %s; not importable %s; device %s; threads %s' % (VERSION, M.paths, M.md5, M.err, args.device, os.environ.get('OMP_NUM_THREADS')))
    rc = 0
    if args.selftest:
        rc = selftest(args, M, LA)
    if args.selftest_slice is not None:
        rc += selftest_slice(args, M, LA)
    if args.selftest or args.selftest_slice is not None:
        if args.star:
            out = run_star(args, M, LA)
            rc += 0 if (out['regression']['pass_'] or not out['regression']['rel']) else 1
        return 1 if rc else 0
    if args.star:
        out = run_star(args, M, LA)
        return 0 if (out['regression']['pass_'] or not out['regression']['rel']) else 1
    os.makedirs(args.out, exist_ok=True)
    clocks = HIST[args.history]['clocks'] if args.clocks.strip().lower() == 'all' else [float(x) for x in args.clocks.split(',') if x.strip()]
    for t in clocks:
        rc = max(rc, run_slice(args, M, LA, args.history, float(t)))
    return rc


if __name__ == '__main__':
    sys.exit(main())
