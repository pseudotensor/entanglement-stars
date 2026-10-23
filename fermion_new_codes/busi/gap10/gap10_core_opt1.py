"""gap10_core_opt1.py -- the GAP10 core with a faster Chebyshev sweep (optimization candidate 1 of
busi/notes/GPU_opt_judgment_v1.md). A drop-in for gap10_core_v2: every name of gap10_core_v2 is re-exported unchanged
(fields, weights, certificates, receivers, geometry, packets, rays, the Taylor stepper); only LatticeH.apply and
cheb_sweep are replaced, with the same signatures and the same returned quantities (the state, P[k], R[k], Nr[k], the face
list, the 1-element-list input form) to roundoff. The run driver takes it through gap10_run_v2_opt1.py (which binds the
name gap10_core_v2 to this module before running gap10_run_v2.py unchanged); the [run] line and the json 'core' field then
say gap10_core_opt1.

Why (the roofline): the production step is memory-bound. v2's apply moves about 80 state buffers of bytes per application
(torch.roll copies, full-size temporaries for each of the 12 shifted terms, the weight fields re-read per batch element and
component) and the recurrence another 9; at B = 4 and L = 225 (2.9 GB per state buffer) that is about 260 GB per
Chebyshev step: 177 ms measured on the H100, 1.5 TB/s against the 3.35 TB/s of HBM3. The flops are irrelevant (about
10 GFLOP per step, 0.3 ms at the FP64 peak).

Two backends, chosen by the environment variable GAP10_OPT1_BACKEND (compiled | eager | auto; default auto):
  eager    : the same complex128 arithmetic as v2, but the 12 shifted terms are accumulated directly into the output
             through narrow views (add_ / addcmul_ with the complex coefficient), no roll, no full-size temporaries, the
             2/dE scale folded into the coefficients: about 51 buffers per step (an expected 1.6 to 1.8x).
  compiled : the whole Chebyshev step (h phi scaled, minus phi_prev, the accumulation into the sum) as fused kernels by
             torch.compile (inductor, Triton) on the real and imaginary parts (float64: Triton has no complex type) of a
             halo-padded state in the layout (B, L+4, L+4, L+4, 4) so that the 4 components of a site are adjacent and
             every weight is read once per site: about 6 to 10 buffers per step (an expected 5 to 10x). The real
             operations are the ones v2's complex arithmetic performs (every coefficient is purely real or purely
             imaginary, so v2's complex products are signed swaps with exact zero partial products), summed in v2's
             order; only FMA contraction and the slab order of the eager backend can differ at the last bit, so the gate
             is a tolerance (gap10_opt_bench_v1.py), not bitwise.
  auto     : compiled on a CUDA device when torch.compile is available and the first compiled step succeeds, else eager;
             the choice is printed once ([opt1] line). For production set the gated backend explicitly: then a failure
             raises instead of falling back.
The sweep keeps the certificate exact: R[k] = ||B T_k psi|| by v2's face formulas on the same T_k vectors (to roundoff),
P[k] by the same gathers and the same Gram products (bitwise on the same vector), Nr[k] the same norms.
"""
import importlib
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gap10_core_v2 as _v2                       # noqa: E402  (the real v2 module, imported before any name binding)
from gap10_core_v2 import *                       # noqa: E402,F401,F403  re-export every public name of v2
from gap10_core_v2 import CD, FD, ALPHA_NZ, BETA_DIAG, axpy_, vec_norm   # noqa: E402  (explicit, used below)
from gap10_core_v2 import _trapz, _nz_table, _spline_interpolant, _allocate_at, _axpy_supported   # noqa: E402,F401  (the tests use C._trapz)

BACKEND_REQUEST = os.environ.get('GAP10_OPT1_BACKEND', 'auto').lower()
if BACKEND_REQUEST not in ('auto', 'compiled', 'eager'):
    raise ValueError("GAP10_OPT1_BACKEND must be auto, compiled or eager")
HALO = 2
_PAD = (0, 0, HALO, HALO, HALO, HALO, HALO, HALO)   # F.pad of (B, X, Y, Z, 4): dims Z, Y, X by 2 each, the component dim by 0
_compiled = {}                                      # name -> compiled function (built lazily)


def _say(msg):
    print(f"[opt1] {msg}", flush=True)


# ------------------------------------------------------------------ the coefficient tables (from v2's alpha tables)
def _onehop_tables():
    """Per axis: the partner permutation perm[row] = col and the signed coefficient of C = val * (-i/2): for axes 0 and 2
    C is purely imaginary (ci[row]), for axis 1 purely real (cr[row]). Asserted from ALPHA_NZ, not assumed."""
    perms, coefs, kinds = [], [], []
    for i in range(3):
        perm = [None] * 4
        coef = [0.0] * 4
        kind = None
        for (row, col, val) in ALPHA_NZ[i]:
            Cc = complex(val) * (-0.5j)
            if Cc.imag == 0.0:
                k, cval = 'real', Cc.real
            elif Cc.real == 0.0:
                k, cval = 'imag', Cc.imag
            else:
                raise ArithmeticError('alpha coefficient neither real nor imaginary')
            if kind is None:
                kind = k
            elif kind != k:
                raise ArithmeticError('mixed coefficient kinds within one axis')
            perm[row] = col
            coef[row] = cval
        perms.append(perm)
        coefs.append(coef)
        kinds.append(kind)
    return perms, coefs, kinds


ONEHOP_PERM, ONEHOP_COEF, ONEHOP_KIND = _onehop_tables()


# ------------------------------------------------------------------ the fused real step (compiled or eager evaluation)
def _step_real(Up, Vp, Uq, Vq, Au, Av, scale, pc, cre, cim, mbeta, Nn, lp, lm, wp, wm, perms, coefs):
    """One Chebyshev step on real parts. States (B, Lp, Lp, Lp, 4) float64 with a zero halo of width 2:
    (Up, Vp) = T_{k-1}, (Uq, Vq) = T_{k-2}, (Au, Av) = the accumulated sum. Returns (Un, Vn, Au', Av') padded, with
        T_k = scale * h T_{k-1} - pc * T_{k-2},    acc' = acc + (cre + i cim) T_k,
    scale, pc, cre, cim 0-d float64 tensors (no recompilation across k). The operations follow v2's complex arithmetic
    term by term: diagonal m beta (times N), per axis the 1-hop difference (times the link weights), its 4 row terms with
    the purely real or purely imaginary coefficient, then the 2-hop difference with the weights a = -w/4."""
    L = Up.shape[1] - 2 * HALO

    def I(t, dx=0, dy=0, dz=0):
        return t[:, HALO + dx:HALO + dx + L, HALO + dy:HALO + dy + L, HALO + dz:HALO + dz + L, :]

    u, v = I(Up), I(Vp)
    hu = u * mbeta
    hv = v * mbeta
    if Nn is not None:
        hu = hu * Nn
        hv = hv * Nn
    for i in range(3):
        sh = [0, 0, 0]
        sh[i] = 1
        up1, vp1 = I(Up, *sh), I(Vp, *sh)
        sh[i] = -1
        um1, vm1 = I(Up, *sh), I(Vp, *sh)
        perm = perms[i]
        up1, vp1, um1, vm1 = up1[..., perm], vp1[..., perm], um1[..., perm], vm1[..., perm]
        if lp is not None:
            du = lp[i] * up1 - lm[i] * um1
            dv = lp[i] * vp1 - lm[i] * vm1
        else:
            du = up1 - um1
            dv = vp1 - vm1
        c = coefs[i]
        if ONEHOP_KIND[i] == 'real':      # out += c (du + i dv)
            hu = hu + c * du
            hv = hv + c * dv
        else:                             # out += (i c)(du + i dv) = -(c dv) + i (c du)
            hu = hu + (-(c * dv))
            hv = hv + c * du
        sh[i] = 2
        up2, vp2 = I(Up, *sh), I(Vp, *sh)
        sh[i] = -2
        um2, vm2 = I(Up, *sh), I(Vp, *sh)
        a, b = wp[i], wm[i]               # a = -w+/4, b = -w-/4: v2's S = (i a) psi(+2), T = (i b) psi(-2), out += S - T
        hu = hu + ((-(a * vp2)) - (-(b * vm2)))
        hv = hv + ((a * up2) - (b * um2))
    nu = hu * scale - pc * I(Uq)
    nv = hv * scale - pc * I(Vq)
    au = I(Au) + (cre * nu - cim * nv)
    av = I(Av) + (cre * nv + cim * nu)
    return F.pad(nu, _PAD), F.pad(nv, _PAD), F.pad(au, _PAD), F.pad(av, _PAD)


def _sqnorm_real(U, V):
    """||T_k||^2 per batch element from the real parts (the halo is zero)."""
    return (U * U + V * V).sum(dim=(1, 2, 3, 4))


def _get_compiled(name, fn):
    if name not in _compiled:
        try:
            dyn = importlib.import_module('torch._dynamo')
            dyn.config.cache_size_limit = max(64, int(dyn.config.cache_size_limit))
        except Exception:
            pass
        _compiled[name] = torch.compile(fn, dynamic=False, fullgraph=True)
    return _compiled[name]


def _compile_available(device):
    """torch.compile with the Triton backend on a CUDA device; on the CPU only with GAP10_OPT1_ALLOW_CPU_COMPILE=1
    (the C++ inductor backend: a tracing check for the box, not a production path)."""
    if not hasattr(torch, 'compile'):
        return False
    try:
        importlib.import_module('torch._dynamo')
        importlib.import_module('torch._inductor')
        if device.type != 'cuda':
            return bool(os.environ.get('GAP10_OPT1_ALLOW_CPU_COMPILE', ''))
        importlib.import_module('triton')
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ layout conversions (once per sweep, cheap)
def _to_real_padded(psi):
    """complex (B, 4, L, L, L) -> (U, V), each (B, L+4, L+4, L+4, 4) float64 contiguous with a zero halo."""
    U = F.pad(psi.real.permute(0, 2, 3, 4, 1), _PAD)
    V = F.pad(psi.imag.permute(0, 2, 3, 4, 1), _PAD)
    return U, V


def _interior(t):
    L = t.shape[1] - 2 * HALO
    return t[:, HALO:HALO + L, HALO:HALO + L, HALO:HALO + L, :]


def _to_complex(U, V):
    """(U, V) padded real layout -> complex (B, 4, L, L, L) contiguous."""
    return torch.complex(_interior(U).permute(0, 4, 1, 2, 3), _interior(V).permute(0, 4, 1, 2, 3)).contiguous()


def _face(U, V, axis, k):
    """The complex (B, 4, L, L) plane at physical index k along the spatial axis (0, 1, 2) of the padded real layout,
    the other 2 axes in order, as v2's psi.narrow(dim, k, 1).squeeze(dim)."""
    L = U.shape[1] - 2 * HALO
    d = axis + 1
    sl = [slice(None), slice(HALO, HALO + L), slice(HALO, HALO + L), slice(HALO, HALO + L), slice(None)]
    sl[d] = k + HALO
    sl = tuple(sl)
    return torch.complex(U[sl].permute(0, 3, 1, 2), V[sl].permute(0, 3, 1, 2))


# ------------------------------------------------------------------ the operator
class LatticeH(_v2.LatticeH):
    """v2's LatticeH (same constructor, same boundary_residual, boundary_vectors, energy) with a slab-accumulating
    apply and the data of the fused real step."""

    def __init__(self, m, wp_np, wm_np, device, ghost_np=None, N_np=None, lp_np=None, lm_np=None):
        super().__init__(m, wp_np, wm_np, device, ghost_np, N_np, lp_np, lm_np)
        dev = self.device
        sh = (1, self.L, self.L, self.L, 1)
        self.wp_r = [torch.tensor(w * (-0.25), dtype=FD, device=dev).view(sh) for w in wp_np]   # = Im of v2's wp, exactly
        self.wm_r = [torch.tensor(w * (-0.25), dtype=FD, device=dev).view(sh) for w in wm_np]
        self.lp_r = None if lp_np is None else [torch.tensor(w, dtype=FD, device=dev).view(sh) for w in lp_np]
        self.lm_r = None if lm_np is None else [torch.tensor(w, dtype=FD, device=dev).view(sh) for w in lm_np]
        self.N_r = None if N_np is None else torch.tensor(N_np, dtype=FD, device=dev).view(sh)
        self.mbeta_r = torch.tensor(self.m * BETA_DIAG, dtype=FD, device=dev)
        self.perm_t = [torch.tensor(p, dtype=torch.int64, device=dev) for p in ONEHOP_PERM]
        self.coef_t = [torch.tensor(c, dtype=FD, device=dev) for c in ONEHOP_COEF]
        self.backend = BACKEND_REQUEST
        if self.backend == 'auto':
            self.backend = 'compiled' if _compile_available(dev) else 'eager'
        elif self.backend == 'compiled' and not _compile_available(dev):
            raise RuntimeError('GAP10_OPT1_BACKEND=compiled but torch.compile with Triton is not available on this device')
        self.backend_used = None
        self.compile_s = 0.0
        self.nstep = 0

    def set_backend(self, name):
        if name not in ('compiled', 'eager'):
            raise ValueError(name)
        if name == 'compiled' and not _compile_available(self.device):
            raise RuntimeError('compiled backend unavailable')
        self.backend = name
        self.backend_used = None

    # ---- the slab-accumulating complex apply (eager backend; also energy(), taylor_step and the tests)
    def _apply_eager(self, psi, scale=1.0):
        L = self.L
        sc = float(scale)
        out = psi * (self.mbeta * sc) if sc != 1.0 else psi * self.mbeta
        if self.N is not None:
            out.mul_(self.N)
        for i in range(3):
            d = i + 2
            for row in range(4):
                col = ONEHOP_PERM[i][row]
                cv = ONEHOP_COEF[i][row]
                C = (complex(cv, 0.0) if ONEHOP_KIND[i] == 'real' else complex(0.0, cv)) * sc
                o, p = out[:, row], psi[:, col]                       # (B, L, L, L); the axis is d - 1
                op, pp = o.narrow(d - 1, 0, L - 1), p.narrow(d - 1, 1, L - 1)   # x <- x + e
                om, pm = o.narrow(d - 1, 1, L - 1), p.narrow(d - 1, 0, L - 1)   # x <- x - e
                if self.lp is None:
                    axpy_(op, pp, C)
                    axpy_(om, pm, -C)
                else:
                    op.addcmul_(pp, self.lp[i][0, 0].narrow(d - 2, 0, L - 1), value=C)
                    om.addcmul_(pm, self.lm[i][0, 0].narrow(d - 2, 1, L - 1), value=-C)
            out.narrow(d, 0, L - 2).addcmul_(psi.narrow(d, 2, L - 2), self.wp[i].narrow(d, 0, L - 2), value=sc)
            out.narrow(d, 2, L - 2).addcmul_(psi.narrow(d, 0, L - 2), self.wm[i].narrow(d, 2, L - 2), value=-sc)
        return out

    def apply(self, psi):
        out = self._apply_eager(psi, 1.0)
        self.napply += 1
        return out

    # ---- the fused real step
    def _step(self, Up, Vp, Uq, Vq, Au, Av, scale, pc, cre, cim):
        args = (Up, Vp, Uq, Vq, Au, Av, scale, pc, cre, cim, self.mbeta_r, self.N_r, self.lp_r, self.lm_r,
                self.wp_r, self.wm_r, self.perm_t, self.coef_t)
        if self.backend == 'compiled':
            if self.backend_used is None:
                t0 = time.time()
                try:
                    with torch.no_grad():
                        out = _get_compiled('step', _step_real)(*args)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                except Exception as e:
                    if BACKEND_REQUEST == 'compiled':
                        raise
                    _say(f"compiled step failed ({type(e).__name__}: {str(e)[:300]}); eager backend for this operator")
                    self.backend = 'eager'
                    self.backend_used = 'eager'
                    return None
                self.compile_s = time.time() - t0
                self.backend_used = 'compiled'
                _say(f"compiled backend (first step incl. compile {self.compile_s:.1f} s; torch {torch.__version__})")
                return out
            with torch.no_grad():
                return _get_compiled('step', _step_real)(*args)
        return None

    def _sqnorm(self, U, V):
        if self.backend == 'compiled' and self.backend_used == 'compiled':
            with torch.no_grad():
                return _get_compiled('sqnorm', _sqnorm_real)(U, V)
        return _sqnorm_real(U, V)

    # ---- faces and residual from the padded real layout (v2's formulas on the same planes)
    def _faces_real(self, U, V):
        """Per axis (f0, f1, fL1, fL2) complex (B, 4, L, L) as v2's narrow().squeeze() planes."""
        L = U.shape[1] - 2 * HALO
        return [(_face(U, V, i, 0), _face(U, V, i, 1), _face(U, V, i, L - 1), _face(U, V, i, L - 2)) for i in range(3)]

    def boundary_residual_real(self, U, V):
        B = U.shape[0]
        total = torch.zeros(B, dtype=FD, device=U.device)
        if self.ghost is None:
            return total
        for i, (f0, f1, fL1, fL2) in enumerate(self._faces_real(U, V)):
            g = self.ghost[i]
            vec = self._alpha_face(f0, i) * ((-0.5j) * g['nN_lo']) + f1 * ((-0.25j) * g['near_lo'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = f0 * ((-0.25j) * g['far_lo'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = self._alpha_face(fL1, i) * ((0.5j) * g['nN_hi']) + fL2 * ((0.25j) * g['near_hi'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
            vec = fL1 * ((0.25j) * g['far_hi'])
            total += (vec.real ** 2 + vec.imag ** 2).sum((1, 2, 3))
        return torch.sqrt(total)

    def boundary_vectors_real(self, U, V):
        out = []
        if self.ghost is None:
            return out
        for i, (f0, f1, fL1, fL2) in enumerate(self._faces_real(U, V)):
            g = self.ghost[i]
            out.append(self._alpha_face(f0, i) * ((-0.5j) * g['nN_lo']) + f1 * ((-0.25j) * g['near_lo']))
            out.append(f0 * ((-0.25j) * g['far_lo']))
            out.append(self._alpha_face(fL1, i) * ((0.5j) * g['nN_hi']) + fL2 * ((0.25j) * g['near_hi']))
            out.append(fL1 * ((0.25j) * g['far_hi']))
        return out


# ------------------------------------------------------------------ projections from the padded real layout
def _receiver_method(project):
    obj = getattr(project, '__self__', None)
    return obj if (isinstance(obj, _v2.Receivers) and getattr(project, '__func__', None) is _v2.Receivers.project) else None


def _padded_indices(rec, L, Lp):
    cache = getattr(rec, '_opt1_cache', None)
    if cache is not None and cache[0] == (L, Lp):
        return cache[1]

    def conv(idx):
        x = idx // (L * L)
        y = (idx // L) % L
        z = idx % L
        return ((x + HALO) * Lp + (y + HALO)) * Lp + (z + HALO)

    site = conv(rec.site_flat) if len(rec.site_pos) else None
    groups = [(conv(g['idx']), g['G']) for g in rec.groups]
    rec._opt1_cache = ((L, Lp), (site, groups))
    return site, groups


def _project_real(rec, U, V):
    """Receivers.project on the padded real state: the same gathered values (site-major for sites, component-major
    for packet groups) and the same Gram products as v2 on the complex state."""
    B, Lp = U.shape[0], U.shape[1]
    L = Lp - 2 * HALO
    site, groups = _padded_indices(rec, L, Lp)
    Uf, Vf = U.view(B, Lp * Lp * Lp, 4), V.view(B, Lp * Lp * Lp, 4)
    outs = []
    if site is not None:
        outs.append(torch.complex(Uf[:, site, :], Vf[:, site, :]).reshape(B, -1))
    for (gidx, G) in groups:
        vals = torch.complex(Uf[:, gidx, :], Vf[:, gidx, :])          # (B, n, 4)
        outs.append(vals.permute(0, 2, 1).reshape(B, -1) @ G)
    return torch.cat(outs, 1)


# ------------------------------------------------------------------ the sweeps
def _cheb_sweep_eager(H, psi, dE, c_end, n, project, residual, faces):
    """v2's cheb_sweep with the slab-accumulating apply and the scale folded in."""
    if isinstance(psi, list):
        psi = psi.pop()
    P = [project(psi)]
    R = [H.boundary_residual(psi)] if residual else None
    if faces is not None:
        faces.append(H.boundary_vectors(psi))
    Nr = [vec_norm(psi)]
    phi_prev = psi
    phi = H._apply_eager(psi, 1.0 / dE)
    H.napply += 1
    acc = psi * complex(c_end[0])
    del psi
    axpy_(acc, phi, c_end[1])
    P.append(project(phi))
    if residual:
        R.append(H.boundary_residual(phi))
    if faces is not None:
        faces.append(H.boundary_vectors(phi))
    Nr.append(vec_norm(phi))
    for k in range(2, n + 1):
        new = H._apply_eager(phi, 2.0 / dE)
        H.napply += 1
        new.sub_(phi_prev)
        axpy_(acc, new, c_end[k])
        P.append(project(new))
        if residual:
            R.append(H.boundary_residual(new))
        if faces is not None:
            faces.append(H.boundary_vectors(new))
        Nr.append(vec_norm(new))
        phi_prev, phi = phi, new
        del new
    del phi_prev, phi
    H.nstep += n
    return acc, torch.stack(P, 0), (torch.stack(R, 0) if residual else None), torch.stack(Nr, 0)


def _cheb_sweep_real(H, psi, dE, c_end, n, project, residual, faces):
    """The fused-step sweep on the padded real layout. Receivers.project is evaluated on the real layout (the same
    numbers); any other project callable, and the face list, go through a complex copy of each T_k (slower, exact)."""
    if isinstance(psi, list):
        psi = psi.pop()
    dev = psi.device
    rec = _receiver_method(project)
    generic = (rec is None) or (faces is not None)
    P = [project(psi)]
    R = [H.boundary_residual(psi)] if residual else None
    if faces is not None:
        faces.append(H.boundary_vectors(psi))
    Nr = [vec_norm(psi)]
    U0, V0 = _to_real_padded(psi)
    del psi
    c0 = complex(c_end[0])
    Au = U0 * c0.real
    Av = V0 * c0.real
    if c0.imag != 0.0:
        Au.sub_(V0 * c0.imag)
        Av.add_(U0 * c0.imag)
    coef = torch.tensor(np.stack([np.real(c_end), np.imag(c_end)], 1), dtype=FD, device=dev)
    s1 = torch.tensor(1.0 / dE, dtype=FD, device=dev)
    s2 = torch.tensor(2.0 / dE, dtype=FD, device=dev)
    pc0 = torch.tensor(0.0, dtype=FD, device=dev)
    pc1 = torch.tensor(1.0, dtype=FD, device=dev)

    def readout(Uk, Vk, k):
        if generic:
            pk = _to_complex(Uk, Vk)
            P.append(project(pk))
            if residual:
                R.append(H.boundary_residual(pk))
            if faces is not None:
                faces.append(H.boundary_vectors(pk))
            del pk
        else:
            P.append(_project_real(rec, Uk, Vk))
            if residual:
                R.append(H.boundary_residual_real(Uk, Vk))
        Nr.append(torch.sqrt(H._sqnorm(Uk, Vk)))

    res = H._step(U0, V0, U0, V0, Au, Av, s1, pc0, coef[1, 0], coef[1, 1])
    if res is None:                                   # the compile failed under 'auto': redo this sweep eagerly
        psi = _to_complex(U0, V0)
        del U0, V0, Au, Av
        return _cheb_sweep_eager(H, psi, dE, c_end, n, project, residual, faces)
    Up, Vp, Au, Av = res
    Uq, Vq = U0, V0
    del U0, V0, res
    readout(Up, Vp, 1)
    for k in range(2, n + 1):
        Un, Vn, Au, Av = H._step(Up, Vp, Uq, Vq, Au, Av, s2, pc1, coef[k, 0], coef[k, 1])
        readout(Un, Vn, k)
        Uq, Vq, Up, Vp = Up, Vp, Un, Vn
        del Un, Vn
    del Up, Vp, Uq, Vq
    acc = _to_complex(Au, Av)
    del Au, Av
    H.napply += n
    H.nstep += n
    return acc, torch.stack(P, 0), (torch.stack(R, 0) if residual else None), torch.stack(Nr, 0)


def cheb_sweep(H, psi, dE, c_end, n, project, residual=True, faces=None):
    """psi(t0 + tau) = sum_k c_end[k] T_k(h/dE) psi(t0); returns (state, P[k], R[k] or None, Nr[k]) as gap10_core_v2.cheb_sweep.
    psi may be a 1-element list (released once the real copies exist). A v2 LatticeH is handled by v2's sweep."""
    if not isinstance(H, LatticeH):
        return _v2.cheb_sweep(H, psi, dE, c_end, n, project, residual=residual, faces=faces)
    box = psi if isinstance(psi, list) else [psi]
    del psi                                   # the inner sweep pops the box: the input is released once its copies exist
    if H.backend == 'compiled':
        return _cheb_sweep_real(H, box, dE, c_end, n, project, residual, faces)
    if H.backend_used is None:
        H.backend_used = 'eager'
        _say("eager backend (slab-accumulating complex apply)")
    return _cheb_sweep_eager(H, box, dE, c_end, n, project, residual, faces)
