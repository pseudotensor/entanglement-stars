#!/usr/bin/env python3
"""
j6_formed_cut.py -- the planar formed cut (J6) by the exact transverse-chain decomposition.
Every transverse channel is computed directly (no interpolation in the channel label M_perp).
numpy, scipy, torch complex128; CPU or CUDA.

Modes
  selftest   stencil vs dense; Chebyshev vs matrix_exp; CF4 and midpoint vs a dense reference ramp (32 sites)
  anchor     flat coefficient by the exact mass-density quadrature and by the 2D midpoint rule (no evolution)
  j6compare  the 48 J6 grid masses (recovered from the supplied flat column) and the 2 J6 ladder masses
  grid       nested 2D quadrant midpoint grids (J6H4), every node a direct channel; Richardson in nk^-2
  mdensity   exact 1D mass-density quadrature (G7/K15 panels), every node direct; the [0, M_lo] tail
  light      light channels on 2 parents; finite-parent control; sub-grid integral vs the J6 continuation
  ladders    time-step ladder; parent ladder with the hold series 0..12 (the wall-transit test)
  wall       pocket table, hard-wall conversion (particles and holes), semiclassical arrival labels per parent
Units a = 1, N = 1, ma = 0. Protocol constants are DECLARED (J6H1-J6H4); the flow is a prescribed control.
"""
import argparse, json, math, time
import numpy as np
import scipy.special as sps
import scipy.integrate as sint
import scipy.optimize as sopt
import torch

KAPPA = 1.0/64.0               # kappa a
T_RAMP = 40.0/KAPPA            # 2560
W_TANH = 25.6                  # v^f = -1 + 0.4 tanh(x/25.6); dv/dx at the cut = 0.4/25.6 = kappa
V_FAR, V_UP = 1.4, 0.6         # |v^f| at x -> -inf (downstream) and x -> +inf (upstream)
ETA_FLAT_REF = 0.78658339      # eq. dr-area
C6_TARGET = 0.786658           # the earlier formed value C6 compares against
C6_TOL = 2.0e-4
M_THRESHOLD = (V_FAR**2 - 1.0)/(2.0*V_FAR)   # w_c(M) = 1.4  <=>  M = 0.342857
T_HAWK = KAPPA/(2.0*math.pi)   # 0.002487
CD, RD = torch.complex128, torch.float64

# ------------------------------------------------------------------ exact flat formulas and the zone
def hF(z):
    z = np.asarray(z, float)
    return np.logaddexp(0.0, -z) + z/(np.exp(np.minimum(z, 700.0)) + 1.0)

def s4_exact(M):
    """eq. dr-area-chain: 4-component half-space entropy of channel M (= 2 x the 2-component chain)."""
    M = np.atleast_1d(np.asarray(M, float))
    eps = np.pi*sps.ellipk(M**2/(1.0 + M**2))/sps.ellipk(1.0/(1.0 + M**2))
    out = np.zeros_like(M); j = 0
    while True:
        z = (2*j + 1)*eps
        out += hF(z)
        if z.min() > 60.0 or j > 200000: break
        j += 1
    return 4.0*out

def rho_mass(M):
    """DERIVED density of M_perp over the zone: (4M/pi^2) K(M^2 (2 - M^2)), parameter convention."""
    M = np.asarray(M, float)
    return 4.0*M/np.pi**2*sps.ellipk(M**2*(2.0 - M**2))

def quad_rho(f, a, b):
    g = lambda M: float(rho_mass(M)*f(M))
    pts = [p for p in (1.0,) if a < p < b]
    tot = err = 0.0
    for lo, hi in zip([a] + pts, pts + [b]):
        r, e = sint.quad(g, lo, hi, limit=800, epsabs=1e-13, epsrel=1e-13)
        tot += r; err += e
    return tot, err

def eta_flat_1d(a=0.0, b=math.sqrt(2.0)):
    return quad_rho(lambda M: float(s4_exact(M)[0]), a, b)

def grid_channels(nk):
    """J6H4 quadrant midpoint grid: channel labels (i <= j) and weights summing to 1."""
    k = (np.arange(nk) + 0.5)*np.pi/(2.0*nk)
    s2 = np.sin(k)**2
    I, J = np.triu_indices(nk)
    return np.sqrt(s2[I] + s2[J]), np.where(I == J, 1.0, 2.0)/float(nk*nk)

def eta_flat_2d(nk):
    M, w = grid_channels(nk)
    return float(np.sum(w*s4_exact(M)))

def trapezoid(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum(0.5*(y[1:] + y[:-1])*(x[1:] - x[:-1])))

# ------------------------------------------------------------------ bands, pocket, semiclassical transit
def w_c(M): return math.sqrt(1.0 + M*M) + M

def pocket_depth(M, w=V_FAR, n=20001):
    s = np.linspace(0.0, 1.0, n)
    return float(max((w*np.sqrt(s*(1.0 - s)) - np.sqrt(M*M + s)).max(), 0.0))

def t_open(M, w_far=V_FAR):
    e = w_c(M)/w_far
    return None if e >= 1.0 else (2.0*T_RAMP/math.pi)*math.asin(math.sqrt(e))

def envelope(t): return math.sin(math.pi*t/(2.0*T_RAMP))**2 if t < T_RAMP else 1.0

def v_profile(x): return -1.0 + 0.4*np.tanh(np.asarray(x, float)/W_TANH)

def omega_plus(k, v, M): return 0.5*v*np.sin(2.0*k) + np.sqrt(M*M + np.sin(k)**2)

def vg_plus(k, v, M):
    s = np.sin(k)
    return v*np.cos(2.0*k) + s*np.cos(k)/np.sqrt(M*M + s*s)

_KGRID = np.linspace(1e-6, 0.5*np.pi, 1501)

def k2_root(E, v, M):
    """the escaping upper-band branch at energy E in flow v < 0: the root of omega_+ = E on the rising
    side beyond the band minimum in (0, pi/2]; None if E lies below the local minimum (turning point)."""
    om = omega_plus(_KGRID, v, M)
    i = int(np.argmin(om))
    if om[i] > E or om[-1] < E: return None
    lo, hi = float(_KGRID[i]), 0.5*math.pi
    for _ in range(50):
        mid = 0.5*(lo + hi)
        if omega_plus(mid, v, M) < E: lo = mid
        else: hi = mid
    return 0.5*(lo + hi)

def transit_label(M, Nx, E, t_start=None, dt=1.0, t_extra=30.0/KAPPA):
    """CONDITIONAL semiclassical estimate: a wall-converted quantum of energy E leaves the downstream wall
    when the far pocket is E deep (or at t_start), moves at the local group velocity of the escaping
    upper-band branch through the ramping profile, waits where the branch does not exist; returns the
    hold label at which it reaches the cut, and the emission time."""
    if t_start is None:
        t = 0.0
        while t < T_RAMP and pocket_depth(M, V_FAR*envelope(t), 4001) < E: t += 8.0
        t_emit = t
    else:
        t_emit = t = float(t_start)
    x = -(Nx - 1)/2.0
    while t < T_RAMP + t_extra:
        v = envelope(t)*float(v_profile(x))
        k = k2_root(E, v, M)
        if k is not None:
            x += dt*float(vg_plus(k, v, M))
            if x >= 0.0: return (t - T_RAMP)*KAPPA, t_emit
        t += dt
    return None, t_emit

def arrival_table(M, Nx):
    """earliest arrival label over the pocket's energies, and the label of the emission at the ramp's end."""
    ep = pocket_depth(M)
    if ep <= 0.0: return None
    labels = [transit_label(M, Nx, f*ep)[0] for f in np.linspace(0.1, 0.95, 8)]
    labels = [l for l in labels if l is not None]
    return (min(labels) if labels else None), transit_label(M, Nx, 0.5*ep, t_start=T_RAMP)[0]

# ------------------------------------------------------------------ hard-wall scattering (homogeneous flow)
def chain_modes(M, v, eps):
    """the 8 Bloch solutions z = e^{ik} of det(h(k) - eps) = 0 for uniform flow v (degree 8 in z)."""
    a = v/(4j)
    p = np.zeros(9, complex)
    p[0] = a*a; p[2] = -2*a*eps + 0.25; p[4] = eps**2 - 2*a*a - M*M - 0.5; p[6] = 2*a*eps + 0.25; p[8] = a*a
    modes = []
    for z in np.roots(p):
        s1, s2 = (z - 1/z)/(2j), (z*z - 1/(z*z))/(2j)
        c1, c2 = (z + 1/z)/2, (z*z + 1/(z*z))/2
        lam = eps - 0.5*v*s2                      # +-sqrt(M^2 + sin^2 k)
        u = np.array([s1, lam - M], complex)      # eigenvector of M sz + sin k sx with eigenvalue lam
        if abs(u).max() < 1e-12: u = np.array([1.0, 0.0], complex)
        prop = abs(abs(z) - 1.0) < 1e-7
        if prop: u = u/np.sqrt(np.vdot(u, u).real)
        vg = float((v*c2 + s1*c1/lam).real) if prop else None
        modes.append(dict(z=z, k=complex(-1j*np.log(z)), lam=complex(lam), u=u, prop=prop, vg=vg))
    return modes

def wall_conversion(M, v, eps, process='particle'):
    """hard wall at the left end of the half-line n >= 0 (psi_{-1} = psi_{-2} = 0), uniform flow v < 0.
    process 'particle', eps > 0: incident occupied pocket mode (lower band, v_g < 0, valley k = 0).
    process 'hole', eps < 0: incident occupied lower-band mode with v_g < 0 (the mirror process).
    Returns the flux fractions of the incident flux into the outgoing lower and upper bands; in both
    processes 'upper_band_fraction' is the conversion probability R_2 (the hole content of the escaping
    lower-band channel equals the fraction sent into the trapped upper-band dip)."""
    modes = chain_modes(M, v, eps)
    cand = [m for m in modes if m['prop'] and m['vg'] < 0 and m['lam'].real < 0]
    if process == 'particle': cand = [m for m in cand if abs(m['k'].real) < math.pi/2]
    if not cand: return None
    inc = min(cand, key=lambda m: abs(m['k'].real))
    unk = [m for m in modes if (m['prop'] and m['vg'] > 0) or ((not m['prop']) and abs(m['z']) < 1.0)]
    Bm = np.zeros((4, len(unk)), complex); rhs = np.zeros(4, complex)
    for col, m in enumerate(unk):
        Bm[0:2, col] = m['u']*m['z']**(-1); Bm[2:4, col] = m['u']*m['z']**(-2)
    rhs[0:2] = -inc['u']*inc['z']**(-1); rhs[2:4] = -inc['u']*inc['z']**(-2)
    A = np.linalg.lstsq(Bm, rhs, rcond=None)[0]
    fl_up = fl_lo = 0.0
    for amp, m in zip(A, unk):
        if m['prop']:
            f = abs(amp)**2*m['vg']/abs(inc['vg'])
            if m['lam'].real > 0: fl_up += f
            else: fl_lo += f
    return dict(n_unknowns=len(unk), resid=float(np.linalg.norm(Bm @ A - rhs)), k_inc=inc['k'].real,
                vg_inc=inc['vg'], upper_band_fraction=fl_up, lower_band_fraction=fl_lo, flux_sum=fl_up + fl_lo)

# ------------------------------------------------------------------ the chain
class Chain:
    """2-component chain J6H2 on Nx open sites; batched stencil action and dense form."""
    def __init__(self, Nx, device):
        self.Nx = Nx
        self.device = torch.device(device)
        self.x = np.arange(Nx) - (Nx - 1)/2.0
        self.vf = np.asarray(v_profile(self.x))
        wf = self.vf[:-2] + self.vf[2:]                                     # v_n + v_{n+2}
        self.wf = torch.tensor(wf, dtype=RD, device=self.device).view(1, -1, 1, 1)
        self.sz = torch.tensor([1.0, -1.0], dtype=RD, device=self.device).view(1, 1, 2, 1)
        self.wmax = float(np.abs(wf).max())
        self.A, self.B = slice(0, Nx//2), slice(Nx//2, Nx)

    def apply(self, Phi, Mb, e):
        """h(M, e) Phi;  Phi (B, Nx, 2, K) complex, Mb (B,) real, e the envelope value."""
        out = Phi*(Mb.view(-1, 1, 1, 1)*self.sz)                            # M sigma_z
        Psw = Phi.flip(2)                                                   # sigma_x Phi
        out[:, :-1] += (-0.5j)*Psw[:, 1:]                                   # (n, n+1): -(i/2) sigma_x
        out[:, 1:] += (0.5j)*Psw[:, :-1]                                    # (n+1, n): +(i/2) sigma_x
        w = e*self.wf
        out[:, :-2] += (-0.125j)*(w*Phi[:, 2:])                             # (n, n+2): -(i/8) w_n
        out[:, 2:] += (0.125j)*(w*Phi[:, :-2])                              # (n+2, n): +(i/8) w_n
        return out

    def dense(self, M, e):
        Nx = self.Nx
        H = torch.zeros((Nx, 2, Nx, 2), dtype=CD, device=self.device)
        idx = torch.arange(Nx, device=self.device)
        H[idx, 0, idx, 0] = M; H[idx, 1, idx, 1] = -M
        i1 = idx[:-1]
        H[i1, 0, i1 + 1, 1] = -0.5j; H[i1, 1, i1 + 1, 0] = -0.5j
        H[i1 + 1, 0, i1, 1] = 0.5j;  H[i1 + 1, 1, i1, 0] = 0.5j
        i2 = idx[:-2]
        w = (e*self.wf.view(-1)).to(CD)
        H[i2, 0, i2 + 2, 0] = -0.125j*w; H[i2, 1, i2 + 2, 1] = -0.125j*w
        H[i2 + 2, 0, i2, 0] = 0.125j*w;  H[i2 + 2, 1, i2, 1] = 0.125j*w
        return H.reshape(2*Nx, 2*Nx)

    def ebound(self, Mmax, emax=1.01):
        """rigorous: ||M sz + sx p|| <= sqrt(1+M^2) (h_0^2 = M^2 + p^2), ||flow term|| <= max|w|/4."""
        return math.sqrt(1.0 + Mmax**2) + 0.25*self.wmax*emax + 0.05

def cheb_coeffs(z, tol=1e-17):
    kmax = int(math.ceil(z + 80 + 12*z**(1.0/3.0)))
    k = np.arange(kmax + 1)
    J = sps.jv(k, z)
    keep = np.nonzero(np.abs(J) > tol)[0]
    K = max(int(keep[-1]) if keep.size else 1, 1)
    c = 2.0*((-1j)**k[:K + 1])*J[:K + 1]
    c[0] = J[0]
    return c

def expm_apply(chain, Phi, Mb, e, tau, Eb):
    """exp(-i tau h(M, e)) Phi by the Chebyshev expansion under the spectral bound Eb."""
    c = cheb_coeffs(tau*Eb)
    T0 = Phi
    T1 = chain.apply(Phi, Mb, e)/Eb
    out = complex(c[0])*T0 + complex(c[1])*T1
    for ck in c[2:]:
        T2 = chain.apply(T1, Mb, e)*(2.0/Eb) - T0
        out += complex(ck)*T2
        T0, T1 = T1, T2
    return out

def transport(chain, Phi0, Mb, stepper, dt, Eb):
    """time-ordered product over the ramp: 'midpoint' (2nd order) or 'cf4' (4th-order Magnus, CF4)."""
    Ns = int(round(T_RAMP/dt)); dt = T_RAMP/Ns
    Phi = Phi0
    if stepper == 'midpoint':
        for n in range(Ns):
            Phi = expm_apply(chain, Phi, Mb, envelope((n + 0.5)*dt), dt, Eb)
    elif stepper == 'cf4':
        c1, c2 = 0.5 - math.sqrt(3.0)/6.0, 0.5 + math.sqrt(3.0)/6.0
        a1, a2 = (3.0 - 2.0*math.sqrt(3.0))/12.0, (3.0 + 2.0*math.sqrt(3.0))/12.0
        for n in range(Ns):
            t0 = n*dt
            e1, e2 = envelope(t0 + c1*dt), envelope(t0 + c2*dt)
            Phi = expm_apply(chain, Phi, Mb, 2.0*(a2*e1 + a1*e2), 0.5*dt, Eb)   # earlier-weighted first
            Phi = expm_apply(chain, Phi, Mb, 2.0*(a1*e1 + a2*e2), 0.5*dt, Eb)
    else:
        raise ValueError(stepper)
    return Phi

def flat_sea(chain, M):
    ev, V = torch.linalg.eigh(chain.dense(M, 0.0))
    occ = ev < 0.0
    return V[:, occ].reshape(chain.Nx, 2, -1).contiguous(), int((ev.abs() < 1e-12).sum())

def comparator_sea(chain, M):
    ev, V = torch.linalg.eigh(chain.dense(M, 1.0))
    return V[:, ev < 0.0].reshape(chain.Nx, 2, -1).contiguous()

def s_binary(lam):
    # builder's repair: the earlier clamp to 1 - 1e-17 rounds to 1.0 in float64 and gave 0*log(0) = nan
    # for eigenvalues within 1.1e-16 of 1 (the self-test printed S_A - S_B = nan); xlogy is 0 at 0.
    lam = lam.clamp(0.0, 1.0)
    return -(torch.xlogy(lam, lam) + torch.xlogy(1.0 - lam, 1.0 - lam)).sum()

def region_entropy(Phi, rows):
    """Tr s(C_R) for C = Phi Phi^dagger restricted to the sites `rows`; Phi (Nx, 2, K)."""
    P = Phi[rows].reshape(-1, Phi.shape[-1])
    G = P @ P.conj().transpose(0, 1) if P.shape[0] <= P.shape[1] else P.conj().transpose(0, 1) @ P
    return float(s_binary(torch.linalg.eigvalsh(G)))

def hold_analysis(chain, PhiT, M, taus):
    """eigenbasis of h_f: occupations, pocket count, comparator, dephased ceiling, hold series."""
    Nx = chain.Nx; K = PhiT.shape[-1]
    ev, V = torch.linalg.eigh(chain.dense(M, 1.0))
    Phif = V.conj().transpose(0, 1) @ PhiT.reshape(2*Nx, K)
    n_a = (Phif.abs()**2).sum(dim=1).clamp(min=0.0)
    pos = ev > 0.0
    frac = (n_a > 1e-6) & (n_a < 1.0 - 1e-6)
    r = dict(M=float(M), Nx=Nx,
             pocket=float(n_a[pos].sum()), hole=float((1.0 - n_a[~pos]).sum()),
             nfrac=int(frac.sum()), efrac_max=float(ev[frac].abs().max()) if int(frac.sum()) else 0.0,
             n_zero_modes=int((ev.abs() < 1e-10).sum()), S_glob_diag=float(s_binary(n_a)),
             orth=float((Phif.conj().transpose(0, 1) @ Phif
                         - torch.eye(K, dtype=CD, device=Phif.device)).abs().max()))
    r['S_g'] = region_entropy(V[:, ~pos].reshape(Nx, 2, -1), chain.A)            # comparator P_-(h_f)
    r['S_diag'] = region_entropy((V*torch.sqrt(n_a).view(1, -1)).reshape(Nx, 2, -1), chain.A)
    series, purity = {}, 0.0
    for tau in taus:
        ph = torch.exp(-1j*ev*(tau/KAPPA)).view(-1, 1)
        Pt = (V @ (ph*Phif)).reshape(Nx, 2, K)
        SA, SB = region_entropy(Pt, chain.A), region_entropy(Pt, chain.B)
        series[float(tau)] = (SA, SB); purity = max(purity, abs(SA - SB))
    r['series'] = series; r['purity'] = purity
    return r

def run_channels(chain, Ms, taus, stepper, dt, batch, tag='', adiab=None):
    Ms = [float(m) for m in Ms]; order = np.argsort(Ms); out = [None]*len(Ms)
    for b0 in range(0, len(Ms), batch):
        idx = order[b0:b0 + batch]; Mb_list = [Ms[i] for i in idx]; t0 = time.time()
        Phis, nzs, Sflat = [], [], []
        for M in Mb_list:
            P0, nz = flat_sea(chain, M)
            Phis.append(P0); nzs.append(nz); Sflat.append(region_entropy(P0, chain.A))
        shortcut = adiab is not None and min(Mb_list) >= adiab
        if shortcut:
            PhiT = torch.stack([comparator_sea(chain, M) for M in Mb_list], 0)
        else:
            Mb = torch.tensor(Mb_list, dtype=RD, device=chain.device)
            PhiT = transport(chain, torch.stack(Phis, 0), Mb, stepper, dt, chain.ebound(max(Mb_list)))
        for j, i in enumerate(idx):
            r = hold_analysis(chain, PhiT[j], Mb_list[j], taus)
            r.update(S_flat=Sflat[j], s4h_exact=0.5*float(s4_exact(Mb_list[j])[0]), nz0=nzs[j], adiabatic=shortcut)
            out[i] = r
        print(f"[{tag}] channels {b0+1}-{b0+len(idx)}/{len(Ms)}  M in [{min(Mb_list):.5f}, {max(Mb_list):.5f}]"
              f"  {'shortcut' if shortcut else 'transported'}  {time.time()-t0:.1f} s", flush=True)
    return out

def print_table(results, taus, weights=None):
    tl = [float(t) for t in taus]
    print("   M        w        S4flat_chain  s4_exact    " + "  ".join(f"S4_hold{t:g}" for t in tl)
          + "   growth      S4_comp    pocket  S4_diag  nfrac  |e|frac  |SA-SB|  orth")
    for k, r in enumerate(results):
        w = weights[k] if weights is not None else float('nan')
        cols = "  ".join(f"{2*r['series'][t][0]:.8f}" for t in tl)
        growth = 2*(r['series'][tl[-1]][0] - r['series'][tl[0]][0])
        print(f"{r['M']:.6f} {w:.3e}  {2*r['S_flat']:.8f}  {2*r['s4h_exact']:.8f}  {cols}  {growth:+.3e}"
              f"  {2*r['S_g']:.8f}  {r['pocket']:7.3f}  {2*r['S_diag']:8.4f}  {r['nfrac']:5d}  {r['efrac_max']:.4f}"
              f"  {r['purity']:.1e}  {r['orth']:.1e}")

def assemble(results, weights, taus):
    tl = [float(t) for t in taus]
    e = {'flat_chain': sum(w*2*r['S_flat'] for w, r in zip(weights, results)),
         'flat_exact': sum(w*2*r['s4h_exact'] for w, r in zip(weights, results)),
         'comparator': sum(w*2*r['S_g'] for w, r in zip(weights, results))}
    for t in tl:
        e[f'hold{t:g}'] = sum(w*2*r['series'][t][0] for w, r in zip(weights, results))
    lo = [(w, r) for w, r in zip(weights, results) if r['M'] < M_THRESHOLD]
    hi = [(w, r) for w, r in zip(weights, results) if r['M'] >= M_THRESHOLD]
    e['excess_hold0_pocket_channels'] = sum(w*2*(r['series'][0.0][0] - r['S_flat']) for w, r in lo)
    e['excess_hold0_gapped_channels'] = sum(w*2*(r['series'][0.0][0] - r['S_flat']) for w, r in hi)
    e['max_|hold0-comparator|_gapped'] = max([2*abs(r['series'][0.0][0] - r['S_g']) for w, r in hi] or [0.0])
    grown = [r['M'] for r in results if 2*(r['series'][tl[-1]][0] - r['series'][0.0][0]) > 1e-6*2*r['series'][0.0][0]]
    e['largest_M_with_growth'] = max(grown) if grown else 0.0
    return e

def dump(obj, path):
    def conv(o):
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, (np.bool_,)): return bool(o)
        return o
    with open(path, 'w') as f: json.dump(conv(obj), f, indent=1)

# ------------------------------------------------------------------ supplied J6 record (2-component values)
J6_GRID = np.array([
 [0.0453,1.49136756,1.49682249,14.36927530],[0.0488,1.46658093,1.47112650,13.82416580],
 [0.0525,1.44176125,1.44556092,13.23008394],[0.0566,1.41690441,1.42008991,12.58378286],
 [0.0609,1.39200583,1.39468381,11.88465444],[0.0655,1.36706042,1.36931767,11.13178731],
 [0.0705,1.34206252,1.34396980,10.32460920],[0.0759,1.31700587,1.31862120,9.46987388],
 [0.0817,1.29188357,1.29325461,8.58007690],[0.0879,1.26668799,1.26785405,7.66154100],
 [0.0947,1.24141078,1.24240433,6.71739348],[0.1019,1.21604276,1.21689080,5.76071371],
 [0.1097,1.19057396,1.19129892,4.82842849],[0.1181,1.16499355,1.16561419,3.95631590],
 [0.1271,1.13928982,1.13982184,3.18046014],[0.1368,1.11345021,1.11390681,2.54655892],
 [0.1472,1.08746133,1.08785362,2.03364639],[0.1585,1.06130901,1.06164640,1.62418389],
 [0.1706,1.03497844,1.03526885,1.32263251],[0.1836,1.00845425,1.00870444,1.12983892],
 [0.1976,0.98172081,0.98193650,1.02249813],[0.2127,0.95476245,0.95494852,0.96529687],
 [0.2290,0.92756387,0.92772448,0.92961776],[0.2465,0.90011059,0.90024929,0.90046221],
 [0.2653,0.87238960,0.87250943,0.87251515],[0.2855,0.84439003,0.84449358,0.84449359],
 [0.3073,0.81610405,0.81619354,0.81619355],[0.3308,0.78752786,0.78760521,0.78760520],
 [0.3561,0.75866288,0.75872972,0.75872972],[0.3833,0.72951697,0.72957471,0.72957471],
 [0.4126,0.70010582,0.70015566,0.70015566],[0.4441,0.67045434,0.67049734,0.67049734],
 [0.4780,0.64059801,0.64063506,0.64063506],[0.5145,0.61058412,0.61061598,0.61061598],
 [0.5538,0.58047269,0.58050004,0.58050004],[0.5961,0.55033709,0.55036051,0.55036051],
 [0.6416,0.52026402,0.52028403,0.52028403],[0.6906,0.49035294,0.49036998,0.49036998],
 [0.7434,0.46071471,0.46072918,0.46072918],[0.8001,0.43146937,0.43148159,0.43148159],
 [0.8613,0.40274331,0.40275360,0.40275360],[0.9270,0.37466562,0.37467426,0.37467426],
 [0.9978,0.34736416,0.34737134,0.34737134],[1.0740,0.32096127,0.32096723,0.32096723],
 [1.1561,0.29556971,0.29557462,0.29557462],[1.2444,0.27128893,0.27129296,0.27129296],
 [1.3394,0.24820207,0.24820536,0.24820536],[1.4417,0.22637375,0.22637643,0.22637643]])
J6_LADDER = {0.1: (1.2234058171, 6.0043188096), 0.7: (0.4849099499, 0.4849099500)}   # midpoint dt 0.5
J6_ASSEMBLED = {12: (0.78618075, 0.78622430, 0.86687213), 24: (0.78648299, 0.78653777, 0.86885236)}

# ------------------------------------------------------------------ Gauss--Kronrod G7/K15 and the panel set
XGK = np.array([0.991455371120813, 0.949107912342759, 0.864864423359769, 0.741531185599394,
                0.586087235467691, 0.405845151377397, 0.207784955007898, 0.0])
WGK = np.array([0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525,
                0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728])
WG = np.array([0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469])

def gk15(a, b):
    c, h = 0.5*(a + b), 0.5*(b - a)
    x = np.concatenate([c - h*XGK[:-1], [c], c + h*XGK[:-1][::-1]])
    wk = np.concatenate([WGK[:-1], [WGK[-1]], WGK[:-1][::-1]])*h
    wg = np.zeros(15)
    for i, g in zip((1, 3, 5), WG[:3]):
        wg[i] = g*h; wg[14 - i] = g*h
    wg[7] = WG[3]*h
    return x, wk, wg

M_LO = 0.0125
DEFAULT_BREAKS = [M_LO, 0.025, 0.0453, 0.07, 0.1, 0.13, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.45, 0.6, 0.8,
                  0.9, 0.97, 0.995, 0.9995, 1.0, 1.0005, 1.005, 1.03, 1.1, 1.25, math.sqrt(2.0)]   # 25 panels

LIGHT_MASSES = [0.004, 0.005, 0.0063, 0.008, 0.01, 0.0125, 0.016, 0.02, 0.025, 0.0315, 0.04, 0.0453, 0.0488]

# ------------------------------------------------------------------ modes
def mode_selftest(args):
    torch.manual_seed(0)
    ch = Chain(64, args.device); M, e = 0.3, 0.77
    Phi = torch.randn(1, 64, 2, 5, dtype=CD, device=ch.device)
    Mb = torch.tensor([M], dtype=RD, device=ch.device)
    H = ch.dense(M, e)
    print("Hermiticity            :", float((H - H.conj().transpose(0, 1)).abs().max()))
    lhs = ch.apply(Phi, Mb, e)[0].reshape(128, 5); rhs = H @ Phi[0].reshape(128, 5)
    print("stencil vs dense       :", float((lhs - rhs).abs().max()))
    tau, Eb = 7.3, ch.ebound(M)
    e1 = expm_apply(ch, Phi, Mb, e, tau, Eb)[0].reshape(128, 5)
    e2 = torch.linalg.matrix_exp(-1j*tau*H) @ Phi[0].reshape(128, 5)
    print("Chebyshev vs matrix_exp:", float((e1 - e2).abs().max()), " terms:", len(cheb_coeffs(tau*Eb)))
    ev = torch.linalg.eigvalsh(H)
    print("spectral radius / bound:", float(ev.abs().max()), "/", Eb,
          "  particle-hole symmetry max|ev + ev_rev|:", float((ev + ev.flip(0)).abs().max()))
    P0, nz = flat_sea(ch, M)
    print("flat sea rank, zero modes, S_A - S_B:", P0.shape[-1], nz,
          region_entropy(P0, ch.A) - region_entropy(P0, ch.B))
    # the steppers against a dense reference ramp on 32 sites (midpoint at dt 0.05)
    ch32 = Chain(32, args.device); M32 = 0.2
    P0, _ = flat_sea(ch32, M32); Mb32 = torch.tensor([M32], dtype=RD, device=ch32.device)
    H0 = ch32.dense(M32, 0.0); Hv = ch32.dense(M32, 1.0) - H0
    dt_ref = 0.05; Ns = int(round(T_RAMP/dt_ref))
    Phi = P0.reshape(64, -1).clone()
    for n in range(Ns):
        Phi = torch.linalg.matrix_exp(-1j*dt_ref*(H0 + envelope((n + 0.5)*dt_ref)*Hv)) @ Phi
    Cref = Phi @ Phi.conj().transpose(0, 1)
    for stp, dt in (('cf4', 64.0), ('cf4', 16.0), ('cf4', 4.0), ('cf4', 1.0), ('midpoint', 2.0), ('midpoint', 0.5)):
        PhiT = transport(ch32, P0.unsqueeze(0), Mb32, stp, dt, ch32.ebound(M32))[0].reshape(64, -1)
        d = float((PhiT @ PhiT.conj().transpose(0, 1) - Cref).abs().max())
        print(f"ramp check (32 sites) {stp:8s} dt {dt:5.2f}: max |C - C_ref| = {d:.2e}")

def mode_anchor(args):
    v1, err = eta_flat_1d()
    print(f"eta_flat by the exact mass density: {v1:.10f}  (quad err {err:.1e})  ref {ETA_FLAT_REF}  diff {v1-ETA_FLAT_REF:+.2e}")
    print(f"int rho dM = {quad_rho(lambda M: 1.0, 0.0, math.sqrt(2.0))[0]:.12f}  (must be 1)")
    prev = None
    for nk in (12, 24, 48, 96, 192, 384, 768, 1536):
        v = eta_flat_2d(nk)
        rich = (4*v - prev)/3 if prev is not None else float('nan')
        print(f"2D midpoint nk {nk:5d}: {v:.10f}  diff {v-ETA_FLAT_REF:+.3e}  nk^2*diff {nk*nk*(v-ETA_FLAT_REF):+.4f}  Richardson {rich:.10f}")
        prev = v
    print(f"2D midpoint nk 24 on the exact s4 vs J6 assembled flat 0.78648299: {eta_flat_2d(24)-0.78648299:+.2e}  (expect ~1e-7: J6 used a spline)")
    print(f"zone weight of M < 0.0453: {quad_rho(lambda M: 1.0, 0.0, 0.0453)[0]:.4e}  (J6 below-grid node weight 6.51e-4)")
    print(f"zone weight of M < {M_LO}: {quad_rho(lambda M: 1.0, 0.0, M_LO)[0]:.4e};  of M < T_H: {quad_rho(lambda M: 1.0, 0.0, T_HAWK)[0]:.2e}")
    print(f"s4/2 at 0.0453: {0.5*s4_exact(0.0453)[0]:.8f}  (J6 flat column 1.49136756; its mass is printed to 4 digits)")
    tot, est = eta_flat_1d(0.0, M_LO)[0], 0.0
    for a, b in zip(DEFAULT_BREAKS[:-1], DEFAULT_BREAKS[1:]):
        x, wk, wg = gk15(a, b); f = rho_mass(x)*s4_exact(x)
        tot += float(np.sum(wk*f)); est += abs(float(np.sum(wk*f) - np.sum(wg*f)))
    print(f"G7/K15 on the panel set + exact tail (flat): {tot:.10f}  diff {tot-ETA_FLAT_REF:+.2e}  K15-G7 est {est:.1e}")

def mode_j6compare(args):
    taus = [0.0, 12.0]; ch = Chain(args.nx, args.device)
    Ms = []
    for Mp, fl, h0, h12 in J6_GRID:
        f = lambda M: 0.5*float(s4_exact(M)[0]) - fl
        try: Ms.append(sopt.brentq(f, Mp*0.99, Mp*1.01, xtol=1e-12))
        except ValueError: Ms.append(float(Mp))
    res = run_channels(ch, Ms, taus, args.stepper, args.dt, args.batch, 'j6grid')
    print("\nJ6 grid: printed M, recovered M, (this code - J6) flat / hold 0 / hold 12 (2-component), growth, pocket, nfrac")
    for (Mp, fl, h0, h12), Mr, r in zip(J6_GRID, Ms, res):
        print(f"{Mp:.4f} {Mr:.7f}  flat {r['S_flat']-fl:+.2e}  hold0 {r['series'][0.0][0]-h0:+.2e}"
              f"  hold12 {r['series'][12.0][0]-h12:+.2e}  growth {r['series'][12.0][0]-r['series'][0.0][0]:+.3e}"
              f"  pocket {r['pocket']:.3f}  nfrac {r['nfrac']}")
    for M, (h0, h12) in J6_LADDER.items():
        for stp, dt in (('midpoint', 0.5), (args.stepper, args.dt)):
            r = run_channels(ch, [M], taus, stp, dt, 1, f'ladder {M} {stp} {dt}')[0]
            print(f"M {M}: {stp} dt {dt}: hold0 {r['series'][0.0][0]:.10f} (J6 {h0:.10f}, diff {r['series'][0.0][0]-h0:+.2e})"
                  f"  hold12 {r['series'][12.0][0]:.10f} (J6 {h12:.10f}, diff {r['series'][12.0][0]-h12:+.2e})  |SA-SB| {r['purity']:.1e}")
    dump(dict(masses=Ms, results=res), args.out or 'j6compare.json')

def mode_grid(args):
    taus = [float(t) for t in args.taus.split(',')]; ch = Chain(args.nx, args.device)
    adiab = args.adiabatic_above if args.adiabatic_above > 0 else None
    prev = None
    for nk in [int(x) for x in args.nk.split(',')]:
        M, w = grid_channels(nk)
        res = run_channels(ch, M, taus, args.stepper, args.dt, args.batch, f'grid nk={nk}', adiab)
        e = assemble(res, w, taus); e['flat_exact_formula'] = eta_flat_2d(nk)
        print(f"\n=== nk {nk}: {len(M)} channels ===")
        if nk <= 24 or args.print_all: print_table(res, taus, w)
        for k_, v_ in e.items(): print(f"  {k_:34s} {v_:.10f}")
        print(f"  hold0 - C6 target {C6_TARGET}: {e['hold0']-C6_TARGET:+.3e}   hold0 - flat ref: {e['hold0']-ETA_FLAT_REF:+.3e}"
              f"   hold0/flat_chain - 1: {e['hold0']/e['flat_chain']-1:+.3e}")
        if nk in J6_ASSEMBLED:
            j = J6_ASSEMBLED[nk]
            print(f"  J6 assembled flat/hold0/hold12 {j}; this code - J6: {e['flat_chain']-j[0]:+.2e} {e['hold0']-j[1]:+.2e}"
                  f" {e.get('hold12', float('nan'))-j[2]:+.2e}")
        if prev is not None and nk == 2*prev[0]:
            for key in ('flat_chain', 'flat_exact', 'hold0'):
                print(f"  Richardson {key}: {(4*e[key]-prev[1][key])/3:.10f}")
        prev = (nk, e)
        dump(dict(nk=nk, M=list(M), w=list(w), eta=e, results=res), (args.out or 'grid') + f'_nk{nk}.json')
    print("\nC6 rule: PASS iff |eta_hold0(converged) - 0.786658| < 2e-4 with the error budget below the margin.")

def mode_mdensity(args):
    taus = [float(t) for t in args.taus.split(',')]; ch = Chain(args.nx, args.device)
    breaks = [float(x) for x in args.breaks.split(',')] if args.breaks else DEFAULT_BREAKS
    nodes, panels = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        x, wk, wg = gk15(a, b); panels.append((a, b, len(nodes), wk, wg)); nodes.extend(x.tolist())
    adiab = args.adiabatic_above if args.adiabatic_above > 0 else None
    res = run_channels(ch, nodes, taus, args.stepper, args.dt, args.batch, 'mdensity', adiab)
    rho = rho_mass(np.array(nodes))
    keys = {'flat_chain': lambda r: 2*r['S_flat'], 'flat_exact': lambda r: 2*r['s4h_exact'], 'comparator': lambda r: 2*r['S_g']}
    for t in taus: keys[f'hold{t:g}'] = (lambda t: (lambda r: 2*r['series'][t][0]))(t)
    print("\npanel      a        b      " + "  ".join(f"{k:>20s}" for k in keys))
    totals = {k: [0.0, 0.0] for k in keys}
    for a, b, i0, wk, wg in panels:
        line = f"[{a:.5f},{b:.5f}]"
        for k, fn in keys.items():
            f = np.array([fn(r) for r in res[i0:i0+15]])*rho[i0:i0+15]
            IK, IG = float(np.sum(wk*f)), float(np.sum(wg*f))
            totals[k][0] += IK; totals[k][1] += abs(IK - IG); line += f"  {IK:.8f}({abs(IK-IG):.0e})"
        print(line)
    # the tail [0, M_lo]: exact flat part; formed excess by a capped power law through the lowest panel
    M_lo = breaks[0]
    tail_flat, tail_ferr = eta_flat_1d(0.0, M_lo)
    w_tail = quad_rho(lambda M: 1.0, 0.0, M_lo)[0]
    Ml = np.array(nodes[:15]); dS = np.array([2*(r['series'][0.0][0] - r['S_flat']) for r in res[:15]])
    mask = dS > 0
    if int(mask.sum()) >= 3:
        p, lna = np.polyfit(np.log(Ml[mask]), np.log(dS[mask]), 1)
        tail_exc = quad_rho(lambda M: math.exp(lna + p*math.log(max(M, KAPPA))), 0.0, M_lo)[0]
    else:
        p, tail_exc = float('nan'), float('nan')
    print(f"\ntail [0,{M_lo}]: zone weight {w_tail:.3e}; exact flat {tail_flat:.10f} (err {tail_ferr:.0e});"
          f" fitted excess power p = {p:.2f}, capped excess estimate {tail_exc:.2e} (carried as +-)")
    print("mass-density totals on the panels (value, summed |K15-G7|):")
    for k, (v, e) in totals.items(): print(f"  {k:12s} {v:.10f}  +- {e:.1e}")
    fl = totals['flat_exact'][0] + tail_flat
    # builder's repair after hreduce's GAP08 review: a failed tail fit no longer adds a silent zero to the value and
    # the error; the tail's formed excess is an estimate (a capped power law), not a bound, and is labelled so.
    tail_ok = (tail_exc == tail_exc)
    h0 = totals['hold0'][0] + tail_flat + (tail_exc if tail_ok else float('nan'))
    err0 = totals['hold0'][1] + (abs(tail_exc) if tail_ok else float('nan')) + tail_ferr
    print(f"\n  flat_exact + tail: {fl:.10f}  (- ref {fl-ETA_FLAT_REF:+.2e}: the panel set's design error)")
    print(f"  flat_chain + tail: {totals['flat_chain'][0]+tail_flat:.10f}  (- ref: finite parent + panels)")
    print(f"  eta_hold0 (formed): {h0:.10f} +- {err0:.1e}   hold0 - C6 target {h0-C6_TARGET:+.3e}   hold0 - flat ref {h0-ETA_FLAT_REF:+.3e}"
          f"   relative excess over flat_chain {h0/(totals['flat_chain'][0]+tail_flat)-1:+.3e}")
    if 'hold12' in totals:
        print(f"  hold12 on the panels only: {totals['hold12'][0]:.8f} (the tail's hold-12 part is not bounded by this run)")
    if not tail_ok:
        print(f"  C6: NOT CERTIFIED (the tail fit failed: the tail's formed excess below M {M_lo} is not estimated; read the light mode)")
    else:
        print(f"  C6: {'PASS' if abs(h0-C6_TARGET)+err0 < C6_TOL else 'NOT PASSED'} under the rule |diff| + error < {C6_TOL}"
              f" (the tail's excess {tail_exc:.2e} is a capped power-law estimate, not a bound; the panel error is the summed |K15-G7|, a resolution diagnostic)")
    print_table(res, taus)
    dump(dict(nodes=nodes, breaks=breaks, totals=totals, tail=dict(flat=tail_flat, excess=tail_exc, weight=w_tail), results=res),
         args.out or 'mdensity.json')

def mode_light(args):
    taus = [0.0, 12.0]; table = {}
    for Nx in [int(x) for x in args.nxs.split(',')]:
        ch = Chain(Nx, args.device)
        res = run_channels(ch, LIGHT_MASSES, taus, args.stepper, args.dt, 1 if Nx >= 4096 else args.batch, f'light Nx={Nx}')
        print(f"\n=== light channels, Nx {Nx} ===  (finite-parent flat error = S4flat_chain - s4_exact)")
        print_table(res, taus); table[Nx] = res
    M = np.array(LIGHT_MASSES); rho = rho_mass(M); i0, i1 = len(M) - 2, len(M) - 1    # 0.0453, 0.0488
    for Nx, res in table.items():
        for t in taus:
            S = np.array([2*r['series'][t][0] for r in res])
            slope = (S[i1] - S[i0])/math.log(M[i1]/M[i0])                 # J6 log-linear continuation
            Scont = S[i0] + slope*np.log(M/M[i0])
            sel = slice(0, i0 + 1)
            direct, cont = trapezoid((rho*S)[sel], M[sel]), trapezoid((rho*Scont)[sel], M[sel])
            print(f"Nx {Nx} hold {t:g}: sub-grid integral [0.004,0.0453]: direct {direct:.6e}  J6-type continuation {cont:.6e}"
                  f"  difference {direct-cont:+.2e}  (tail below 0.004 has zone weight {0.004**2/math.pi:.1e})")
    if len(table) > 1:
        Nxs = sorted(table)
        print("\nparent dependence (S4 at Nx minus S4 at the smallest parent), hold 0 and hold 12:")
        for k, m in enumerate(LIGHT_MASSES):
            diffs = "  ".join(f"Nx{Nx} h{t:g}: {2*(table[Nx][k]['series'][t][0]-table[Nxs[0]][k]['series'][t][0]):+.2e}"
                              for Nx in Nxs[1:] for t in taus)
            print(f"  M {m:.5f}  {diffs}")
    dump({str(k): v for k, v in table.items()}, args.out or 'light.json')

def mode_ladders(args):
    taus_series = [float(t) for t in range(13)]
    ch = Chain(args.nx, args.device)
    print("=== time-step ladder (2-component values) ===")
    for M in (0.1, 0.7):
        # builder's repair: CF4 at dt 64 to 8 lies outside its convergence range on this chain (32-site test:
        # 2.7e-3 at dt 64, 1.2e-3 at 16, 4.3e-5 at 4, 3.0e-10 at 1 against a midpoint dt 0.25 reference)
        for stp, dt in (('midpoint', 1.0), ('midpoint', 0.5), ('cf4', 4.0), ('cf4', 2.0), ('cf4', 1.0), ('cf4', 0.5)):
            r = run_channels(ch, [M], [0.0, 12.0], stp, dt, 1, f'dt {stp} {dt}')[0]
            print(f"M {M}: {stp:8s} dt {dt:5.1f}: hold0 {r['series'][0.0][0]:.10f}  hold12 {r['series'][12.0][0]:.10f}"
                  f"  |SA-SB| {r['purity']:.1e}  orth {r['orth']:.1e}")
    print("\n=== parent ladder with the hold series (the wall-transit test; 4-component values) ===")
    for Nx in [int(x) for x in args.nxs.split(',')]:
        chN = Chain(Nx, args.device)
        for M in (0.0453, 0.1, 0.2):
            r = run_channels(chN, [M], taus_series, args.stepper, args.dt, 1, f'parent Nx={Nx} M={M}')[0]
            ser = "  ".join(f"{t:g}:{2*r['series'][t][0]:.4f}" for t in taus_series)
            arr = arrival_table(M, Nx)
            print(f"Nx {Nx} M {M}: S4 hold series {ser}")
            print(f"   S4 comparator {2*r['S_g']:.4f}  S4 dephased ceiling {2*r['S_diag']:.4f}  pocket count {r['pocket']:.2f}"
                  f"  nfrac {r['nfrac']}  |eps| of fractional modes <= {r['efrac_max']:.4f} (eps_p {pocket_depth(M):.4f})")
            print(f"   semiclassical arrival labels (CONDITIONAL): earliest {arr[0] if arr and arr[0] is not None else 'none'},"
                  f" ramp-end emission {arr[1] if arr and arr[1] is not None else 'none'}")

def mode_wall(args):
    print("M       w_c     t_open  eps_p   0.8M    R2 particle/hole at E = 0.01, 0.03, 0.06, 0.10 (|v| = 1.4)   v_g(k2, eps_p/2)"
          "   arrival labels (earliest, ramp-end) Nx 512 | 1024 | 2048")
    for M in (0.0, 0.0453, 0.1, 0.15, 0.2, 0.2465, 0.2653, 0.2855, 0.3, M_THRESHOLD):
        ep = pocket_depth(M); to = t_open(M)
        convs = []
        for E in (0.01, 0.03, 0.06, 0.10):
            if E < ep:
                wp = wall_conversion(M, -V_FAR, E, 'particle'); wh = wall_conversion(M, -V_FAR, -E, 'hole')
                convs.append(f"{wp['upper_band_fraction']:.3f}/{wh['upper_band_fraction']:.3f}" if wp and wh else "   -   ")
            else: convs.append("   -   ")
        k2 = k2_root(0.5*ep, -V_FAR, M) if ep > 0 else None
        vg = float(vg_plus(k2, -V_FAR, M)) if k2 is not None else float('nan')
        arrs = []
        for Nx in (512, 1024, 2048):
            a = arrival_table(M, Nx)
            arrs.append("  -  " if a is None else f"({a[0] if a[0] is not None else 'none'}, {a[1] if a[1] is not None else 'none'})")
        print(f"{M:.4f}  {w_c(M):.4f}  {to if to else float('nan'):6.0f}  {ep:.4f}  {0.8*M:.4f}   "
              + "  ".join(convs) + f"   {vg:.3f}   " + " | ".join(arrs))
    print("R2: flux fraction of the incident occupied mode sent into the upper band (particles: the escaping k_2 branch;"
          " holes: the trapped dip edge, i.e. the hole content of the escaping lower-band channel). Flux sums print 1 in the json.")
    print("analytic check at M = 0, E -> 0: |r_b| = |r_2| = sin(0.3875)/sin(0.775) = 0.540, fractions 0.50/0.50.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['selftest', 'anchor', 'j6compare', 'grid', 'mdensity', 'light', 'ladders', 'wall'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--nx', type=int, default=1024)
    ap.add_argument('--nxs', default='512,1024,2048')
    ap.add_argument('--nk', default='24,48')
    ap.add_argument('--taus', default='0,12')
    ap.add_argument('--stepper', default='cf4', choices=['cf4', 'midpoint'])
    ap.add_argument('--dt', type=float, default=1.0)   # builder's repair: 16 is not converged (see mode_ladders)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--breaks', default='')
    ap.add_argument('--adiabatic-above', type=float, default=0.0, dest='adiabatic_above')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--print-all', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"device {args.device}  Nx {args.nx}  stepper {args.stepper} dt {args.dt}  kappa {KAPPA}  T_ramp {T_RAMP}"
          f"  adiabatic shortcut above {args.adiabatic_above or 'off'}")
    globals()['mode_' + args.mode](args)

if __name__ == '__main__':
    main()
