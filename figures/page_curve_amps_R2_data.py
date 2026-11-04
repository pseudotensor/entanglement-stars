"""Page figure, route 2 (data script). Spec: WRITE_PAGE_SIGNED_v1.md sec. (3), FIGURE SPEC A-J (followed as written).

Writes <internal>/page_curve_amps_R2.npz (everything the figure script plots) and page_curve_amps_R2_report.json
(every REQUIRED number with its convergence diagnostic). Figure: page_curve_amps_R2.py loads the npz.

Method (spec B, I):
  route 1  first-order system psi' = (i q s3 - V s1) psi in stable (Riccati) scattering variables:
           phi1 = psi1 e^{-iqx}, phi2 = psi2 e^{iqx}; R = phi2/phi1, L = ln phi1;
           R' = -V e^{2iqx} + V e^{-2iqx} R^2, L' = -V e^{-2iqx} R (x-derivatives), integrated in y = ln(rho - 1) from rho_max to rho_min;
           Gamma = exp(-2 Re L(-inf)), unitarity Gamma + |R|^2 = 1 (the current identity |A_in|^2 - |A_ref|^2 = 1 divided by |A_in|^2).
  route 2  the independently reconstructed u'' + [q^2 - V^2 + V'] u = 0, u = psi1 + psi2, integrated as (u, u_x) in y;
           A_in = (i q u + u_x) e^{-iqx}/(2iq) at rho_min, Gamma = |A_in|^-2.
  Outer endpoint: each route starts from its own asymptotic series in 1/rho (route 1: (c1, c2) recurrences of the first-order system;
           route 2: g of u = e^{iqx} g from the second-order equation), summed to the smallest term; the smallest term is reported.
  Inner endpoint: rho_min - 1 = 1e-30; residual bound |Delta ln Gamma| <= 4 k (1 + 1e-30) sqrt(rho_min - 1) (V <= k sqrt(rho - 1), dx <= d rho rho/(rho-1)).
  Endpoint extension (I.2): rho_max x 10 and rho_min - 1 x 1e-10, each independently; changes in Gamma reported.
  Quadrature (I.4): composite Gauss-Legendre in ln q on [ln 1e-3, ln 3], 20 nodes per panel; 20 panels (400 nodes per channel), doubled to
           40 and 80 panels; diagnostics are the changes between levels.
  Spectral functionals (B, D): <g>_nu = sum_k (64 k/2 pi) int g_k dq; C0 = <q n>, D0 = <s(n)>, calR = D0/(4 pi C0), T_W = <W(n)>.
  Panel (a) (C): closed forms. Panel (b) (E-G): uniform reference; exact interval envelope by finite LPs on the quadrature nodes,
           solved by HiGHS dual simplex and by HiGHS interior point independently, plus an independent Lagrangian dual bound
           (two equality multipliers, per-node polygon maximization), I.6 residuals and normalized gaps.
  Pair response (D, I.7): direct four-state integration against the closed form.
Box only; nice 10; <= 16 threads.
"""
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
import mpmath as mp

OUT = sys.argv[1] if len(sys.argv) > 1 else "<internal>/page_fig_r2"
NPROC = 16
RHO_MIN1 = 1e-30          # rho_min - 1
RTOL, ATOL = 1e-12, 1e-14
KS = list(range(1, 9))
QMIN, QMAX = 1e-3, 3.0
SPEC_C0, SPEC_D0, SPEC_R = 0.005237315471, 0.107876529936, 1.6391110176
F_LIST = {"1": 1.0, "0.887": 0.887, "f_-": 0.7883924135, "f_+": 0.9830837141}
W_AVG = 1.0 / 25.0
LP_THR = 2e-9             # LP node threshold on wt s / D0 (see envelope): above HiGHS's small_matrix_value (1e-9), below
                          # which it drops matrix entries; entropy rows are passed scaled by 1/D0
# Figure geometry shared with page_curve_amps_R2.py (spec A: 7.0 x 2.4 in canvas): axes rectangles in figure fractions,
# inset rectangle in panel-(b) axes fractions. The refinement of J measures path shifts in these rendered units (pt).
LAYOUT = dict(fig_in=[7.0, 2.4], ax_a=[0.072, 0.175, 0.385, 0.70], ax_b=[0.585, 0.175, 0.385, 0.70], inset=[0.05, 0.08, 0.40, 0.50])


def layout_pt():
    fw, fh = LAYOUT["fig_in"]
    ax_w, ax_h = LAYOUT["ax_b"][2] * fw * 72, LAYOUT["ax_b"][3] * fh * 72
    return ax_w, ax_h, LAYOUT["inset"][2] * ax_w, LAYOUT["inset"][3] * ax_h
LEVELS = (1, 2, 4) if os.environ.get("PFIG_QUICK") else (20, 40, 80)   # panels of 20 Gauss-Legendre nodes per channel


# ---------------------------------------------------------------- scattering
def V_of(k, rm1):
    """V_k = k sqrt(1 - 1/rho)/rho with rho - 1 = rm1 (stable near the horizon)."""
    rho = 1.0 + rm1
    return k * np.sqrt(rm1 / rho) / rho


def x_of(rm1):
    return 1.0 + rm1 + np.log(rm1)


def series_route1(k, q, u, nmax=400):
    """Outgoing Jost column psi = e^{iqx} (c1, c2), c = sum a_n u^n, u = 1/rho.
    (1-u)(-u^2) c1' = -V c2 ; (1-u)(-u^2) c2' + 2iq c2 = -V c1 (u-derivatives), V(u) = k u sqrt(1-u)."""
    vc = [k * float(mp.binomial(0.5, j)) * (-1) ** j for j in range(nmax + 2)]   # V = sum_j vc[j] u^{j+1}
    a = np.zeros(nmax + 2, complex); b = np.zeros(nmax + 2, complex); a[0] = 1.0
    def conv(coef, n):  # [V * coef]_n = sum_{j} vc[j] coef[n-1-j]
        return sum(vc[j] * coef[n - 1 - j] for j in range(0, n))
    terms1, terms2 = [a[0]], [0.0]
    s1, s2 = complex(a[0]), 0j
    best = None
    for n in range(1, nmax):
        b[n] = (-conv(a, n) + (n - 1) * b[n - 1] - ((n - 2) * b[n - 2] if n >= 2 else 0)) / (2j * q)
        a[n] = (conv(b, n + 1) + (n - 1) * a[n - 1]) / n
        t1, t2 = a[n] * u ** n, b[n] * u ** n
        mag = abs(t1) + abs(t2)
        if best is not None and mag > best[0]:
            break
        s1 += t1; s2 += t2
        best = (mag, n)
        if mag < 1e-18:
            break
    return s1, s2, best[0]


def series_route2(k, q, u, nmax=400):
    """u-function e^{iqx} g, g = sum g_n u^n; D = d/dx = (1-u)(-u^2) d/du; 2iq Dg + D^2 g + W g = 0, W = -V^2 + DV."""
    V = np.zeros(nmax + 3)
    for j in range(nmax + 1):
        V[j + 1] = k * float(mp.binomial(0.5, j)) * (-1) ** j
    def Dop(c):
        out = np.zeros(len(c), dtype=c.dtype)
        for n in range(len(c)):
            out[n] = (-(n - 1) * c[n - 1] if n >= 1 else 0) + ((n - 2) * c[n - 2] if n >= 2 else 0)
        return out
    DV = Dop(V.astype(complex))
    V2 = np.convolve(V, V)[: nmax + 3]
    W = -V2 + DV.real
    g = np.zeros(nmax + 3, complex); g[0] = 1.0
    h = np.zeros(nmax + 3, complex)
    sg, sh = 1.0 + 0j, 0j
    best = None
    for n in range(2, nmax):
        # unknown g[n-1]; h_n = -(n-1) g_{n-1} + (n-2) g_{n-2}; (Dh)_n uses h_{n-1}, h_{n-2}
        h[n - 1] = -(n - 2) * g[n - 2] + ((n - 3) * g[n - 3] if n >= 3 else 0)
        Dh_n = -(n - 1) * h[n - 1] + (n - 2) * h[n - 2]
        Wg_n = sum(W[j] * g[n - j] for j in range(2, n + 1))
        g[n - 1] = (2j * q * (n - 2) * g[n - 2] + Dh_n + Wg_n) / (2j * q * (n - 1))
        h[n] = -(n - 1) * g[n - 1] + (n - 2) * g[n - 2]
        tg, th = g[n - 1] * u ** (n - 1), h[n - 1] * u ** (n - 1)
        mag = abs(tg) + abs(th)
        if best is not None and mag > best[0]:
            break
        sg += tg; sh += th
        best = (mag, n)
        if mag < 1e-18:
            break
    return sg, sh, best[0]


def rho_max_for(k, q):
    return max(400.0, 60.0 * (k * k + 1) / q)


def solve_route1(k, q, rmax_fac=1.0, rmin1=RHO_MIN1):
    rmax = rho_max_for(k, q) * rmax_fac
    c1, c2, tail = series_route1(k, q, 1.0 / rmax)
    x0 = x_of(rmax - 1.0)
    R0 = c2 * np.exp(2j * q * x0) / c1
    L0 = np.log(c1)
    def rhs(y, Y):
        rm1 = math.exp(y); rho = 1.0 + rm1
        x = 1.0 + rm1 + y
        V = V_of(k, rm1)
        R = Y[0] + 1j * Y[1]
        em = np.exp(-2j * q * x); ep = 1.0 / em
        dR = rho * (-V * ep + V * em * R * R)
        dL = rho * (-V * em * R)
        return [dR.real, dR.imag, dL.real, dL.imag]
    sol = solve_ivp(rhs, [math.log(rmax - 1.0), math.log(rmin1)], [R0.real, R0.imag, L0.real, L0.imag],
                    method="DOP853", rtol=RTOL, atol=ATOL)
    R = sol.y[0, -1] + 1j * sol.y[1, -1]
    L = sol.y[2, -1] + 1j * sol.y[3, -1]
    Gam = math.exp(-2.0 * L.real)
    return Gam, abs(R) ** 2, tail, sol.nfev


def solve_route2(k, q, rmax_fac=1.0, rmin1=RHO_MIN1):
    rmax = rho_max_for(k, q) * rmax_fac
    u = 1.0 / rmax
    g, h, tail = series_route2(k, q, u)
    x0 = x_of(rmax - 1.0)
    e = np.exp(1j * q * x0)
    U0, Ux0 = e * g, e * (1j * q * g + h)
    def rhs(y, Y):
        rm1 = math.exp(y); rho = 1.0 + rm1
        V = V_of(k, rm1)
        # V_x = (rho-1)/rho dV/drho; dV/drho = k [ (1/(2 rho^2 sqrt(1-1/rho))) / rho - sqrt(1-1/rho)/rho^2 ]
        sq = math.sqrt(rm1 / rho)
        dVdrho = k * (1.0 / (2.0 * rho ** 3 * sq) - sq / rho ** 2) if sq > 0 else 0.0
        Vx = (rm1 / rho) * dVdrho
        Wq = q * q - V * V + Vx
        u_ = Y[0] + 1j * Y[1]; ux = Y[2] + 1j * Y[3]
        du = rho * ux
        dux = -rho * Wq * u_
        return [du.real, du.imag, dux.real, dux.imag]
    sol = solve_ivp(rhs, [math.log(rmax - 1.0), math.log(rmin1)], [U0.real, U0.imag, Ux0.real, Ux0.imag],
                    method="DOP853", rtol=RTOL, atol=ATOL)
    uu = sol.y[0, -1] + 1j * sol.y[1, -1]; ux = sol.y[2, -1] + 1j * sol.y[3, -1]
    xm = x_of(rmin1)
    Ain = (1j * q * uu + ux) * np.exp(-1j * q * xm) / (2j * q)
    Aref = (1j * q * uu - ux) * np.exp(1j * q * xm) / (2j * q)
    Gam = 1.0 / abs(Ain) ** 2
    return Gam, abs(Aref) ** 2 / abs(Ain) ** 2, tail, sol.nfev


def node_job(args):
    k, q = args
    G1, R2, tail, nf = solve_route1(k, q)
    return (k, q, G1, R2, tail, nf)


def full_check(args):
    """Both routes, both endpoint extensions, unitarity (I.1-I.3)."""
    k, q = args
    G1, R1, t1, _ = solve_route1(k, q)
    G2, R2, t2, _ = solve_route2(k, q)
    G1o, _, _, _ = solve_route1(k, q, rmax_fac=10.0)
    G1i, _, _, _ = solve_route1(k, q, rmin1=RHO_MIN1 * 1e-10)
    G2o, _, _, _ = solve_route2(k, q, rmax_fac=10.0)
    G2i, _, _, _ = solve_route2(k, q, rmin1=RHO_MIN1 * 1e-10)
    return dict(k=k, q=q, Gamma_route1=G1, Gamma_route2=G2, rel_route_diff=(G2 - G1) / G1,
                ext_outer_rel_r1=(G1o - G1) / G1, ext_inner_rel_r1=(G1i - G1) / G1,
                ext_outer_rel_r2=(G2o - G2) / G2, ext_inner_rel_r2=(G2i - G2) / G2,
                unitarity_r1=G1 + R1 - 1.0, unitarity_r2=G2 + R2 - 1.0,
                series_tail_r1=t1, series_tail_r2=t2,
                inner_bound_rel=4 * k * math.sqrt(RHO_MIN1))


# ---------------------------------------------------------------- spectral functionals
def nodes_for(npan, npp=20):
    xg, wg = np.polynomial.legendre.leggauss(npp)
    edges = np.linspace(math.log(QMIN), math.log(QMAX), npan + 1)
    lq, w = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        lq.append(0.5 * (b - a) * xg + 0.5 * (a + b)); w.append(0.5 * (b - a) * wg)
    lq = np.concatenate(lq); w = np.concatenate(w)
    q = np.exp(lq)
    return q, w * q   # dq = q d ln q


def fermi(q):
    return 1.0 / (np.exp(4 * np.pi * q) + 1.0)


def s_of(n):
    return -n * np.log(n) - (1 - n) * np.log1p(-n)


def W_of(n):
    t = 2 * n - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(t) < 1e-300, 1.0, t / np.arctanh(t))
    return n * (1 - n) * r / 2.0


def functionals(table):
    """table: dict (k) -> (q, wq, Gamma)."""
    rows = []
    for k in KS:
        q, wq, G = table[k]
        n = G * fermi(q)
        wt = (64 * k / (2 * np.pi)) * wq
        rows.append((k, q, wt, n))
    C0 = sum((wt * q * n).sum() for k, q, wt, n in rows)
    D0 = sum((wt * s_of(n)).sum() for k, q, wt, n in rows)
    TW = sum((wt * W_of(n)).sum() for k, q, wt, n in rows)
    return C0, D0, D0 / (4 * np.pi * C0), TW, rows


# ---------------------------------------------------------------- panel (a)
def tau_of(z, R):
    return 1.0 - np.maximum(0.0, 1.0 - z / R) ** 1.5


def F_of(z, f, R):
    return np.minimum(z, f * (1.0 - z / R))


# ---------------------------------------------------------------- panel (b): LP envelope
def envelope(zc, R, f, wt, s, Wn, TW, D0):
    P = f * R / (R + f); D = R - P; beta = f / R
    zm, zp = max(0.0, zc - W_AVG / 2), min(R, zc + W_AVG / 2)
    dz = zp - zm
    out = dict(zc=zc, z_minus=zm, z_plus=zp, dz=dz)
    r0 = 0.5 * (1 + beta)
    phi = max(0.0, zp - max(zm, P)) / dz
    out["c_ref_early"] = r0 * phi
    if zp <= P:
        out.update(c_min=0.0, c_max=0.0, analytic_zero=True)
        return out
    a = max(zm, P) - P; b = zp - P; ell = b - a
    nn = len(wt)
    # variables x = [A (nn), B (nn)]
    lbA = np.maximum(0.0, P - a) * np.ones(nn); ubA = (P + a) * np.ones(nn)
    lbB = np.zeros(nn); ubB = max(D - b, 0.0) * np.ones(nn)
    bounds = list(zip(np.r_[lbA, lbB], np.r_[ubA, ubB]))
    from scipy.sparse import csr_matrix, hstack, vstack, identity
    I = identity(nn, format="csr")
    A_ub = vstack([hstack([-I, I]), hstack([I, -I])]).tocsr()          # B - A <= ell, A - B <= ell
    b_ub = ell * np.ones(2 * nn)
    A_eq = csr_matrix(np.vstack([np.r_[wt * s, np.zeros(nn)], np.r_[np.zeros(nn), wt * s]]))
    b_eq = np.array([D0 * (P - beta * a), D0 * (P - beta * b)])
    cvec = np.r_[wt * Wn, -wt * Wn]                                     # O = <W (A - B)>
    # Columns: every node whose normalized entropy weight wt s/D0 is at least LP_THR is its own column (HiGHS drops matrix
    # entries below its small_matrix_value). The remaining nodes, sorted by W/s, are grouped into consecutive blocks whose
    # summed weight reaches LP_THR; a block shares one (A, B) (identical node bounds, so the restriction is feasible for the
    # full problem). A remainder below LP_THR joins the last block, or, with no block, is held at the uniform history
    # A = P - beta a, B = P - beta b. The Lagrangian dual below runs over ALL nodes separately, so its normalized gap
    # encloses the effect of the grouping (I.5).
    es = wt * s; ew = wt * Wn
    big = es / D0 >= LP_THR
    gid = np.full(nn, -1)
    gid[big] = np.arange(int(big.sum()))
    ng = int(big.sum())
    small = np.where(~big)[0]
    small = small[np.argsort(Wn[small] / np.where(s[small] > 0, s[small], 1.0))]
    acc_, start = 0.0, ng
    for i in small:
        gid[i] = ng; acc_ += es[i] / D0
        if acc_ >= LP_THR:
            ng += 1; acc_ = 0.0
    if acc_ > 0:
        if ng > start:
            gid[gid == ng] = ng - 1            # remainder joins the last block
        else:
            gid[gid == ng] = -1                # no block: held at the uniform history
    Au_, Bu_ = P - beta * a, P - beta * b
    inb = gid >= 0
    G = csr_matrix((np.ones(int(inb.sum())), (np.where(inb)[0], gid[inb])), shape=(nn, ng))   # node -> column
    es_g = G.T @ es; ew_g = G.T @ ew
    Ig = identity(ng, format="csr")
    A_ub_b = vstack([hstack([-Ig, Ig]), hstack([Ig, -Ig])]).tocsr()
    A_eq_b = csr_matrix(np.vstack([np.r_[es_g, np.zeros(ng)], np.r_[np.zeros(ng), es_g]]))
    hs = float(es[~inb].sum())
    b_eq_b = b_eq - np.array([hs * Au_, hs * Bu_])
    cvec_b = np.r_[ew_g, -ew_g]
    bounds_b = [(lbA[0], ubA[0])] * ng + [(0.0, ubB[0])] * ng
    out["lp_columns"] = dict(nodes=nn, singletons=int(big.sum()), blocks=ng - int(big.sum()), held=int((~inb).sum()),
                             blocked_entropy_share=float(es[(~big) & inb].sum()) / D0, held_entropy_share=hs / D0)
    res = {}
    for sense, sgn in (("min", 1.0), ("max", -1.0)):
        sols = {}
        for meth in ("highs-ds", "highs-ipm"):
            # solved with entropy rows scaled by 1/D0 and objective by 1/T_W (residuals and multipliers reported unscaled)
            r = linprog(sgn * cvec_b / TW, A_ub=A_ub_b, b_ub=ell * np.ones(2 * ng), A_eq=A_eq_b / D0, b_eq=b_eq_b / D0,
                        bounds=bounds_b, method=meth,
                        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
            if r.status != 0:
                raise RuntimeError(f"LP {meth} {sense} at z = {zc}: {r.message}")
            Av = np.full(nn, Au_); Bv = np.full(nn, Bu_)
            Av[inb] = r.x[:ng][gid[inb]]; Bv[inb] = r.x[ng:][gid[inb]]
            x = np.r_[Av, Bv]
            eres = np.abs(A_eq @ x - b_eq) / (D0 * R)
            ires = max(0.0, (A_ub @ x - b_ub).max(), (np.r_[lbA, lbB] - x).max(), (x - np.r_[ubA, ubB]).max()) / R
            sols[meth] = dict(O=float(cvec @ x), status=r.status, eq_res=float(eres.max()), ineq_res=float(ires),
                              marg_eq=list(np.asarray(r.eqlin.marginals) * TW / D0) if hasattr(r, "eqlin") and r.eqlin is not None else None)
            if meth == "highs-ds":
                # J: reconstructed optimizing history, inventory linear through (0,P),(a,A),(b,B),(D,0); coincident knots merged
                knots = [(0.0, np.full(nn, P)), (a, Av), (b, Bv), (D, np.zeros(nn))]
                mk = [knots[0]]
                for t_, I_ in knots[1:]:
                    if t_ - mk[-1][0] > 1e-14:
                        mk.append((t_, I_))
                    else:
                        mk[-1] = (t_, I_)
                slope_v = max(float((np.abs((I2 - I1) / (t2 - t1)) - 1.0).max()) for (t1, I1), (t2, I2) in zip(mk[:-1], mk[1:]))
                neg_v = max(float((-I_).max()) for _, I_ in mk)
                ent_v = max(abs(float((wt * s * I_).sum()) / D0 - (P - beta * t_)) for t_, I_ in mk) / R
                sols[meth]["history"] = dict(max_slope_excess=slope_v, max_negative_inventory=neg_v / R, max_entropy_law_res=ent_v,
                                             end_inventory=float(np.abs(mk[-1][1]).max()) / R, n_knots=len(mk))
        # independent Lagrangian dual bound: g(lam) = sum_i wt_i max_{(A,B) in poly_i} [sgn'(W(A-B)) - lam1 s A - lam2 s B] + lam . b_eq
        sg = -sgn  # maximize sg * O
        # per-node polygon {lbA <= A <= ubA, 0 <= B <= ubB, |B - A| <= ell}: candidate set containing every vertex
        # (A-bound lines meet diagonals or B-bounds; B-bound lines meet diagonals), infeasible candidates masked
        cA, cB = [], []
        for Av_ in (lbA, ubA):
            cA += [Av_, Av_]; cB += [np.maximum(0.0, Av_ - ell), np.minimum(Av_ + ell, ubB)]
        for Bv_ in (np.zeros(nn), ubB):
            for Av_ in (Bv_ - ell, Bv_ + ell):
                cA.append(np.clip(Av_, lbA, ubA)); cB.append(Bv_)
        cA = np.array(cA); cB = np.array(cB)
        tol = 1e-12 * max(1.0, R)
        ok = ((cA >= lbA - tol) & (cA <= ubA + tol) & (cB >= -tol) & (cB <= ubB + tol) & (np.abs(cB - cA) <= ell + tol))
        if not ok.any(axis=0).all():
            raise RuntimeError("empty node polygon")
        def gdual(lam):
            val = sg * Wn * (cA - cB) - lam[0] * s * cA - lam[1] * s * cB
            val = np.where(ok, val, -np.inf)
            return lam[0] * b_eq[0] + lam[1] * b_eq[1] + float((wt * val.max(axis=0)).sum())
        m = sols["highs-ds"]["marg_eq"]
        lam0 = np.array([-sg * m[0], -sg * m[1]]) if m is not None else np.zeros(2)
        from scipy.optimize import minimize
        mi = sols["highs-ipm"]["marg_eq"]
        cand = [lam0, -lam0, np.array([-m[0], -m[1]]) if m is not None else np.zeros(2), np.array(m) if m is not None else np.zeros(2)]
        if mi is not None:
            cand += [-np.array(mi), np.array(mi)]
        best = min(cand, key=gdual)
        rr = minimize(gdual, best, method="Nelder-Mead", options={"xatol": 1e-14, "fatol": 1e-16, "maxiter": 4000})
        dual_bound = min(gdual(best), rr.fun)          # upper bound on max sg*O
        primal = sg * sols["highs-ds"]["O"]
        gap = (dual_bound - primal) / (2 * TW * dz)
        cbar = lambda O: (ell * TW + O) / (2 * TW * dz)
        res[sense] = dict(cbar_ds=cbar(sols["highs-ds"]["O"]), cbar_ipm=cbar(sols["highs-ipm"]["O"]),
                          ds_ipm_diff=cbar(sols["highs-ds"]["O"]) - cbar(sols["highs-ipm"]["O"]),
                          eq_res=max(sols[m_]["eq_res"] for m_ in sols), ineq_res=max(sols[m_]["ineq_res"] for m_ in sols),
                          dual_gap_norm=float(gap), status=[sols[m_]["status"] for m_ in sols], history=sols["highs-ds"]["history"])
    out.update(c_min=res["min"]["cbar_ds"], c_max=res["max"]["cbar_ds"], lp=res, a=a, b=b, ell=ell, analytic_zero=False)
    return out


# ---------------------------------------------------------------- pair response (I.7)
def pair_checks(n):
    mp.mp.dps = 40
    n = mp.mpf(n)
    # Fock basis |n_d n_a> ordered (00, 01, 10, 11); Jordan-Wigner with d first
    sm = mp.matrix([[0, 1], [0, 0]]); Z = mp.matrix([[1, 0], [0, -1]]); I2 = mp.eye(2)
    def kron(A, B):
        M = mp.matrix(4, 4)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        M[2 * i + k, 2 * j + l] = A[i, j] * B[k, l]
        return M
    cd = kron(sm, I2); ca = kron(Z, sm)
    cdd = cd.T; cad = ca.T
    psi = mp.matrix([0, mp.sqrt(1 - n), mp.sqrt(n), 0])     # sqrt(1-n)|0_d 1_a> + sqrt(n)|1_d 0_a>
    exp = lambda O: (psi.T * O * psi)[0]
    K = exp(cdd * ca)
    j = 1j * (mp.conj(K) * cdd * ca - K * cad * cd)
    nd, na = exp(cdd * cd), exp(cad * ca)
    p = [(1 - nd) * (1 - na), (1 - nd) * na, nd * (1 - na), nd * na]
    # W = int_0^1 Tr[w^s j w^{1-s} j] ds, w diagonal: sum_ij |j_ij|^2 int p_i^s p_j^{1-s} ds (direct integration)
    Wd = mp.mpf(0)
    for i in range(4):
        for jj in range(4):
            c = abs(j[i, jj]) ** 2
            if c == 0 or p[i] == 0 or p[jj] == 0:
                continue
            Wd += c * mp.quad(lambda s: p[i] ** s * p[jj] ** (1 - s), [0, 1])
    if n == 0 or n == 1:
        closed = mp.mpf(0)                      # Lambda = 0 rule: W(0) = W(1) = 0
    else:
        closed = mp.mpf(1) / 8 if n == mp.mpf(1) / 2 else n * (1 - n) * (2 * n - 1) / mp.log(n / (1 - n))
    var = sum(p[i] * sum(abs(j[i, jj]) ** 2 for jj in range(4)) for i in range(4))   # Tr w j^2 (centred: Tr w j = 0)
    numcov = exp(cdd * cd * cad * ca) - nd * na
    # complete pure pair sigma = |psi><psi|: sigma^s = sigma on (0,1), so the centred canonical covariance is
    # int Tr[sigma^s dj sigma^{1-s} dj] ds = <psi|dj|psi>^2 (zero); the ordinary variance <dj^2> is recorded beside it
    dj = j - exp(j) * mp.eye(4)
    pure_cov = exp(dj) ** 2
    pure_var = mp.re(exp(dj * dj))
    # dephased state: diag of |psi><psi| in the Fock basis -> cross block removed, number covariance kept
    rd = mp.diag([abs(psi[i]) ** 2 for i in range(4)])
    trr = lambda O: sum((rd * O)[i, i] for i in range(4))
    rho_deph_K = abs(trr(cdd * ca))
    deph_numcov = trr(cdd * cd * cad * ca) - trr(cdd * cd) * trr(cad * ca)
    return dict(n=mp.nstr(n, 15), K=mp.nstr(abs(K), 20), K2_vs_n1mn=mp.nstr(abs(K) ** 2 - n * (1 - n), 5),
                W_direct=mp.nstr(Wd, 25), W_closed=mp.nstr(closed, 25),
                rel_diff=mp.nstr((Wd - closed) / closed, 5) if closed != 0 else mp.nstr(Wd, 5),
                product_variance=mp.nstr(var, 20), variance_formula=mp.nstr(n * (1 - n) * (n ** 2 + (1 - n) ** 2), 20),
                number_cov=mp.nstr(numcov, 20), number_cov_formula=mp.nstr(-n * (1 - n), 20),
                centred_canonical_cov_pure=mp.nstr(abs(pure_cov), 5), ordinary_variance_pure=mp.nstr(pure_var, 20),
                dephased_cross_block=mp.nstr(rho_deph_K, 5),
                dephased_number_cov=mp.nstr(deph_numcov, 20))


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rep = {"spec": "WRITE_PAGE_SIGNED_v1.md sec. (3) A-J", "route": 2, "rho_min_minus_1": RHO_MIN1, "rtol": RTOL}
    levels = {}
    with Pool(NPROC) as pool:
        for npan in LEVELS:
            q, wq = nodes_for(npan)
            jobs = [(k, float(qq)) for k in KS for qq in q]
            cache = f"{OUT}/scatter_cache_{npan}.npz"   # route-1 node solutions; reused only for identical jobs and settings
            if os.path.exists(cache) and np.array_equal(np.load(cache)["jobs"], np.array(jobs)) and float(np.load(cache)["rtol"]) == RTOL \
                    and float(np.load(cache)["rmin1"]) == RHO_MIN1:
                out = [tuple(r_) for r_ in np.load(cache)["out"]]
                out = [(int(o[0]), float(o[1]), float(o[2]), float(o[3]), float(o[4]), int(o[5])) for o in out]
            else:
                out = pool.map(node_job, jobs, chunksize=4)
                np.savez(cache, jobs=np.array(jobs), out=np.array(out, dtype=float), rtol=RTOL, rmin1=RHO_MIN1)
            G = {k: np.array([o[2] for o in out if o[0] == k]) for k in KS}
            table = {k: (q, wq, G[k]) for k in KS}
            C0, D0, R, TW, rows = functionals(table)
            uni = max(abs(o[2] + o[3] - 1.0) for o in out)
            levels[npan] = dict(nodes_per_channel=len(q), C0=C0, D0=D0, R=R, TW=TW, max_unitarity=uni, table=table, rows=rows,
                                max_series_tail=max(o[4] for o in out))
            print(npan, len(q), C0, D0, R, TW, uni, time.time() - t0, flush=True)
        # two routes + endpoint extension at the REQUIRED points and on a node subsample
        pts = [(k, q) for k in (1, 2, 4, 8) for q in (0.01, 0.1, 0.3, 1.0, 3.0)]
        chk = pool.map(full_check, pts)
        q40, _ = nodes_for(LEVELS[1])
        sub = [(k, float(qq)) for k in KS for qq in q40[::10]]
        chk_sub = pool.map(full_check, sub)
    fin = levels[LEVELS[2]]; mid = levels[LEVELS[1]]; lo = levels[LEVELS[0]]
    def rel(a, b): return abs(a - b) / abs(b)
    rep["C0"] = fin["C0"]; rep["D0"] = fin["D0"]; rep["calR"] = fin["R"]; rep["T_W"] = fin["TW"]
    rep["quadrature_levels"] = {str(k): {kk: v for kk, v in d.items() if kk not in ("table", "rows")} for k, d in levels.items()}
    rep["conv_diag"] = {"C0_rel_40_80": rel(mid["C0"], fin["C0"]), "D0_rel_40_80": rel(mid["D0"], fin["D0"]),
                        "TW_rel_40_80": rel(mid["TW"], fin["TW"]), "R_rel_40_80": rel(mid["R"], fin["R"]),
                        "C0_rel_20_40": rel(lo["C0"], mid["C0"]), "D0_rel_20_40": rel(lo["D0"], mid["D0"]), "TW_rel_20_40": rel(lo["TW"], mid["TW"])}
    rep["spec_reproduction_rel_diff"] = {"C0": (fin["C0"] - SPEC_C0) / SPEC_C0, "D0": (fin["D0"] - SPEC_D0) / SPEC_D0, "calR": (fin["R"] - SPEC_R) / SPEC_R}
    rep["Gamma_points"] = chk
    rep["route_check_subsample"] = {"n": len(chk_sub), "max_rel_route_diff": max(abs(c["rel_route_diff"]) for c in chk_sub),
                                    "max_ext_outer": max(max(abs(c["ext_outer_rel_r1"]), abs(c["ext_outer_rel_r2"])) for c in chk_sub),
                                    "max_ext_inner": max(max(abs(c["ext_inner_rel_r1"]), abs(c["ext_inner_rel_r2"])) for c in chk_sub),
                                    "max_unitarity": max(max(abs(c["unitarity_r1"]), abs(c["unitarity_r2"])) for c in chk_sub)}
    rep["max_current_residual_I3"] = max(max(abs(c["unitarity_r1"]), abs(c["unitarity_r2"])) for c in chk + chk_sub)
    # panel (a)
    R = fin["R"]
    pa = {}
    for lab, f in F_LIST.items():
        zP = f * R / (R + f); tP = 1 - (R / (R + f)) ** 1.5
        pa[lab] = dict(z_P=zP, tau_P=tP, F_f_at_zP=float(F_of(zP, f, R)))
    rep["panel_a"] = pa
    # panel (b) at both finest levels
    def panel_b(level):
        rows = level["rows"]; R_ = level["R"]; TW_ = level["TW"]; D0_ = level["D0"]
        wt = np.concatenate([r[2] for r in rows]); n = np.concatenate([r[3] for r in rows])
        s = s_of(n); Wn = W_of(n)
        keep = ~((wt == 0) | (s == 0))
        wt, s, Wn = wt[keep], s[keep], Wn[keep]
        f = 1.0
        P = f * R_ / (R_ + f); beta = f / R_; r0 = 0.5 * (1 + beta)
        zx = P + (W_AVG / 2) * (R_ - f) / (R_ + f)
        return dict(R=R_, TW=TW_, D0=D0_, wt=wt, s=s, Wn=Wn, P=P, r0=r0, zx=zx, dropped=int((~keep).sum()))
    pb = panel_b(fin); pbm = panel_b(mid)
    R_, P, r0, zx = pb["R"], pb["P"], pb["r0"], pb["zx"]
    rep["r_0"] = r0; rep["z_x"] = zx; rep["tau_x"] = float(tau_of(zx, R_)); rep["dropped_nodes"] = pb["dropped"]
    named = {"P - w/2": P - W_AVG / 2, "P": P, "z_x": zx, "P + w/2": P + W_AVG / 2, "P + 0.1": P + 0.1, "P + 0.5": P + 0.5, "calR - w/2": R_ - W_AVG / 2}
    rep["named_centres"] = {}
    for lab, z in named.items():
        e = envelope(z, R_, 1.0, pb["wt"], pb["s"], pb["Wn"], pb["TW"], pb["D0"])
        em = envelope(z, pbm["R"], 1.0, pbm["wt"], pbm["s"], pbm["Wn"], pbm["TW"], pbm["D0"])
        rep["named_centres"][lab] = dict(z=z, tau=float(tau_of(z, R_)), c_ref_early=e["c_ref_early"], c_min=e["c_min"], c_max=e["c_max"],
                                         lp=e.get("lp"), conv_c_min=abs(e["c_min"] - em["c_min"]), conv_c_max=abs(e["c_max"] - em["c_max"]))
    # whole post-Page interval: a = 0, b = D -> both extrema r_0 (J acceptance)
    D = R_ - P
    wt, s, Wn, TW, D0 = pb["wt"], pb["s"], pb["Wn"], pb["TW"], pb["D0"]
    # emulate an interval [P, calR] by a centre with half width D/2
    save = globals()["W_AVG"]
    globals()["W_AVG"] = D
    ew = envelope(P + D / 2, R_, 1.0, wt, s, Wn, TW, D0)
    globals()["W_AVG"] = save
    rep["whole_post_page"] = dict(c_min=ew["c_min"], c_max=ew["c_max"], r_0=r0, dmin=ew["c_min"] - r0, dmax=ew["c_max"] - r0, lp=ew.get("lp"))
    # plotted centres (J): 201 even + P +- w/2, P, z_x, w/2, calR - w/2 and every local maximum of the envelope edges,
    # refined until inserting a midpoint moves no rendered path (main axes or inset, in pt) by 0.1 pt or more
    tP1 = float(tau_of(P, R_)); tx = float(tau_of(zx, R_))
    t_in0, t_in1 = tP1 - 2 * (tx - tP1), tP1 + 4 * (tx - tP1)
    ax_w, ax_h, in_w, in_h = layout_pt()
    def to_pt(t, c, inset):
        if inset:
            return np.array([(t - t_in0) / (t_in1 - t_in0) * in_w, (c - 0.35) / 0.30 * in_h])
        return np.array([t * ax_w, c / 1.05 * ax_h])
    def seg_dev(p0, pm, p1):
        d = p1 - p0; L2 = float(d @ d)
        if L2 == 0:
            return float(np.hypot(*(pm - p0)))
        u = min(1.0, max(0.0, float((pm - p0) @ d) / L2))
        return float(np.hypot(*(pm - p0 - u * d)))
    E = {}
    def evaluate(zl, pool):
        zl = [z for z in zl if z not in E]
        for z, e in zip(zl, pool.starmap(envelope, [(z, R_, 1.0, wt, s, Wn, TW, D0) for z in zl])):
            E[z] = e
    def dev(z0, zm, z1):
        out = 0.0
        t0_, tm_, t1_ = (float(tau_of(z, R_)) for z in (z0, zm, z1))
        insets = [False] + ([True] if (t1_ >= t_in0 and t0_ <= t_in1) else [])
        for key in ("c_min", "c_max", "c_ref_early"):
            for ins in insets:
                out = max(out, seg_dev(to_pt(t0_, E[z0][key], ins), to_pt(tm_, E[zm][key], ins), to_pt(t1_, E[z1][key], ins)))
        return out
    def refine(pending, pool, log):
        rounds = 0
        while pending and rounds < 30:
            rounds += 1
            mids = [0.5 * (z0 + z1) for z0, z1 in pending]
            evaluate(mids, pool)
            nxt = []
            for (z0, z1), zm in zip(pending, mids):
                if dev(z0, zm, z1) >= 0.1 and z1 - z0 > 1e-9:
                    keep.add(zm); nxt += [(z0, zm), (zm, z1)]
            log.append(dict(round=rounds, tested=len(pending), split=len(nxt) // 2))
            pending = nxt
        return pending
    zs0 = list(np.linspace(0.0, R_, 201)) + [P, P - W_AVG / 2, P + W_AVG / 2, zx, W_AVG / 2, R_ - W_AVG / 2]
    keep = set(float(z) for z in zs0 if 0 <= z <= R_)
    rlog = []
    with Pool(NPROC) as pool:
        evaluate(sorted(keep), pool)
        zl = sorted(keep)
        left = refine(list(zip(zl[:-1], zl[1:])), pool, rlog)
        # local maxima of c_min and c_max (prominence > 1e-9 against both neighbours), located to 1e-7 in z and inserted
        from scipy.optimize import minimize_scalar
        maxima = []
        zl = sorted(keep)
        for key in ("c_min", "c_max"):
            cv = np.array([E[z][key] for z in zl])
            for i in range(1, len(zl) - 1):
                if cv[i] - cv[i - 1] > 1e-9 and cv[i] - cv[i + 1] > 1e-9:
                    rr = minimize_scalar(lambda z: -envelope(z, R_, 1.0, wt, s, Wn, TW, D0)[key], bounds=(zl[i - 1], zl[i + 1]),
                                         method="bounded", options={"xatol": 1e-7})
                    maxima.append(dict(curve=key, z=float(rr.x), value=float(-rr.fun), grid_value=float(cv[i])))
        for m_ in maxima:
            keep.add(m_["z"])
        evaluate(sorted(keep), pool)
        zl = sorted(keep)
        nbr = [(a_, b_) for a_, b_ in zip(zl[:-1], zl[1:]) if any(a_ == m_["z"] or b_ == m_["z"] for m_ in maxima)]
        left += refine(nbr, pool, rlog)
        zs = sorted(keep)
        final_dev = max(dev(z0, 0.5 * (z0 + z1), z1) for z0, z1 in zip(zs[:-1], zs[1:]) if 0.5 * (z0 + z1) in E) if len(zs) > 1 else 0.0
        envs = [E[z] for z in zs]
        envs_m = pool.starmap(envelope, [(z, pbm["R"], 1.0, pbm["wt"], pbm["s"], pbm["Wn"], pbm["TW"], pbm["D0"]) for z in zs])
    rep["refinement"] = dict(rounds=rlog, unresolved_segments=len(left), maxima=maxima, n_centres=len(zs),
                             max_tested_midpoint_shift_pt=final_dev, inset_tau=[t_in0, t_in1], layout_pt=[ax_w, ax_h, in_w, in_h])
    cmin = np.array([e["c_min"] for e in envs]); cmax = np.array([e["c_max"] for e in envs]); cref = np.array([e["c_ref_early"] for e in envs])
    cmin_m = np.array([e["c_min"] for e in envs_m]); cmax_m = np.array([e["c_max"] for e in envs_m])
    lp_eq = max([e["lp"][sd]["eq_res"] for e in envs if not e["analytic_zero"] for sd in ("min", "max")] + [0])
    lp_in = max([e["lp"][sd]["ineq_res"] for e in envs if not e["analytic_zero"] for sd in ("min", "max")] + [0])
    lp_gap = max([abs(e["lp"][sd]["dual_gap_norm"]) for e in envs if not e["analytic_zero"] for sd in ("min", "max")] + [0])
    lp_dsipm = max([abs(e["lp"][sd]["ds_ipm_diff"]) for e in envs if not e["analytic_zero"] for sd in ("min", "max")] + [0])
    held = max([e["lp_columns"]["blocked_entropy_share"] + e["lp_columns"]["held_entropy_share"] for e in envs if not e["analytic_zero"]] + [0])
    blocks = max([e["lp_columns"]["blocks"] for e in envs if not e["analytic_zero"]] + [0])
    rep["envelope_grid"] = dict(n_centres=len(zs), max_eq_res=lp_eq, max_ineq_res=lp_in, max_norm_gap=lp_gap, max_ds_ipm_diff=lp_dsipm,
                                max_grouped_entropy_share=held, max_blocks=blocks,
                                conv_max_ordinate_mid_fin=float(max(np.abs(cmin - cmin_m).max(), np.abs(cmax - cmax_m).max())))
    tz = tau_of(np.array(zs), R_)
    rep["tau_conv_mid_fin"] = float(np.abs(tau_of(np.array(zs), pbm["R"]) - tz).max())
    # panel (a) grid: closed forms; 2001 even + every z_P, refined the same way on the main axes (y range 0.7)
    za = sorted(set(list(np.linspace(0.0, R_, 2001)) + [pa[l]["z_P"] for l in pa]))
    rep["panel_a_points"] = len(za)
    def dev_a(f):
        out = 0.0
        for z0, z1 in zip(za[:-1], za[1:]):
            zm = 0.5 * (z0 + z1)
            p = [np.array([float(tau_of(z, R_)) * ax_w, float(F_of(z, f, R_)) / 0.7 * ax_h]) for z in (z0, zm, z1)]
            out = max(out, seg_dev(p[0], p[1], p[2]))
        return out
    rep["panel_a_max_midpoint_shift_pt"] = {l: dev_a(f) for l, f in F_LIST.items()}
    # acceptance checks computable here
    zz = np.linspace(0, R_, 2001)
    acc = {}
    for lab, f in F_LIST.items():
        F = F_of(zz, f, R_)
        acc[f"F_{lab} >= 0"] = bool((F >= 0).all())
        acc[f"F_{lab}(0) = F_{lab}(calR) = 0"] = bool(abs(F_of(0.0, f, R_)) == 0 and abs(F_of(R_, f, R_)) < 1e-15)
    acc["ordering 0 <= c_min <= c_ref <= c_max <= 1"] = bool(((cmin >= -1e-9) & (cmin <= cref + 1e-9) & (cref <= cmax + 1e-9) & (cmax <= 1 + 1e-9)).all())
    acc["whole post-Page extrema = r_0"] = bool(abs(ew["c_min"] - r0) < 1e-7 and abs(ew["c_max"] - r0) < 1e-7)
    # F_f maxima at z_P with value z_P, and the capacity envelope F_f <= min(z, f(1 - z/calR))
    for lab, f in F_LIST.items():
        F = F_of(zz, f, R_)
        acc[f"F_{lab} max = z_P at z_P"] = bool(abs(F.max() - pa[lab]["z_P"]) <= (zz[1] - zz[0]) and abs(pa[lab]["F_f_at_zP"] - pa[lab]["z_P"]) < 1e-15)
    # uniform release satisfies every constraint of E: theta = r_0 after P, I = F_f, <s theta> = r_0 D0, I(calR) = 0
    unif_res = abs(float((wt * s).sum()) * r0 - r0 * D0) / (D0 * R_)
    acc["uniform release satisfies E"] = bool(unif_res < 1e-12 and abs(F_of(R_, 1.0, R_)) < 1e-15 and 0 <= r0 <= 1 and abs(P - (2 * r0 - 1) * D) < 1e-12)
    rep["uniform_release_residual"] = unif_res
    # reconstructed optimizing histories for intervals crossing P and ending at calR
    hist = {}
    for lab in ("P - w/2", "P", "z_x", "P + w/2", "calR - w/2"):
        lp = rep["named_centres"][lab]["lp"]
        if lp:
            hist[lab] = {sd: lp[sd]["history"] for sd in ("min", "max")}
    hist["whole post-Page"] = {sd: ew["lp"][sd]["history"] for sd in ("min", "max")}
    rep["reconstructed_histories"] = hist
    acc["reconstructed optimizing histories admissible (crossing P, ending at calR)"] = bool(all(
        h["max_slope_excess"] < 1e-9 and h["max_negative_inventory"] < 1e-9 and h["max_entropy_law_res"] < 1e-9 for d in hist.values() for h in d.values()))
    acc["LP feasibility < 1e-9 and normalized gap < 1e-7 (all centres)"] = bool(lp_eq < 1e-9 and lp_in < 1e-9 and lp_gap < 1e-7)
    rep["acceptance_computed"] = acc
    rep["pair_response_I7"] = [pair_checks(n) for n in ("0.5", "0.1", "1e-3", "1e-12")]
    rep["pair_singular_limits_I7"] = [pair_checks(n) for n in ("0", "1")]   # W_direct is the absolute value; closed form 0
    np.savez(f"{OUT}/page_curve_amps_R2.npz", R=R_, C0=fin["C0"], D0=fin["D0"], TW=TW, P=P, r0=r0, zx=zx, w=W_AVG,
             f_names=np.array(list(F_LIST.keys())), f_vals=np.array(list(F_LIST.values())),
             zs=np.array(zs), cmin=cmin, cmax=cmax, cref=cref, za=np.array(za), inset_tau=np.array([t_in0, t_in1]),
             fig_in=np.array(LAYOUT["fig_in"]), ax_a=np.array(LAYOUT["ax_a"]), ax_b=np.array(LAYOUT["ax_b"]), inset=np.array(LAYOUT["inset"]),
             node_wt=wt, node_s=s, node_W=Wn)
    rep["elapsed_s"] = time.time() - t0
    json.dump(rep, open(f"{OUT}/page_curve_amps_R2_report.json", "w"), indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("C0", "D0", "calR", "T_W", "spec_reproduction_rel_diff", "conv_diag", "r_0", "z_x", "tau_x")}, indent=1, default=float))


if __name__ == "__main__":
    main()
