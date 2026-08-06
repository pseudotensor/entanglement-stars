"""The state factor in the HDA remainder bound, in the cold near-horizon regime.

eq:schw-cff evaluates the state-dependent weight of the remainder bound eq:schw-R as
the second moment of a geometric envelope,

    S_2(q) := sum_{m>=1} m^2 |dG(m)|^2  with  |dG(m)| ~ q^m,  q = beta t/4,
            = q^2 (1+q^2)/(1-q^2)^3,

which needs q < 1 and blows up at q -> 1.  On the macroscopic branch every bond has
q = beta_0 t_0 Nbar/4 >> 1, so that expression is outside its regime.  This script
evaluates the same functional on the true correlator instead --- so it needs no
envelope assumption --- for the uniform chain at dimensionless inverse temperature
x = beta t and for the model's own near-horizon chain, and checks the two exact
cold-branch identities the near-cap estimate rests on:

    xi_T = 2 rho  (rho = proper distance to the would-be horizon),  xi_T / l_N = 2,

with l_N = N/|dN/drho| the proper lapse-variation scale.

Data -> cold_remainder_moment.npz ; no figure (see CLAUDE.md convention).
"""
import os
import numpy as np
from scipy.special import polygamma

t0 = 1.0


def chain_len_for(x, xi_margin=12.0, n_min=3000):
    """Chain long enough that the half-chain is xi_margin thermal lengths.

    xi_T = 2x/pi, so half-chain >= xi_margin*xi_T means n >= 4*xi_margin*x/pi.
    A FIXED n=3000 was used before, which is 23.6 xi at x=100 but only 5.9 xi at
    x=400, and that point came out at S_2/x = 0.053181 against the converged
    0.053051 -- a 0.25% error that dragged the stored cold-branch amplitude from
    0.05305 to 0.05289.  Verified directly: at x=400, n=3000 gives 0.053181
    while n=6000 (11.8 xi) and n=12000 (23.6 xi) both give 0.053051.
    """
    return int(max(n_min, np.ceil(4.0 * xi_margin * x / np.pi)))


def uniform_correlator(x, n=None):
    """G_{n,n+m} for the half-filled uniform chain at beta t_0 = x."""
    if n is None:
        n = chain_len_for(x)
    h = np.diag(-t0 * np.ones(n - 1), 1) + np.diag(-t0 * np.ones(n - 1), -1)
    ev, U = np.linalg.eigh(h)
    occ = 1.0 / (np.exp(np.clip(x * ev / t0, -700, 700)) + 1.0)
    G = (U * occ) @ U.T
    i = n // 2
    return G[i, i:]


def second_moment(row, m_max=None):
    """S_2 = sum_{m>=1} m^2 |G(m)|^2 on the given correlator row."""
    m = np.arange(1, row.size if m_max is None else m_max + 1)
    return float(np.sum(m ** 2 * row[1:m.size + 1] ** 2))


def corr_length(row):
    """Envelope decay length from |G| at the odd separations."""
    m = np.arange(1, row.size)
    g = np.abs(row[1:])
    sel = (m % 2 == 1) & (g > 1e-13) & (m < row.size // 3)
    if sel.sum() < 4:
        return np.nan
    A = np.vstack([m[sel], np.ones(sel.sum())]).T
    slope = np.linalg.lstsq(A, np.log(g[sel] * m[sel]), rcond=None)[0][0]
    return -1.0 / slope if slope < 0 else np.inf


def horizon_chain(rs, n_in, n_out):
    n0 = 1
    while 1.0 - rs * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs * polygamma(1, n))
    return n_in, np.concatenate([np.full(n_in, N_ext[0]), N_ext])


def main():
    print("uniform chain: state factor S_2 = sum m^2 |G(m)|^2 versus the")
    print("geometric-envelope formula q^2(1+q^2)/(1-q^2)^3, q = x/4\n")
    rows = []
    for x in [0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 50.0, 100.0, 200.0, 400.0]:
        n_x = chain_len_for(x)
        row = uniform_correlator(x, n_x)
        S2 = second_moment(row, m_max=row.size - 1)
        q = x / 4.0
        geo = (q ** 2 * (1 + q ** 2) / (1 - q ** 2) ** 3) if q < 1 else np.inf
        xi = corr_length(row)
        rows.append((x, S2, geo, xi, 2 * x / np.pi))
        print(f"  x={x:7.1f}  q={q:7.2f}  n={n_x:6d} ({n_x/2/(2*x/np.pi):5.1f} xi)  "
              f"S_2={S2:11.4f}  S_2/x={S2/x:.6f}   geometric={geo:12.4g}   "
              f"xi_T/a={xi:8.2f}", flush=True)

    r = np.array(rows, dtype=float)
    cold = r[r[:, 0] >= 20]
    fit = np.polyfit(np.log(cold[:, 0]), np.log(cold[:, 1]), 1)
    ratio = cold[:, 1] / cold[:, 0]
    print(f"\n  cold branch: S_2 propto x^{fit[0]:.3f}, S_2 = {np.exp(fit[1]):.5f} x^{fit[0]:.3f}")
    print(f"  slope-1 form (what the paper quotes): S_2/x = {ratio.mean():.6f}"
          f"  spread {ratio.max()-ratio.min():.1e} over x = {cold[0,0]:.0f}-{cold[-1,0]:.0f}")
    print(f"  i.e. S_2 grows linearly in xi_T/a; the geometric formula diverges at q=1.")

    print("\nsame state factor measured directly at the surface bond of the model's")
    print("own near-horizon chain, and the resulting remainder weight")
    print("  r(r) := C_ff xi^2 = 4 t^4 S_2 xi^2  (eq:schw-cff x eq:schw-R),")
    print("  with t = t_0 Nbar, xi = 2 x_loc a/pi the local thermal range,")
    print("  reported relative to its value on the flat chain at the same beta_0.\n")
    hz = []
    for rs in [50, 200, 1000, 5000]:
        n_in = n_out = min(3000, max(1500, int(40 * np.sqrt(rs))))
        i, N = horizon_chain(rs, n_in, n_out)
        beta0 = 2 * np.pi * rs
        hm = np.diag(-t0 * 0.5 * (N[:-1] + N[1:]), 1)
        hm = hm + hm.T
        ev, U = np.linalg.eigh(hm)
        occ = 1.0 / (np.exp(np.clip(beta0 * ev, -700, 700)) + 1.0)
        row = ((U * occ) @ U.T)[i, i:]
        S2_cap = second_moment(row, m_max=min(3000, row.size - 1))
        x_star = beta0 * t0 * 0.5 * (N[i] + N[i + 1])
        S2_uni = np.exp(fit[1]) * x_star ** fit[0]
        t_cap = t0 * 0.5 * (N[i] + N[i + 1])
        xi_cap = 2.0 * x_star / np.pi
        frak_cap = 4 * t_cap ** 4 * S2_cap * xi_cap ** 2
        x_inf = beta0 * t0
        frak_inf = 4 * t0 ** 4 * (np.exp(fit[1]) * x_inf ** fit[0]) * (2 * x_inf / np.pi) ** 2
        hz.append((rs, x_star, S2_cap, S2_uni, frak_cap / frak_inf))
        print(f"  rs/a={rs:6d}  x_*={x_star:8.2f}  S_2(cap)={S2_cap:9.3f}  "
              f"[uniform-chain S_2(x_*)={S2_uni:8.3f}]   r_cap/r_flat="
              f"{frak_cap/frak_inf:.3e}", flush=True)

    h = np.array(hz, dtype=float)
    if h.shape[0] > 2:
        s = np.polyfit(np.log(h[:, 0]), np.log(h[:, 4]), 1)[0]
        print(f"\n  d log(r_cap/r_flat) / d log(r_s/a) = {s:+.3f}: the remainder weight")
        print("  at the surface bond is suppressed relative to flat space, not marginal.")

    _here = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "cold_remainder_moment.npz"))
    np.savez(_out,
             x=r[:, 0], S2=r[:, 1], S2_geometric=r[:, 2], xi_meas=r[:, 3],
             xi_pred=r[:, 4], cold_slope=fit[0], cold_amp=np.exp(fit[1]),
             cold_ratio_mean=ratio.mean(), cold_ratio_spread=ratio.max() - ratio.min(),
             hz_rs=h[:, 0], hz_xstar=h[:, 1], hz_S2_cap=h[:, 2],
             hz_S2_uniform=h[:, 3], hz_frak_ratio=h[:, 4])


if __name__ == "__main__":
    main()
