"""Does the cubic-lattice nearest-neighbour bond count enter the Gauss-law matching?

On a cubic lattice a coordinate sphere of radius R is crossed by (3/2)*4*pi*R^2/a^2
nearest-neighbour bonds, not 4*pi*R^2/a^2 = g_n: each axis contributes
<|n_e|> = 1/2 of the sphere area, and 3 * (1/2) = 3/2.

The shell chain instead assigns g_n radial channels per boundary (eq:shell-H), and
the absolute Gauss-law matching that fixes G (eq:sc-GM, eq:sc-G) uses kappa_n^flat
= g_n t_0^2.  This script asks which count the *flux* of a radial field actually
realises, by summing the discrete flux bond by bond on a3 Z^3.

The two counts differ because the flux weights each crossing bond by its direction
cosine as well: a bond along e contributes a|n_e| Phi' to the potential difference
and has crossing density |n_e|/a^2, so the flux-weighted count is
sum_e <n_e^2> * 4 pi R^2/a^2 = 4 pi R^2/a^2 = g_n exactly, while the raw count is
sum_e <|n_e|> * 4 pi R^2/a^2 = (3/2) g_n.

Measures, for Phi = -1/r on a3 Z^3 with unit bond conductance:
  raw_ratio  = (# NN bonds crossing |x| = R) / (4 pi R^2 / a^2)      -> 3/2
  flux_ratio = (sum_crossing kappa (Phi_out - Phi_in)) / (4 pi kappa / a)  -> 1
The second is the quantity that enters eq:sc-GM; the first is the quantity the
3/2 refers to.  Also checks the Dirichlet-form (energy) count, which controls the
continuum Laplacian normalisation, by the same weighting.

Data -> cubic_bond_flux.npz ; no figure (see CLAUDE.md convention).
"""
import os
import numpy as np

A = 1.0          # lattice spacing
KAPPA = 1.0      # per-bond conductance


def counts(R, L=None):
    """Raw and flux-weighted NN-bond counts across the sphere |x| = R on a Z^3."""
    L = int(np.ceil(R)) + 3 if L is None else L
    ax = np.arange(-L, L + 1, dtype=float) * A
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    inside = r < R

    raw = 0
    flux = 0.0
    energy = 0.0
    # Phi = -1/r (regularised at the origin, which is deep inside and never on a
    # crossing bond for R >> a); flux of a Newtonian field is r-independent.
    with np.errstate(divide="ignore"):
        Phi = np.where(r > 0, -1.0 / np.maximum(r, 1e-30), 0.0)

    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        cross = inside[lo] ^ inside[hi]          # bond straddles the sphere
        raw += int(cross.sum())
        dPhi = Phi[hi] - Phi[lo]
        # outward difference: + when the inner site is the low-index one
        sgn = np.where(inside[lo], 1.0, -1.0)
        flux += float((KAPPA * dPhi * sgn * cross).sum())
        # Dirichlet energy of the radial shell R <= |bond midpoint| < R + a:
        # this is what eq:shell-E has to reproduce with one bond per shell pair.
        rmid = 0.5 * (r[lo] + r[hi])
        shell = (rmid >= R) & (rmid < R + A)
        energy += float((KAPPA * dPhi**2 * shell).sum())

    return raw, flux, energy


def main():
    print("Phi = -1/r on a Z^3, unit bond conductance.")
    print("  raw   = (# NN bonds crossing |x|=R) / (4 pi R^2/a^2)   [the 3/2 count]")
    print("  flux  = (discrete Gauss flux) / (4 pi kappa GM / a)    [enters eq:sc-GM]")
    print("  ener  = (shell Dirichlet energy) / g_n kappa (a Phi')^2 [enters eq:shell-E]\n")
    Rs = [5, 10, 20, 30, 40, 60, 80]
    rows = []
    for R in Rs:
        raw, flux, energy = counts(float(R))
        g = 4.0 * np.pi * (R / A) ** 2
        raw_ratio = raw / g
        # Newtonian flux through a sphere: kappa_n a Phi'(R) with kappa_n = g_n kappa
        # gives 4 pi kappa GM/a with GM = 1; so normalise by 4 pi kappa / a.
        flux_ratio = flux / (4.0 * np.pi * KAPPA / A)
        # shell Dirichlet energy: g_n-channel prediction is g_n kappa (a Phi'(R))^2
        energy_ratio = energy / (g * KAPPA * A**2 / R**4)
        rows.append((R, raw, raw_ratio, flux_ratio, energy_ratio))
        print(f"  R/a={R:4d}  bonds={raw:8d}  raw/g_n={raw_ratio:.4f}  "
              f"flux/pred={flux_ratio:.5f}  energy/pred={energy_ratio:.5f}")

    r = np.array([(a, b, c, d, e) for a, b, c, d, e in rows], dtype=float)
    _here = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "cubic_bond_flux.npz"))
    np.savez(_out, R_over_a=r[:, 0], n_bonds=r[:, 1],
             raw_over_gn=r[:, 2], flux_over_pred=r[:, 3], energy_over_pred=r[:, 4])
    print(f"\nraw count      -> {r[-1,2]:.4f}   (analytic 3/2 = 3<|n_e|> )")
    print(f"Gauss-law flux -> {r[-1,3]:.5f}   (analytic 1   = 3<n_e^2>  = g_n channels)")
    print(f"Dirichlet form -> {r[-1,4]:.5f}   (analytic 1   = 3<n_e^2>  = g_n channels)")
    print("\nThe 3/2 is a raw bond count.  The flux and the Dirichlet form, which are")
    print("what eq:sc-GM and eq:sc-discrete-laplacian match, both realise exactly g_n.")


if __name__ == "__main__":
    main()
