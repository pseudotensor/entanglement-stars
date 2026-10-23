"""Fixed parameter choices for the native-action test (integ65 sec. 4 item 5 defaults, confirmed by the lead).

Masses: the lightest species of the two run inventories in lattice units, m = sqrt(.01) = 0.1 and sqrt(.04) = 0.2.
Contacts U = .01, .1.  k = 1 (main) and 10, 100, 1e4 (free-limit sequence).  b near 1 for M'_b, sigma near 1
for R = d_sigma H_sigma, each with two step sizes for the step checks.  Twisted-boundary momenta P = theta / L_x
on a symmetric grid.  Register: two readings (one particle in modes f_0, f_1) appended after the matter modes.
"""
import numpy as np

MASSES = {"inv1": 0.1, "inv2": 0.2}
U_VALUES = (0.01, 0.1)
K_MAIN = 1.0
K_FREE_LIMIT = (1.0, 10.0, 100.0, 1e4)
B_STEPS = (1e-3, 1e-4)
SIGMA_STEPS = (1e-3, 1e-4)
P_GRID = np.linspace(-0.05, 0.05, 9)
LATTICES = ("ring4", "plaq", "cube")
REGISTER_READINGS = 2


def twist_for(lattice_name, P):
    """Twist vector theta with momentum P along x: ring L = 4, plaquette L_x = 2 (the cube is open)."""
    Lx = {"ring4": 4, "plaq": 2}[lattice_name]
    return (P * Lx, 0.0, 0.0)


def register_modes(nmatter):
    """Indices of the register modes appended after nmatter matter modes."""
    return [nmatter + r for r in range(REGISTER_READINGS)]


def register_projector(sector, nmatter):
    """Pi = 1 on states with exactly one register particle (diagonal in the occupation basis)."""
    reg = register_modes(nmatter)
    occ = sector.occupations()
    return np.diag((occ[:, reg].sum(axis=1) == 1).astype(float))
