"""C5_v1d_generator_check_v1.py: with the barrier on, E = M(infinity) must still generate the shells' motion: by central differences
dE/dR_i = -n_i (dk_i/dt) and dE/dk_i = n_i (dR_i/dt) of dust_comparator_v1d's rhs (the centrifugal force included), for random shells
with L2 in [0, 30], the density kernel, the continuum and sine symbols. Also the rest release's velocity residual with the barrier."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust_comparator_v1d as D

rng = np.random.default_rng(11)
S = 40
R = rng.uniform(15.0, 200.0, S)
k = rng.uniform(-0.2, 0.3, S)
n = rng.uniform(0.5, 5.0, S)
L2 = np.where(rng.uniform(size=S) < 0.25, 0.0, rng.uniform(0.05, 30.0, S))
for disp in ("continuum", "sine"):
    grid = D.Grid(0.5, 1.5 * R.max() + 40.0)
    solver = D.FieldSolver(grid, D.Kernel(4.0, "density"), 0.25, lapse=True)
    sysm = D.ShellSystem(n, R, D.Dispersion(disp, 0.2), solver, 0.25, 0.2, 64.0, L2=L2)
    y = sysm.pack(R, k, np.zeros(S))
    dy = sysm.rhs(0.0, y)
    dR, dk = dy[:S], dy[S:2 * S]
    gR, gk = -n * dk, n * dR
    scale = max(np.abs(gR).max(), np.abs(gk).max())
    worst = 0.0
    for i in range(S):
        d = 1e-5
        Rp, Rm = R.copy(), R.copy(); Rp[i] += d; Rm[i] -= d
        fd = (sysm.fields(Rp, k)["E"] - sysm.fields(Rm, k)["E"]) / (2 * d)
        worst = max(worst, abs(fd - gR[i]) / scale)
        kp, km = k.copy(), k.copy(); kp[i] += d; km[i] -= d
        fd = (sysm.fields(R, kp)["E"] - sysm.fields(R, km)["E"]) / (2 * d)
        worst = max(worst, abs(fd - gk[i]) / scale)
    # the same with the centrifugal force removed from the rhs, to show the check sees it
    sysm2 = D.ShellSystem(n, R, D.Dispersion(disp, 0.2), solver, 0.25, 0.2, 64.0, L2=L2)
    F = sysm2.fields(R, k)
    gR_nocf = n * (F["sgn"] * F["dN"] * F["eps"] + F["dv"] * F["q"])
    worst_nocf = 0.0
    for i in range(S):
        d = 1e-5
        Rp, Rm = R.copy(), R.copy(); Rp[i] += d; Rm[i] -= d
        fd = (sysm2.fields(Rp, k)["E"] - sysm2.fields(Rm, k)["E"]) / (2 * d)
        worst_nocf = max(worst_nocf, abs(fd - gR_nocf[i]) / scale)
    k0, info, F0 = D.release(sysm2, "rest")
    print("%s: generator identity with the barrier, largest |central difference - rhs|/scale %.2e over %d shells (all dR and dk); without the centrifugal term %.2e; rest release residual %.1e in %d iterations; E %.6f"
          % (disp, worst, S, worst_nocf, info["velocity_residual"], info["iterations"], F["E"]))
