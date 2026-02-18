#!/usr/bin/env python3
"""
Full 3D cubic-lattice solver — GPU-accelerated via PyTorch.

Key optimization: never form the full N×N correlation matrix.
Only compute needed bond elements via batched dot products on GPU.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from scipy.optimize import least_squares
import time
import torch

DEVICE = torch.device('cuda:0')
torch.set_default_dtype(torch.float64)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def binary_entropy_torch(x):
    x = torch.clamp(x, 1e-30, 1.0 - 1e-30)
    return -x * torch.log(x) - (1.0 - x) * torch.log(1.0 - x)


class CubicLattice3D:
    """3D cubic lattice within sphere of radius R_max."""

    def __init__(self, R_max):
        self.R_max = R_max
        t0 = time.time()

        coords = np.mgrid[-R_max:R_max+1, -R_max:R_max+1, -R_max:R_max+1]
        coords = coords.reshape(3, -1).T
        r2 = np.sum(coords**2, axis=1)
        mask = r2 <= R_max * R_max
        self.sites = coords[mask]
        self.N_sites = len(self.sites)
        self.r = np.sqrt(np.sum(self.sites**2, axis=1).astype(float))
        self.shell_of = np.round(self.r).astype(int)
        self.N_shell = self.shell_of.max() + 1

        self.shell_sites = [np.where(self.shell_of == n)[0] for n in range(self.N_shell)]
        self.G_n = np.array([len(s) for s in self.shell_sites])

        # Build neighbor pairs
        coord_to_idx = {}
        for i in range(self.N_sites):
            coord_to_idx[tuple(self.sites[i])] = i

        row, col = [], []
        for i in range(self.N_sites):
            x, y, z = self.sites[i]
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                j = coord_to_idx.get((x+dx, y+dy, z+dz), -1)
                if j >= 0:
                    row.append(i)
                    col.append(j)
        self.row = np.array(row, dtype=np.int64)
        self.col = np.array(col, dtype=np.int64)

        self.shell_row = self.shell_of[self.row]
        self.shell_col = self.shell_of[self.col]

        fwd = self.shell_col == self.shell_row + 1
        self.fwd_row = self.row[fwd]
        self.fwd_col = self.col[fwd]
        self.fwd_shell = self.shell_row[fwd]

        intra = (self.shell_col == self.shell_row) & (self.row < self.col)
        self.intra_row = self.row[intra]
        self.intra_col = self.col[intra]
        self.intra_shell = self.shell_row[intra]

        # Pre-convert to torch tensors on GPU
        self.t_row = torch.tensor(self.row, dtype=torch.long, device=DEVICE)
        self.t_col = torch.tensor(self.col, dtype=torch.long, device=DEVICE)
        self.t_fwd_row = torch.tensor(self.fwd_row, dtype=torch.long, device=DEVICE)
        self.t_fwd_col = torch.tensor(self.fwd_col, dtype=torch.long, device=DEVICE)
        self.t_fwd_shell = torch.tensor(self.fwd_shell, dtype=torch.long, device=DEVICE)
        self.t_intra_row = torch.tensor(self.intra_row, dtype=torch.long, device=DEVICE)
        self.t_intra_col = torch.tensor(self.intra_col, dtype=torch.long, device=DEVICE)
        self.t_intra_shell = torch.tensor(self.intra_shell, dtype=torch.long, device=DEVICE)
        self.t_shell_of = torch.tensor(self.shell_of, dtype=torch.long, device=DEVICE)

        # Core site indices on GPU
        self.core_sites_per_shell = []
        for n in range(self.N_shell):
            self.core_sites_per_shell.append(
                torch.tensor(self.shell_sites[n], dtype=torch.long, device=DEVICE))

        elapsed = time.time() - t0
        print(f"  Lattice: R={R_max}, {self.N_sites} sites, {self.N_shell} shells, "
              f"{len(self.row)//2} bonds ({len(self.fwd_row)} fwd, {len(self.intra_row)} intra) "
              f"[{elapsed:.1f}s]", flush=True)
        print(f"  Shells: {list(self.G_n)}", flush=True)


class ThreeDModel:
    """Full 3D self-consistent closure equation — GPU accelerated.

    Key: never form full G matrix. Compute bond elements via batched dot products.
    """

    def __init__(self, lat, t0=1.0, V0=0.01, n_core=2, beta0=0.1, cstar_sq=0.5):
        self.lat = lat
        self.t0 = t0
        self.V0 = V0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.N_shell = lat.N_shell
        self.n_call = 0

        # Precompute bg/tgt
        t_s = time.time()
        self.rho_bg = self._energy_profile_np(np.zeros(self.N_shell), smeared=False, add_V0=False)
        self.rho_tgt = self._energy_profile_np(np.zeros(self.N_shell), smeared=False, add_V0=True)
        self.t_rho_src = torch.tensor(self.rho_bg - self.rho_tgt, dtype=torch.float64, device=DEVICE)

        # Background MI per forward bond for normalization
        self.mi_bg_bonds_t = self._compute_mi_background_gpu()

        print(f"  bg/tgt in {time.time()-t_s:.1f}s  "
              f"src_norm={np.sum(np.abs(self.rho_tgt - self.rho_bg)):.6f}", flush=True)

    def _gpu_eigh(self, H_t):
        """GPU eigendecomposition, return (evals, evecs) on GPU."""
        return torch.linalg.eigh(H_t)

    def _fermi(self, evals):
        """Fermi function on GPU."""
        be = torch.clamp(self.beta0 * evals, -500, 500)
        return 1.0 / (torch.exp(be) + 1.0)

    def _build_H_gpu(self, Phi_shell_t, smeared=False, add_V0=False):
        """Build dense Hamiltonian on GPU."""
        N = self.lat.N_sites
        H = torch.zeros((N, N), dtype=torch.float64, device=DEVICE)

        if smeared:
            lapse = 1.0 + Phi_shell_t / self.cstar_sq
            Nbar = 0.5 * (lapse[self.lat.t_shell_of[self.lat.t_row]]
                        + lapse[self.lat.t_shell_of[self.lat.t_col]])
            H[self.lat.t_row, self.lat.t_col] = -self.t0 * torch.abs(Nbar)
        else:
            H[self.lat.t_row, self.lat.t_col] = -self.t0

        if add_V0:
            for n in range(min(self.n_core, self.N_shell)):
                idx = self.lat.core_sites_per_shell[n]
                H[idx, idx] += self.V0

        return H

    def _energy_profile_np(self, Phi_shell, smeared=False, add_V0=False):
        """Energy per shell — full approach for initialization (returns numpy)."""
        Phi_t = torch.tensor(Phi_shell, dtype=torch.float64, device=DEVICE)
        H = self._build_H_gpu(Phi_t, smeared=smeared, add_V0=add_V0)
        evals, evecs = self._gpu_eigh(H)
        f = self._fermi(evals)

        rho = torch.zeros(self.N_shell, dtype=torch.float64, device=DEVICE)

        # Forward bonds: G[i,j] = sum_k V[i,k]*f[k]*V[j,k]
        V_fwd_i = evecs[self.lat.t_fwd_row]  # K_fwd × N
        V_fwd_j = evecs[self.lat.t_fwd_col]  # K_fwd × N
        G_fwd = (V_fwd_i * V_fwd_j) @ f      # K_fwd

        if smeared:
            lapse = 1.0 + Phi_t / self.cstar_sq
            Nbar_fwd = 0.5 * (lapse[self.lat.t_shell_of[self.lat.t_fwd_row]]
                            + lapse[self.lat.t_shell_of[self.lat.t_fwd_col]])
            hop_fwd = self.t0 * torch.abs(Nbar_fwd)
        else:
            hop_fwd = self.t0
        rho.scatter_add_(0, self.lat.t_fwd_shell, 2.0 * hop_fwd * G_fwd)

        # Intra-shell bonds
        V_intra_i = evecs[self.lat.t_intra_row]
        V_intra_j = evecs[self.lat.t_intra_col]
        G_intra = (V_intra_i * V_intra_j) @ f

        if smeared:
            hop_intra = self.t0 * torch.abs(lapse[self.lat.t_intra_shell])
        else:
            hop_intra = self.t0
        rho.scatter_add_(0, self.lat.t_intra_shell, 2.0 * hop_intra * G_intra)

        # On-site V0
        if add_V0:
            for n in range(min(self.n_core, self.N_shell)):
                idx = self.lat.core_sites_per_shell[n]
                V_core = evecs[idx]  # G_n × N
                G_diag_core = (V_core * V_core) @ f  # G_n
                rho[n] += self.V0 * G_diag_core.sum()

        return rho.cpu().numpy()

    def _compute_mi_background_gpu(self):
        """Background MI per forward bond at Phi=0 (uniform chain).

        Used to normalize conductances: kappa_bond = t0^2 * MI/MI_bg.
        """
        H = self._build_H_gpu(
            torch.zeros(self.N_shell, dtype=torch.float64, device=DEVICE),
            smeared=False, add_V0=False)
        evals, evecs = torch.linalg.eigh(H)
        del H
        be = torch.clamp(self.beta0 * evals, -500, 500)
        f = 1.0 / (torch.exp(be) + 1.0)

        V_i = evecs[self.lat.t_fwd_row]
        V_j = evecs[self.lat.t_fwd_col]
        a = (V_i * V_i) @ f
        d = (V_j * V_j) @ f
        b = (V_i * V_j) @ f

        tr = a + d
        det = a * d - b * b
        disc = torch.clamp(tr * tr - 4.0 * det, min=0.0)
        lam1 = 0.5 * (tr + torch.sqrt(disc))
        lam2 = 0.5 * (tr - torch.sqrt(disc))
        mi = (binary_entropy_torch(a) + binary_entropy_torch(d)
              - binary_entropy_torch(lam1) - binary_entropy_torch(lam2))
        del evecs
        return torch.clamp(mi, min=1e-30)

    def _conductances_and_energy(self, Phi_shell):
        """Compute both conductances and energy profile in one pass (shares eigh for smeared H).
        Returns (kappa, rho_sigma) as numpy arrays.
        """
        Phi_t = torch.tensor(Phi_shell, dtype=torch.float64, device=DEVICE)
        lapse = 1.0 + Phi_t / self.cstar_sq

        # Smeared Hamiltonian — shared between conductances and energy
        H_smeared = self._build_H_gpu(Phi_t, smeared=True, add_V0=False)
        evals, evecs = self._gpu_eigh(H_smeared)
        f = self._fermi(evals)

        # ── Conductances ──
        V_fwd_i = evecs[self.lat.t_fwd_row]
        V_fwd_j = evecs[self.lat.t_fwd_col]

        a = (V_fwd_i * V_fwd_i) @ f  # G[i,i]
        d = (V_fwd_j * V_fwd_j) @ f  # G[j,j]
        b = (V_fwd_i * V_fwd_j) @ f  # G[i,j]

        tr = a + d
        det = a * d - b * b
        disc = torch.clamp(tr * tr - 4.0 * det, min=0.0)
        lam1 = 0.5 * (tr + torch.sqrt(disc))
        lam2 = 0.5 * (tr - torch.sqrt(disc))

        mi = (binary_entropy_torch(a) + binary_entropy_torch(d)
              - binary_entropy_torch(lam1) - binary_entropy_torch(lam2))

        mi_ratio = mi / self.mi_bg_bonds_t
        kappa = torch.zeros(self.N_shell - 1, dtype=torch.float64, device=DEVICE)
        kappa.scatter_add_(0, self.lat.t_fwd_shell, self.t0**2 * mi_ratio)

        # ── Energy profile (smeared, no V0) ──
        rho = torch.zeros(self.N_shell, dtype=torch.float64, device=DEVICE)

        # Forward bonds
        G_fwd = b  # already computed
        Nbar_fwd = 0.5 * (lapse[self.lat.t_shell_of[self.lat.t_fwd_row]]
                        + lapse[self.lat.t_shell_of[self.lat.t_fwd_col]])
        hop_fwd = self.t0 * torch.abs(Nbar_fwd)
        rho.scatter_add_(0, self.lat.t_fwd_shell, 2.0 * hop_fwd * G_fwd)

        # Intra-shell bonds
        V_intra_i = evecs[self.lat.t_intra_row]
        V_intra_j = evecs[self.lat.t_intra_col]
        G_intra = (V_intra_i * V_intra_j) @ f
        hop_intra = self.t0 * torch.abs(lapse[self.lat.t_intra_shell])
        rho.scatter_add_(0, self.lat.t_intra_shell, 2.0 * hop_intra * G_intra)

        return kappa.cpu().numpy(), rho.cpu().numpy()

    def _conductances_only(self, Phi_shell):
        """Conductances only (for proxy residual)."""
        Phi_t = torch.tensor(Phi_shell, dtype=torch.float64, device=DEVICE)

        H = self._build_H_gpu(Phi_t, smeared=True)
        evals, evecs = self._gpu_eigh(H)
        f = self._fermi(evals)

        V_fwd_i = evecs[self.lat.t_fwd_row]
        V_fwd_j = evecs[self.lat.t_fwd_col]

        a = (V_fwd_i * V_fwd_i) @ f
        d = (V_fwd_j * V_fwd_j) @ f
        b = (V_fwd_i * V_fwd_j) @ f

        tr = a + d
        det = a * d - b * b
        disc = torch.clamp(tr * tr - 4.0 * det, min=0.0)
        lam1 = 0.5 * (tr + torch.sqrt(disc))
        lam2 = 0.5 * (tr - torch.sqrt(disc))

        mi = (binary_entropy_torch(a) + binary_entropy_torch(d)
              - binary_entropy_torch(lam1) - binary_entropy_torch(lam2))

        mi_ratio = mi / self.mi_bg_bonds_t
        kappa = torch.zeros(self.N_shell - 1, dtype=torch.float64, device=DEVICE)
        kappa.scatter_add_(0, self.lat.t_fwd_shell, self.t0**2 * mi_ratio)
        return kappa.cpu().numpy()

    def residual(self, Phi_shell):
        """Full self-consistent residual.

        Both rho_sigma and rho_tgt are evaluated in the same gravitational
        field (lapse-smeared hopping), so their difference is localized to
        the core.
        """
        self.n_call += 1
        kappa, rho_sigma = self._conductances_and_energy(Phi_shell)
        rho_tgt_sm = self._energy_profile_np(Phi_shell, smeared=True, add_V0=True)

        N = self.N_shell
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi_shell[:N-1] - Phi_shell[1:N])
        lhs[1:N]  += kappa * (Phi_shell[1:N]  - Phi_shell[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * (rho_sigma - rho_tgt_sm)
        F[N-1] = Phi_shell[N-1]
        return F

    def residual_proxy(self, Phi_shell):
        """Proxy: MI conductances, frozen source (1 eigh call)."""
        self.n_call += 1
        kappa = self._conductances_only(Phi_shell)

        N = self.N_shell
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi_shell[:N-1] - Phi_shell[1:N])
        lhs[1:N]  += kappa * (Phi_shell[1:N]  - Phi_shell[:N-1])

        src = self.rho_bg - self.rho_tgt
        F = lhs - (self.beta0 / self.cstar_sq) * src
        F[N-1] = Phi_shell[N-1]
        return F


def solve(model, mode='full', seed=None, tol=1e-7):
    N = model.N_shell
    lb = np.full(N, -model.cstar_sq * 5.0)
    ub = np.full(N, model.cstar_sq * 0.5)
    lb[-1] = -1e-8; ub[-1] = 1e-8

    x0 = np.clip(seed if seed is not None else np.zeros(N), lb + 1e-10, ub - 1e-10)
    func = model.residual if mode == 'full' else model.residual_proxy
    model.n_call = 0

    result = least_squares(func, x0, method='trf', bounds=(lb, ub),
                          ftol=1e-12, xtol=1e-12, max_nfev=300)
    F = func(result.x)
    res = np.max(np.abs(F[:-1]))
    return result.x, res < tol, res, model.n_call


def print_result(label, Phi, res, nfev, elapsed, model):
    cstar_sq = model.cstar_sq
    lapse = 1.0 + Phi / cstar_sq
    N = len(lapse)
    # Extract rs
    r = np.arange(1, N+1, dtype=float)
    GM = -Phi * r
    n_core = model.n_core
    i_lo, i_hi = min(n_core+1, N-1), min(N-2, N//2)
    if i_hi > i_lo:
        i_peak = i_lo + np.argmax(GM[i_lo:i_hi])
        rs = 2.0 * GM[i_peak] / cstar_sq
    else:
        rs = 0.0
    stag = np.max(np.abs(lapse[1:min(7,N):2] - lapse[2:min(7,N):2])) if N > 3 else 0
    print(f"  {label}: |F|={res:.2e}, nfev={nfev}, {elapsed:.1f}s, "
          f"min_N={lapse.min():.6f}@{np.argmin(lapse)}, rs={rs:.4f}, stag={stag:.4e}", flush=True)
    n_show = min(16, N)
    print(f"    lapse[0:{n_show}]: {' '.join(f'{lapse[i]:.6f}' for i in range(n_show))}", flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rmax', type=int, default=10)
    parser.add_argument('--ncore', type=int, default=3)
    args = parser.parse_args()

    R_MAX = args.rmax
    t0 = 1.0
    cstar_sq = 0.5
    n_core = args.ncore

    print("=" * 90, flush=True)
    print(f"FULL 3D CUBIC LATTICE SOLVER — GPU (R_max={R_MAX}, n_core={n_core})", flush=True)
    print("=" * 90, flush=True)

    lat = CubicLattice3D(R_MAX)

    results = {}
    for bt0 in [0.10, 0.50, 1.00, 2.11]:
        beta0 = bt0 / t0
        V0 = 0.003 / beta0

        print(f"\n{'─'*90}", flush=True)
        print(f"bt0 = {bt0}, V0 = {V0:.6f}, β₀V₀ = 0.003, n_core = {n_core}", flush=True)
        print(f"{'─'*90}", flush=True)

        model = ThreeDModel(lat, t0=t0, V0=V0, n_core=n_core,
                            beta0=beta0, cstar_sq=cstar_sq)

        # Proxy
        t_s = time.time()
        Phi_p, _, res_p, nfev_p = solve(model, mode='proxy')
        print_result("PROXY", Phi_p, res_p, nfev_p, time.time()-t_s, model)

        # Full from zero
        t_s = time.time()
        Phi_f, _, res_f, nfev_f = solve(model, mode='full')
        print_result("FULL/0", Phi_f, res_f, nfev_f, time.time()-t_s, model)

        # Full from proxy seed
        t_s = time.time()
        Phi_fp, _, res_fp, nfev_fp = solve(model, mode='full', seed=Phi_p)
        print_result("FULL/p", Phi_fp, res_fp, nfev_fp, time.time()-t_s, model)

        results[bt0] = {
            'proxy': Phi_p, 'full0': Phi_f, 'fullp': Phi_fp,
            'res_p': res_p, 'res_f': res_f, 'res_fp': res_fp,
        }

    # Save results
    outfile = f'full_numerics/3d_results_R{R_MAX}.npz'
    np.savez(outfile, **{f'{k}_{sk}': sv for k, v in results.items() for sk, sv in v.items()})
    print(f"\nResults saved to {outfile}", flush=True)
    print("\nDone.", flush=True)
