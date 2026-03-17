# Entanglement Stars

**Self-consistent Schwarzschild exterior from entanglement closure on quantum lattice**

Jonathan C. McKinney

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18785987.svg)](https://doi.org/10.5281/zenodo.18785987)

## Abstract

Quantum correlations between neighboring sites of a lattice behave like conductances in a resistor network.
We show that when a localized energy defect is placed on the lattice, the conductances define a geometry that can be made mutually consistent with the quantum state.
In spherical symmetry this self-consistency produces an unscreened 1/*r* far-field gravitational potential whose strength sets the mass.
Because the conductances scale as the squared lapse (*N*<sup>2</sup>), the closure equation reduces to Laplace's equation whose solution is exactly the radial co-metric, and the exterior takes the Schwarzschild form including the post-Newtonian expansion.
We refer to the resulting object as an *entanglement star*, supported by conductances that play the role of an effective degeneracy pressure.
The lapse remains strictly positive because conductances are suppressed before the lapse reaches zero, replacing the classical singularity by a conical interior geometry within the model.
Tracking mode-by-mode evaporation in the unitary Gaussian state, the radiation entropy exhibits a Page-like curve.
Within the closure framework we derive expressions corresponding to surface gravity, Hawking temperature, and the first law.
Self-consistent numerical calculations validate key analytic results for free fermions, an interacting XXZ chain, and a 3D cubic lattice.
Replacing the defect with a Tolman-Klein equilibrium gives a self-consistent stellar interior, whose astrophysical limit leads to gravitationally relevant entanglement concentrated at the Schwarzschild radius.

## Paper figures

All nine publication figures are in [`fermion/numerical/figures/`](fermion/numerical/figures/).

| Fig | File | Description | Script | Data |
|-----|------|-------------|--------|------|
| 1 | [twostate_potential.pdf](fermion/numerical/figures/twostate_potential.pdf) | Gravitational potential profiles | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 2 | [cometric_conductance_pn.pdf](fermion/numerical/figures/cometric_conductance_pn.pdf) | Co-metric, conductance, PN residual | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 3 | [twostate_embedding.pdf](fermion/numerical/figures/twostate_embedding.pdf) | Embedding diagram | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 4 | [twostate_thermodynamics.pdf](fermion/numerical/figures/twostate_thermodynamics.pdf) | Surface gravity and Hawking temperature | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 5 | [tqi_profile.pdf](fermion/numerical/figures/tqi_profile.pdf) | QI temperature profile | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 6 | [twostate_temperature_sweep.pdf](fermion/numerical/figures/twostate_temperature_sweep.pdf) | Temperature continuation | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 7 | [tov_tk_chemical.png](fermion/numerical/figures/tov_tk_chemical.png) | TK star chemical-potential scan: profiles, core size, Schwarzschild radius vs &mu; | [tov_tk_chemical_el_fig.py](tov_tk_chemical_el_fig.py) | [tov_tk_chemical_el.npz](tov_tk_chemical_el.npz) |
| 8 | [tov_tk_final_flux.png](fermion/numerical/figures/tov_tk_final_flux.png) | TK star exterior diagnostics: *w*-flux, lapse vs Schwarzschild, residual | [tov_tk_el_flux_fig.py](tov_tk_el_flux_fig.py) | [tov_tk_chemical_el.npz](tov_tk_chemical_el.npz) |
| 9 | [evaporation_gaussian_dynamics.pdf](fermion/numerical/figures/evaporation_gaussian_dynamics.pdf) | Unitary Page curve | [evaporation_gaussian_dynamics.py](full_numerics/evaporation_gaussian_dynamics.py) | generated at runtime |

## Code organization by paper section

### Core library ([`full_numerics/`](full_numerics/))

Shared modules used by all solvers and plotters.

| File | Purpose |
|------|---------|
| [physics.py](full_numerics/physics.py) | Base shell-chain Hamiltonian and MI computation |
| [physics_twostate.py](full_numerics/physics_twostate.py) | High-temperature two-state model |
| [physics_twostate_exact.py](full_numerics/physics_twostate_exact.py) | Exact two-state model (full MI) |
| [solver.py](full_numerics/solver.py) | Newton solver, rs extraction (vacuum BC) |
| [solve_twostate.py](full_numerics/solve_twostate.py) | Analytic high-T two-state solver |
| [solve_twostate_exact.py](full_numerics/solve_twostate_exact.py) | Newton solver with finite-difference Jacobian |
| [compare_observables.py](full_numerics/compare_observables.py) | Smeared vs fixed observable comparison |
| [\_\_init\_\_.py](full_numerics/__init__.py) | Package init |

### Sec. 11c: Full self-consistent solution ([`full_numerics/`](full_numerics/))

Main computational pipeline for Figs. 1–6. The GPU Newton solver
[compute_exact_data_gpu.py](compute_exact_data_gpu.py) produces
[exact_solutions.npz](fermion/numerical/data/exact_solutions.npz), from which
[plot_exact_figures.py](full_numerics/plot_exact_figures.py) generates Figs. 1–6.

| File | Purpose | Output |
|------|---------|--------|
| [compute_exact_data_gpu.py](compute_exact_data_gpu.py) | GPU Newton solver (vacuum BC) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | Plot Figs. 1–6 from pre-computed data | Figs. 1–6 |
| [pseudoarc_continuation.py](pseudoarc_continuation.py) | Pseudo-arclength continuation for fold identification | [pseudoarc_fold.npz](pseudoarc_fold.npz) |
| [compute_temp_sweep.py](full_numerics/compute_temp_sweep.py) | Temperature-continuation sweep | contributes to [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |

#### Additional Sec. 11c analysis

| File | Purpose |
|------|---------|
| [test_bw_property.py](full_numerics/test_bw_property.py) | Bisognano-Wichmann entanglement temperature verification |
| [study_nonadjacent_mi.py](full_numerics/study_nonadjacent_mi.py) | Non-adjacent MI truncation error |
| [study_observable_jacobian.py](full_numerics/study_observable_jacobian.py) | Observable Jacobian sensitivity |

### Sec. 11c: Three-dimensional cubic-lattice verification ([`full_numerics/`](full_numerics/))

| File | Purpose | Data |
|------|---------|------|
| [solve_3d_persite.py](full_numerics/solve_3d_persite.py) | Per-site 3D Picard solver (R=10, 4169 sites) | [3d_bkm_el_R10.npz](full_numerics/3d_bkm_el_R10.npz) |
| [solve_3d_bkm_el.py](full_numerics/solve_3d_bkm_el.py) | 3D BKM+EL Newton solver | [3d_bkm_el_R10.npz](full_numerics/3d_bkm_el_R10.npz) |

### Sec. 11c: XXZ universality test ([`full_numerics/`](full_numerics/))

| File | Purpose |
|------|---------|
| [solve_xxz_twostate.py](full_numerics/solve_xxz_twostate.py) | XXZ spin chain Picard solver |
| [solve_xxz_bkm_el.py](full_numerics/solve_xxz_bkm_el.py) | XXZ BKM+EL Newton solver |

### Sec. 11e: Tolman-Klein stellar interior (root level)

Self-consistent TK star solutions. All scripts and data files live at the repository root.

| File | Purpose | Data |
|------|---------|------|
| [tov_tk_chemical_el.py](tov_tk_chemical_el.py) | TK chemical-potential / EL solver | [tov_tk_chemical_el.npz](tov_tk_chemical_el.npz) |
| [tov_tk_chemical_el_fig.py](tov_tk_chemical_el_fig.py) | Plot Fig. 7 | [tov_tk_chemical_el.npz](tov_tk_chemical_el.npz) |
| [tov_tk_el_flux_fig.py](tov_tk_el_flux_fig.py) | Plot Fig. 8 | [tov_tk_chemical_el.npz](tov_tk_chemical_el.npz) |
| [tov_tk_bt5_eigenvalue.py](tov_tk_bt5_eigenvalue.py) | Jacobian eigenvalue analysis at &beta;<sub>0</sub>t<sub>0</sub>=5 | [tov_tk_bt5_eigenvalue.npz](tov_tk_bt5_eigenvalue.npz) |
| [tov_tk_bt10_mu_scan.py](tov_tk_bt10_mu_scan.py) | &mu;-scan with eigenvalue analysis at &beta;<sub>0</sub>t<sub>0</sub>=10 | [tov_tk_bt10_mu_scan.npz](tov_tk_bt10_mu_scan.npz), [tov_tk_bt10_eigenvalue.npz](tov_tk_bt10_eigenvalue.npz) |
| [tov_tk_beta_chain.py](tov_tk_beta_chain.py) | Chained continuation at &beta;<sub>0</sub>t<sub>0</sub>=12,15 | [tov_tk_beta_chain.npz](tov_tk_beta_chain.npz) |
| [tov_convergence_el.py](tov_convergence_el.py) | Full EL convergence and robustness study | — |

### Sec. 11d: Quasi-static evaporation and Page curve ([`full_numerics/`](full_numerics/))

| File | Purpose | Output |
|------|---------|--------|
| [evaporation_gaussian_dynamics.py](full_numerics/evaporation_gaussian_dynamics.py) | Unitary Page-curve circuit (Fig. 9) | [evaporation_gaussian_dynamics.pdf](fermion/numerical/figures/evaporation_gaussian_dynamics.pdf) |
| [evaporation_selfconsistent.py](full_numerics/evaporation_selfconsistent.py) | Quasi-static Page curve from self-consistent closure (V0 scan + area-law) | [evaporation_selfconsistent.npz](fermion/numerical/data/evaporation_selfconsistent.npz) |
| [greybody_shell_chain.py](full_numerics/greybody_shell_chain.py) | Greybody factors from shell-chain transfer matrix | [greybody_shell_chain.npz](fermion/numerical/data/greybody_shell_chain.npz) |
| [weak_link_thermal_conductance.py](full_numerics/weak_link_thermal_conductance.py) | Cap-limited thermal conductance | [weak_link_conductance.npz](fermion/numerical/data/weak_link_conductance.npz) |

### Pre-computed data

| File | Used by | Paper section |
|------|---------|---------------|
| [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | Sec. 11c (Figs. 1–6) |
| [pseudoarc_fold.npz](pseudoarc_fold.npz) | [pseudoarc_continuation.py](pseudoarc_continuation.py) | Sec. 11c (fold bifurcation) |
| [3d_bkm_el_R10.npz](full_numerics/3d_bkm_el_R10.npz) | [solve_3d_bkm_el.py](full_numerics/solve_3d_bkm_el.py) | Sec. 11c (3D verification) |
| [tov_tk_chemical_el.npz](tov_tk_chemical_el.npz) | [tov_tk_chemical_el_fig.py](tov_tk_chemical_el_fig.py), [tov_tk_el_flux_fig.py](tov_tk_el_flux_fig.py) | Sec. 11e (Figs. 7–8) |
| [tov_tk_bt5_eigenvalue.npz](tov_tk_bt5_eigenvalue.npz) | [tov_tk_bt5_eigenvalue.py](tov_tk_bt5_eigenvalue.py) | Sec. 11e |
| [tov_tk_bt10_mu_scan.npz](tov_tk_bt10_mu_scan.npz) | [tov_tk_bt10_mu_scan.py](tov_tk_bt10_mu_scan.py) | Sec. 11e |
| [tov_tk_bt10_eigenvalue.npz](tov_tk_bt10_eigenvalue.npz) | [tov_tk_bt10_mu_scan.py](tov_tk_bt10_mu_scan.py) | Sec. 11e |
| [tov_tk_beta_chain.npz](tov_tk_beta_chain.npz) | [tov_tk_beta_chain.py](tov_tk_beta_chain.py) | Sec. 11e |
| [evaporation_selfconsistent.npz](fermion/numerical/data/evaporation_selfconsistent.npz) | [evaporation_selfconsistent.py](full_numerics/evaporation_selfconsistent.py) | Sec. 11d (closure Page curve) |
| [greybody_shell_chain.npz](fermion/numerical/data/greybody_shell_chain.npz) | [greybody_shell_chain.py](full_numerics/greybody_shell_chain.py) | Sec. 11d (greybody) |
| [weak_link_conductance.npz](fermion/numerical/data/weak_link_conductance.npz) | [weak_link_thermal_conductance.py](full_numerics/weak_link_thermal_conductance.py) | Sec. 11d (thermal conductance) |

## Reproducing the figures

### Requirements

Python 3.9+ with:

```bash
pip install numpy scipy matplotlib
```

GPU scripts additionally require `torch` or `cupy`.

### Paper figures (from pre-computed data)

```bash
python3 full_numerics/plot_exact_figures.py          # Figs. 1–6
python3 tov_tk_chemical_el_fig.py                    # Fig. 7
python3 tov_tk_el_flux_fig.py                        # Fig. 8
python3 full_numerics/evaporation_gaussian_dynamics.py  # Fig. 9
```

### Recompute paper data from scratch

```bash
python3 compute_exact_data_gpu.py        # regenerates exact_solutions.npz (~GPU hours)
python3 tov_tk_chemical_el.py            # regenerates tov_tk_chemical_el.npz
python3 pseudoarc_continuation.py        # regenerates pseudoarc_fold.npz
```

### Supporting computations

```bash
python3 full_numerics/solve_3d_persite.py --R 10    # 3D cubic lattice (Picard)
python3 full_numerics/solve_3d_bkm_el.py            # 3D BKM+EL Newton
python3 full_numerics/solve_xxz_twostate.py         # XXZ chain (Picard)
python3 full_numerics/solve_xxz_bkm_el.py           # XXZ chain (Newton)
python3 full_numerics/greybody_shell_chain.py        # Greybody factors
python3 full_numerics/weak_link_thermal_conductance.py  # Thermal conductance
python3 full_numerics/test_bw_property.py            # Bisognano-Wichmann check
python3 full_numerics/study_nonadjacent_mi.py        # Non-adjacent MI
python3 -m full_numerics.study_observable_jacobian    # Observable Jacobian
```

## Citation

```bibtex
@article{McKinney2026EntanglementStars,
  author  = {McKinney, Jonathan C.},
  title   = {Entanglement Stars: Self-consistent Schwarzschild exterior
             from entanglement closure on quantum lattice},
  year    = {2026},
  doi     = {10.5281/zenodo.18785987}
}
```

## Data availability

All pre-computed datasets are included in this repository and archived at
[Zenodo](https://zenodo.org/records/18785987)
(DOI: [10.5281/zenodo.18785987](https://doi.org/10.5281/zenodo.18785987)).

## License

All code in this repository is released under the MIT License.
The manuscript text and figures are copyright the author.
