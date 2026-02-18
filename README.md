# Entanglement Stars

**Emergent Schwarzschild geometry from entanglement closure on a quantum lattice**

Jonathan C. McKinney

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18686080.svg)](https://doi.org/10.5281/zenodo.18686080)

## Abstract

Can spacetime geometry and black hole physics emerge from quantum information alone, with no assumed metric, connection, gravitational action, or holographic duality?
We treat mutual information (defining graph conductances) and modular Hamiltonians (generating state-dependent deformations) as primary observables.
The conductances define an emergent spatial geometry via a Dirichlet form; in a spherically symmetric shell reduction, the modular generators reproduce the radial sector of the hypersurface-deformation algebra with structure function *h*<sup>rr</sup>, up to a remainder bounded by *&xi;*<sup>2</sup>/*&ell;*<sup>2</sup> (correlation length over curvature scale).
We implement the construction with free fermions, so every quantity is computable from the single-particle correlation matrix.

A nonlinear closure equation, with conductances depending on its solution, yields the Schwarzschild radial factor: in the analytic high-temperature closure where the conductance scales as *&kappa;* &prop; *N&#x0304;*<sup>2</sup> (*N&#x0304;* the bond-averaged lapse), the exterior Euler&ndash;Lagrange equation reduces in *w* = *N*<sup>2</sup> (*N* the lapse) to Laplace's equation.
A two-state closure (comparing reconstructed and target states in the same gravitational field) removes the screening term from linearization around a polarizable medium, recovering a Schwarzschild exterior.

The resulting *entanglement star* is a horizonless compact object: conductance feedback suppresses further redshift in the deep interior, capping the classical singularity while preserving Schwarzschild behavior outside.
The absence of an event horizon dissolves the information paradox: information escapes on sub-evaporation timescales, and an explicit unitary dynamics produces a Page curve and avoids the firewall obstruction.
Surface gravity, Hawking temperature, a mutual-information area law, and the first law follow from modular and Kubo&ndash;Martin&ndash;Schwinger compatibility in the continuum limit.

**Keywords:** emergent gravity, quantum entanglement, black hole thermodynamics

## Paper figures

All seven publication figures are in [`fermion/numerical/figures/`](fermion/numerical/figures/).
The data pipeline uses BC-corrected (asymptotically Newtonian) GM extraction
throughout.

| Fig | PDF | Description | Script | Data |
|-----|-----|-------------|--------|------|
| 1 | [twostate_potential.pdf](fermion/numerical/figures/twostate_potential.pdf) | Gravitational potential profiles | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 2 | [cometric_conductance_pn.pdf](fermion/numerical/figures/cometric_conductance_pn.pdf) | Co-metric, conductance, PN residual | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 3 | [twostate_embedding.pdf](fermion/numerical/figures/twostate_embedding.pdf) | Embedding diagram | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 4 | [twostate_thermodynamics.pdf](fermion/numerical/figures/twostate_thermodynamics.pdf) | Surface gravity, Hawking temperature | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 5 | [tqi_profile.pdf](fermion/numerical/figures/tqi_profile.pdf) | QI temperature profile | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 6 | [twostate_temperature_sweep.pdf](fermion/numerical/figures/twostate_temperature_sweep.pdf) | Temperature continuation | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| 7 | [evaporation_gaussian_dynamics.pdf](fermion/numerical/figures/evaporation_gaussian_dynamics.pdf) | Unitary Page curve | [evaporation_gaussian_dynamics.py](full_numerics/evaporation_gaussian_dynamics.py) | generated at runtime |

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

### Sec. 10: Emergent Schwarzschild geometry ([`fermion/numerical/`](fermion/numerical/))

Analytic and self-consistent solvers exploring the weak-field (Newtonian),
post-Newtonian, and vacuum boundary condition regimes discussed in
Secs. 10a-10f.

| File | Purpose | Output figure(s) |
|------|---------|-------------------|
| [solve_selfconsistent.py](fermion/numerical/solve_selfconsistent.py) | Single-state self-consistent solver (vacuum BC) | [potential_profile](fermion/numerical/figures/potential_profile.pdf), [cometric_comparison](fermion/numerical/figures/cometric_comparison.pdf), [lapse_profile](fermion/numerical/figures/lapse_profile.pdf), [conductance_profile](fermion/numerical/figures/conductance_profile.pdf), [tqi_profile](fermion/numerical/figures/tqi_profile.pdf), [pn_comparison](fermion/numerical/figures/pn_comparison.pdf), [cometric_conductance_pn](fermion/numerical/figures/cometric_conductance_pn.pdf), [max_mass](fermion/numerical/figures/max_mass.pdf) |
| [solve_exact_vacuum.py](fermion/numerical/solve_exact_vacuum.py) | Exact solver with vacuum BC | [vacuum_bc_potential_profiles](fermion/numerical/figures/vacuum_bc_potential_profiles.pdf), [vacuum_bc_rPhi_test](fermion/numerical/figures/vacuum_bc_rPhi_test.pdf), [vacuum_bc_summary](fermion/numerical/figures/vacuum_bc_summary.pdf), [vacuum_bc_bt211_focused](fermion/numerical/figures/vacuum_bc_bt211_focused.pdf), [vacuum_bc_1overr_persistence](fermion/numerical/figures/vacuum_bc_1overr_persistence.pdf), [vacuum_bc_weak_field_check](fermion/numerical/figures/vacuum_bc_weak_field_check.pdf) |
| [investigate_pn_vs_temperature.py](fermion/numerical/investigate_pn_vs_temperature.py) | Post-Newtonian coefficients vs temperature | [pn_vs_temperature](fermion/numerical/pn_vs_temperature.pdf), [mi_vs_lapse](fermion/numerical/mi_vs_lapse.pdf), [cometric_comparison_exact_vs_ht](fermion/numerical/cometric_comparison_exact_vs_ht.pdf) |
| [investigate_log_concavity.py](fermion/numerical/investigate_log_concavity.py) | Log-concavity and Schwarzschild emergence | [log_concavity_landscape](fermion/numerical/log_concavity_landscape.pdf), [alpha2_vs_parameters](fermion/numerical/alpha2_vs_parameters.pdf), [F_comparison_schwarzschild](fermion/numerical/F_comparison_schwarzschild.pdf), [schwarzschild_impossibility](fermion/numerical/schwarzschild_impossibility.pdf) |
| [make_vacuum_screening_figures.py](fermion/numerical/scripts/make_vacuum_screening_figures.py) | Vacuum vs Yukawa screening comparison | [vacuum_screening_profile](fermion/numerical/figures/vacuum_screening_profile.pdf) |

### Sec. 10e: Black hole thermodynamics ([`full_numerics/`](full_numerics/))

| File | Purpose | Output figure(s) |
|------|---------|-------------------|
| [test_bw_property.py](full_numerics/test_bw_property.py) | Bisognano-Wichmann verification | [bw_test](fermion/numerical/figures/bw_test.pdf) |

### Sec. 11c: Full self-consistent solution ([`full_numerics/`](full_numerics/))

Main computational pipeline for Figs. 1-6. The Newton solver
[compute_exact_data.py](full_numerics/compute_exact_data.py) produces
[exact_solutions.npz](fermion/numerical/data/exact_solutions.npz), from which
[plot_exact_figures.py](full_numerics/plot_exact_figures.py) generates all six figures.

| File | Purpose | Output |
|------|---------|--------|
| [compute_exact_data.py](full_numerics/compute_exact_data.py) | Main Newton solver (vacuum BC) | [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) |
| [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | Plot Figs. 1-6 from pre-computed data | 15 PDFs including Figs. 1-6 |

#### Additional Sec. 11c analysis

| File | Purpose | Output figure(s) |
|------|---------|-------------------|
| [study_nonadjacent_mi.py](full_numerics/study_nonadjacent_mi.py) | Non-adjacent MI structure | [nonadjacent_mi](fermion/numerical/figures/nonadjacent_mi.pdf) |
| [study_observable_jacobian.py](full_numerics/study_observable_jacobian.py) | Observable Jacobian sensitivity | [observable_jacobian](fermion/numerical/figures/observable_jacobian.pdf) |
| [study_wavelength_origin.py](full_numerics/study_wavelength_origin.py) | Wavelength origin analysis | [wavelength_origin](fermion/numerical/figures/wavelength_origin.pdf) |
| [study_wavelength_stability.py](full_numerics/study_wavelength_stability.py) | Wavelength stability | [wavelength_stability](fermion/numerical/figures/wavelength_stability.pdf) |

### Sec. 11c: Three-dimensional cubic-lattice verification ([`full_numerics/`](full_numerics/))

Confirms that the spherical shell reduction is self-consistent by solving
the closure equation on a full 3D cubic lattice (Sec. ex-3d).

| File | Purpose | Output |
|------|---------|--------|
| [solve_3d_persite.py](full_numerics/solve_3d_persite.py) | Per-site 3D solver (R=10, 4169 sites) | [persite_3d_R10_bt0.10](fermion/numerical/figures/persite_3d_R10_bt0.10.pdf) |
| [solve_3d.py](full_numerics/solve_3d.py) | Shell-averaged 3D solver | — |
| [solve_3d_vacuum.py](full_numerics/solve_3d_vacuum.py) | 3D vacuum boundary conditions (CPU) | — |
| [solve_3d_vacuum_gpu.py](full_numerics/solve_3d_vacuum_gpu.py) | 3D vacuum boundary conditions (GPU) | — |
| [solve_3d_lm.py](full_numerics/solve_3d_lm.py) | Levenberg-Marquardt 3D solver | — |
| [solve_3d_bt010.py](full_numerics/solve_3d_bt010.py) | 3D at specific temperature | — |
| [solve_3d_mc.py](full_numerics/solve_3d_mc.py) | Monte Carlo 3D solver | — |
| [screening_length_3d.py](full_numerics/screening_length_3d.py) | 3D screening length analysis | — |

### Sec. 11c: Classical Ising and quantum XXZ universality tests

Non-Gaussian universality tests (Sec. ex-ising). The 1D Ising, XXZ, and
2D Ising Monte Carlo solvers confirm that Schwarzschild emergence is not
specific to free fermions.

**Solvers** ([`full_numerics/`](full_numerics/)):

| File | Purpose |
|------|---------|
| [solve_ising_twostate.py](full_numerics/solve_ising_twostate.py) | 1D Ising two-state closure |
| [solve_ising_v2.py](full_numerics/solve_ising_v2.py) | Ising solver (variant) |
| [solve_xxz_twostate.py](full_numerics/solve_xxz_twostate.py) | XXZ spin chain two-state closure |

**2D Ising Monte Carlo** ([`fermion/numerical/`](fermion/numerical/)):

| File | Purpose | Output |
|------|---------|--------|
| [ising_hda.py](fermion/numerical/ising_hda.py) | 2D Ising MC: conductances + co-metric + HDA | [ising_hda_results.pkl](fermion/numerical/data/ising_hda_results.pkl) |
| [ising_hda_plot.py](fermion/numerical/ising_hda_plot.py) | Plot Ising HDA results | [ising_hda_conductances](fermion/numerical/figures/ising_hda_conductances.pdf), [ising_hda_cometric](fermion/numerical/figures/ising_hda_cometric.pdf), [ising_hda_remainder](fermion/numerical/figures/ising_hda_remainder.pdf) |
| [ising_crossover.py](fermion/numerical/ising_crossover.py) | 2D Ising EFT crossover study | [ising_crossover_results.pkl](fermion/numerical/data/ising_crossover_results.pkl) |
| [ising_crossover_plot.py](fermion/numerical/ising_crossover_plot.py) | Plot Ising crossover results | [ising_crossover_xi](fermion/numerical/figures/ising_crossover_xi.pdf), [ising_crossover_conductance](fermion/numerical/figures/ising_crossover_conductance.pdf), [ising_crossover_remainder](fermion/numerical/figures/ising_crossover_remainder.pdf), [ising_crossover_isotropy](fermion/numerical/figures/ising_crossover_isotropy.pdf) |
| [investigate_ising_pn.py](fermion/numerical/investigate_ising_pn.py) | Ising post-Newtonian coefficients | [ising_pn_comparison](fermion/numerical/ising_pn_comparison.pdf), [ising_pn_blocksize](fermion/numerical/ising_pn_blocksize.pdf), [ising_pn_F_blocksize](fermion/numerical/ising_pn_F_blocksize.pdf) |

### Sec. 11d: Quasi-static evaporation and Page curve ([`full_numerics/`](full_numerics/))

Unitary Gaussian dynamics, greybody factors, and thermal transport
(Secs. ev-massloss, ev-page, ff-evaporation-dynamics).

| File | Purpose | Output |
|------|---------|--------|
| [evaporation_gaussian_dynamics.py](full_numerics/evaporation_gaussian_dynamics.py) | Unitary Page-curve circuit (Fig. 7) | [evaporation_gaussian_dynamics](fermion/numerical/figures/evaporation_gaussian_dynamics.pdf) |
| [evaporation_nonthermality.py](full_numerics/evaporation_nonthermality.py) | Evaporation non-thermality diagnostics | [evaporation_nonthermality](fermion/numerical/figures/evaporation_nonthermality.pdf) |
| [greybody_shell_chain.py](full_numerics/greybody_shell_chain.py) | Greybody factors from shell-chain transfer matrix | [greybody_shell_chain](fermion/numerical/figures/greybody_shell_chain.pdf) |
| [weak_link_thermal_conductance.py](full_numerics/weak_link_thermal_conductance.py) | Cap-limited thermal conductance | [weak_link_conductance](fermion/numerical/figures/weak_link_conductance.pdf) |

### Pre-computed data

| File | Used by | Paper section |
|------|---------|---------------|
| [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz) | [plot_exact_figures.py](full_numerics/plot_exact_figures.py) | Sec. 11c (Figs. 1-6) |
| [ising_hda_results.pkl](fermion/numerical/data/ising_hda_results.pkl) | [ising_hda_plot.py](fermion/numerical/ising_hda_plot.py) | Sec. 11c (Ising universality) |
| [ising_crossover_results.pkl](fermion/numerical/data/ising_crossover_results.pkl) | [ising_crossover_plot.py](fermion/numerical/ising_crossover_plot.py) | Sec. 11c (Ising crossover) |
| [greybody_shell_chain.npz](fermion/numerical/data/greybody_shell_chain.npz) | [greybody_shell_chain.py](full_numerics/greybody_shell_chain.py) | Sec. 11d (greybody) |
| [evaporation_nonthermality.npz](fermion/numerical/data/evaporation_nonthermality.npz) | [evaporation_nonthermality.py](full_numerics/evaporation_nonthermality.py) | Sec. 11d (evaporation) |
| [weak_link_conductance.npz](fermion/numerical/data/weak_link_conductance.npz) | [weak_link_thermal_conductance.py](full_numerics/weak_link_thermal_conductance.py) | Sec. 11d (thermal conductance) |
| [3d_results_R10.npz](full_numerics/3d_results_R10.npz) | [solve_3d_persite.py](full_numerics/solve_3d_persite.py) | Sec. 11c (3D verification) |
| [3d_lm_results_R10.npz](full_numerics/3d_lm_results_R10.npz) | [solve_3d_lm.py](full_numerics/solve_3d_lm.py) | Sec. 11c (3D verification) |

### Generated figures

51 PDF figures in [`fermion/numerical/figures/`](fermion/numerical/figures/) (7 paper + 44 supplementary).
10 additional analysis PDFs in [`fermion/numerical/`](fermion/numerical/) (top-level).

## Reproducing the figures

### Requirements

Python 3.9+ with:

```bash
pip install numpy scipy matplotlib
```

Some scripts additionally use `numba` (Ising solvers) or `cupy` (GPU 3D
solver); these are optional.

### Paper figures (from pre-computed data)

Regenerate all 7 publication figures from the included [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz):

```bash
python3 full_numerics/plot_exact_figures.py          # Figs. 1-6
python3 full_numerics/evaporation_gaussian_dynamics.py  # Fig. 7
```

### Recompute paper data from scratch

Rerun the main Newton solver (~7 hours on an i9 16-core CPU):

```bash
python3 full_numerics/compute_exact_data.py
```

This regenerates [exact_solutions.npz](fermion/numerical/data/exact_solutions.npz), after which
`plot_exact_figures.py` reproduces Figs. 1-6.

### Supplementary figures from pre-computed data

```bash
# Ising universality plots (from pre-computed .pkl data)
python3 fermion/numerical/ising_hda_plot.py          # Ising HDA conductances, co-metric, remainder
python3 fermion/numerical/ising_crossover_plot.py    # Ising crossover xi, conductance, remainder, isotropy

# Evaporation and thermal transport (compute + plot)
python3 full_numerics/evaporation_nonthermality.py   # Evaporation non-thermality diagnostics
python3 full_numerics/greybody_shell_chain.py        # Greybody factors
python3 full_numerics/weak_link_thermal_conductance.py  # Cap-limited thermal conductance
```

### Sec. 10 analysis figures

```bash
# Self-consistent solver (vacuum BC) -- 8 figures
python3 fermion/numerical/solve_selfconsistent.py

# Exact solver with vacuum BC -- 6 figures
python3 fermion/numerical/solve_exact_vacuum.py

# Investigation scripts -- 10 figures
python3 fermion/numerical/investigate_pn_vs_temperature.py
python3 fermion/numerical/investigate_log_concavity.py
python3 fermion/numerical/investigate_ising_pn.py
python3 fermion/numerical/scripts/make_vacuum_screening_figures.py
```

### Sec. 11c analysis figures

```bash
python3 full_numerics/test_bw_property.py            # Bisognano-Wichmann verification
python3 full_numerics/study_nonadjacent_mi.py        # Non-adjacent MI structure
python3 full_numerics/study_observable_jacobian.py    # Observable Jacobian sensitivity
python3 full_numerics/study_wavelength_origin.py      # Wavelength origin analysis
python3 full_numerics/study_wavelength_stability.py   # Wavelength stability
python3 -m full_numerics.compare_observables          # Smeared vs fixed observable comparison
```

### Universality tests

```bash
python3 full_numerics/solve_3d_persite.py --R 10    # 3D cubic lattice
python3 full_numerics/solve_3d_vacuum.py             # 3D vacuum BC (CPU)
python3 full_numerics/solve_3d_vacuum_gpu.py         # 3D vacuum BC (GPU, requires cupy)
python3 full_numerics/solve_ising_twostate.py        # 1D Ising
python3 full_numerics/solve_xxz_twostate.py          # XXZ chain
python3 fermion/numerical/ising_hda.py               # 2D Ising MC (data generation)
python3 fermion/numerical/ising_crossover.py         # 2D Ising crossover (data generation)
```

## Citation

```bibtex
@article{McKinney2025EntanglementStars,
  author  = {McKinney, Jonathan C.},
  title   = {Entanglement Stars: Emergent Schwarzschild geometry
             from entanglement closure on a quantum lattice},
  year    = {2026},
  doi     = {10.5281/zenodo.18686080}
}
```

## Data availability

All pre-computed datasets are included in this repository and archived at
[Zenodo](https://zenodo.org/records/18686080)
(DOI: [10.5281/zenodo.18686080](https://doi.org/10.5281/zenodo.18686080)).

## License

All code in this repository is released under the MIT License.
The manuscript text and figures are copyright the author.
