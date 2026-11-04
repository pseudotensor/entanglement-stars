# Entanglement Stars: Black-Hole Geometry from Quantum Correlations on a Lattice

Jonathan C. McKinney

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23963163.svg)](https://doi.org/10.5281/zenodo.23963163)

This repository holds the numerical codes, the computed data, and the figures of the paper, together with supporting detailed proofs and derivations that the paper cites but does not require. The paper cites it as Ref. [McKinneyCode2026].

## Abstract

We study lattice fermions on fixed cubic sites, a Dirac fermion field, assuming no light cone, vacuum, or metric. One principle is added: energy is a charge. A clock rate weights the fermion's energy and a flow velocity its momentum, and the energy in any set of sites is a flux of these fields through its outgoing bonds, as Gauss's law reads charge from flux. Newton's constant is the only measured input. For slowly varying fields the principle and the fermion's symmetries leave exactly Einstein's constraint equations, and the light cone makes clock rate, flow velocity, and spacing a metric. Outside a spherical positive-energy source the constraints give exactly Schwarzschild's vacuum, a trapping surface at 2GE, and Hawking's temperature. Its first law gives the Bekenstein-Hawking entropy, supplied in full by entanglement across the surface at a spacing near two Planck lengths. A computed collapse of a cloud above the degenerate maximum mass forms an *entanglement star*, its trapping surface inside the matter. Operationally the surface is no horizon: short-wavelength lattice modes outrun long-wavelength light and cross it outward. A second, massless Dirac field, the radiation, develops outgoing flow. With supply, balance, area-sized storage, and release assumed, its entropy follows a Page curve peaking near half the evaporation time. Without smooth geometry, we define a dynamical singularity as failure of unique continuation of state, clock, and flow. For our collapse simulation, every stored slice keeps a positive clock, and traced rays cross the regular centre.

## Layout

| Folder | Contents |
|--------|----------|
| `figures/` | The four paper figures, the scripts that draw them, and the arrays they plot. |
| `code/` | The solvers, readouts and checks behind the paper's numbers. |
| `proofs/` | Supporting derivations and notes the paper cites; not required for the paper's results. |
| `data/` | Result files that hold the numbers printed in the paper. |
| `MANIFEST.tsv` | Every file with its size, the md5 of the original file, whether its text was neutralized, and where it came from. |

`code/`, `proofs/` and `data/` share four subfolders, one per independent calculation team. Matching names across the three folders belong together.

| Subfolder | Role |
|-----------|------|
| `solve_a` | Primary route: the collapse engine, the light-field (Page) runs, ladders and readouts. |
| `busi` | Readouts of the stored collapse histories, figure data, the light-run drivers and cold re-checks. |
| `hreduce` | Derivation checks for the model and the exterior, vacuum entanglement coefficients, and second-route records. |
| `solve_b` | Second, independently written route for the collapse, cut, packet and Page numbers. |
| `code/orient` | Small library for the planar and spherical entanglement coefficient used by `solve_a`. |

## Paper figures

| Figure | File | Script | Data |
|--------|------|--------|------|
| Fig. 1, structure of the entanglement star (Sec. VI) | `figures/star_structure.tex` | none, plain TikZ input | values from the reference collapse slice at T/r_s = 15, stated in Sec. VI |
| Fig. 2, collapse of 640 fermions (Sec. VI B) | `figures/item5_collapse_v3.1.pdf` | `code/busi/figures/figure_item5_collapse_v3_1.py` | `figures/fig_item5_collapse_v3_data.npz`, built by `code/busi/figures/figures_data_v1.py` from the reference collapse and `data/solve_a/ladders/ladders_readout_v1_3.*` (build log `figures/figures_data_v1.log`) |
| Fig. 3, test light on the doubled-domain collapse (Sec. VII C) | `figures/page_v5.pdf` | `figures/page_v5_fig.py` | `figures/page_v5_data.npz`, byte-identical to the `fig_page_v5_data.npz` named in the paper source, from `figures_data_v1.py` with `data/solve_a/page_v4.1/PAGE_fig_v4.1_data.npz`, `data/busi/w1/v6s_B2hold/` and the light runs of `code/solve_a/page/page_history_v6o.py` and `page_history_v6s.py` |
| Fig. 4, Page curve from the emission law (Sec. VIII D) | `figures/page_curve_amps.pdf` | `figures/page_curve_amps_R1.py`, then `figures/page_curve_amps_postedit.py` | `figures/page_curve_amps_R1.csv`, with the independent second route in `figures/page_curve_amps_R2.py`, `page_curve_amps_R2_data.py`, `page_curve_amps_R2.csv`, `page_curve_amps_R2_report.json` |

Figure 4 is `page_curve_amps_R1.pdf` with two text edits: the panel (a) title reads "(a) Page curve", and the peak label of the dashed f = 0.887 curve is removed. `page_curve_amps_postedit.py` makes both edits and reproduces the installed figure.

## Code by paper section

| Code | Purpose | Paper |
|------|---------|-------|
| `code/hreduce/declared_*_check.py`, `declared_exterior_*.py`, `hstar_scaling_conversions.py` | Short numerical checks of the displayed equations derived in `proofs/hreduce/declared_*.tex` | Secs. II to IV, VIII |
| `code/hreduce/final3_vacuum_controls/`, `partner_controls/`, `partner_final2_controls/`, `p33_p19_controls/` | Controls for the vacuum reference, the partner admissibility and the priority arithmetic | Secs. II C, IV B |
| `code/solve_a/mark3_layer_r1/` | Second-gradient vacuum polarization W_2 (`vaclaw_w2_v3.py`), the reference-law screen over 18 slices (`ref707_*`), and the massless cubic coefficient gamma_11 (`vrc_k0_v1.py`) | Sec. II C, Eq. (dm-reference), Sec. VI A |
| `code/solve_b/vrc/vrcb_cubicK0_v1.py` | Second route for gamma_11 | Sec. II C |
| `code/solve_a/strand2/`, `strand0/` | Vacuum kernel and the frozen native source action test | Secs. II C, III |
| `code/solve_a/strandS/`, `code/solve_a/obs/`, `code/orient/` | Graded entanglement cut, planar and spherical coefficient eta | Sec. IV B, Appendix A |
| `code/hreduce/results/sphere_directional/`, `code/hreduce/results/eta_massive_scalar_v1/` | Certified sphere mean and directional eta, massive eta(0.2) and its environment continuation | Sec. IV B, IV F, Appendix A |
| `code/hreduce/results/web40e_controls/`, `web40e_h3/`, `web40f_controls/`, `scaling_asymptotics/` | Elliptic integrals, conversion, fold-width constants and surface-lapse asymptotics | Secs. IV E, VIII A, VIII B |
| `code/solve_a/strand5/` | Cold continuum star family and its maximum mass (`strand5_star_v3.py`) | Sec. VI |
| `code/solve_a/engine/` | The collapse engine and its versions. The reference runs use `engine_collapse_v36d.py`. Later variants, including the Newton engine, are written from it by the `build_engine_collapse_*.py` patch builders here and in `code/busi/`. `engine_release_quadrature_v1.py` gives the release timing | Sec. VI A |
| `code/busi/iso/`, `rd13/`, `gatecomp/`, `w2/`, `prim/`, `core/`, `audit/` | Builders of the engine variants used by the controls, the assignment primitive, and the confirming-residual audit | Sec. VI A |
| `code/solve_a/ladders/`, `code/solve_a/compositions/`, `code/solve_a/attrib/` | Size, width and step ladders, the isotropic bridge member, and formation attribution | Sec. VI B |
| `code/busi/redshift/`, `code/solve_b/redshift/` | Received-redshift readouts on the stored histories | Sec. VI B |
| `code/busi/hf/`, `code/solve_a/core_hf/`, `code/busi/notcomp/` | Hartree-Fock core level and the second-order mass operator on the Hartree slices | Sec. VI C |
| `code/busi/compC/`, `code/busi/w11/`, `code/busi/w9/` | Core readouts, heavy-sea truncation runs, and the solved-flow ray map | Secs. VI C, VI E |
| `code/solve_b/singq/` | Inverse bound on the full Jacobian over the stored records, light weight 32 | Sec. VI D |
| `code/busi/readings/` | Cold re-checks of curvature, evolved cut, packet, Newton formation and size-128 formation readouts | Secs. VI D, VI E, VII D |
| `code/solve_a/flow/` | Flow-chart one-particle operator | Secs. V, VI E |
| `code/solve_a/page/`, `code/busi/w1/`, `code/busi/pagetests/` | Light-field (radiation) propagation on the collapse: `page_history_v6o.py` (18 channels) and `page_history_v6s.py` (held background), with their core, operator and readout modules | Sec. VII C |
| `code/solve_a/page/scat32/a1_rates_curve.py` | Frozen-field greybody luminosity, number and entropy rates | Secs. VII B, VIII C |
| `code/solve_a/page_v4/`, `code/solve_a/page_verify/` | Readout of the light runs for Fig. 3, and the conformal control | Sec. VII C |
| `code/busi/o5/`, `code/busi/gap10/`, `code/busi/route2/` | Cross-surface transfer and retarded response across the trapping radius, and the 13-site second-route cluster | Sec. VII D |
| `code/solve_a/firstlaw/`, `code/busi/w6w7/` | First-law records on the collapse | Secs. IV F, VI E |
| `code/hreduce/second_route/` | Second-route hold, horizon, core-well and residual-audit computations | Secs. VI, VIII D |
| `code/solve_b/code/` | Second-route runs (`run_*_secondroute_v1.py`) for curvature, evolved cut, packet and Page balance, the ray window, and the bound and entropy checks | Secs. VI to VIII |

## Proofs

Everything under `proofs/` is supporting material. The paper states its results and their hypotheses in full, and these files give the longer derivations, checks and records behind them. None of them is required to follow the paper.

- `proofs/derivations_from_the_paper.md` holds derivations and proof details removed from the paper for length, supporting the cited results. Each section names the paper section and the result it supports.
- `proofs/notes/` holds the detailed derivation, review and readout notes that the paper's source names for its cited results, under their original file names. `notes/REPO_DERIVATIONS_20261102.md` archives, verbatim, the longer passages removed from the paper between 1 and 4 November 2026. The notes use the working section numbers of their date, mapped to the submitted numbering in Part B of `derivations_from_the_paper.md`.
- proofs/notes/pointer_gaps/ holds the notes that back the paper's remaining repository pointers, numbered as in the pointer audit.
- The files below are self-contained derivations, listed by the result they support.

| File | Supports |
|------|----------|
| `hreduce/final4_foundation_*.tex`, `final4_momentum_alternative.tex`, `final4_vacuum_independence.tex`, `foundational_arguments_final3.tex`, `main_theorem_bridge_conditions.tex` | The Gauss law, source allocation, bond and momentum sources, operator ordering, preservation, sign, and vacuum independence (Secs. II, III, Appendix C) |
| `hreduce/final4_count_short.tex`, `final4_flow_ward_short.tex`, `final4_lapse_drift_short.tex`, `final4_shell_lapse_short.tex`, `final4_graded_cut.tex`, `final4_entropy_consistency.tex` | Short proofs: state count, flow Ward identity, lapse drift, shell lapse, graded cut, entropy consistency |
| `hreduce/declared_light_cone.tex`, `declared_ds_closure.tex`, `declared_bw_flow.tex`, `declared_faithful_vacuum.tex`, `declared_comoving.tex`, `declared_h2_continuum.tex` | Dispersion, energy-density algebra, vacuum modular flow and acceleration temperature, comoving matter, continuum Gauss law (Secs. II to IV) |
| `hreduce/declared_exterior.tex`, `declared_worked_example.tex`, `EXTERIOR_lattice_corrections_v1.tex` | Schwarzschild exterior, rays, and lattice corrections (Sec. IV D) |
| `hreduce/declared_local_area.tex`, `declared_area_law.tex`, `declared_entropy.tex`, `ETA_sphere_enclosure_v2.tex`, `solve_b/AREA_proofs_v1.tex` | Area law, sphere-enclosure bounds on eta, and the entropy and first-law match (Sec. IV B, IV F, Appendix A) |
| `hreduce/declared_exchange_kernel.tex`, `FIRSTLAW_rows_v1.tex` | Exchange kernel and the first-law rows (Sec. IV F) |
| `hreduce/declared_scaling.tex`, `declared_scaling_asymptotics.tex`, `solve_b/SCALING_theorems_v1.tex` | Astrophysical scaling, surface asymptotics, formation scales and the light-radiation capacity bound (Sec. VIII) |
| `hreduce/CORE_regime_second_route_v1.tex`, `solve_b/CORE_regime_theorem_v1.tex` | Core hydrodynamic regime (Sec. VI C) |
| `hreduce/ASYMPTOTIC_object_second_route_v1.tex`, `ASYMPTOTIC_agreement_v1.tex`, `solve_b/ASYMPTOTIC_object_v1.tex` | Late-stage asymptotics from two independent derivations (Sec. VIII B) |
| `hreduce/EVOLVEDCUT_theorem_v1.tex`, `solve_b/EVOLVEDCUT_purity_bound_v1.md`, `EVOLVEDCUT_purity_bound_channels_v1.md`, `EVOLVEDCUT_error_statement_v1.md` | Entanglement across the evolved trapping cut (Secs. VI E, VII D) |
| `hreduce/GRAFT_purity_theorem_v1.tex`, `hreduce/PURIFICATION_second_route_v1.tex`, `solve_b/PURIFICATION_bound_v1.tex` | Purity of the grafted state and purification at an exterior cut (Sec. VIII D) |
| `hreduce/LIGHTCHANNEL_identities_second_route_v1.tex`, `hreduce/PAGE_expectation_second_route_v1.tex`, `solve_b/CONFORMAL_cut_formula_second_route_v1.tex` | Light-channel identities, the Page expectation, and the conformal cut formula (Secs. VII, VIII D) |
| `solve_b/strand_X_page_correlations.tex`, `strand_X_causal_sea.tex`, `strand_X_nonlinear_response.tex`, `strand_X_gaussian_result.tex` | Light-algebra entropy and correlations, the causal sea and pair response, the nonlinear causal state, and the native Gaussian preparation (Secs. VII, VIII D) |
| `solve_b/T10_entropy.tex`, `solve_b/C3_analytical.tex`, `solve_b/strand_B_web40f_fold_width_scope.tex` | Entropy inventory of the ordinary cut, the C3 coefficient test, and the fold-width comparator premise 2GE = 59.05a (Secs. IV F, VI, VIII A) |

## Data

| Folder | Holds |
|--------|-------|
| `data/solve_a/engine/` | Reference collapse records: step controls tau = 4a and 8a (`C3prod19/`), the run from the start to T/r_s = 40 (`v36d_from0/`), and the Newton-route member (`p64_newton/`) |
| `data/solve_a/ladders/` | Ladder readout behind Fig. 2 and the ladder ranges in Sec. VI B |
| `data/solve_a/strand5/`, `data/solve_a/attrib/`, `data/solve_a/capacity_r1/` | Cold star family, formation attribution, and the pocket capacity ceilings |
| `data/solve_a/page/`, `page_v4.1/`, `page_verify/`, `data/busi/w1/` | Frozen-field scattering and rates, the light-run readout for Fig. 3, the conformal control, and the held-background light run |
| `data/busi/redshift/`, `data/busi/eta/` | Registry of the stored histories used by the redshift readout, and the directional eta table |
| `data/hreduce/results/sphere_directional/`, `eta_massive_scalar_v1/` | Certified eta enclosures and the massive coefficient |
| `data/hreduce/results/second_route/`, `page_balance_compare_v1/`, `web40e_*`, `web40f_controls/`, `scaling_asymptotics/` and the top-level `data/hreduce/*.json` | Second-route hold and core-well tables, Page balance comparison, and scalar controls |
| `data/solve_b/results/` | Second-route results: curvature, evolved cut, packet, Page, ladders arithmetic, the bound runs, ray window, and the UV and pair-response integrals |
| `data/solve_b/singq/`, `brk1/`, `items13/`, `page_pk5/` | Jacobian inverse-bound table, the rung-one recession bounds, the reference-projector weighted totals, and the pocket validation |

## How to run

Python 3.10 or later with `numpy`, `scipy`, `mpmath` and `matplotlib` runs the figure scripts and most checks. GPU runs of the engine and the light field need `torch` or `cupy`. Some checks need `numba`, `jax`, `python-flint` or `clarabel`, and the Figure 4 post-edit needs `pymupdf`.

The figure scripts run from `figures/`:

```bash
cd figures
python page_v5_fig.py page_v5_data.npz page_v5
python ../code/busi/figures/figure_item5_collapse_v3_1.py fig_item5_collapse_v3_data.npz item5_collapse_v3.1
python page_curve_amps_postedit.py page_curve_amps_R1.pdf page_curve_amps.pdf
```

The production runs were made on a GPU compute host. Absolute paths to that host were replaced by `<internal>/<name>` in Python and by `${QI2_ROOT}/<name>` in shell scripts. Set them to your own locations before running. Internal reviewer and team names in comments were replaced by neutral labels, and no logic was changed. The notes under `proofs/notes/` had the same treatment in their text but keep their original file names, so the paper's source comments still find them. Text files marked 1 in the `neutralized` column of `MANIFEST.tsv` therefore differ from the originals, so md5 assertions inside some scripts refer to the original files listed there. `figures/page_curve_amps_R1.py` keeps its embedded sources byte for byte, because it checks their sha256 sums. Binary arrays are byte-identical to the originals.

## What was left out

Result files over 2 MB that no cited proof or note names were left out, as were scratch directories, per-record subdirectories whose numbers are summarized in the run's top-level files, typesetting logs, engineering probes, and operations scripts. Two of the larger inputs are not included: the 34 MB field arrays of the reference collapse, whose plotted values are in `figures/fig_item5_collapse_v3_data.npz`, and the per-record arrays of the second-route holds. Copies are available on request.

## Paper and how to cite

The paper is archived at Zenodo, where the PDF can be read: [zenodo.org/records/23963163](https://zenodo.org/records/23963163) (DOI [10.5281/zenodo.23963163](https://doi.org/10.5281/zenodo.23963163); all versions: [10.5281/zenodo.19078442](https://doi.org/10.5281/zenodo.19078442)).

```bibtex
@article{McKinney2026EntanglementStars,
  author  = {McKinney, Jonathan C.},
  title   = {Entanglement Stars: Black-Hole Geometry from Quantum Correlations on a Lattice},
  year    = {2026},
  doi     = {10.5281/zenodo.23963163},
  url     = {https://zenodo.org/records/23963163}
}
```
