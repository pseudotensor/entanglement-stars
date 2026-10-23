# Proofs and derivations for the fermion paper

Derivations behind statements in the fermion paper (lattice fermion, Gauss laws, collapse, radiation and entropy). Files are copied as issued (markdown or LaTeX fragments). Mathematical content is unchanged. Two mechanical edits were made to every file: internal absolute paths were replaced by `<internal>/<basename>`, and internal reviewer or lane names were replaced by neutral labels ("reviewer A", "reviewer B", lane names). Version suffixes (`_v1`, `_v1.2`) follow the internal issue numbering. Where a file says "supersedes" or "kept as issued", only the latest version is included here.

Subdirectories are named after the working group that produced each file: `busi/`, `hreduce/` and `solve_b/`. The matching code is in `../fermion_new_codes/<group>/` and the data in `../data/<group>/`.

## Lattice momentum non-conservation (paper Sec. V A)

| File | What it is |
|------|------------|
| `busi/NONCONS_universal_v1.2.md` | The universal lattice non-conservation theorem. No local lattice momentum whose leading term is a non-constant function of quasi-momentum is conserved once a conversion vertex is present, under the stated nonvanishing-amplitude hypothesis (A). Also shows the counterexample that refutes the unrestricted wording. |
| `busi/NONCONS_indep_G2G4_v1.md` | Independent recomputation of the numerical check C1c (the G2 and G4 coefficients) used by the theorem above. |

## Conversion vertex: gate, amplitudes, force and stress

| File | What it is |
|------|------------|
| `hreduce/VERTEX_gate_v1.md` | Exact on-site conversion-vertex gate on the paper's lattice fermion. |
| `hreduce/VERTEX_gate_filtered_v1.md` | The same gate for the filtered (bond) converter. |
| `hreduce/VERTEX_amplitudes_v1.md` | Axial-pair conversion amplitudes for both converters. |
| `hreduce/VERTEX_force_event_v1.md` | Momentum transfer at the retained conversion points (the first-order on-shell force). |
| `hreduce/VERTEX_force_moving_v1.md` | Momentum transfer for a moving pair, for both converters. |
| `hreduce/VERTEX_stress_v1.md` | Interacting momentum balance and stress built from the vertex. |

## Dark-codeword theorem (paper section "Radiation, correlations and entropy")

| File | What it is |
|------|------------|
| `busi/MOCK_recovery_audit_v1.2.md` | Channel-optimal decoders on the saved n = 3 mock states. Contains the dark-subspace (dark-codeword) theorem: a symmetry of the vertex forces the empty-output sector to 1/4 at every n, and it gives the certified upper bounds on recovery. |

## Foundations: Gauss laws, sources, flow (paper section "The model" and the appendices on the boundary charge and the Gauss laws)

| File | What it is |
|------|------------|
| `hreduce/final4_foundation_gauss.tex` | The finite Gauss law and the question it answers. |
| `hreduce/final4_foundation_allocation.tex` | Allocation of the energy density between the bond and the site. |
| `hreduce/final4_foundation_bond.tex` | The bond-level source. |
| `hreduce/final4_foundation_momentum.tex` | The lattice momentum source: result and premises. |
| `hreduce/final4_momentum_alternative.tex` | Alternative momentum source and why it is not used. |
| `hreduce/final4_foundation_ordering.tex` | Operator ordering in the sources. |
| `hreduce/final4_foundation_preservation.tex` | Preservation of the constraints under the evolution. |
| `hreduce/final4_foundation_sign.tex` | Sign of the gravitational coupling. |
| `hreduce/final4_vacuum_independence.tex` | Independence of the results from the choice of vacuum reference. |
| `hreduce/final4_count_short.tex`, `final4_flow_ward_short.tex`, `final4_lapse_drift_short.tex`, `final4_shell_lapse_short.tex` | Short proofs: state count, flow Ward identity, lapse drift, shell lapse. |
| `hreduce/foundational_arguments_final3.tex` | Collected foundational arguments (earlier round). |
| `hreduce/main_theorem_bridge_conditions.tex` | Bridge conditions connecting the lattice statements to the main continuum theorem. |

## Declared-equation derivations (paper sections "The model" and "Properties of the fermion and of the vacuum exterior", and the appendices)

Each `declared_*.tex` derives one group of the paper's displayed equations. The bracket tags in each file mark the status of each step: X means exact, R means it rests on a stated reduction, N means numerical.

| File | Paper topic |
|------|-------------|
| `hreduce/declared_light_cone.tex` | Dispersion and propagation. |
| `hreduce/declared_ds_closure.tex` | The energy-density algebra. |
| `hreduce/declared_bw_flow.tex`, `declared_faithful_vacuum.tex` | Vacuum modular flow and the acceleration temperature. |
| `hreduce/declared_comoving.tex` | Comoving matter. |
| `hreduce/declared_exterior.tex` | The vacuum exterior of a spherical source: metric, axial sector, rays. |
| `hreduce/declared_worked_example.tex` | The point-source exterior (appendix). |
| `hreduce/declared_local_area.tex`, `declared_area_law.tex` | Local area estimates and the area law (appendix). |
| `hreduce/declared_entropy.tex` | Entanglement entropy and the first-law match. |
| `hreduce/declared_exchange_kernel.tex` | The exchange kernel. |
| `hreduce/declared_h2_continuum.tex` | Continuum Gauss law and its consistency. |
| `hreduce/declared_scaling.tex`, `declared_scaling_asymptotics.tex` | Astrophysical scaling and surface asymptotics. |
| `hreduce/EXTERIOR_lattice_corrections_v1.tex` | Lattice corrections to the exterior. |
| `hreduce/FIRSTLAW_rows_v1.tex` | The first-law rows. |
| `hreduce/ETA_sphere_enclosure_v1.tex` | Sphere-enclosure bound for the massive environment. |

## Collapse, radiation, cuts and entropy (paper sections on collapse, radiation and astrophysical scaling)

| File | What it is |
|------|------------|
| `hreduce/EVOLVEDCUT_theorem_v1.tex` | Entanglement across the evolved trapping cut. |
| `solve_b/EVOLVEDCUT_purity_bound_v1.md`, `EVOLVEDCUT_purity_bound_channels_v1.md`, `EVOLVEDCUT_error_statement_v1.md` | Evolved-cut entropy from the pure-state complement: the purity bound, its channel companion and the error statement. |
| `hreduce/GRAFT_purity_theorem_v1.tex` | Purity theorem for the grafted state. |
| `hreduce/PURIFICATION_second_route_v1.tex`, `solve_b/PURIFICATION_bound_v1.tex` | Purification at a geometric exterior cut: an independent derivation and the conditional bound. |
| `hreduce/LIGHTCHANNEL_identities_second_route_v1.tex` | Identities of the light (massless) channel. |
| `hreduce/PAGE_expectation_second_route_v1.tex` | Expectation for the radiation (Page) curve. |
| `hreduce/ASYMPTOTIC_object_second_route_v1.tex`, `ASYMPTOTIC_agreement_v1.tex`, `solve_b/ASYMPTOTIC_object_v1.tex` | Black-object asymptotics: the derived statement, its limits and the agreement between two independent derivations. |
| `hreduce/CORE_regime_second_route_v1.tex`, `solve_b/CORE_regime_theorem_v1.tex` | Core hydrodynamic regime: the conditional estimate and its correction (scaling section, "Core, shedding and persistence"). |
| `solve_b/CONFORMAL_cut_formula_second_route_v1.tex` | The conformal cut formula. |
| `solve_b/AREA_proofs_v1.tex` | Three conditional area proofs and their scope. |
| `solve_b/SCALING_theorems_v1.tex` | Formation scales, the coordinate-rest clock and a light-radiation capacity bound (scaling section). |
| `solve_b/strand_X_page_correlations.tex` | Light-algebra entropy and correlations across a surface, with an exact purity identity. |
| `solve_b/strand_X_causal_sea.tex` | Causal sea, the complete pair response and the scalar rows. |
| `solve_b/strand_X_nonlinear_response.tex` | Nonlinear causal state, clock elimination and a finite-parent test. |
| `solve_b/strand_X_gaussian_result.tex` | Bounded native Gaussian preparation result. |
| `solve_b/T10_entropy.tex` | Entropy-inventory test of the ordinary cut at the Newton match. |
| `solve_b/C3_analytical.tex` | C3 analytical consequences and the coefficient test. |

`MANIFEST.tsv` lists each file with the md5 of the original issued file. Hashes cited inside a document refer to the original records, which are kept off-repository on the group's compute host.
