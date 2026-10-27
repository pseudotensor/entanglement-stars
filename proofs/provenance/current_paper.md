# Current fermion paper: provenance and reproducibility record

## 1. Status and scope

This record separates supplied facts from information still requiring
extraction. It is not a claim that the numerical initial-value problem
has been independently reconstructed.

Implementation facts below are transcribed from PFIX_VALUES_v1.md,
the supplied read-only extraction of code, launch scripts, records and
logs. That extraction ran no simulation or numerical validation; apart
from hashing files it only read code, records and logs. Its marker
numbers identify its source sections. Code-level facts, recorded
settings, inferred arithmetic and unlocated information are
distinguished. No implementation is inferred from a version name
alone, and no missing item is replaced by a default.

For source locators below:
SOLVE is /home/jon/qi2/.wt-claude-alt-solve/solve;
BUSI is /home/jon/qi2/.wt-claude-busi/busi;
SOLVE_B is /home/jon/qi2/.wt-codex-solve/solve;
DEL is /home/jon/qi2_fable_20261013/deliveries;
BOX is 192.168.1.93.
These are historical private locations, not public download links.

The top-level "README.md" describes a different historical construction:
"Because the conductances scale as the squared lapse" and
"Tracking mode-by-mode evaporation ... a Page-like curve."
Those sentences are not definitions or results of the current paper.
Only the explicitly identified current artifacts may support it.

[VALUE NEEDED: immutable current-paper commit and companion-archive commit, including the exact bibliography target for McKinneyCode2026; obtain from the two Git commit records and the bibliography file used by ms_fermion_new.tex.]

The supplied "data_README.md" says that text provenance was neutralized,
numerical values were unchanged, and binary arrays are byte-identical.
The manifest header is "file size_bytes original_md5 neutralized".
Therefore an original_md5 for a neutralized text file is not a hash
of its distributed bytes. It must not be presented as such.

[VALUE NEEDED: SHA256 of distributed source, data, figure and manifest bytes, and resolvable locations for required omitted arrays; obtain from the archived files, data/MANIFEST.tsv, fermion_new_codes/MANIFEST.tsv and SKIPPED_LARGE.tsv.]

The original_md5 strings quoted in Section 5 are confirmed against data/MANIFEST.tsv
when that marker is filled, and rows for the five new proof files are added to
proofs/MANIFEST.tsv at the same commit.

Missing values below are not defaults. No new full simulation is needed
to extract them.

## 2. Principal preparation and dimensions

The supplied "05_star.tex" states:
"ma=0.2 and G/a^2=1/4",
"nominal r_s=64a",
"N_B=640",
"released at rest relative to the lattice at 4r_s",
"2GE_0=59.07a",
"Gaussian kernel of width sigma=4a",
"channels k=1 to 10", with dust in "k=1 to 5",
and "tau of 8 lattice units".

These are dimensionless run settings and physical counts, not a list
of different physical masses. The nominal radius uses 2GmN_B and
must not replace the charge radius 2GE.

For each two-component radial channel with n_k sites, a pure
number-conserving initial covariance can be recorded unambiguously as

C_k(0)=Z_k Z_k^dagger, Z_k^dagger Z_k=I,

where Z_k contains every occupied initial orbital, including the sea.
This equation is a representation of a pure covariance, not evidence
that the actual preparation is P_k+U_kU_k^dagger.

The supplied text does not determine Z_k. In particular it does not
determine whether localized orbitals were projected, orthogonalized,
filled fractionally across identical copies, or accompanied by holes.
Particle count and coordinate rest cannot choose among those possibilities.

[VALUE NEEDED: complete initial occupied-column construction for the principal collapse and its five-column and isotropic preparations, including radial envelopes, phases, projections, orthogonalization, sea occupations, copy multiplicities and seeds or an explicit deterministic statement; extract the executed preparation function and launch arguments through the lineage of engine_collapse_v36d.py and its actual successors, and the initial covariance/column payloads named by that lineage.]

The preparation, chain-length, source-map and reference-continuation
extractions in Sections 2 and 3 apply to every ladder member listed
in Section 7, including the zeta=0.1 member at r_s/a=96, the releases
from 2r_s, and the E_0/E_max=1.14 members. Each member retains its own
parameters and preparation; the listed settings do not specify a
Cartesian product of runs.

The code listing contains
"fermion_new_codes/solve_a/engine/engine_collapse_v36d.py",
"fermion_new_codes/solve_a/engine/engine_collapse_v36r.py",
"fermion_new_codes/busi/iso/build_engine_collapse_v36n.py",
and many other versions. Their presence does not identify the producer.

[VALUE NEEDED: actual radial chain lengths for the reference collapse, each step/width/channel-count control, both long interior completions, principal light propagation, held continuations, conformal control, cubic transfer and feedback control; read launch arguments and matrix/column shapes from the primary payloads referenced by the lineage and stored_identity.json records.]

For every extracted preparation, small matrix checks must report
||C-C^dagger||, extremal eigenvalues or an equivalent projector
certificate, ||C^2-C||, the weighted excitation count, the initial
source arrays, and the precise coordinate-rest criterion.
No missing seed is to be guessed.

## 3. Chain operator and discrete update

The manuscript specifies r_n=na, n=1,...,n_k, open endpoints,

h0,k=m sigma3+sigma1(T-T^dagger)/(2ia)+sigma2 k/r_n,
Q_r=(T^2-T^{dagger 2})/(4ia),

with the stated allocations and back-averaged fields. The chain
weight is 16k. This is a separate radial discretization, not a
restriction of the full cubic zone.

For fields u_j held during step j, the reported exponential has

U_k,j=exp[-i tau_j h_k(W^dagger u_j)],
C_k,j+1=U_k,j C_k,j U_k,j^dagger.                        (1)

Equation (1) is all that can be reconstructed about the update from
the supplied operator and method prose. To complete the discrete
initial-value problem one must specify the actual source map S_j,
row solver R_j and reference continuation T_j in

u_j=R_j(S_j({C_k,j},{P_k,j};u_j)),
P_k,j+1=T_j(P_k,j; actual field path and subdivisions).  (2)

The old wording combines fields "at the start of each step" with
"midpoint sources" and explicitly says "the per-step choice is not read".
Neither a midpoint arithmetic average nor a half-step covariance
is silently substituted for S_j.

[VALUE NEEDED: literal equations or pseudocode for S_j, R_j and T_j, including which covariance and reference enter each iteration, relaxation, row quadrature, boundary conditions, stopping/failure rules and the fields used in the exponential; extract the executed step and field-solve functions selected by the primary run lineage, not an unpinned engine_fields version.]

[VALUE NEEDED: actual reference path, overlap/assignment rule, degeneracy and zero-crossing convention, subdivision limits and per-step subdivision choices for every principal run and control; extract the executed continuation functions and logged assignment/subdivision records associated with the same lineage.]

The historical "declared_local_area.tex" describes an earlier
specification whose ambiguity rule "halves the interval at most three
times". That is not evidence that the current runs used it.
A changing spectral reference is not a propagated projector.

The reported 10^-3 iteration-change threshold is a stopping criterion.
For a verified self-consistency map F mapping a closed ball to itself,
with Lipschitz constant q<1 there, one could prove

||u-u_*||<=||F(u)-u||/(1-q).

Without that ball and q, no field-error bound follows. Ray errors
would additionally require a bound on the derivative of the ray
observable on the same ball. No such certificate is recorded here.

[VALUE NEEDED: actual numerical precision, exponential/eigensolver algorithms and tolerances, software versions, randomization settings, and measured purity/charge residual definitions; obtain from each run's execution metadata and the executed propagation and eigensolver functions.]

## 4. Data-to-generator prescription for the light field and rays

The manuscript specifies a massless test field, no source feedback in
the principal light calculation, propagation from T=0 to T/r_s=100,
eighteen displayed channels, and an appended vacuum exterior.
It distinguishes linear temporal interpolation from cubic-spline
and piecewise-constant controls.

A reproducible generator requires the ordered operations:

1. choose the actual stored geometry/source record and its units;
2. reconstruct dense radial fields from the stored arrays;
3. apply the stated spatial interpolation and endpoint extension;
4. append the exterior at its actual join, with its charge choice;
5. apply the actual back-averaging, in its actual order;
6. interpolate in time with the run's specified rule;
7. assemble the endpoint-symmetrized chain matrix;
8. apply the declared propagation algorithm and reference/comparator.

These steps are requirements for extracting the implementation,
not an assertion that this order was used.

[VALUE NEEDED: exact spatial and temporal interpolation, averaging order, exterior join radius, charge interpolation/holding choice, taper or matching rule, origin/far-boundary treatment and time substeps for principal light and ray generators; obtain from the executed page_history/page_ops/raymap functions selected by stored_identity.json, registry_v1.json and their primary input bindings.]

For entropy the current "07_page_curve.tex" defines the comparator
as the negative projector of the Hamiltonian held at the same record.
It is not automatically the continued reference. Extract its actual
matrix, occupation convention, cut-site rule and endpoint entropy
treatment separately.

[VALUE NEEDED: entropy and flux payload selectors, cut-site inclusions, held-vacuum construction, eigenvalue endpoint handling, propagation-floor computation, channel weighting and tail/common-mode statistic for each displayed sum; obtain from the actual figure and spectral-analysis functions and their saved input bindings.]

For received redshift also extract the event grid, clocks, launch
positions, receiver normalization, arrival-window mask and interpolation
variants. A maximum over samples is not a supremum over all emissions.

[VALUE NEEDED: complete event selection and receiver-frequency normalization for the headline redshifts, with integration tolerances and certified-window masks; obtain from data/busi/redshift/registry_v1.json, its referenced ray records and the executed raymap producer.]

## 5. Artifact inventory and what it proves

These are entries of the code listing and the data manifest.
They identify extraction targets, not independently verified numerical
contents.

- The code listing includes
  "fermion_new_codes/busi/figures/figure_item5_collapse_v3_1.py",
  "fermion_new_codes/busi/figures/figure_page_v5.py", and
  "fermion_new_codes/busi/figures/figures_data_v1.py".
  The current paper includes "item5_collapse_v3.1.pdf" and "page_v5.pdf".
  A source-to-figure dependency is not established by similar names.

- The data manifest lists
  "busi/redshift/registry_v1.json", size 27828,
  original_md5 efcfad78bb6e2c942f21cd4ef9028270, neutralized 1.

- It lists
  "busi/perf_host/LINEAGES_h2default_v1.tsv", size 1564,
  original_md5 fc7e1573f63762ff228b4d1b67721824, neutralized 0.

- It lists
  "solve_b/results/LADDERS_v1p3_arithmetic_v1/scalar_records/P64_s4.json",
  size 718405, original_md5 90f6ad73da2d997472f73d643d4d1fde,
  neutralized 0.
  Its relationship to the present principal history must be checked,
  rather than inferred from P64_s4.

- It lists
  "solve_b/results/PAGE_secondroute_v1/stored_identity.json",
  size 1716, original_md5 44190ee612906414e5e642f83da954a7,
  neutralized 1, and
  "solve_b/results/PAGE_secondroute_v1/spectral_entropy.json",
  size 8882205, original_md5 d9b6056712e38f0131660626b7b53b5d,
  neutralized 1.
  Recomputing sums from spectra does not independently reconstruct
  either covariance or held projector.

- It lists
  "hreduce/results/sphere_directional/eta111_certificate.json",
  size 385264, original_md5 eb0cd966f9fe3c302d2111b31be8ea67,
  neutralized 0, and
  "hreduce/results/eta_massive_scalar_v1/p111_result.json",
  size 27665, original_md5 f1c50aef7cb737646f7eb992e6ea4bcd,
  neutralized 1.
  The payloads and interval arithmetic must be read before their
  endpoints are independently certified.

The analytical proof files have their own original_md5 entries.
Those are provenance identifiers of issued sources, not certificates
of the propositions or of later manuscript compatibility.

[VALUE NEEDED: assertion-to-artifact map for all principal figures and numerical result families, including exact producer functions, executed source hashes, launch arguments, dataset hashes, payload selectors, analysis functions, independent-route dependencies and final PDF hashes; obtain from the figure scripts, their input bindings, execution records and current paper build.]

The result families requiring rows in that map are: formation and
width/step/channel controls; size and parameter ladders; redshift;
core density and spectra; exchange, variance and branch witness;
curvature and packet continuation; first-law/evolution residuals;
light excess and collar flux; held drains; the Hawking comparator rates of the luminosity equation and the conformal
companion; feedback; cross-root
transfer; conversion fits; flat-area coefficients and interval
enclosures; continuum maximum-mass constants; and conditional-figure
arithmetic. Shared input operands must be distinguished from
independent implementations.

## 6. Validation without full simulation

After extracting the missing metadata, perform only targeted checks:

- reconstruct one initial covariance per preparation;
- replay one stored step using its actual reference subdivisions;
- rebuild one stored generator on each interpolation/extension variant;
- reproduce selected cut eigenvalues and weighted sums from stored arrays;
- evaluate resonant and probe formulas by exact algebra or small matrices;
- verify hashes, dependency joins and masks;
- compare the two claimed independent calculations at matched inputs.

For the asymptotic discussion use chi=a/r_s ->0 at fixed ma,
g=G/a^2 and binding fraction, in the stated supra-maximal domain.
No finite-chain result or solver tolerance supplies a uniform
chi-limit by itself.

Earlier measured builds, as quoted in Fable's PLEN round 2 answer
(pdflatex, bibtex, pdflatex, pdflatex):
- BUILD_REPORT_pfix_v1.txt, PFIX edits applied: 32 pages, zero LaTeX
  errors, zero undefined references or citations; in
  fermion_new/07_page_curve.tex the converter, momentum-balance and
  current displays exceeded the column by 20.5pt, 44.6pt and 9.9pt.
  Reserves 1, 1+2 and 1+2+3 each left 32 pages, the last with 14
  bibliography lines on page 32.
- BUILD_REPORT_plen1b_v1.txt, PLEN round 1b paper edits applied:
  30 pages, page 30 holding 22 non-blank lines; zero LaTeX errors,
  zero undefined references or citations; no overfull boxes; the
  three to-do marks of sec:page-effective verbatim. The reported
  section and figure locations give about 5 pages for Sec. V plus
  sec:page-effective with Figs. 2 and 3 and their captions.

Latest supplied measurement:
- The lead reports that PAPER_flat_plen2.tex, with the PLEN round 1b
  edits, Fable's amendments and Edits 55-71 applied, builds on the
  compute box to 30 pages, with 27 lines on page 30; zero errors,
  zero undefined references, no overfull boxes, and the three to-do
  marks verbatim.
- The latest report's file identity, compiler version and invocation,
  dependency hashes and final PDF SHA256 are not supplied. The
  earlier section locations and approximate Page-material area are
  not a fresh measurement of this build.

Targets remain at most 29 pages including every appendix and the
bibliography (requirements_new.txt, Secs. 1 and 5), and clearly less
than four pages for Sec. V plus sec:page-effective including both
figures and captions. The reported 30-page build fails the total
length target.

After the present corrections, rebuild and record the compiler,
class, bibliography and figure identities and final PDF SHA256.
Count every page, inspect every display and table, report errors,
undefined references and citations and overfull widths, verify the
three to-do marks verbatim, and measure the occupied page-equivalent
area of the specified Page material. Estimated savings do not
certify either length target.

The publication gate is at most 30 pages. Word reduction alone does
not certify that gate.

## 7. Retained size and parameter ladder readouts

The following inventory retains the numerical readouts of the corresponding
paragraph of 05_star.tex. That paragraph stays in the paper unless the
length reserve replaces it with its summary. This is a record of reported
results, not a new evaluation.

Across sizes and masses the formation is a first-record dependence on
two and three members, no law fitted. Along r_s/a=32, 64 and 128
at zeta=0.2 the root forms at T/r_s=15.81, 12.125 and 11.19
to 11.56, at 0.73, 0.66 and 0.63 to 0.68 of 2GE (the 128 values
at sigma/a=8 and 16, the five-column sigma/a=4 solve
failing there; the isotropic sigma/a=4 member forms at 11.5 and
0.70, on one route); along zeta=0.4 to 0.1 at r_s/a=64,
except 96 for the 0.1 member, it forms later, 5.375 to 21.17, at a
larger fraction of 2GE, 0.37 to 0.73; the (0.05,326) records,
certified to 0.49r_s, hold no root, and the members at
E0/Emax=1.14 form no root to 40 from 4r_s, while the
(0.2,64) and (0.1,64) counts released from 2r_s form. The
128,sigma/a=16 formation is outside the gate. At r_s/a=64,
the isotropic preparation forms 0.25r_s later, within 1.4 percent
in radius at matched clocks 12.75 to 16. The formation record moves
later and its root fraction rises with sigma/a, and it moves
earlier with a finer step, by fractions of r_s per factor of two;
the 32 and zeta=0.4 points fail the averaging-scale criterion,
which the 128 pair meets at formation only. At nominal r_s/a=256
formation inside the gate is reported at one width, sigma/a=8,
with no width span and no width law established there.
An isotropic sigma/a=4 member at r_s/a=128 past its first
certified formation record can be computed and is not required here.
A fourth size, r_s/a=256, and the
(0.05,326) member's continuation past 0.49r_s can be computed
and are not required for the claims.

## 8. Retained received-map and dispersive-ray readouts

The following inventory retains the numerical readouts of the corresponding
paragraph of 05_star.tex. That paragraph stays in the paper unless the
length reserve replaces it with its summary. This is a record of reported
results, not a new evaluation.

For outward radial continuum light emitted by a clock comoving with the
flow, the received fraction at infinity is g=omega_infinity/omega_em.
On stationary fields it equals N+v at emission and tends to
zero as the emission point approaches their trapping root; on evolving
fields it requires ray propagation. On its slices the solved history
tracks no material surface, its trapping root staying below the
instantaneous 2GE on every record, so the marginally bound
surface-peeling control describes none of its records.
The received map is computed on the back-averaged fields the matter
couples to, with the receiver at 4r_s, on sampled test clocks
comoving with the flow from T=0 and on launches beside the root.

Within the solver-certified window, to T/r_s=40, the largest
received ln(1/g) is 10.3 on the reference collapse
(3x10^4; from a clock whose front, emitted at 0.30r_s
before the first root formed, the root caught) and 15.6 at
sigma/a=2, from the clocks. From the launches, which exceed
the clocks (3.1 to 6.0) elsewhere, the maxima are 3.6 at sigma/a=8,
6.5 at tau=4, 7.6 and 3.5 at r_s/a=128 with sigma/a=8 and 16,
and 3.7 and 4.1 on the two alternative interior completions.
These are maxima over sampled events. The deepest arrive at the
window's end still growing, on flights almost wholly past the
one-term hydrodynamic gate.

At this size the late redshift depends at order one on width, step,
interior completion and propagation background. The dense fields
of the same solve move the clocks' values by up to 2.3.
The clocks' integration error indicator is 10^-7; other propagation
variants at headline events and fixed arrival change values by 0.05.
Sensitivity to field perturbations at the certification ceiling is
0.4 per 10^-3, with no bound on field error established.
The launches' values move by up to 1.4 under the propagation
background and variants.

While a root stands, a hovering front's redshift grows with the
integral of its slope over the hover, int kappa_R dt.
This relation is selected on seven of ten sampled clocks of the
reference collapse, with fitted slope 0.85 to 1.25.
Median kappa_R across members is 0.74 to 1.21 of 1/(4GE).
No receiver-time law is reported. Members with sigma/a>=8 at both
sizes and both completions have some blueshifted sampled arrivals
inside the window, so the samples give no uniform redshift bound.

These are long-wave continuum-map values. A quantum received at
omega r_s=0.1 with the reference maximum would have had ak=48
at emission. Lattice-dispersion rays from the same events with
ak=0.05 to 0.3 arrive earlier and, on the four histories with the
largest redshifts, less redshifted. Their ln(1/g) reaches at most
6.6, 5.4, 4.5 and 3.7, near the respective continuum crossover
ln(2/(ak)^2)=6.7, 5.3, 3.9 and 3.1. Some rays exceed that scale;
it is not a bound. Where the root closes early the dispersive
same-event ray can instead be more redshifted, by up to 0.16.
No packet is propagated and no finite-band received spectrum is
computed.
