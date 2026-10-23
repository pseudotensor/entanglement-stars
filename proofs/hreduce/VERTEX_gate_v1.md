# VERTEX gate v1

Scope: this answers inbox hreduce #97 for the local heavy–light axial-pair converter on the paper’s sine Dirac lattice at commit `daaf6fbd05ff029e771f8fc1ebcd8f24ccf17dca`. It determines the operator, its symmetries, two-particle amplitudes, energy current and restricted Hartree field. It does not change the manuscript, calculate a conversion rate or Page curve, or establish an interacting gapless branch. The author and the sole independent red team rederived the claims and read the actual exact-check outputs. The numerical implementation is **incomplete in the amplitude stage**; its completed checks and partial returns are identified below. All numerical records remain on the box.

Tags: **[R]** re-derived; **[N]** own numerical/exact-symbolic result with the stated controls; **[S]** supplied definition or proposal. No absent expected-value record is counted as numerical evidence.

| Requested item | Verdict | Evidence and limit |
|---|---|---|
| 1. On-site term and specified discrete/light-chiral symmetries | **PASS [R,N]** | Symmetric-phase Dirac-mass protection. A common physical multivalley Lorentz symmetry is a stronger claim and **fails**. |
| 2. Threshold cancellation and first nonzero/intervalley amplitudes | **PASS analytically [R]; numerical verification INCOMPLETE** | The threshold tensor returned exactly zero. Nonzero, symbolic-slope and intervalley target checks did not return. |
| 3. Summed free two-step current remains unchanged | **FAIL [R,N]** | The interacting current has a nonzero quartic correction, with an exact matrix element \(+U\). The corrected-current identity passes. |
| 4. Mean field vanishes on number-conserving species-product Hartree states | **PASS [R,N]** | This is a restricted mean-field statement. It does not make exact interacting evolution, the vacuum or entropy history unchanged. |

## 1. Literal term, symmetries and mass protection

Use dimensionless site CAR, \(\{c_{A,x,a},c^\dagger_{B,y,b}\}=\delta_{AB}\delta_{xy}\delta_{ab}\), and component order upper up, upper down, lower up, lower down. The paper’s free symbol is \(h_A(\boldsymbol p)=\boldsymbol\alpha\cdot\boldsymbol d+m_A\beta\), \(d_i=\sin(ap_i)/a\), with \(m_L=0\). Set

\[
\alpha_i=\tau_x\otimes\sigma_i,\quad
\beta=\tau_z\otimes I,\quad
\gamma^0=\beta,\quad \gamma^i=i\tau_y\otimes\sigma_i,\quad
\gamma_5=\tau_x\otimes I,\quad C_D=-\tau_x\otimes\sigma_y.
\]

The Dirac-matrix contraction uses \(\eta=\operatorname{diag}(1,-1,-1,-1)\); this convention does not change the paper’s geometric metric convention. The antisymmetric pairing matrices \(M^\mu=C_D\gamma^\mu\gamma_5\) are

\[
M^0=\tau_z\otimes\sigma_y,\quad
M^1=\tau_y\otimes\sigma_z,\quad
M^2=i\tau_y\otimes I,\quad
M^3=-\tau_y\otimes\sigma_x.
\]

Define, with **no factor \(1/2\)** in the pairing,

\[
\Phi_A^\mu(x)=c_{A,x}^{T}M^\mu c_{A,x},\qquad
O_x=\Phi_L^{0\dagger}\Phi_H^0-\sum_{i=1}^3\Phi_L^{i\dagger}\Phi_H^i,
\]
\[
W_x=O_x+O_x^\dagger,\qquad
H=H_{0,H}+H_{0,L}+U\sum_xW_x,\qquad
U=\lambda_X/a^3,\quad\lambda_X\in\mathbb R.
\]

Here \(\psi=a^{-3/2}c\), so \(\lambda_X\) has units of length squared and \(U\) of energy. A common phase of \(C_D\) cancels from \(O_x\). This is the requested derivative-free on-site term. [R]

Hermiticity is explicit. The charge identities are

\[
[N_H,O_x]=-2O_x,\quad [N_L,O_x]=2O_x,\quad
[N_H+N_L,W_x]=0.
\]

Both species parities, and hence total fermion parity, are conserved. Separate continuous heavy and light numbers are not. A fixed reference constant in the Dirac charge does not change these commutators. [R,N]

The decisive light-chiral identity is

\[
\gamma_5^TM^\mu+M^\mu\gamma_5=0.
\]

It makes every \(\Phi_L^\mu\) invariant under \(c_L\mapsto e^{i\theta\gamma_5}c_L\). The massless free operator and the interaction therefore commute with \(Q_{5L}=\sum_xc_L^\dagger\gamma_5c_L\). This also holds with the paper’s scalar clock/flow backgrounds and with the corrected current derived below. Reference subtraction must preserve this symmetry. [R,N]

For a proper spatial rotation \(S_R=I\otimes u_R\), the spatial pairings rotate by a real orthogonal matrix while \(M^0\) is fixed. This includes the cubic group. Inversion uses \(P=\beta\) and site reversal:

\[
P^TM^0P=M^0,\qquad P^TM^iP=-M^i.
\]

Time reversal is \(T=U_TK\), where \(K\) here denotes complex conjugation and \(U_T=I\otimes i\sigma_y\), with

\[
U_T^T(M^0)^*U_T=-M^0,\qquad
U_T^T(M^i)^*U_T=M^i.
\]

Each contracted term contains two equal signs, and real \(\lambda_X\) is unchanged. The free hopping transforms with momentum/site reversal. These are operator symmetries, with backgrounds transformed accordingly; a particular inhomogeneous state need not share them. [R,N]

The matrices anticommuting with all three \(\alpha_i\) are exactly the span of \(\tau_y\otimes I\) and \(\tau_z\otimes I\). For the Pauli-tensor basis ordered by \(a,b=0,x,y,z\), the Gram matrix of \(P\mapsto(\{\alpha_i,P\})_{i=1}^3\) is

\[
\operatorname{diag}(48,16,16,16,48,16,16,16,0,32,32,32,0,32,32,32).
\]

Both kernel matrices violate the exact light-chiral symmetry. Total charge excludes a Majorana mass and species parity excludes a heavy–light bilinear mass. Thus a symmetry-preserving perturbative expansion about a symmetric phase cannot generate a light Dirac or pseudoscalar mass at any order. **This does not prove a gapless interacting pole or exclude spontaneous symmetry breaking.** [R,N]

There is also a necessary Lorentz qualification. The pairing is a Lorentz vector under a common spinor transformation, so its formal contraction is a scalar. Its boost identities are

\[
\alpha_i^TM^0+M^0\alpha_i=2M^i,\qquad
\alpha_i^TM^j+M^j\alpha_i=2\delta_{ij}M^0.
\]

This does not give the unfiltered lattice interaction a common physical boost after separate valley-dependent cone redefinitions. In ordered pair basis \((01,02,03,12,13,23)\), let \(K_{ab;cd}=4\sum_\mu\eta_{\mu\mu}(M^\mu_{ab})^*M^\mu_{cd}\), and let \(B_3\) be the one-body lift of \(\alpha_z\) to that wedge space. With

\[
v_0=(1,0,0,0,0,-1)^T,\qquad v_3=(0,0,1,1,0,0)^T,
\]
\[
B_3K+KB_3=0,\quad B_3v_0=-2v_3,\quad
Kv_0=8v_0,\quad Kv_3=-8v_3,
\]
\[
(-B_3K+KB_3)v_0=32v_3\ne0.
\]

The last expression is the mixed-valley physical-boost obstruction. It returned exactly in the on-site checks. An unqualified common multivalley Lorentz claim is **FAIL [R,N]**, while the specified lattice symmetries and common-spinor algebra pass.

## 2. Amplitudes: analytic PASS, numerical stage incomplete

Use CAR-normalized positive-band spinors, rather than covariant \(\bar u u=2m\) normalization:

\[
u_m(\boldsymbol d,\xi)=\sqrt{\frac{E+m}{2E}}
\begin{pmatrix}\xi\\(\boldsymbol\sigma\cdot\boldsymbol d)\xi/(E+m)\end{pmatrix},
\qquad E=\sqrt{m^2+|\boldsymbol d|^2},\quad\xi^\dagger\xi=1.
\]

For ordered distinct two-particle modes on \(V_s\) sites,

\[
\langle L_1L_2|V_X|H_1H_2\rangle
=\frac{U}{V_s}\delta_{p_1+p_2,k_1+k_2\ ({\rm mod}\ 2\pi/a)}\,\mathcal A,
\]
\[
\mathcal A=4\sum_\mu\eta_{\mu\mu}
(u_{L1}^{T}M^\mu u_{L2})^*(u_{H1}^{T}M^\mu u_{H2}).
\]

The factor four follows from \(\Phi a^\dagger(u)a^\dagger(v)|0\rangle=-2u^TMv|0\rangle\). Reversing an ordered pair reverses its sign. Identical occupied modes are excluded by CAR. The spin averages below are tensor normalization checks, not occupation or phase-space weights. [R]

For opposite heavy sine momenta \(\boldsymbol d,-\boldsymbol d\),

\[
J_H^0=\frac mE\xi_1^T\sigma_y\xi_2,\qquad
J_H^i=-\frac iE\xi_1^T\sigma_y(\boldsymbol d\times\boldsymbol\sigma)_i\xi_2.
\]

For opposite light directions \(\boldsymbol n,-\boldsymbol n\), \(J_L^0=0\) and the spatial expression replaces \(\boldsymbol d/E\) by \(\boldsymbol n\). The identity \((\boldsymbol\sigma\cdot\boldsymbol n)^T\sigma_y(\boldsymbol\sigma\cdot\boldsymbol n)=-\sigma_y\) gives the temporal cancellation. At heavy rest the spatial heavy currents vanish, so the **same-valley centre-of-momentum threshold amplitude is exactly zero**. This uses opposite sine momenta at finite spacing; it is not a claim about every moving or inhomogeneous pair. The incoming spin singlet is dark in this particular channel. [R]

The first generic spatial amplitude is linear in relative momentum. Its exact spin sum is

\[
\frac14\sum_{s_1s_2t_1t_2}|\mathcal A_{t_1t_2;s_1s_2}|^2
=\frac{16}{E^2}\bigl(|\boldsymbol d|^2+(\boldsymbol n\cdot\boldsymbol d)^2\bigr).
\]

It follows from the Pauli trace identity for \(\sigma_y(\boldsymbol n\times\boldsymbol\sigma)_i\). For collinear momenta along \(z\), only

\[
\mathcal A_{\uparrow\uparrow;\uparrow\uparrow}
=\mathcal A_{\downarrow\downarrow;\downarrow\downarrow}
=-8d/E=-16r/(1+r^2),\qquad r=d/(E+m)
\]

are nonzero. Their derivative at \(r=0\) is \(-16\). At \(ma=3/5\), \(ad=4/5\), \(aE=1\), the entries are \(-32/5\), the collinear spin average \(512/25\), and the perpendicular spin average \(256/25\). **These nonzero values and the derivative are analytic results [R], not returned numerical results of the attempts below.**

For intervalley threshold conversion write valley centres \(K_b=\pi b/a\), \(b\in\{0,1\}^3\). Incoming valley XOR \(c\) equals outgoing valley XOR by crystal-momentum conservation. For outgoing sine directions \(\boldsymbol n,-D_c\boldsymbol n\), \(D_c=\operatorname{diag}((-1)^{c_i})\),

\[
B_c(\boldsymbol n)=\frac12\sigma_y
\left[I-(\boldsymbol\sigma\cdot\boldsymbol n)(\boldsymbol\sigma\cdot D_c\boldsymbol n)\right],
\]
\[
\mathcal A_{t_1t_2;s_1s_2}=4[B_c(\boldsymbol n)_{t_1t_2}]^*(\sigma_y)_{s_1s_2},
\qquad
\frac14\sum|\mathcal A|^2=16\sum_{i:c_i=1}n_i^2.
\]

The last identity uses \(\operatorname{Tr}(B_c^\dagger B_c)=1-\boldsymbol n\cdot D_c\boldsymbol n\). For \(c=0\) all threshold entries vanish, including conversion into another same-valley pair. For \(c\ne0\), directions in flipped components give nonzero threshold amplitudes. In particular \(c=(0,0,1)\), \(\boldsymbol n=\hat z\) gives \(B_c=\sigma_y\) and \(\mathcal A_{\uparrow\downarrow;\uparrow\downarrow}=4\). At \(ma=3/5\), incoming rest momenta \(0,\pi\hat z/a\) and outgoing \(q\hat z,(\pi/a-q)\hat z\), with \(\sin(aq)=3/5\), conserve both energy and crystal momentum. For a general chosen direction the on-shell light momenta require \(|ma n_i|\le1\). Deleting these allowed channels would change the operator. **Universal lattice \(p\)-wave suppression is false.** This classification and its nonzero example are [R]; the planned intervalley numerical tensors did not return.

The returned exact amplitude prefix contains the \(ma=3/5\) same-valley threshold case. Its saved currents are \(J_H=(\sigma_y,0,0,0)\), \(J_L=(0,iI,-\sigma_z,0)\), its tensor and spin average are zero, and all eleven returned leg/energy/tensor comparison residuals are exact zeros. This is directly inspected partial [N] evidence for the threshold result. The producer did **not** record their Boolean validation decisions. It also saved seven normalized leg records and the next collinear \(J_H^0=(3/5)\sigma_y\), but no nonzero amplitude tensor. Thus the analytic classification passes while the frozen numerical amplitude plan remains **INCOMPLETE**.

No amplitude here determines an occupation-integrated forward/inverse rate, a conversion clock or a Page curve. Those require state occupancies, final-state availability and phase space.

## 3. Interacting energy current: the unchanged-current claim fails

Allocate each free hopping bond equally to its endpoints and the whole \(UW_x\) to site \(x\), using the paper’s free-energy convention. Set \(D_i^E=\sum_xx_i e_x\) and \(\mathcal Q_i=i[H,D_i^E]\). For the free bond Hamiltonian \(T_{xy}\), the exact local identity is

\[
\boxed{\mathcal Q_i=\mathcal Q_{0,i}
+\frac{iU}{2}\sum_{\langle x,y\rangle}(y_i-x_i)[T_{xy},W_y-W_x].}
\]

The on-site mass–interaction commutators cancel between the two cross terms at the same position. Even interactions on distinct sites commute, so there is no \(U^2\) term. \(\mathcal Q_{0,i}\) is the summed free two-step current. The derivation uses actual bond displacements, not a discontinuous torus position. [R]

The correction is nonzero. Take one open \(z\)-bond with sites \(0,a\), each ordered \(H_0,H_1,H_2,H_3,L_0,L_1,L_2,L_3\), then the second site; label modes from zero. Its hopping is

\[
T=-\frac{i}{2a}\sum_Ac_{A,0}^\dagger\alpha_zc_{A,a}
+\frac{i}{2a}\sum_Ac_{A,a}^\dagger\alpha_zc_{A,0}.
\]

In the full 120-dimensional two-particle sector,

\[
\boxed{\langle4,7|\mathcal Q_z-\mathcal Q_{0,z}|0,9\rangle=U.}
\]

The independent CAR construction returns exactly \(+1\) for the coefficient of \(U\). The direct dipole commutator, its expansion and the bond formula agree. With \(j^E_{0\to1}=i[e_0,e_1]\), both local continuity equations and its first-moment identity return zero residuals. The separate three-site free-chain check reproduces the summed two-step convention, with all eight forward \(0\to2\) entries \(-i/4\) at \(a=1\). All 18 current comparisons returned true. [R,N]

Therefore **FAIL** applies to keeping only the free two-step current as the interacting energy current. It does not reject the converter itself: the displayed corrected current is available and passes. A subsequent flow coupling must specify the local allocation of this quartic bond correction. No interacting gravitational-source prescription is inferred here.

## 4. Hartree vanishing and its exact limits

For any species-product density matrix \(\rho_H\otimes\rho_L\) with \([\rho_A,N_A]=0\), each \(\langle\Phi_A^\mu\rangle=0\). Factorization therefore makes both the interaction mean and each partial-trace Hartree potential zero. Neither factor must be Gaussian. This proves **PASS** for the requested Hartree statement. [R]

The Gaussian first-variation extension also has an explicit test. Let \(B_{ac}=\langle l_a^\dagger h_c\rangle\), \(F^L_{ab}=\langle l_a^\dagger l_b^\dagger\rangle\), \(F^H_{dc}=\langle h_dh_c\rangle\), and \(a<b,c<d\). With the ordered tensor \(K\) defined above,

\[
\langle W\rangle_{\rm Wick}
=2\operatorname{Re}\sum_{a<b,c<d}K_{ab;cd}
\bigl(B_{ac}B_{bd}-B_{ad}B_{bc}+F^L_{ab}F^H_{dc}\bigr).
\]

Its value and full first differential vanish at \(B=F^L=F^H=0\). The exact polynomial and all 56 real-coordinate first derivatives were returned and checked at that origin. This establishes the branch restriction without a finite-difference approximation. It does not apply Wick’s theorem to a general non-Gaussian product state; that case uses the preceding selection/factorization proof. [R,N]

The exact connected dynamics is nonzero even on the product fixture:

\[
\langle H_0H_1|W|H_0H_1\rangle=0,\qquad
W|H_0H_1\rangle=4|L_0L_1\rangle-4|L_2L_3\rangle,
\]
\[
\langle W^2\rangle=32,\qquad
\langle[W,[W,N_L]]\rangle=-128.
\]

These exact returned controls give the initial interaction-induced term \(\langle N_L(t)\rangle=64U^2t^2+\cdots\), not a depletion rate. Moreover the normalized globally number-conserving Slater state

\[
|\chi\rangle=\tfrac12(H_0^\dagger+L_0^\dagger)(H_1^\dagger+L_1^\dagger)|0\rangle
\]

has returned \(\langle\chi|W|\chi\rangle=2\). It mixes heavy parity and has nonzero cross-species covariance, so it is outside the species-product branch. Total number conservation alone is insufficient. [R,N]

The exact state can change at order \(Ut\) while the specified species-diagonal one-body means have no linear response. Smooth fixed-cut entropy first variations can obey the same selection restriction; a rank-boundary entropy need not have a Taylor series. Transferring such an order statement to geometry requires regular differentiable feedback and a regular readout; a trapping-root birth fold need not respond analytically. There is no uniform-in-time, moving-cut, or unrestricted entropy/geometry equality. The empty CAR vacuum used to define these finite bases is not the filled Dirac-sea reference, and these checks do not establish unchanged interacting vacuum polarization. [R]

## Actual check completion and retained missing work

The first attempt ended on the native-integer CAR callback defect after completing stage A and retaining the original B prefix. The single missing-only continuation repaired that boundary, reused A and all returned B objects, and ended at its fixed handled deadline. No accepted stage was rerun.

| Stage | Actual state |
|---|---|
| A: spinor/discrete/mass algebra | 74 comparisons complete and true; inherited unchanged. |
| B: on-site Fock, boost, Hartree and connected controls | 80 comparisons complete and true: 14 decisions from retained residuals plus 66 new comparisons. |
| C: interacting current and free-current convention | All 31 required raw objects and 18 comparisons complete and true. |
| D: amplitudes | Seven leg branches and 15 of 557 unconditional objects returned. Eleven threshold residuals are zero; all 430 producer decisions remain missing. |

D still lacks 542 unconditional raw objects and 66 of 73 leg obligations. Eleven of its planned residuals returned, while 419 did not. Its missing Boolean-decision list correctly still contains all 430 names. File presence is complete, but amplitude raw completion and validation are false. The result remains `target_complete=false`, `status=incomplete`; the reviewer’s inspection of saved zero residuals does not alter that record.

The continuation producer ended after 94.06769008934498 s with a handled `SIGALRM`; the wrapper’s child interval was 94.41200954467058 s, maximum RSS 80276 KiB, one CPU/thread and nice 10. The fixed settings were 2 GiB, a 94 s handled signal and a 104 s external kill; returned work was preserved before the handled stop. It reported no scientific or custody errors. Terminal metadata completed, all fifteen predecessor hashes/identities were unchanged, both logs are empty and no unfinished temporary is listed. An empty scientific-error list does not validate absent comparisons.

The four scientific questions are answered above, with the numerical amplitude limitation explicit. The frozen numerical plan is not described as complete. No third allocation, wrapper retry or recomputation is proposed. Any later expressly authorized completion would have to reuse all accepted A/B/C results, the seven legs, threshold objects and eleven residuals, and compute only the missing D work. No numerical record leaves the box.

## SOURCES

SOURCES: inbox task `<internal>/hreduce.md` #97; actual paper source `<internal>/03_model.tex` (`eq:dm-fermion`, `eq:dm-current`), SHA256 `c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb`, pinned to paper commit `daaf6fbd05ff029e771f8fc1ebcd8f24ccf17dca`.

The supplied [S] proposal inputs were read in full as immutable copies, without treating their proposed rates or signed conclusions as proof:

- `<internal>/JOINT_VISION_page_mock_signed_v1.md`; `fe8648e986ce8382b22c13a3ba6b27289d49229ae32f034182e1098428253279`.
- `<internal>/ReviewA_pcq4_v1.md`; `1a4818713fb720ce41e3a90e0aa6d9a7e16c92b351d4bd718e9434371848dcb2`.
- `<internal>/ReviewA_pcq5_v1.md`; `eafb1c6487282f65bf3edd5e3416643d2b2451a3e2a6ea39e267a5578c1167b2`.
- `<internal>/ReviewB_pcq5_v1.md`; `2bacd47b093cdef79448dd5e4a6c3c4923c7829cfc5855d2ea117bb43630db9f`.
- `<internal>/ReviewB_pcq6_v1.md`; `9279dacb90acc5bd804629e6058e52765469181e3b823e6dadcdc02be6c149af`.

The full independent derivation, literal finite contract, source assessments and actual-output judgment are in `<internal>/VERTEX_gate_author_v1.tex`; the sole independent derivation/red team and comparison are in `<internal>/redteam_VERTEX_gate_v1.tex`. The immutable mathematical snapshots are `<internal>/VERTEX_gate_author_v1.tex` (`7b32632b79d26b771192a86fb721eaeb85326938ba589fb953c502a594943d7c`) and `<internal>/VERTEX_gate_missing_contract_v1.tex` (`1c002616f0711c78687163bf3dbebb9b31e22c600773c7447321d2cf8ce357eb`).

The reviewed own code is `<internal>/vertex_gate_exact_v1.py` (`50fa4a6c44b91295a59ac3d91181902aa963166bc328fca6c616ecc6a901f26b`), `<internal>/vertex_gate_missing_v1.py` (`34c21167be2acd8cea0662d380aa8fc47ded65862176e272d3b0ef2646d167fe`), and `<internal>/run_vertex_gate_missing_v1.py` (`c0137dbdd72571b325eacc00aa0147ff1244248e5a536026470f5347704c0d38`).

Actual numerical records, read in place on box `compute-host`:

| Absolute box path | SHA256 |
|---|---|
| `<internal>/spinor_algebra.json` | `af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb` |
| `<internal>/on_site.json` | `37155ac55d9ba1ce59d456afcb3bee2060c8784d42ba6060a83a1005cbc5c5f8` |
| `<internal>/vertex_gate_result.json` | `e8898fbdda68834f507b55ee94232deab6791139e225c25ef1d35ccec4c59547` |
| `<internal>/execution.json` | `4f48ecac00cb4591c85fb5911ddbf456246e4aff70148aaf8abf29e6675e8497` |
| `<internal>/on_site_completion.json` | `77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d` |
| `<internal>/bond_current.json` | `d93c8604e18ee2d7ee22d8900ba25b65bcd9e1435bb1c7380322c0d12d3ffd8f` |
| `<internal>/amplitudes.json` | `af9c99e96e10da536cdecf779e6f4f5e01a6754c1e8d3e1997aea1110a5604df` |
| `<internal>/reuse_manifest.json` | `a02609c538e8019d69cc314c054fb103abc415bc6b55eba86e777991f7a19bc8` |
| `<internal>/vertex_gate_missing_result.json` | `52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d` |
| `<internal>/execution.json` | `be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9` |

The complete custody index, including both empty continuation logs and all staged source files, is `<internal>/LARGE_FILES.tsv`. Those logical index entries do not imply local copies of numerical records.
