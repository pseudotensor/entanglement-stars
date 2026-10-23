# Momentum transfer at the retained conversion points

**PASS for the named exact matrix checks; the normalized source-free criterion is not evaluable. [R,N]** All 54 specified raw objects and all 21 decisions are complete, with every decision true. Both converters give zero pair-charge difference and zero full first-order current at the retained points. Every incoming pair charge is also zero, so no defined fraction tests the declared bound \(1/10\). The source-free approximation receives no clearance from this result.

Scope: inbox hreduce #104, for the on-site axial-pair converter and the actual filtered, kernel-averaged converter at \(ma=3/5\). The calculation uses the named threshold, finite collinear point, relative-momentum derivative and \(c=001\) intervalley point. It reuses accepted spinors, amplitudes and current jets. The derivative is not a finite event; a vanishing amplitude supplies no probability-conditioned event. The result supplies no moving-pair test, rate history, interacting-vacuum subtraction or general vanishing-force claim. The manuscript is unchanged. Tags: [R] re-derived; [N] numerical with controls; [S] single source. Inherited numerical evidence is identified explicitly.

SOURCES: [inbox #104](<internal>/hreduce.md); [pinned reviewer A proposal](<internal>/ReviewA_stress_v1.md); [pinned reviewer B proposal](<internal>/ReviewB_stress_v1.md); [pinned model source](<internal>/03_model.tex); [frozen mathematical specification](<internal>/frozen_spec.tex); [frozen missing-only contract](<internal>/force_event_missing_contract_v1.tex); [complete author derivation and assessment](<internal>/VERTEX_force_event_author_v1.tex); [independent peer assessment](<internal>/redteam_VERTEX_force_event_v1.tex). Accepted predecessors and box-only output identities are given below. Proposal expectations are not known-answer controls.

| Requested quantity | Result | Disposition |
|---|---|---|
| Pair-charge difference \(\Delta q_z\) | Zero at all retained finite points; zero relative derivative | **PASS [R,N]** |
| Full first-order current \(Q_{1,z}\) | Zero matrices at all four retained rows, with nonzero insertion and gradient controls | **PASS [R,N]**, for the declared allocation and on-shell domain |
| Force identity residual | Exactly zero for both converters | **PASS [R,N]**; zero-transfer checks do not test the sign away from zero |
| Leading bond share and event transfer | Zero where a finite nonzero transition exists; derivative and absent-event rows remain unassigned | **PASS for the stated conditional definition [R]**, using the controlled current elements |
| Normalized source-free bound | Every denominator \(q_{HH,z}=0\) | **UNASSESSED**; no source-free clearance |
| Baseline coupling | \(aU=G/a^2\); decimal not fixed by \(ma\) | **PASS as a sourced relation [S,R]**; no numerical physical baseline |

## Operators, allocation and normalization

Use \(\hbar=c=1\), \(H=H_0+UV\), \(U=\lambda_X/a^3\in\mathbb R\), and \(V=\sum_xW_x\). The heavy mass is \(m\), the light mass is zero, and the free one-body Hamiltonian is \(h_A=m_A\beta+\boldsymbol\alpha\cdot\mathbf d\), with \(d_i=\sin(ap_i)/a\). In the inherited component basis,

\[
\alpha_i=\tau_x\otimes\sigma_i,\qquad \beta=\tau_z\otimes I_2,
\qquad \eta=\operatorname{diag}(1,-1,-1,-1),
\]
\[
M^0=\tau_z\otimes\sigma_y,\quad M^1=\tau_y\otimes\sigma_z,
\quad M^2=i\tau_y\otimes I_2,\quad M^3=-\tau_y\otimes\sigma_x.
\]

The Pauli matrices \(\tau_i,\sigma_i\) act on the upper/lower and spin factors. The shift convention is \((T_j u)_x=u_{x+ae_j}\).

The on-site definition is

\[
\Phi_A^\mu=c_A^TM^\mu c_A,\qquad
W_x=\sum_\mu\eta_{\mu\mu}\Phi_{L,x}^{\mu\dagger}\Phi_{H,x}^\mu+\mathrm{h.c.},
\tag{1}
\]

with no factor \(1/2\) in \(\Phi\). For the filtered converter, set

\[
F_b=\prod_j\frac{1+(-1)^{b_j}(T_j+T_j^\dagger)/2}{2},\quad
\Phi_{A,b,x}^\mu=(F_bc_A)_x^TM^\mu(F_bc_A)_x,
\quad b\in\{0,1\}^3,
\]
\[
O_{b,xy}=\sum_\mu\eta_{\mu\mu}\Phi_{L,b,x}^{\mu\dagger}\Phi_{H,b,y}^\mu,
\quad W_x=\frac12\sum_{b,y}g_{(x-y)/a}
(O_{b,xy}+O_{b,yx}+O_{b,xy}^\dagger+O_{b,yx}^\dagger).
\tag{2}
\]

Thus the interaction energy is split equally between the pair endpoints. Retain the actual lattice overlap kernel,

\[
B_a(y)=a^{-3}\mathbf1_{[-a/2,a/2]^3}(y),\qquad
K_\ell(y)=\frac{9009}{1024\pi\ell^3}(1-|y|^2/\ell^2)_+^5,
\quad k_{\ell,a}=K_\ell*B_a,
\]
\[
g_r=a^3\int k_{\ell,a}(y)k_{\ell,a}(y-ar)\,d^3y,
\qquad \kappa(P)=\sum_{r\in\mathbb Z^3}g_r e^{iaP\cdot r}.
\tag{3}
\]

Its evenness and cell partition give \(\kappa(0)=1\) and \(\partial_{P_z}\kappa(0)=\partial_{P_z}\kappa(\pi\hat z/a)=0\). The intervalley value \(\kappa_\pi=\kappa(\pi\hat z/a)\) remains unspecified. No kernel quadrature or continuum replacement is used. [R; inherited kernel scope]

Normalize positive-band spinors by \(u_A^\dagger u_A=I_2\) and plane waves on \(V_s\) lattice sites. The spin-pair order is \((\uparrow\uparrow,\uparrow\downarrow,\downarrow\uparrow,\downarrow\downarrow)\). Stripping \(U/V_s\) and the crystal-momentum Kronecker delta from the conversion matrix element gives

\[
\mathcal A_{t_1t_2;s_1s_2}=4\sum_\mu\eta_{\mu\mu}
(J_L^\mu)^*_{t_1t_2}(J_H^\mu)_{s_1s_2},\qquad
J_A^\mu=u_{A1}^TM^\mu u_{A2}.
\tag{4}
\]

The factor four follows from the canonical anticommutation relations and the pair normalization in (1). For the filtered tensor, \(\mathcal A_F=\kappa(P)S\mathcal A\), where

\[
f_b(p)=\prod_j\frac{1+(-1)^{b_j}\cos(ap_j)}2,
\qquad S=\sum_b\prod_{l=1}^4f_b(p_l).
\tag{5}
\]

The current raw matrix is \(\mathcal Q_1=V_s\langle f|Q_1|i\rangle\). The force and identity-residual raw matrices are in units \(aV_s\langle f|i[V,Q_0]|i\rangle\); multiplying by \(U/(aV_s)\) restores the leading physical force. Relative derivatives use the same stripped convention.

## Retained kinematics and exact charge differences

The free contribution to the flow-coupled momentum is the two-step energy-current charge,

\[
q_z(p)=\frac{\sin(2ap_z)}{2a}=\frac{\sin(ap_z)\cos(ap_z)}a,
\qquad \Delta q_z=q_{L1,z}+q_{L2,z}-q_{H1,z}-q_{H2,z}.
\tag{6}
\]

The following signed pairs are ordered as \((\cos ap_z,\sin ap_z)\). All transverse momenta are zero.

| Label | Incoming heavy legs | Outgoing light legs | Energy of each particle |
|---|---|---|---|
| Same-valley threshold | \((1,0),(1,0)\) | \((4/5,3/5),(4/5,-3/5)\) | \(3/(5a)\) |
| Finite collinear point | \((3/5,4/5),(3/5,-4/5)\) | \((0,1),(0,-1)\) | \(1/a\) |
| Intervalley \(c=001\) | \((1,0),(-1,0)\) | \((4/5,3/5),(-4/5,3/5)\) | \(3/(5a)\) |

The slope differentiates the opposite-pair family in \(r=d/(E+m)\) at \(r=0\), with total pair momentum zero. It is not a derivative of a common pair boost. Its heavy sine derivatives are \(+6/5,-6/5\); cosine and energy derivatives vanish. The light spinor directions remain fixed.

The returned charge matrix, with columns \((aq_{H1},aq_{H2},aq_{L1},aq_{L2},aq_{HH},aq_{LL},a\Delta q)\) and rows threshold, collinear, relative derivative, intervalley, is

\[
\begin{pmatrix}
0&0&12/25&-12/25&0&0&0\\
12/25&-12/25&0&0&0&0&0\\
6/5&-6/5&0&0&0&0&0\\
0&0&12/25&-12/25&0&0&0
\end{pmatrix}.
\tag{7}
\]

The third row contains derivatives, not finite single-particle charges. At the three finite points, the returned individual group velocities and the two pair means are

\[
\begin{pmatrix}
0&0&4/5&-4/5&0&0\\
12/25&-12/25&0&0&0&0\\
0&0&4/5&-4/5&0&0
\end{pmatrix}.
\tag{8}
\]

Both exact controls pass. [R,N] At the mixed point, the outgoing sine momenta have the same sign but their cosine signs differ. The charges cancel. Distinct crystal modes are retained even when their spinor matrices coincide.

## Full current, force identity and nonzero intermediate controls

Let \(D_0=\sum_xx_ze_{0,x}\) and \(D_W=\sum_xx_zW_x\), with \(e_{0,x}=d\Gamma(\{\Pi_x,h_0\}/2)\) and (1) or (2). Here \(\Pi_x\) projects onto a site and \(d\Gamma\) is the number-conserving many-body lift, summed over species. The corrected momentum is

\[
Q_z=Q_{0,z}+UQ_{1,z}+U^2Q_{2,z},\qquad
Q_{1,z}=i[H_0,D_W]+i[V,D_0],\quad Q_{2,z}=i[V,D_W].
\tag{9}
\]

The quadratic term vanishes on-site and is generally present for the filtered converter. It is not a global operator identity that \(Q_1=i[V,D_0]\). For regular normalizable free eigenvectors on a defined dipole domain, the other commutator is proportional to \(E_f-E_i\). For plane waves its derivative of the momentum delta must be combined with the contact from the second commutator before imposing the energy shell.

For the endpoint allocation (2), the off-shell scalar multiplying the component pair tensor is

\[
w(F,I)=\frac{\kappa(P_f)+\kappa(P_i)}2\sum_b\prod_lf_b(p_l).
\tag{10}
\]

Canceling the derivative-delta terms first gives the regular stripped current

\[
\mathcal Q_1^F=\kappa(P)S\mathcal Q_1^O
+\mathcal A\sum_{l=1}^4(aE_l)\partial_{\vartheta_{lz}}w,
\qquad \vartheta_l=ap_l,
\tag{11}
\]
\[
C_A^\mu=(c_{A1}\alpha_z u_{A1})^TM^\mu u_{A2}
+u_{A1}^TM^\mu(c_{A2}\alpha_z u_{A2}),\qquad c_{Al}=\cos(ap_{Al,z}),
\]
\[
(\mathcal Q_1^O)_{t_1t_2;s_1s_2}
=2\sum_\mu\eta_{\mu\mu}
\left[(C_L^\mu)^*_{t_1t_2}(J_H^\mu)_{s_1s_2}
+(J_L^\mu)^*_{t_1t_2}(C_H^\mu)_{s_1s_2}\right].
\tag{12}
\]

The scalar derivatives in (11) do not differentiate external spinors. The factor two in (12) combines the pair factor four with the symmetric energy allocation. The zero pair group velocities in (8) justify the qualified on-shell reduction at these points; no \((E_f-E_i)\delta'\) term is discarded without its contact. [R]

Writing \(Y=\sigma_y\), \(Z=\sigma_z\), the new heavy insertion stacks are \((0,0,0,2Y)\) at threshold and \((0,0,0,18Y/25)\) at the collinear point. The three light insertion stacks and the heavy relative derivative are zero. At the mixed point, the admitted heavy jet is reused as \(C_H=(0,2iI,-2Z,0)\); the new \(J_L^1,J_L^2\) vanish and the admitted \(J_L^0=Y\). These nonzero intermediate values give \(\mathcal Q_1^O=0\) for every retained row. [R,N]

For the finite filtered point, only the \(000,001\) leg-filter rows survive:

\[
(4/5,4/5,1/2,1/2),\qquad(1/5,1/5,1/2,1/2),
\quad S=17/100,\quad \mathcal A_F=-\frac{136}{125}D,
\quad D=\operatorname{diag}(1,0,0,1).
\tag{13}
\]

This is the newly evaluated finite point at \(ma=3/5\); the accepted filtered finite example at \(ma=3/10\) is not relabelled. For the gradient of the four-leg product, the nonzero threshold row is \(000=(0,0,-27/100,27/100)\). The nonzero collinear rows are

\[
000=(-2/25,2/25,-4/25,4/25),\qquad
001=(1/50,-1/50,1/100,-1/100).
\tag{14}
\]

All other entries and all mixed-point gradients vanish. The three energy-weighted gradient sums in (11) are zero. Analytically, this cancellation holds along the opposite-pair family, so its relative derivative also vanishes. The mixed outgoing filter products are \(9/100\) in the first two rows, while the admitted incoming products are zero. Thus the filtered current is zero in all four cases, with its nonzero gradients explicitly retained. [R,N]

The exact first-order force identity is

\[
\langle f|i[V,Q_{0,z}]|i\rangle=-i\Delta q_zV_{fi}.
\tag{15}
\]

The full first-order force is \(i([H_0,Q_1]+[V,Q_0])\); its regular first term vanishes on shell. The returned force matrices and the identity residuals are zero for both converters, with the product rule retained in the slope row. The entrywise interference matrices \(\mathcal A\odot\overline{\mathcal Q_1}\), with \(\odot\) denoting entrywise multiplication, also vanish. All controls pass. The zero-transfer checks do not numerically determine the commutator sign at nonzero transfer; that sign follows from (15). [R,N]

For the filtered mixed row, the tensor, current, force and residual records mean coefficients of the unassigned real \(\kappa_\pi\); the interference is its squared-factor coefficient. All those coefficients are zero. No choice \(\kappa_\pi=1\) is made.

## Per-event definition fixed by second-order perturbation theory

Put \(\omega=E_f-E_i\), \(z_{fi}=V_{fi}(Q_1)_{if}\), and prepare a free energy eigenstate without imposed species coherence. With a constant interaction, the relative transition amplitude and probability at leading order are

\[
b_f(t)=UV_{fi}\frac{e^{-i\omega t}-1}{\omega},\qquad
P_{fi}(t)=\frac{2U^2|V_{fi}|^2(1-\cos\omega t)}{\omega^2}.
\tag{16}
\]

The bond-momentum contribution at the same order is

\[
B_{fi}(t)=\frac{2U^2}{\omega}
\left[(\cos\omega t-1)\Re z_{fi}+\sin\omega t\Im z_{fi}\right],
\]
\[
\frac{B_{fi}(t)}{P_{fi}(t)}
=-\frac{\omega\Re z_{fi}}{|V_{fi}|^2}
+\frac{\omega\cot(\omega t/2)\Im z_{fi}}{|V_{fi}|^2},
\qquad P_{fi}(t)\ne0.
\tag{17}
\]

The common \(U^2\) cancels. At exact degeneracy the bond share is \(2\Im z_{fi}/(t|V_{fi}|^2)\). A coherent finite-system transition therefore has no universal constant bond share in general. Here the full returned current and interference vanish, so its leading bond share is zero wherever a nonzero transition probability exists. The charge transfer conditioned on that transition is \(\Delta q_f+B_{fi}/P_{fi}=0\). [R; current elements N]

For a secular scattering definition, take a continuum of final states before the weak-coupling kinetic limit. Assume a smooth locally integrable density, integrable finite-band weights, regular \(V,Q_1\) matrix elements and an existing principal-value contribution; subtract the nonconverting initial dressing. Then

\[
\frac1t\int P_{fi}(t)\,df\longrightarrow
2\pi U^2\int |V_{fi}|^2\delta(\omega)\,df,
\qquad \frac1t\int B_{fi}(t)\,df\longrightarrow0.
\tag{18}
\]

Thus the **conditional leading secular bond weight is zero**, and \(\delta_f=\Delta q_f\). This weight was derived and fixed in the frozen specification before the computation. The filtered \(U^2Q_2\) has an initial diagonal expectation in a free eigenstate; after that subtraction its conversion-dependent change lies beyond this leading event count. Equation (18) is not a numerical rate calculation or a proof of the kinetic assumptions for a production state. [R]

## What the event table does and does not decide

The amplitude labels below are inherited accepted tensors, except the new filtered collinear tensor (13). Let \(T_{001}\) have the \(2\times2\) block \(\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\) in zero-based spin rows and columns \(1,2\), with all other entries zero.

| Retained row | On-site tensor | Filtered tensor | Event and transfer |
|---|---|---|---|
| Same-valley threshold | \(0\) | \(0\) | No conversion event at this exact point; no event-conditioned fraction |
| Finite collinear point | \(-(32/5)D\) | \(-(136/125)D\) | Nonzero named spin transitions; leading bond share and conditional secular \(\delta_f\) zero |
| Relative-\(r\) slope at zero | \(-16D\) | \(-(324/25)D\) | Derivative coefficients, not finite events |
| \(c=001,\mathbf n=\hat z\) intervalley | \(4T_{001}\) | \(0\) for every \(\kappa_\pi\) | On-site named spin transitions have zero conditional transfer; filtered event absent |

Every \(q_{HH,z}\) is zero, including its relative-family derivative. All eight normalized entries therefore remain null. The exact status is `not_evaluable_zero_denominator`, with no eligible same-valley normalized event and `source_free_clearance=false`. **The one-tenth criterion is UNASSESSED on this retained set.** It is neither a vacuous PASS nor a measured inequality failure.

The proposed \(-0.72\) does not describe these kinematics. Analytically, \(-2(ma)^2\) belongs to a different near-rest family with incoming \(P/2,P/2\), outgoing \(P/2\pm q(P)\), and a derivative in total pair momentum as \(P\to0\). The retained slope differentiates relative momentum at zero total momentum. No such moving-pair case is added or used as a numerical control here.

For \(p=\pi b/a+k\), equation (6) gives \(q_z(p)=\sin(2ak_z)/(2a)\) exactly. The conditional leading secular transfer depends on the residual momenta, and the retained mixed outgoing residuals are opposite. This statement concerns that charge and event definition; it does not make intervalley amplitudes, higher-order dressed charges or all interacting forces valley independent. The nonzero stress and force witnesses in the accepted stress result remain valid.

## Baseline coupling

The signed vision and section two of the pinned coefficient proposal declare \(\lambda_X=\xi_XG\) with \(\xi_X=1\). Therefore

\[
aU=\frac{\lambda_X}{a^2}=g,\qquad g=\frac{G}{a^2},\qquad
(aU)^2=g^2.
\tag{19}
\]

At the conditional entropy-matched spacing \(a=2\sqrt{\eta_{\rm ent}}\,\ell_P\), this becomes \(aU=1/(4\eta_{\rm ent})\), squared \(1/(16\eta_{\rm ent}^2)\). Here \(\eta_{\rm ent}\) is the entropy-area coefficient and \(\ell_P^2=G\). The separately declared matched-lattice control \(\lambda_X=a^2\) gives \(aU=1\). These are definition-dependent relations. Fixing \(ma=3/5\) does not fix \(G/a^2\), so the physical decimal baseline remains null. No production force or interacting-sea subtraction follows from (19). [S for the coefficient declaration; R for the conversion of units]

## Evidence, reuse and exact identities

The complete box-only record contains 108 admission records and 138 scientific records: 54 raw objects plus separately retained actual, expected, residual and decision parts for 21 comparisons. The kinematic, on-site and filtered groups have respectively \(5/2\), \(23/8\) and \(26/11\) raw/decision counts. Every residual is zero and every Boolean is true. All requested objects and comparisons are present; the terminal process is reaped with return code zero and no operational error. All 32 input identities agree across admission and terminal postcheck.

The accepted primary matrices, seven spinors, threshold currents, collinear and intervalley tensors, heavy current jets and filtered threshold/slope controls were read through explicit cache pointers. They were not recomputed. The new current insertions, gradient contacts, finite filtered tensor and charge/force readouts are the missing products. These are shared-primary exact algebraic controls, not independent state evolutions. The broader amplitude D plan remains incomplete.

The immutable `<internal>/run_v1` record contains a schema-order admission failure before any new scientific value. The changed computation preserves that entire record and admits the same 70 unique prior raw names by membership, retaining their actual order and the ordered 24 decisions. It writes exclusively to `<internal>/continuation_v1`. No accepted product was replayed and no failed record was replaced.

The original author and sole independent peer derived the targets, read both complete actual sources, and independently read all complete outputs before comparing their judgments. The new exact numerical cut is accepted; the physical normalization limitation is retained. No further numerical allocation is needed for this result.

The five immutable definition inputs are under `<internal>/inputs`:

| File | SHA256 |
|---|---|
| `03_model.tex` | `c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb` |
| `ReviewA_stress_v1.md` | `c46038400a7d07b77f5d4003bf10ce6bae950ea057640c272cdfded3b66c2e62` |
| `ReviewB_stress_v1.md` | `9e04db48aa229c0d7cdf23d69abc1f172bf188f04278a8c63f37636697c238f8` |
| `ReviewA_pcq5_v1.md` | `eafb1c6487282f65bf3edd5e3416643d2b2451a3e2a6ea39e267a5578c1167b2` |
| `JOINT_VISION_page_mock_signed_v1.md` | `fe8648e986ce8382b22c13a3ba6b27289d49229ae32f034182e1098428253279` |

The exact executed files and contract are:

- `<internal>/vertex_force_event_missing_v1.py`; SHA256 `4d539308534e8608f9cad62b8b0e85fd729d9677dc9f32913f7b220763616950`.
- `<internal>/run_vertex_force_event_missing_v1.py`; SHA256 `e212483f507a24bcbbb90b26c1b44f13df6b3b1494eb2469bfc7b523a6bcfd93`.
- `<internal>/force_event_missing_contract_v1.tex`; SHA256 `de1571fb0874920e88d6c69e0794beb48dceb8cb2a76fb14f31a1f5bcd2a4f35`.
- `<internal>/frozen_spec.tex`; SHA256 `f15fb7580caa666bf33c2f0dd337b500293633530ba4a6e97974d7593fc1b1c9`.

All raw files remain on box `compute-host`, with the common full directory `<internal>/continuation_v1`:

| File in that directory | Bytes | SHA256 |
|---|---:|---|
| `admission.jsonl` | 121157 | `9c740a48b51f412c05a1a1395a54f663586c3a0d4676fc1781b0445f6cef87e3` |
| `kinematics.jsonl` | 18848 | `c4a7378a012e523d83f437101a6b407b9ffd20a3f99cc2e25a60ce3aacf08710` |
| `onsite.jsonl` | 52709 | `d9b15c34b3a520ed4577dec3861c1c47a6c83520b6c6cb75abbb1ca24879c253` |
| `filtered.jsonl` | 35744 | `f5e68a408c9801b1512051d5fbd141d1c08e6df46aabb5bbaeb05c569c984253` |
| `vertex_force_event_result.json` | 45420 | `9e266b89ebf74fda3244f7133df4311fe6e803648db429466c07cc94661fe48a` |
| `execution.json` | 35750 | `32bec6929b678b95a9a8ea9465151f1d514ac91258a59f2f52ea08ad629729ee` |
| `stdout.log`, `stderr.log` | 0 each | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The exact cache selections and failed-run links are retained in the admission journal. Their accepted interpretations are in `<internal>/VERTEX_gate_v1.md`, `<internal>/VERTEX_gate_filtered_v1.md`, `<internal>/VERTEX_amplitudes_v1.md` and `<internal>/VERTEX_stress_v1.md`. The box records are indexed in `<internal>/LARGE_FILES.tsv`.
