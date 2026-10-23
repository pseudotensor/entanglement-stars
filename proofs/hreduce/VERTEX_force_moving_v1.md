# Moving-pair momentum transfer for the two converters

**Part A FAIL and Part B FAIL for both the on-site and filtered converters. [R,N]** All 92 prescribed raw objects and all 28 numerical controls are complete, with every control passing. Every actual charge and angular normalizer is nonzero. The finite Part B point requires the explicit band-edge shell completion below. The full filtered-current derivative is an exact function of an unassigned kernel moment; that moment is not set to zero.

Scope: inbox hreduce #106 at \(ma=3/5\). This result tests the specified moving-pair response and the leading angular response under the stated secular assumptions. It does not establish a rate, a finite coherent-time identity, a production-history force bound or an interacting-reference subtraction. The manuscript is unchanged. Tags: [R] re-derived; [N] numerical with controls; [S] single source. Proposal values are readout expectations, not numerical controls.

SOURCES: [inbox #106](<internal>/hreduce.md); [pinned reviewer A proposal](<internal>/ReviewA_fevent_v1.md); [pinned reviewer B proposal](<internal>/ReviewB_fevent_v1.md); [pinned model](<internal>/03_model.tex); [frozen mathematical specification](<internal>/frozen_spec.tex); [complete author derivation and assessment](<internal>/VERTEX_force_moving_author_v1.tex); [independent peer derivation and assessment](<internal>/redteam_VERTEX_force_moving_v1.tex). Full source and box-only output identities are given below.

| Physical gate | On-site converter | Filtered converter | Domain |
|---|---|---|---|
| Part A: \(R\le1/10\) | **FAIL**, \(R=144/119\) | **FAIL**, \(R=144/119\) | Regular moving same-valley shell with finite nonzero weight |
| Part B: both collinear ratios and \(\lvert F_{\rm iso}\rvert\) at most \(1/10\) | **FAIL** | **FAIL** | Leading nonzero-weight limit, completed finite endpoint and specified principal-valley angular shell |
| Numerical controls | **PASS** | **PASS** | Exact jets and identities; controlled integration of derived angular weights |

## Operators, allocation and units

Use \(H=H_0+UV\), with \(U\) real and excluded from all stripped matrices. The heavy mass is \(m\), the light mass is zero, and

\[
h_A(p)=m_A\beta+\sum_i\alpha_i\frac{\sin(ap_i)}a,
\qquad \alpha_i=\tau_x\otimes\sigma_i,\quad
\beta=\tau_z\otimes I_2.
\]

The Pauli matrices \(\tau_i\) act on the upper/lower factor and \(\sigma_i\) on spin. Set \(Y=\sigma_y\), \(Z=\sigma_z\),

\[
(M^0,M^1,M^2,M^3)=
(\tau_z\otimes Y,\tau_y\otimes Z,i\tau_y\otimes I_2,-\tau_y\otimes\sigma_x),
\qquad \eta=(1,-1,-1,-1).
\]

For the on-site converter,

\[
\Phi^\nu_{A,x}=c_{A,x}^{T}M^\nu c_{A,x},\qquad
W_x=\sum_\nu\eta_\nu\Phi^{\nu\dagger}_{L,x}\Phi^\nu_{H,x}+\mathrm{h.c.},
\qquad V=\sum_xW_x.
\]

There is no factor \(1/2\) in the pair definition. For the filtered converter, use the same pair with \(c_A\) replaced by \(F_bc_A\), where

\[
F_b=\prod_i\frac{1+(-1)^{b_i}(T_i+T_i^\dagger)/2}{2},\qquad
f_b(p)=\prod_i\frac{1+(-1)^{b_i}\cos(ap_i)}2,
\quad b\in\{0,1\}^3.
\]

Here \((T_i u)_x=u_{x+ae_i}\). Equal filter labels are contracted. Define

\[
O_{b,xy}=\sum_\nu\eta_\nu\Phi^{\nu\dagger}_{L,b,x}\Phi^\nu_{H,b,y},\qquad
W_x=\frac12\sum_{b,y}g_{(x-y)/a}
(O_{b,xy}+O_{b,yx}+O_{b,xy}^\dagger+O_{b,yx}^\dagger).
\]

This fixes the equal-endpoint interaction-energy allocation. Retain the actual lattice overlap kernel,

\[
B_a(y)=a^{-3}\mathbf1_{[-a/2,a/2]^3}(y),\qquad
K_\ell(y)=\frac{9009}{1024\pi\ell^3}(1-|y|^2/\ell^2)_+^5,
\quad k_{\ell,a}=K_\ell*B_a,
\]
\[
g_r=a^3\int k_{\ell,a}(y)k_{\ell,a}(y-ar)\,d^3y,
\qquad \kappa(P)=\sum_{r\in\mathbb Z^3}g_r e^{iaP\cdot r}.
\]

The even finite-range kernel satisfies \(\kappa(0)=1\). Write \(\kappa_B(B)=\kappa(B\hat z/a)\); then \(\kappa_B'(0)=0\). No continuum transform replaces this lattice Gram symbol. [R; inherited operator definitions]

Normalize positive-band spinors by \(u_A^\dagger u_A=I_2\), with plane waves on \(V_s\) lattice sites. The spin-pair order is \((\uparrow\uparrow,\uparrow\downarrow,\downarrow\uparrow,\downarrow\downarrow)\). The stripped conversion tensor and spin average are

\[
\mathcal A_{t_1t_2;s_1s_2}
=4\sum_\nu\eta_\nu(J_L^\nu)^*_{t_1t_2}(J_H^\nu)_{s_1s_2},
\quad J_A^\nu=u_{A+}^{T}M^\nu u_{A-},\qquad
w=\frac14\sum_{s,t}|\mathcal A_{t;s}|^2.
\]

Thus the physical conversion matrix element is \(U\mathcal A/V_s\), with its crystal-momentum delta understood. For the filtered operator,

\[
\mathcal A_F=\kappa(P)S\mathcal A_O,\qquad
S=\sum_b\prod_{l=1}^4 f_b(p_l).
\]

The current matrix is \(\mathcal Q_1=V_s\langle LL|Q_1|HH\rangle\); the force matrix is \(\mathcal F=aV_s\langle LL|i[V,Q_0]|HH\rangle\). Restoring the leading physical force multiplies the latter by \(U/(aV_s)\).

## Moving shells and nonzero normalizers

Set \(\mu=ma=3/5\), \(B=aP_z\), \(\bar q=aq\), and \(e=aE\). Primes and dots denote \(\partial_B|_0\); additional relative derivatives are marked explicitly. A derivative in physical \(P_z\) is \(a\partial_B\); the normalized ratios are unchanged.

For collinear pairs,

\[
\theta_{H\pm}=B/2\pm h(B),\qquad
\theta_{L\pm}=B/2\pm k(B),\qquad
e_H(\theta)=\sqrt{\mu^2+\sin^2\theta},\quad e_L(\theta)=|\sin\theta|.
\]

The shell equates the sums of the two heavy and two light energies. The flow-coupled free charge and pair charges are

\[
q_z(p)=\frac{\sin(2ap_z)}{2a},\qquad
\bar q_{HH}=\sin B\cos2h(B),\qquad
\bar q_{LL}=\sin B\cos2k(B).
\]

This is the two-step energy-current charge, not a logarithm of the discrete translation operator. The actual charge derivatives are:

| Case | Base point | \(\bar q'_{HH}\) | \(\bar q'_{LL}\) | \(\Delta\bar q'\) | Signed response \(\Delta\bar q'/\bar q'_{HH}\) |
|---|---|---:|---:|---:|---:|
| Part A | \(\sin h=9/20,\ \sin k=3/4\) | \(119/200\) | \(-1/8\) | \(-18/25\) | \(-144/119\) |
| Part B leading limit | \(h\to0,\ \sin k\to3/5\) | \(1\) | \(7/25\) | \(-18/25\) | \(-18/25\) |
| Part B completed endpoint | \(\sin h=4/5,\ k=\pi/2\) | \(-7/25\) | \(-1\) | \(-18/25\) | \(18/7\) |

Every incoming normalizer was proved nonzero in the frozen specification and is nonzero in the actual returned controls. [R,N]

Part A fixes \(h\). Since \(\partial_k(e_{L+}+e_{L-})=\sqrt7/2\ne0\), the same-valley shell is smooth and even, with \(k'(0)=0\). Its actual tensor and weights are

\[
D=\operatorname{diag}(1,0,0,1),\qquad
\mathcal A_{O,A}=-\frac{24}{5}D,\quad w_{O,A}=\frac{288}{25},
\]
\[
S_A=\frac{16537+320\sqrt{2233}}{51200}>0,\quad
\mathcal A_{F,A}=S_A\mathcal A_{O,A},\quad
w_{F,A}=\frac{4519184121}{2048000000}
       +\frac{148833\sqrt{2233}}{3200000}>0.
\]

The leading Part B row uses \(r=\sin h/(e+\mu)\), with \(\partial_r\sin h|_0=2\mu\). Its accepted tensors \(\mathcal A_{O,r}=-16D\) and \(\mathcal A_{F,r}=-324D/25\) are reused. The response divides out the common leading \(r^2\) weight. At exactly \(r=0\), the amplitude vanishes and there is no probability-conditioned conversion event.

The finite endpoint does not admit the fixed-\(h\) moving shell. With \(E_{HH}^{(a)}=e_H(B/2+h)+e_H(B/2-h)\), the returned curvature gives

\[
E_{HH}^{(a)}(B)-2\cos(B/2)
=\frac{153}{1250}B^2+O(B^4)>0,
\]

while the collinear light pair is bounded above by \(2\cos(B/2)\). The declared completion of the same base point and first jet is

\[
k(B)=\pi/2,\qquad h(B)=h_0-\frac{51}{400}B^2+O(B^4).
\]

The implicit-function derivative \(\partial_h E_{HH}^{(a)}=24/25\ne0\), \(h'(0)=0\), \(h''(0)=-51/200\), and the completed second-order residual is exactly zero. This is a real local band-edge completion. It is not an interior low-energy same-valley shell and does not validate the nonexistent fixed-\(h\) family. The Part B verdict uses this completion explicitly. [R,N]

## Full current, contacts and the force sign

For the stated energy allocation, let \(D_0=\sum_x x_z e_{0,x}\), \(D_W=\sum_x x_zW_x\), and \(\mathscr D=D_0+UD_W\). The total current is

\[
Q=i[H,\mathscr D]=Q_0+UQ_1+U^2Q_2,\qquad
Q_1=i[H_0,D_W]+i[V,D_0],\qquad Q_2=i[V,D_W].
\]

The two terms of \(Q_1\) must be combined before imposing the energy shell. Their derivative-delta contacts have opposite derivative-operator coefficients. Canceling them before positive-band projection gives the regular filtered matrix

\[
\mathcal Q_1^F=\kappa_B(B)S\mathcal Q_1^O
+\mathcal A_O\sum_{l=1}^4 e_l\partial_{\theta_{lz}}w,
\qquad w=\tfrac12[\kappa_B(B_f)+\kappa_B(B_i)]S.
\]

The inherited endpoint allocation is part of this identity. Imposing an on-shell zero on the derivative-delta term before adding the commutators would lose a contact contribution. [R]

For \(d=\sin h,c=\cos h,e=\sqrt{\mu^2+d^2},C=\cos k\), the normalized heavy spinors and common-boost jets are

\[
u_{H\pm}=\binom{NI_2}{\pm LZ},\quad
N=\sqrt{\frac{e+\mu}{2e}},\quad L=\frac d{\sqrt{2e(e+\mu)}},\quad
\dot u_{H\pm}=\frac{\mu c}{4e^2}\binom{\mp LI_2}{NZ}.
\]

The collinear light spinors \(u_{L\pm}=2^{-1/2}(I_2;\pm Z)\) have zero common-boost spinor derivative in these sign patches. Their energies and phases still vary. With \(c_{A\pm}=\cos\theta_{A\pm}\), define

\[
C_A^\nu=(c_{A+}\alpha_z u_{A+})^TM^\nu u_{A-}
+u_{A+}^TM^\nu(c_{A-}\alpha_z u_{A-}).
\]

The retained contractions give

\[
J_H=(\mu Y/e,idI_2/e,-dZ/e,0),\qquad J_L=(0,iI_2,-Z,0),
\]
\[
\dot C_H=(\mu c^2Y/e^2,-idI_2,dZ,0),\qquad
\dot C_L=(0,-ieI_2,eZ,0).
\]

The leading row uses the actual cached-stack scalings \(\dot C_L=-\mu J_L\), \(\partial_r\dot C_H=-\mu J_{H,r}\). The signed eigenvector and normalization jets pass. The current contractions, including both commutators, yield \(\dot{\mathcal Q}_1^O=8dD=-e\mathcal A_{O,0}\), with the leading row additionally differentiated in \(r\). [R,N]

For the filtered current let

\[
S=\frac{(1+c^2)(1+C^2)+4cC}{8},\quad
G=\sum_l\dot e_lS_l+\frac e2\sum_{l,j}S_{lj}
=-2eS+\frac{\mu^2c}{8e}\{2C+c(1+C^2)\}.
\]

Here \(S_l=\partial_{\theta_{lz}}S\) and \(S_{lj}=\partial_{\theta_{lz}}\partial_{\theta_{jz}}S\) act on independent leg momenta before the common boost is substituted. The complete derivative is

\[
\dot{\mathcal Q}_1^F
=S\dot{\mathcal Q}_1^O+\mathcal A_{O,0}(G-2eS h_\kappa),
\qquad h_\kappa=-\kappa_B''(0)=\sum_r g_r r_z^2.
\]

The actual exact coefficients below multiply the spin matrix \(D=\operatorname{diag}(1,0,0,1)\). The leading row is \(\partial_r\partial_B\); the other rows are \(\partial_B\).

| Case | On-site current | Filtered constant coefficient | Filtered \(h_\kappa\) coefficient |
|---|---:|---:|---:|
| Part A | \(18/5\) | \(10105947/3200000+603\sqrt{2233}/10000\) | \(148833/64000+9\sqrt{2233}/200\) |
| Part B leading | \(48/5\) | \(486/25\) | \(1944/125\) |
| Part B completed endpoint | \(32/5\) | \(9876/3125\) | \(272/125\) |

All moment coefficients are nonzero. The actual record leaves \(h_\kappa\) null and retains both coefficient matrices. Thus \(\kappa_B(0)=1\) and \(\kappa_B'(0)=0\) suffice for the stated amplitude/force gate, but do not determine a scalar full-current derivative. No kernel quadrature or assignment of \(h_\kappa\) was made. [R,N]

The force is calculated by inserting the actual incoming and outgoing charges in the contraction. Its independent identity is

\[
\dot{\mathcal F}+i\Delta\bar q'\mathcal A_0=0.
\]

The actual force coefficients, in the same units and derivative order, are:

| Case | On-site coefficient of \(-iD\) | Filtered coefficient of \(-iD\) |
|---|---:|---:|
| Part A | \(432/125\) | \(446499/400000+27\sqrt{2233}/1250\) |
| Part B leading | \(288/25\) | \(5832/625\) |
| Part B completed endpoint | \(576/125\) | \(2448/3125\) |

All six force coefficients are nonzero and all six Ward residuals vanish exactly. These moving-pair checks therefore test the commutator sign at nonzero transfer. The raw Part A filtered Ward matrix retains explicitly encoded zero entries; its comparison residual is an empty zero matrix. The original records are unchanged. [N]

## Event definition and the two gates

For a coherent transition with \(\omega=E_f-E_i\), \(z=V_{fi}(Q_1)_{if}\), let \(P_{fi}(t)\) be the leading conversion probability and \(B_{fi}(t)\) the \(O(U^2)\) interaction-current contribution from \(UQ_1\). The latter is distinct from the boost coordinate \(B=aP_z\). The inherited second-order expression is

\[
\frac{B_{fi}(t)}{P_{fi}(t)}
=-\frac{\omega\operatorname{Re}z}{|V_{fi}|^2}
+\frac{\omega\cot(\omega t/2)\operatorname{Im}z}{|V_{fi}|^2},
\qquad P_{fi}(t)\ne0.
\]

It is not generally zero in a finite coherent system. The adopted event definition takes the continuum before the weak-coupling kinetic limit, assumes regular matrix elements and integrable band weights with an existing principal value, and subtracts nonconverting initial dressing. Under those conditions the leading secular bond share vanishes and \(\delta=\Delta q\). The initial \(U^2Q_2\) term is subtracted; its conversion change lies beyond the leading event count. The common \(U^2\) cancels in the ratio. A finite regular \(h_\kappa\) does not alter this conditional secular conclusion. [R]

Part A uses \(R=|\Delta\bar q'|/\bar q'_{HH}=144/119\), for each converter. Part B tests the leading ratio \(-18/25\), the completed finite ratio \(18/7\), and the isotropic angular ratio below. Their absolute values exceed \(1/10\). The nonzero weights cancel only in these conditional ratios; no rate or branch population is inferred.

## Angular measure, amplitude weight and convergence

Let \(\rho=|a\mathbf d_H|\to0\), where \(d_{H,i}=\sin(ap_{H,i})/a\), with incoming direction \(u\). Write the principal-valley outgoing shell as \(\sin\theta_{L,i}=\mu n_i\), \(n\in S^2\), and \(c_i=\sqrt{1-\mu^2n_i^2}\ge4/5\). Its coarea measure is

\[
d^3\theta_L\,\delta(2e_L-2\mu)
=\frac{\mu^2}{2c_xc_yc_z}\,d\Omega(n).
\]

The radial shell continuation is regular since \(\partial_t E_{LL}=2\ne0\) at \(\sin\theta_i=tn_i,t=\mu\). It has zero first radial derivative in the common boost. [R]

Write \(\mathcal A=\rho\mathcal A^{(1)}+O(\rho^3)\). The angular spin trace is derived from \(T_j=-iY\sigma_j\) and

\[
\mathcal A^{(1)}(n,u)=-\frac4\mu
\sum_{b,d}[(n\cdot u)\delta_{bd}-u_bn_d]
\overline{T_b}\otimes_{\rm ent}T_d,\qquad
\operatorname{Tr}(T_b^\dagger T_d)=2\delta_{bd}.
\]

Here \((\overline T_b\otimes_{\rm ent}T_d)_{t_1t_2;s_1s_2}
=\overline{(T_b)_{t_1t_2}}(T_d)_{s_1s_2}\); this is the amplitude index order. Consequently

\[
\frac14\sum_{s,t}|\mathcal A|^2
=\frac{16\rho^2}{\mu^2}[1+(u\cdot n)^2]+O(\rho^4).
\]

Only the incoming \(000\) filter survives at this order. With \(f_0=\prod_i[(1+c_i)/2]\), the filtered amplitude has factor \(f_0^2\), and its squared weight has factor \(f_0^4\). The actual \(\kappa(0)=1\) is retained. These leading weights are positive in every direction. [R; inherited collinear normalization checked numerically]

Define

\[
w_O=\frac{\mu^2}{2c_xc_yc_z},\quad w_F=w_Of_0^4,
\quad L_z=1+n_z^2,\quad L_{\rm iso}=4/3,
\]
\[
\mathcal D_{X,j}=\int w_XL_j\,d\Omega,\quad
\mathcal N_{X,j}=\int w_XL_j(-2\mu^2n_z^2)\,d\Omega,\quad
F_{X,j}=\mathcal N_{X,j}/\mathcal D_{X,j}.
\]

The isotropic factor averages the incoming direction analytically. The common leading \(16\rho^2/\mu^2\) cancels; this is a leading response rather than a rate at the absent threshold event. The nonzero denominator bounds are

\[
\mathcal D_{O,z}\ge18\pi/25,\qquad
\mathcal D_{O,\mathrm{iso}}\ge24\pi/25,\qquad
\mathcal D_{F,j}\ge(9/10)^{12}\mathcal D_{O,j}^{\rm lower}>0.
\]

The returned finest-grid values are:

| Converter and incoming directions | Denominator | Numerator | Response |
|---|---:|---:|---:|
| On-site, axial \(u=\hat z\) | 3.697425993007294 | −1.0658214478040744 | −0.28826038704217327 |
| On-site, isotropic | 3.6974259930072937 | −0.8873822383217504 | −0.24 |
| Filtered, axial \(u=\hat z\) | 2.492544747435713 | −0.7175922478253013 | −0.2878954323943623 |
| Filtered, isotropic | 2.492544747435713 | −0.598210739384571 | −0.23999999999999996 |

The three fixed grids use \(16,32,64\) Gauss-Legendre nodes in \(n_z\) and \(4n\) midpoint azimuths. All nodes, weights, directions, cosines, coarea factors, filters, weighted measures, charge responses, moments and refinements are retained. Both refinements pass the \(10^{-8}\), \(10^{-10}\) scaled tolerances on all 20 moments and all four ratios. Their largest absolute moment changes are \(2.220446049250313\times10^{-15}\) and \(4.440892098500626\times10^{-16}\); the ratio changes are at most \(4.996003610813204\times10^{-16}\) and \(1.1102230246251565\times10^{-16}\). [N]

Uniform monomials, finite/positive weights, the sphere and light-shell identities, \(c_i\ge4/5\), and the denominator lower bounds all pass. The separately accumulated normalized second moments agree across isotropic axes within \(1.3877787807814457\times10^{-17}\); their axial \(x-y\) differences are at most \(2.7755575615628914\times10^{-17}\), both below the fixed \(2\times10^{-11}\) tolerance. These controls divide by the corresponding angular denominators. Axial \(z\) moments were not forced equal to the transverse ones. Cubic symmetry independently gives the isotropic ratio \(-2\mu^2/3=-6/25\); this value was not a quadrature control. [R,N]

The numerical credit is controlled integration of the independently derived trace weights. It is not an independent angular-spinor reconstruction or an integral over unselected valleys. The refinement evidence is empirical, with no certified remainder or all-digit enclosure. The gate failures are separate from the passing controls and from the printed precision.

## Evidence, provenance and limits

The complete finite calculation returned 92 raw objects and 28 true decisions, with empty missing lists, no operational error and an owned terminal child. Its exact scalars are rational combinations of the 16 real basis elements generated by \(\sqrt2,\sqrt5,\sqrt7,\sqrt{319}\), with separate real and imaginary coefficients. The angular arrays use binary64. The execution records one CPU/thread, nice ten and a two-GiB memory ceiling. All thirteen input/source identities match their terminal postchecks; both logs are empty. All 264 complete journal records remain on the box.

The author and sole peer independently derived the formulas and independently assessed the returned records. The numerical route shares the explicitly admitted component matrices, spinors, finite/leading amplitudes and filter gradients from the accepted predecessor chain. It does not regenerate those operands or supply an independent state evolution. The angular trace-weight derivation and its numerical integration are distinct kinds of evidence. No independent numerical re-execution is claimed, and the older broad amplitude plan remains incomplete.

The physical outcome retires the declared per-event clearance route at these benchmarks. It does not decide whether the force can be neglected on an evolving physical branch. That requires its rate-weighted, reference-consistent contribution, including inverse conversion and blocking, compared with the actual source balance. No production source-free clearance, unitary-lead result, general momentum no-go or history bound is supplied. The full filtered-current derivative remains parameterized by \(h_\kappa\); no averaging-width independence or chosen numerical kernel moment is inferred.

The literal implementation and provenance sources are:

- [Executed producer](<internal>/vertex_force_moving_v1.py), SHA256 `a21181d62b67caf7789dc7f15e8de4843d054775735d92a54efd4119464003d0`.
- [Executed wrapper](<internal>/run_vertex_force_moving_v1.py), SHA256 `50f85cf9290c988496d8c0b630cc062723bd01de41871475bd1753f7e473b2af`.
- [Frozen specification](<internal>/frozen_spec.tex), SHA256 `2ffc93746466b9efd176cffd1297b7b999e1420057dec43851aa3e87095a0439`.
- [Accepted force-event definitions and allocation](<internal>/VERTEX_force_event_v1.md), SHA256 `afd8faf0521ba288872347aa7e36fb7bc6c0396d0a37079c5fccec4589e5bd1e`.
- [Filtered operator and kernel derivation](<internal>/VERTEX_gate_filtered_v1.md); [amplitude normalization and accepted derivatives](<internal>/VERTEX_amplitudes_v1.md); [stress and force scope](<internal>/VERTEX_stress_v1.md).

The six admitted box predecessors are `<internal>/spinor_algebra.json`, `<internal>/admission.jsonl`, `<internal>/onsite.jsonl`, `<internal>/filtered.jsonl`, `<internal>/vertex_force_event_result.json`, and `<internal>/execution.json`. Their exact primary selections, nested original pointers, source hashes and descriptor joins are preserved in the admission journal and frozen specification.

The following files are on box `compute-host`; no numerical file was copied into the worktree. The [owned bulk index](<internal>/LARGE_FILES.tsv) records the same paths and identities.

| Actual box path | SHA256 |
|---|---|
| `<internal>/admission.jsonl` | `90749827af6e72e60bab107ba499e421547e79d64105282194279ed77598aa14` |
| `<internal>/exact.jsonl` | `f55eb2df68b2934bf4231c1ee212e71a8099cd0cc9c6b8bef3950f49c26d3f5e` |
| `<internal>/quadrature.jsonl` | `a3f6459f118ad8f479fd35b2f7979dabe094f647d0d79dd8f850d0bd39b46971` |
| `<internal>/vertex_force_moving_result.json` | `25dd8379f1a49cd3ce65fa187c606b45f2033c8261b44fba219989c76364fa85` |
| `<internal>/execution.json` | `d93bb6884cf9ea52b23fcdbb32aedd23620ba13c2ac788209f7ec312fa2758a6` |
| `<internal>/stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `<internal>/stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
