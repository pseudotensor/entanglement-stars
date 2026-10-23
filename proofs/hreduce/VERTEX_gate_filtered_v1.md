# VERTEX_gate_filtered_v1

Scope: inbox hreduce #99, the three-dimensional finite-range, kernel-averaged axial-pair converter constructed below. The proposal inputs supply ideas and definitions, not proof of the verdicts. This is a candidate extension of the free sine-fermion declaration in the pinned paper at commit 98f8f81ecce7c01feec55d365c4a8045ae9f3578. No manuscript was changed. The independent author and original sole peer derived the gate, compared their derivations, read the complete actual implementation, and independently read all returned records before the sole output comparison.

SOURCES: [frozen mathematical contract](<internal>/frozen_spec.tex), [complete author record](<internal>/VERTEX_gate_filtered_author_v1.tex), [independent peer record](<internal>/redteam_VERTEX_gate_filtered_v1.tex), and box compute-host:<internal>/vertex_gate_filtered_result.json. Full absolute source and raw-output identities appear below.

The single new finite exact attempt completed: all 117 prescribed raw objects and all 45 decisions returned, with every decision passing. Those checks support the following distinct physical conclusions.

| Item | Verdict | Exact scope |
|---|---|---|
| 1. Operator, symmetries, locality and leading mixed-valley obstruction | **PASS, scoped** | The specified averaged pair converter has the stated exact symmetries and finite range. Its leading bare cone-centre vertex removes the offending \(32v_3\) channel. Finite-momentum leakage survives; no all-orders common-velocity or Lorentz claim is certified. |
| 2. Threshold and first nonzero amplitudes | **PASS** | Threshold zeros, the specified collinear first derivatives, and both finite same/cross-valley amplitudes pass the complete new exact plan. These are normalized amplitudes, not rates or histories. |
| 3. Unchanged free summed two-step current | **FAIL** | The interacting current contains \(U\mathcal Q_1+U^2\mathcal Q_2\). The quadratic term has an explicit nonzero exact witness. The corrected current is given below. |
| 4. Product Hartree vanishing | **PASS for the converter potential; FAIL for the blanket full-flow claim** | The converter and its partial-trace Hartree potentials vanish for species-product states with each species number conserving. The complete flow correction need not vanish on that same class, even for a Gaussian Slater product. |

Tags: [R] re-derived; [X] two independent routes; [N] numerical with controls; [S] single source. New [N] claims refer only to the finite exact checks recorded here. The current/wedge amplitude routes and Gram/direct-commutator routes share the explicitly admitted old operands; they do not represent independently evolved states. Supplied proposal and paper definitions remain [S].

## 1. The actual changed operator

Use \(\hbar=c=1\), lattice sites \(x=an\), two four-component fields \(c_{H,x},c_{L,x}\), and canonical CAR. The light mass is zero and the heavy mass is positive. In the inherited component basis,

\[
\alpha_i=\tau_x\otimes\sigma_i,\qquad
\beta=\tau_z\otimes I,\qquad \gamma_5=\tau_x\otimes I,
\]
\[
M^0=\tau_z\otimes\sigma_y,\quad
M^1=\tau_y\otimes\sigma_z,\quad
M^2=i\tau_y\otimes I,\quad
M^3=-\tau_y\otimes\sigma_x,\qquad
\eta=\operatorname{diag}(1,-1,-1,-1).
\]

Round 14 discusses a transition-current/light-current operator with a one-heavy to three-light channel. It is not the requested axial-pair converter. Here only its finite-filter and averaging construction is adapted: filter each leg, form the two axial pairs, average both pairs, and contract only equal filter labels. The resulting interaction changes the two species numbers by two and retains both species parities. It does not inherit the round-14 one-heavy decay statement. [R]

For \(b\in\{0,1\}^3\), put \(s_{bi}=(-1)^{b_i}\) and

\[
F_b=\prod_{i=1}^3\frac{1+s_{bi}(T_i+T_i^\dagger)/2}{2},
\qquad c_{A,b}=F_bc_A,\qquad
\Phi^\mu_{A,b,x}=c_{A,b,x}^{T}M^\mu c_{A,b,x}.
\]

There is **no factor \(1/2\)** in the pair definition. Each one-axis stencil is \((s/4,1/2,s/4)\), so each field filter has support on 27 sites. Its symbol is

\[
f_b(p)=\prod_i\frac{1+s_{bi}\cos(ap_i)}2,\qquad
\sum_b f_b(p)=1,\qquad f_b(\pi c/a)=\delta_{bc}.
\]

The filtered fields overlap. They are not independent canonical valley modes and the filters are not projectors away from the corners. [R,N]

Retain the paper's actual averaging kernel:

\[
B_a(z)=a^{-3}\mathbf1_{[-a/2,a/2]^3}(z),\qquad
K_\ell(z)=\frac{9009}{1024\pi\ell^3}(1-|z|^2/\ell^2)_+^5,
\]
\[
k_{\ell,a}=K_\ell*B_a,\qquad
(W_\ell F)(y)=a^3\sum_xk_{\ell,a}(y-x)F_x.
\]

Define the finite lattice overlap and its symbol by

\[
g_r=a^3\int d^3y\,k_{\ell,a}(y)k_{\ell,a}(y-ar),\qquad
\kappa(P)=\sum_r g_r e^{iaP\cdot r}.
\]

The coefficients are nonnegative, even and cubic invariant. The cell partition of unity gives \(\sum_rg_r=1\), hence \(\kappa(0)=1\). The convolution Gram representation gives \(0\leq\kappa(P)\leq1\). No numerical kernel width, quadrature or continuum replacement of these lattice coefficients is used. [R]

With real \(U=\lambda_X/a^3\), the selected candidate is

\[
\boxed{
V_F=\lambda_X\sum_b\int d^3y\,
\left[(W_\ell(a^{-3}\Phi_{L,b}))^\dagger
\eta\,W_\ell(a^{-3}\Phi_{H,b})+\mathrm{h.c.}\right]
=U\sum_{b,x,y}g_{(x-y)/a}(O_{b,xy}+O_{b,xy}^\dagger)
}
\]
\[
O_{b,xy}=\sum_\mu\eta_{\mu\mu}
\Phi^{\mu\dagger}_{L,b,x}\Phi^\mu_{H,b,y}.
\]

All four legs in a summand have the same \(b\). That restriction, together with the leg filters, removes the leading mixed-valley channel. Pair-centre averaging alone would leave it: twice a valley momentum is reciprocal, so its multiplier would be \(\kappa(0)=1\). The chosen operator includes both the leg filters and the actual \(W_\ell\); a leg-filter-only operator was not substituted. [R]

For a definite energy allocation, use

\[
W_x=\frac12\sum_{b,y}g_{(x-y)/a}
(O_{b,xy}+O_{b,yx}+O_{b,xy}^\dagger+O_{b,yx}^\dagger),
\qquad e_x=e_{0,x}+UW_x.
\]

Then \(W_x^\dagger=W_x\) and \(U\sum_xW_x=V_F\). Its support about \(x\) is bounded by \(2\ell+2\sqrt3a\); a conservative maximum leg separation is \(2\ell+3\sqrt3a\). Thus locality is finite range, not on-site. Infinite-lattice displacement identities apply directly. A finite torus requires a consistent local lift of those displacements. [R]

Each \(O\) obeys

\[
[N_H,O]=-2O,\qquad [N_L,O]=2O.
\]

Therefore total number and both species parities are exact, whereas separate species numbers are not. An odd heavy-number sector cannot empty completely into light through this term alone. The inherited identity
\(\gamma_5^TM^\mu+M^\mu\gamma_5=0\) commutes through the spin-scalar convolutions, so the light axial charge remains exact. The heavy mass breaks heavy axial symmetry. Real even filters and kernel preserve the inherited inversion, time-reversal and cubic transformations, with background, state and reference transformed along. Hermiticity follows from real \(U\) and the explicit adjoint. [R]

Explicitly, use inversion \(P=\beta\) and time reversal \(T=U_T\mathcal K\), \(U_T=I\otimes i\sigma_y\). Then \(P^TM^0P=M^0\), \(P^TM^iP=-M^i\), whereas \(U_T^TM^{0*}U_T=-M^0\) and \(U_T^TM^{i*}U_T=M^i\). Each sign appears in both contracted pairs and cancels. Proper cubic rotations rotate the spatial pairing index and permute the filter labels. These are inherited exact spin identities combined with the new spatial controls, not newly repeated on-site calculations. [R,N]

The light Dirac-mass exclusion retains its symmetric-phase, symmetry-preserving perturbative meaning. It does not prove a gapless interacting pole, an unchanged sea or absence of spontaneous symmetry breaking. For the declared number-conserving species-product reference, cross-species and anomalous contractions vanish, so normal ordering of this converter adds no hidden scalar or bilinear contact. [R]

## 2. What happens to the multivalley obstruction and amplitudes

For \(p=K_c+k\), \(K_c=\pi c/a\),

\[
f_b(K_c+k)=
\prod_{i:b_i=c_i}\cos^2(ak_i/2)
\prod_{i:b_i\ne c_i}\sin^2(ak_i/2).
\]

The four-leg multiplier is

\[
\mathcal F(p_1,p_2,k_1,k_2)
=\sum_b f_b(p_1)f_b(p_2)f_b(k_1)f_b(k_2).
\]

At exact corners it is one precisely when all four valley labels agree. Of 512 momentum-conserving corner quartets, eight survive and 504 are removed. The 16 one-axis corner patterns and their tensor factorization were checked exactly. [R,N]

For the spin witness, use the ordered antisymmetric pair basis \((01,02,03,12,13,23)\) and define
\[
K_{ab;cd}=4\sum_\mu\eta_{\mu\mu}(M^\mu_{ab})^*M^\mu_{cd},
\qquad a<b,\quad c<d,
\]
\[
B_i(u\wedge v)=(\alpha_i u)\wedge v+u\wedge(\alpha_i v),
\qquad v_0=|01\rangle-|23\rangle,\quad
v_3=|03\rangle+|12\rangle.
\]
The lift uses the inherited full-\(\alpha_i\) convention, with no additional factor \(1/2\). These are the same admitted \(K,B_i,v_0,v_3\) definitions as in the unfiltered gate.

For the physical frame of cone \(b\), the pair boost is \(B_i^{(b)}=s_{bi}B_i\). Every surviving diagonal block satisfies the inherited identity \(B_i^{(b)}K+KB_i^{(b)}=0\). The opposite-frame block that previously gave

\[
(-B_3K+KB_3)v_0=32v_3
\]

has zero coefficient at the corners. The new zero was obtained by multiplying the actual selector by the cached nonzero old vector. It does not revise the old obstruction. [R,N]

For a conserving mixed-valley quartet with \(|ak_{\lambda i}|\leq\epsilon<\pi/4\), some coordinate has two of each corner bit. Each of its two filter products has two wrong-filter factors, giving

\[
0\leq\mathcal F\leq\epsilon^4/8.
\]

The diagonal factor is \(1-\tfrac14a^2\sum_\lambda|k_\lambda|^2+O(a^4\Lambda^4)\); the kernel also has small-\(P\) derivative corrections. This removes the order-one offending channel in the leading bare continuum vertex. It does not establish equality of renormalized velocities or Lorentz symmetry to all perturbative orders. Smooth geometry alone does not ensure small matter momenta. [R]

Use \(V_s\) lattice sites, Fourier normalization \(V_s^{-1/2}\), \(u^\dagger u=I_2\), and ordered two-particle states. Then

\[
\langle L_1L_2|V_F|H_1H_2\rangle
=\frac{U}{V_s}\delta_{p_1+p_2,k_1+k_2\ {\rm mod}\ 2\pi/a}\,
\mathcal A_F,
\]
\[
\mathcal A_F=
4\kappa(P)\mathcal F
\sum_\mu\eta_{\mu\mu}
(u_{L1}^TM^\mu u_{L2})^*
(u_{H1}^TM^\mu u_{H2}).
\]

The factor four follows from the no-half pairing. \(\kappa\) sees total pair momentum, not one-particle transfer. Exchanging either pair changes the sign; identical occupied modes are excluded. [R,X,N]

The same-valley centre-of-momentum rest-threshold tensor remains zero: the heavy spatial pairings and light temporal pairing vanish. The old mixed-XOR thresholds with the two heavy particles at distinct exact corners are now killed by their incoming filter product before a spinor contraction. At \(am=3/5\), the two tested threshold weights are \(81/100\) and \(1/100\), and both filtered tensors are zero. [R,N]

For the first nonzero collinear term, set \(r=d/(E+m)\), so \(d=2mr/(1-r^2)\) and \(E=m(1+r^2)/(1-r^2)\). For \(0<am<1\), the two nonzero diagonal spin-tensor slopes at threshold are

\[
\left.\frac{d\mathcal A_{F,+}}{dr}\right|_0
=-16\left(\frac{1+\sqrt{1-(am)^2}}2\right)^2,\qquad
\left.\frac{d\mathcal A_{F,-}}{dr}\right|_0
=-16\left(\frac{1-\sqrt{1-(am)^2}}2\right)^2.
\]

The minus case is a heavy pair near zero converting to a light pair near \(\pi\hat z/a\), not the old mixed-XOR incoming threshold. At \(am=3/5\), the exact checked slopes are \(-324/25\) and \(-4/25\). The differentiated eigen-equations and spinor normalization identities pass. These are new derivative operands built from admitted rest spinors. [R,N]

A second exact on-shell pair of cases uses \(a=1\),

\[
m_H=3/10,\quad d_H=2/5,\quad E_H=E_L=1/2,
\quad p_{H1,H2}=\pm q_H\hat z,\quad
\cos q_H=\sqrt{21}/5,\quad \sin q_H=2/5.
\]

The outgoing light momenta are either \(\pm(\pi/6)\hat z\), or \((\pi\mp\pi/6)\hat z\), ordered with sine momenta \(+1/2,-1/2\). Their signed phase products confirm total momentum zero modulo \(2\pi\), so \(\kappa(P)=1\). The normalized heavy and light spinors are admitted unchanged from the old cache; the heavy mass, sine momentum and energy are scaled together by \(1/2\).

For row and column spin order \((\uparrow\uparrow,\uparrow\downarrow,\downarrow\uparrow,\downarrow\downarrow)\), the exact results are

\[
S_\pm=\frac{161\pm60\sqrt7}{400},\qquad
\boxed{\mathcal A_{F,\pm}
=-\frac{32}{5}S_\pm\operatorname{diag}(1,0,0,1)},
\qquad
\frac14\|\mathcal A_{F,\pm}\|_F^2=\frac{512}{25}S_\pm^2.
\]

Current contractions and ordered-wedge contractions agree, including exchange signs and reverse adjoints. These are the two [X] contraction routes, with common admitted spinors. All corresponding exact residuals vanish. [R,X,N]

Since \(161^2-60^2\,7=721>0\), \(S_->0\): finite intervalley leakage survives. The retained internal-spin boost witness is \(32S_-v_3\ne0\). At nonzero momentum, derivatives of the filters also enter a full boost Ward identity; this spin witness is not presented as that full identity. The certified repair is removal of the leading corner channel, not exact Lorentz invariance of the finite lattice. No amplitude here establishes a rate, a population transfer history, a Page curve or an interacting dispersion law. [R,N]

## 3. The corrected interacting current

Here \(e_{0,x}\) is the sum of the two free many-body site energies. For the declared endpoint allocation, set \(D_i^0=\sum_xx_ie_{0,x}\) and \(D_i^W=\sum_xx_iW_x\). Then

\[
\boxed{
\mathcal Q_i=i[H,D_i^0+UD_i^W]
=\mathcal Q_{0,i}+U\mathcal Q_{1,i}+U^2\mathcal Q_{2,i}
}
\]
\[
\mathcal Q_{1,i}
=i[H_0,D_i^W]+i[\sum_xW_x,D_i^0]
=i\sum_{x,y}(y_i-x_i)[e_{0,x},W_y],
\]
\[
\mathcal Q_{2,i}=\frac{i}{2}\sum_{x,y}(y_i-x_i)[W_x,W_y].
\]

Both linear commutators are required. The filtered \(W_x\) overlap, so the on-site argument setting the quadratic term to zero no longer applies. The local law is

\[
j^E_{xy}=i[e_x,e_y]=-j^E_{yx},\qquad
\dot e_x=-\sum_yj^E_{xy},\qquad
\mathcal Q_i=\frac12\sum_{x,y}(y_i-x_i)j^E_{xy}.
\]

The free term is \(\mathcal Q_{0,i}=\sum_{A=H,L}c_A^\dagger[T_i^2-(T_i^\dagger)^2]/(4ia)c_A\).

A compatible local flow allocation is
\[
\mathcal P_{x,i}=\mathcal P_{0,x,i}
+\frac12\sum_y(y_i-x_i)(j^E_{xy}-j^{E,0}_{xy}).
\]
It sums to the corrected current and reduces to the paper's allocation at \(U=0\). This specifies an interaction-current extension; it does not certify an interacting gravitational source law or branch. [R]

The nonzero \(U^2\) term can be proved without a full lattice Fock space. In the exact two-particle \(HH\oplus LL\) sector, let \(\theta=aP\) and

\[
L_b(\theta,q)=f_b(\theta/2+q)f_b(\theta/2-q),\qquad
S=LL^\dagger,
\]

Here the symbols \(f_b\) and \(\kappa\) are written as functions of dimensionless momenta, \(q\in[-\pi,\pi]^3\), and the inner product in \(q\) uses measure \(d^3q/(2\pi)^3\). Primes denote dimensionless \(\theta_i\) derivatives, or \(\vartheta=\theta_z\) derivatives in the witness below. The spatial pair operator has rank at most eight and equals \(\kappa L^\dagger L\). In the ordered antisymmetric spin basis \((01,02,03,12,13,23)\), its spin factor is the admitted \(K\). The actual interaction dipole and quadratic current reduce to

\[
D_i^W=\tau_x^{HL}\otimes iaL^\dagger
\left(\kappa\partial_{\theta_i}+\frac12\kappa_i'\right)L\otimes K,
\]
\[
\boxed{
\mathcal Q_{2,i}|_{N=2}
=aI_{HL}\otimes L^\dagger
(\kappa\kappa_i'S+\kappa^2S_i')L\otimes K^2.
}
\]

The derivative in the dipole acts on \(L\) and on the wavefunction to its right. Keeping both derivatives is essential. The finite Fourier overlaps are

\[
S_{bc}(\theta)=\prod_i
\begin{cases}
A(\theta_i),&b_i=c_i,\\
B(\theta_i),&b_i\ne c_i,
\end{cases}
\quad
A(t)=\frac{18+16\cos t+\cos2t}{128},\quad
B(t)=\frac{2+\cos2t}{128}.
\]

These identities are exact Laurent-polynomial identities, not a mode truncation. [R,X]

For a witness already inside the Hartree class, occupy heavy components \(0\) and \(2\) at the same momentum \(p=(0,0,\vartheta/2a)\), with the light vacuum. The modes are distinct and the state is a normalized species-product Slater state. Put \(\theta=(0,0,\vartheta)\), \(l_b=f_b(\theta/2)^2\) in the dimensionless convention, and

\[
h=\sum_rg_r r_z^2=-\kappa''(0)\geq0.
\]

This is the actual finite lattice moment of the compact kernel, not its continuum variance. With the cached \((K^2)_{02;02}=64\),

\[
\langle\mathcal Q_{2,z}\rangle
=\frac{64a}{V_s}\,
l^T(\kappa\kappa'S+\kappa^2S')l,
\]
\[
\boxed{
\frac{V_s}{a}
\left.\frac{d\langle\mathcal Q_{2,z}\rangle}{d\vartheta}\right|_0
=-\frac{6125}{8192}-\frac{42875}{32768}h
=-\frac{1225(20+35h)}{32768}<0.
}
\]

The exact finite check saves the full \(8\times125\) Laurent jet matrices. Its Gram route and independently derived differentiated-commutator contraction return the same coefficient pair \((-6125/8192,-42875/32768)\), with separate known-answer and strict-sign checks. The \(64\) spin factor was reused as an admitted scalar, not recomputed. [R,X,N]

For a finite torus, take even side length \(L\geq6\) and \(La>8\ell+8\sqrt3a\), with compatible grid momenta and a local displacement lift. The finite momentum sum equals the Laurent constant coefficient without aliasing. The derivative is of the analytic trigonometric formula describing compatible arbitrarily small momenta on increasing finite boxes; it is not a derivative of a seam-discontinuous torus position. No finite-box spectrum or production-history flux was computed.

The negative slope for every allowed \(h\geq0\) proves that the quadratic current is not identically zero. Therefore the summed free two-step operator is only \(\mathcal Q_0\), not the full interacting current. **Item 3 fails as an unchanged-current claim.** [R,N]

## 4. The precise Hartree statement

Let \(\rho=\rho_H\otimes\rho_L\), with \([\rho_A,N_A]=0\) for each species. Every filtered anomalous pair expectation vanishes, independently of correlations within a species. Consequently

\[
\operatorname{Tr}(\rho V_F)=0,\qquad
\operatorname{Tr}_L(\rho_LV_F)=0,\qquad
\operatorname{Tr}_H(\rho_HV_F)=0.
\]

Thus the converter potential has zero mean and zero species partial-trace Hartree field. Gaussianity is unnecessary. Clock weighting of its local energy terms preserves this conclusion. For a Gaussian first variation, the inherited Wick polynomial still has its homogeneous mixed/anomalous-covariance form, so its value and first derivative vanish at zero mixed and anomalous covariances. The accepted old Wick controls were reused. [R]

This scope does not include arbitrary species-entangled states, states constrained only by total number, anomalous product factors, the interacting sea or later connected dynamics. A zero product Hartree field does not make the interaction inert.

The linear current correction still changes the species numbers by two, so its product-state mean vanishes. The quadratic commutator contains species-number-conserving terms. The Slater witness above has nonzero \(\mathcal Q_2\), despite belonging to the stated product class. Therefore a full clock-and-flow generator using the corrected current does **not** have an identically vanishing interaction correction on that class; a uniform nonzero flow already detects the difference. The potential-only Hartree statement passes, while its blanket extension to the full generator fails. [R,N]

No new first-order entropy, geometry or long-time assertion is inferred. The previous fixed-cut, regular-feedback and finite-time qualifications remain in force.

## Exact run and provenance

| New stage | Required raw objects | Returned decisions | Outcome |
|---|---:|---:|---|
| F: stencils, corner selector and symmetry/support controls | 14 | 10 | Complete; all true |
| A: signed kinematics, threshold, finite and first-jet amplitudes | 70 | 24 | Complete; all true |
| C: Laurent current jets and two slope contractions | 33 | 11 | Complete; all true |

The journals contain all 354 records, including 57 admission/cache records and separate actual, expected, residual and decision records for every comparison. The raw phase differences retain exact unsimplified expressions; their returned comparison residual is zero. Both inherited and newly constructed objects have explicit operand references. Every journal ends at a complete newline, with no trailing partial bytes.

The wrapper reaped the child with exit code zero, no signal, empty stdout/stderr and no custody error. Peak child RSS was 64512 KiB. Actual execution used one CPU, one numerical thread, nice 10 and a 2 GiB address-space limit, within all frozen resource bounds. Python 3.12.12 and SymPy 1.14.0 are recorded. All twelve initial, child and terminal identities agree. The complete execution record retains the resource measurements.

Old A74 and B80 remain inherited accepted controls. Old D remains incomplete with 430 missing decisions. No accepted old spinor, current, Fock product or stage was replayed to complete this new plan. No further allocation is needed for this scoped gate. All numerical records remain on the box.

## SOURCES

Immutable local premises and mathematical records:

- [Task inbox, entry 99](<internal>/hreduce.md); scope and requested deliverable.
- [Pinned reviewer B gate ruling](<internal>/ReviewB_vgate_v1.md); SHA256 4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b.
- [Pinned reviewer A round 14](<internal>/ReviewA_pc14_v1.md); SHA256 918430260445b6829bc5ed237a21507f6c7aba754a0162c92d4785ea0cce82ac.
- [Pinned reviewer B round 14](<internal>/ReviewB_pc14_v1.md); SHA256 e4d950ac3443f4c867aa7f80128806a38ec8651dbc86c5b72812c1cb1fbd715a.
- [Pinned paper model](<internal>/03_model.tex); equations dm-fermion, dm-average and dm-current; SHA256 c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb.
- [Frozen mathematical contract](<internal>/frozen_spec.tex); SHA256 9c800965d8e050a1357ebef24c06fe16298625d339a8dba860af65db686fd217.
- [Author derivation and complete source/output judgments](<internal>/VERTEX_gate_filtered_author_v1.tex) and [original sole peer derivation and judgments](<internal>/redteam_VERTEX_gate_filtered_v1.tex).
- [Previous unfiltered gate](<internal>/VERTEX_gate_v1.md); SHA256 3cbca4d133da6ebc02c23dcbfd5fc961233f95580abd421b53c6cfc1cbc8f8b3; accepted predecessor scope and incomplete old amplitude plan.
- [Executed producer](<internal>/vertex_gate_filtered_exact_v1.py); SHA256 24a490eff17d3ee61626fd681f95766a3533450ba8c8a310d5b0a5a9059477d7.
- [Executed wrapper](<internal>/run_vertex_gate_filtered_exact_v1.py); SHA256 bfe2701974a15edc64f5a9cc9ed3ddb8f6a35290f3c9b995267913dbb7302b7c.

The following numerical paths are on box compute-host, read in place. The five predecessor identities are preserved in both admission and terminal postchecks.

| Absolute box predecessor path | SHA256 |
|---|---|
| <internal>/spinor_algebra.json | af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb |
| <internal>/on_site_completion.json | 77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d |
| <internal>/amplitudes.json | af9c99e96e10da536cdecf779e6f4f5e01a6754c1e8d3e1997aea1110a5604df |
| <internal>/vertex_gate_missing_result.json | 52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d |
| <internal>/execution.json | be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9 |

| Absolute box new-output path | SHA256 |
|---|---|
| <internal>/admission.jsonl | b3a692e0f8fe1b429fde3ca5e3f5c6603616e4d101f87d6ca340c0a35f9ef10d |
| <internal>/filters.jsonl | efa41dd9d9a282143ec2ba3b9afc9e5f8a1373baf7940a69c3e202f4e1262802 |
| <internal>/amplitudes.jsonl | 34a06923b52f850766c76f0e712e3f21cb9a06809a0b214243369b2933cc4dfb |
| <internal>/current.jsonl | daae7293de7df50c64a732ad34cd9bab396b0f04cf4379aa029f73d1f5beb64a |
| <internal>/vertex_gate_filtered_result.json | 32300d99c630cb5ff9d9b8615e932a23c7dbf4b0b8bd75bfe5b24d30d2c17027 |
| <internal>/execution.json | b5d726a8c3ba9a333c9bf391ae33abf92b75ece5174dd66a755eb68373377869 |
| <internal>/stdout.log | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| <internal>/stderr.log | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

Bulk custody is indexed at [LARGE_FILES.tsv](<internal>/LARGE_FILES.tsv). No numerical record is attached to this local delivery.

