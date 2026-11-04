
## 1. Both coefficients remain free with the reference fixed

**Yes. Neither coefficient is selected by any requirement stated in the paper.** This is fixed-reference nonuniqueness, not merely a freedom to repartition the reference.

The current file, `00_PAPER_flat_ea20cd88475_nocomment.tex`, defines the function `\mathcal W[u]` in `eq:dm-reference` by
> `\mathcal W[u]&=\Tr(P_\star h[u])+W_2[u]-DW_2[u_\star](u)`

Its `sec:model-gravity` describes the next-gradient coefficients as
> “undetermined coefficients and no bound.”

Hold that reference, the fermion generator, preparation, averaging, \(a,m,G\), and boundary conditions fixed. Define
\[
f=N^{-1},\qquad
\mathscr D=\nabla\times\nabla\times,\qquad
k_4=\frac{a^2}{16\pi G},
\]
\[
S_1=\int f|B|^2\,d^3x,\qquad
S_2=\int fC_{ij}C_{ji}\,d^3x,
\]
where \(\varpi=\nabla\times v\), \(C_{ij}=\partial_i\varpi_j\), and \(B=\mathscr Dv\). The two-parameter witness is
\[
\boxed{
\mathcal W=\mathcal W_{\rm decl},\qquad
\mathscr A_{\lambda_1,\lambda_2}
=\mathscr A_2+k_4(\lambda_1S_1+\lambda_2S_2),
\qquad (\lambda_1,\lambda_2)\in\mathbb R^2 .
}
\]

### Functional derivatives and the four requirements

Integration by parts, using \(\nabla\cdot\varpi=0\), gives
\[
S_2=\int \varpi_i\varpi_j\,\partial_i\partial_jf\,d^3x
\]
modulo a boundary term. Consequently the added laws are
\[
\boxed{
\begin{aligned}
\rho_4&=-\frac{k_4}{N^2}
 \left(\lambda_1|B|^2+\lambda_2C_{ij}C_{ji}\right),\\
J_4&=2k_4\left[
 \lambda_1\mathscr D(fB)
 +\lambda_2\nabla\times\bigl((\nabla\nabla f)\varpi\bigr)
 \right].
\end{aligned}}
\tag{1}
\]

These meet all four requirements.

* **Common generator.** Neither \(h[N,v]\) nor its energy, current, or spin assignments changes.
* **Response identity.** Both \(S_i[sN,sv]=sS_i[N,v]\). Euler’s identity and symmetry of the functional Hessian therefore give
  \[
  \int(N\,\delta\rho_4+v\cdot\delta J_4)\,d^3x=0.
  \]
* **Momentum and angular-momentum fluxes.** With
  \[
  S_{ij}^{v}=\tfrac12(\partial_iv_j+\partial_jv_i),\qquad
  C_{ij}=2\epsilon_{jab}\partial_aS^v_{ib},
  \]
  each density is a function of \(N\) and \(\partial_kS^v_{ij}\). If
  \(P_{kij}\) is its derivative with respect to \(\partial_kS^v_{ij}\), taken symmetrically in \(i,j\), then
  \[
  J_{4j}=\partial_i\tau_{ij},\qquad
  \tau_{ij}=\partial_kP_{kij}=\tau_{ji}.
  \]
  Thus momentum has a local flux, and symmetry eliminates the torque term in the divergence of its moment. Uniform shifts and rigid turns leave \(C\), hence both laws, unchanged.
* **Potential-flow charge.** On every \(v=\nabla\phi\), \(C=B=0\), so both densities and their **full first functional derivatives** vanish at every positive clock.

Both terms are local, isotropic, regular for \(N>0\), time-reversal even, and contain no field rates. Their restriction to the entire spherical field space is zero. Thus the radial equations, spherical constraint Jacobians, boundary functional \(E[C]\), and its quantum expansion are identical throughout the witness family.

### Genuine two-parameter independence

At constant clock, \(S_2\) is a boundary term, whereas \(S_1\) is nonzero on a transverse Fourier mode. This first distinguishes \(\lambda_1\). The printed test then distinguishes \(\lambda_2\):
\[
v=(0,0,\cos x+\cos y),\qquad
f=1+\epsilon\cos x\cos y,\qquad |\epsilon|<1,
\]
\[
\int_{\mathbb T^3}fC_{ij}C_{ji}\,d^3x=-4\epsilon\pi^3.
\]
Therefore no nonzero linear combination of \(S_1,S_2\) is a divergence on the variable-clock domain. Rescaling the box makes this test arbitrarily smooth relative to \(\ell\).

### Frame dragging: density nonzero, momentum law zero

`02_F_lambda_matching_val.md`, §1, identifies the perturbation as
> “\(\delta v\propto\hat z\times x/r^3\)”.

For its general orientation, write
\[
\delta v=\varepsilon\,\frac{s\times x}{r^3},\qquad r>0.
\]
Then
\[
\varpi=-\varepsilon\nabla\left(\frac{s\cdot x}{r^3}\right),\qquad B=0,
\]
but \(C\) is a nonzero symmetric Hessian. Explicitly,
\[
C_{ij}C_{ji}
=\frac{18\varepsilon^2}{r^8}
 \left(|s|^2+2(s\cdot\hat r)^2\right).
\]
Equation (1) therefore gives
\[
\boxed{
J_{4,\lambda_2}\big|_{N=1}=0,\qquad
\rho_{4,\lambda_2}\big|_{N=1}
=-k_4\lambda_2 C_{ij}C_{ji}=O(\varepsilon^2).
}
\]
The \(\lambda_2\) density does **not** vanish there. Its unit-clock momentum law vanishes identically, and its energy correction starts at second order. The first-order frame-dragging result survives.

---

## 2. The two measurements and the sea’s contribution

### Uniform-clock transverse response fixes \(\lambda_1\)

Within the displayed two-parameter response sector,
\[
J_T(Q)=\frac{Q^2}{16\pi G}v_T
       +2k_4\lambda_1Q^4v_T+\cdots .
\tag{2}
\]
Thus the isotropically projected \(Q^4\) coefficient fixes \(\lambda_1\), with no sensitivity to \(\lambda_2\).

`03_A_brk_lambda.md`, §6, gives
> `J^{\rm sea,sub}_{4,T}=-2I_4^{\rm iso}Q^4v_T`

and hence, after moving the static sea polarization to the law side,
\[
c_{\rm eff}=\lambda_1+16\pi\frac{G}{a^2}I_4^{\rm iso},
\qquad I_4^{\rm iso}=[I4].
\]

### A clock-modulated transverse response fixes \(\lambda_2\)

Put \(N=1+\eta n\). Equation (1) gives the explicit variable-clock measurement
\[
\boxed{
\left.\partial_\eta J_4[1+\eta n,v]\right|_{\eta=0}
+2k_4\lambda_1\mathscr D(nB)
=-2k_4\lambda_2\nabla\times
 \left[(\nabla\nabla n)(\nabla\times v)\right].
}
\tag{3}
\]

A particularly simple Fourier implementation uses
\[
q=Qe_x,\qquad p=Qe_y,\qquad t=e_z,
\]
with input \(v_t(q)\), clock modulation \(\delta N(p-q)\), and measured output \(J_t(p)\). Using independent complex Fourier amplitudes,
\[
\boxed{
M_4^{\rm law}
:=[Q^4]\frac{\delta^2J_t(p)}
 {\delta N(p-q)\,\delta v_t(q)}
=-2k_4(\lambda_1-\lambda_2).
}
\tag{4}
\]
Subtract the second-gradient response before extracting \([Q^4]\). Once (2) fixes \(\lambda_1\), (4) fixes \(\lambda_2\). Equivalently, measure the energy-law cross term on the two superposed transverse flows.

### The sea’s share is a mixed three-point integral, not zero by a divergence argument

**The sea fixes a share of this specified \(\lambda_2\) measurement.** No complete fourth-order invariant basis is needed to define it.

To make that statement precise, let \(\chi_4(F)\) and \(M_4(F)\) denote the isotropically averaged responses (2) and (4) of a functional \(F\), without their lower-order terms. The map
\[
\boxed{
P(F)=\frac{\chi_4(F)}2S_1+
     \frac{\chi_4(F)+M_4(F)}2S_2
}
\tag{5}
\]
is a projection onto \(\operatorname{span}\{S_1,S_2\}\): substitution of \(F=\alpha S_1+\beta S_2\) returns exactly that functional. This supplies an operationally defined share even when the full band functional has other components.

Let \(M_4^{\rm sea,iso}\) be the sea’s response (4), averaged over rotations of the orthonormal frame. Then
\[
\boxed{
I_{4,2}^{\rm iso}
=I_4^{\rm iso}-\frac12M_4^{\rm sea,iso},\qquad
\lambda_{2,\rm eff}^{\rm mix}
=\lambda_2+16\pi\frac{G}{a^2}I_{4,2}^{\rm iso}.
}
\tag{6}
\]
Here is a fully specified integral for the new quantity.

Use dimensionless momenta and \(\mu=ma>0\):
\[
H(k)=\mu\beta_{\rm D}+\alpha_i\sin k_i,\qquad
G_k(\zeta)=(\zeta-H(k))^{-1}.
\]
The clock vertex is
\[
E(k',k)=\tfrac12[H(k')+H(k)].
\]
The exact flow vertex follows from \(h[N,v]\) in the current paper’s `eq:dm-hamiltonian`, specifically
> `\tfrac12\sum_i\{v^i,Q_i\}`  
> `\tfrac18\sum_k\{(\nabla\times v)^k,\Sigma^{\rm lat}_k\}`.

Writing \(q_i^0(k)=\sin(2k_i)/2\) and \(c_i(k)=\cos k_i\), it is
\[
\begin{aligned}
V_t(k',k)={}&
 \tfrac12t_i[q_i^0(k')+q_i^0(k)]I\\
&+\frac{i}{8}\sum_r[(k'-k)\times t]_r
 [c_i(k')c_j(k')+c_i(k)c_j(k)]\Sigma_r ,
\end{aligned}
\tag{7}
\]
where \((i,j,r)\) is cyclic.

For each rotation \(R\), put
\[
q=zRe_x,\qquad p=zRe_y,\qquad t=Re_z,\qquad h=p-q.
\]
With normalized Haar measure \(dR\),
\[
\boxed{
M_4^{\rm sea,iso}
=\int_{SO(3)}dR\int_{\rm BZ}\frac{d^3k}{(2\pi)^3}
[z^4]\oint_{\Gamma_-}\frac{d\zeta}{2\pi i}\,
\operatorname{tr}(T_1+T_2),
}
\tag{8}
\]
where
\[
\begin{aligned}
T_1={}&G_kV_t(k,k+p)G_{k+p}
 E(k+p,k+q)G_{k+q}V_t(k+q,k),\\
T_2={}&G_kV_t(k,k+p)G_{k+p}
 V_t(k+p,k+h)G_{k+h}E(k+h,k).
\end{aligned}
\]
The contour encloses the negative bands and excludes the positive bands.

This is directly reducible to a Brillouin-zone integral of projector traces. For three resolvents, the residue weight is zero when all three band signs agree. With one negative energy it is
\[
\frac{1}{(e_-+e_{+1})(e_-+e_{+2})},
\]
and with one positive energy it is
\[
-\frac{1}{(e_++e_{-1})(e_++e_{-2})}.
\]
Thus no frequency integration or unspecified vertex remains.

The normalization follows from the cubic band-energy term
\(\frac13\oint\Tr(G\delta h)^3\). Its six permutations reduce to the two cyclic orderings in (8). As a check, setting \(p=q\), so that the clock modulation is uniform, gives the negative of the two-point Hessian, hence the fourth-order coefficient \(+2I_4^{\rm iso}\). The four-component trace already counts spin and valleys.

The declared \(W_2\) subtraction removes the second-gradient part. With identical back-averaging in band and reference, its averaging-generated fourth-order contribution cancels as well.

There is no valid zero proof from \(S_2|_{N=\mathrm{constant}}\) being a divergence. Indeed,
\[
P_+(k')E(k',k)P_-(k)
=\frac{e(k')-e(k)}2P_+(k')P_-(k)
\]
is generally nonzero at second order in transfer. Equation (8) includes this clock vertex. **The share vanishes exactly if \(M_4^{\rm sea,iso}=2I_4^{\rm iso}\), not because uniform-clock momentum is blind to \(S_2\).**

No numerical value or sign for (8) is asserted here. To obtain one, evaluate its fourth Taylor coefficient, Brillouin-zone integral, and rotational average at the same \(ma\) and reference convention, with enclosed arithmetic and quadrature errors. The merged paragraph below retains this share analytically, so it needs no second numerical placeholder.

### Scheme statement

For either term,
\[
\mathcal W\longmapsto\mathcal W+d_1S_1+d_2S_2,\qquad
\lambda_i\longmapsto\lambda_i-\frac{d_i}{k_4}
\tag{9}
\]
preserves the coupled equations. The literal `eq:dm-reference` fixes \(d_1=d_2=0\). It does not fix either \(\lambda_i\).

---

## 3. Draft check and replacement paragraph

The following changes to `06_F_lambda_r8merge.md` are necessary.

* Its **“free, not unmeasured”** distinction is unsupported. The coefficients are free under the declarations and measurable through additional matching data.
* **“\(B\) vanishes with its variations”** is false. The density \(|B|^2/N\) and its first variations vanish at \(B=0\), while \(\delta B=\mathscr D\delta v\) need not.
* **“no uniform-clock quantity”** must become **“no uniform-clock momentum response.”** The local energy law at unit clock detects the second invariant.
* The clause **“the share of \(c_4'\) it carries is defined only against the complete fourth-order basis, which is not classified here”** is replaced by the explicit response projection and integral (5)–(8). A pure-clock term contributes nothing to the mixed derivative with two flow insertions.
* The printed paragraph already uses \(\lambda_1,\lambda_2\). Its unindexed \(\lambda\) in `eq:sc-clock-exact` creates no clash with them, so the draft’s renaming to \(c_4,c_4'\) is unnecessary.

The replacement has **nine sentences**, two more than the printed paragraph, and no semicolons.

> The second is the order of the derivation. Beyond the leading laws, the four requirements leave at least two coefficients free: with \(\varpi=\nabla\times v\), \(C_{ij}=\partial_i\varpi_j\) and \(B=\nabla\times\varpi\), the density \(\mathcal A+a^2[\lambda_1|B|^2+\lambda_2C_{ij}C_{ji}]/(16\pi GN)\) satisfies them for every real \(\lambda_1,\lambda_2\), with the fermion generator and declared reference unchanged. These local isotropic terms are degree one, invariant under uniform shifts and rigid turns, and quadratic in second derivatives of \(v\), so momentum is a symmetric-tensor divergence and its moment has a flux, while both vanish with their full first variations on every potential flow at every positive clock. Their independence requires variable clock, since \(C_{ij}C_{ji}=\partial_i(\varpi_j\partial_j\varpi_i)\) is a divergence at constant clock, whereas \(v=(0,0,\cos x+\cos y)\) and \(1/N=1+\epsilon\cos x\cos y\), \(|\epsilon|<1\), give \(\int C_{ij}C_{ji}/N\,d^3x=-4\epsilon\pi^3\) on the \(2\pi\)-periodic box. The unit-clock transverse momentum law \((q^2/16\pi G)(1+2\lambda_1a^2q^2)v_T\) fixes \(\lambda_1\) alone, while the \(q^4\) response mixing \(v_z(qe_x)\) into \(J_z(qe_y)\) through \(\delta N(q(e_y-e_x))\) is \(-a^2(\lambda_1-\lambda_2)/(8\pi G)\), fixing \(\lambda_2\) equivalently to the energy law on superposed transverse flows. Moving the sea polarization to the law side gives \(c_{\rm eff}=\lambda_1+16\pi(G/a^2)I_4^{\rm iso}\) and \(\lambda_{2,\rm eff}^{\rm mix}=\lambda_2+16\pi(G/a^2)(I_4^{\rm iso}-M_4^{\rm sea,iso}/2)\) for these response projections, with \(I_4^{\rm iso}=[I4]\) at \(ma=0.2\) and \(M_4^{\rm sea,iso}\) the sea’s corresponding rotationally averaged mixed-response coefficient from the three-vertex band integral. Adding \(\int(\mathsf d_1|B|^2+\mathsf d_2C_{ij}C_{ji})/N\,d^3x\) to \(\mathcal W\) and shifting \(\lambda_i\) by \(-16\pi G\mathsf d_i/a^2\) leaves the coupled equations unchanged, but Eq.~\eqref{eq:dm-reference} fixes \(\mathsf d_1=\mathsf d_2=0\) and no declaration selects either law coefficient. The added terms leave every radial or potential flow unchanged, and on the exterior’s frame-dragging field \(B=0\) while \(C_{ij}C_{ji}\ne0\), its unit-clock momentum law vanishing and its energy law starting at second order in the perturbation, so no result here changes. For fixed coefficients their relative \(O((aq)^2)\) effect is parametrically below the \(O((\ell q)^2)\) averaging control.

lambda_1, lambda_2: free under the paper, fixed by added uniform-clock transverse and variable-clock mixed-response data respectively; merged: needs [I4]