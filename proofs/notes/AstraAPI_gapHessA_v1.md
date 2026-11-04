
## Result and scope

**The restriction to errors parallel to the computed Hessian is unnecessary.** An all-direction error bound follows from a validated constraint solve and a residual bound for its linear response. Its propagation to the second-order charge is governed by a positive semidefinite Wick Gram matrix. That matrix gives the **optimal constant for a specified Hessian-error norm**, including off-diagonal, indefinite, and nonparallel errors.

The numerical conclusion on the eighteen slices cannot be certified from the supplied files alone. They contain neither the Hessians nor the response residuals, covariances, or slice manifest. Below I give the proof, explicit computable constants, propagation formulas, numerical acceptance margins, and a two-route computation specification.

Three distinctions matter:

* In `00_P3_current_nc.tex`, the interior paragraph says:
  > “For the interior deviation \(4\sum_j(\widehat M_j/M_j-I)^2\), the bound \(\langle\mathcal B\rangle\le0.080\) … implies overlap at least 0.92 with that test subspace, not the branch”.

  Thus that number is **not** a boundary-charge branch certificate.
* The same paragraph says:
  > “Every operator of this paragraph is a held-source operator, built from the fixed reference projector's response … instead of Eq.~\eqref{eq:dm-reference}'s reference response”.

  Certifying numerical error for that operator does not certify the declared-reference operator.
* The subsequent boundary-charge paragraph reports:
  > “matched nominal upper endpoints below \(10^{-3}\) on five ten-channel stored slices”,

  but also:
  > “neither actual-error inclusion nor the declared-reference branch calculation is established”.

No implementation functions or numerical arrays are supplied. The algorithms below are a specification, not a claim to have inspected or executed existing routines.

---

## 1. An all-direction bound for the computed Hessian

### 1.1 Use the constraint response, not a relative rescaling of the Hessian

Let
\[
x_\mu=\operatorname{Tr}(A_\mu C),\qquad \mu=1,\ldots,d,
\]
be the retained bare moments, and let \(u\in\mathbb R^f\) denote the boundary-normalized field variables.

Use an affine-in-source form of the discrete constraints:
\[
F(u,x)=f(u)-Bx=0,\qquad J=F_u=f_u.
\tag{1}
\]
The reference moments are included in \(f\). For example, expressing the rows as geometric densities minus state-minus-reference densities has this form.

This is the finite-dimensional version of the function named in `00_P3_current_nc.tex`, `sec:singularity`:
> “the boundary-normalized constraints \(F(C,u)=0\), \(u=(N,v)\), with the full reference response”.

Write \(L u\) for the coefficients of the back-averaged one-body generator in the \(A_\mu\) basis.

The printed boundary identity, `eq:dr-boundary-variation`,
> “\(\delta E=\Tr(\bar h\,\delta C)\)”,

then gives
\[
\nabla_xE=L u,\qquad
U:=D_xu=J^{-1}B,\qquad
\boxed{V_E=D_x^2E=LJ^{-1}B=LU.}
\tag{2}
\]

**Important discrete gate:** (2) may be used only if the implemented finite-parent rows and quadratures preserve that identity, or its discrepancy is bounded. A continuum identity does not automatically hold for every row discretization. Section 1.3 supplies a route that does not rely on it.

All norms below must use declared coordinate scales. If \(x=S\widetilde x\), transform both objects:
\[
\widetilde V=S^{\mathsf T}VS,\qquad
\widetilde A_\mu=A_\mu/S_{\mu\mu}.
\]
A Hessian-error norm without its moment normalization is not an error specification.

### 1.2 Validated root and response residual

Let \(\widehat u\) be the computed fields, \(R\) an approximate inverse of \(J\), and
\[
\mathcal U=\{u:\|u-\widehat u\|_2\le r\}.
\]
Suppose directed-rounding bounds establish
\[
\eta=\|RF(\widehat u,x)\|_2,\qquad
\kappa=\sup_{u\in\mathcal U}\|I-RJ(u)\|_2<1,
\qquad
\eta\le(1-\kappa)r.
\tag{3}
\]
Then \(u\mapsto u-RF(u,x)\) maps \(\mathcal U\) into itself and is a contraction. It has a unique root there, and
\[
\|J(u)^{-1}\|_2\le
\beta:=\frac{\|R\|_2}{1-\kappa}.
\tag{4}
\]

Let \(\widehat U\) be the computed response matrix and bound its **full residual**
\[
r_U=
\sup_{u\in\mathcal U}\|J(u)\widehat U-B\|_F.
\tag{5}
\]
At the exact root,
\[
\widehat U-U=J^{-1}(J\widehat U-B),
\]
so
\[
\boxed{\|U-\widehat U\|_F\le\beta r_U.}
\tag{6}
\]

Consequently, if \(\widehat V_E=\operatorname{sym}(L\widehat U)\), apart from an enclosed assembly error \(\epsilon_{\rm asm}\),
\[
\boxed{
\|V_E-\widehat V_E\|_F
\le
\epsilon_{V_E}:=
\|L\|_2\,\frac{\|R\|_2}{1-\kappa}\,r_U
+\epsilon_{\rm asm}.
}
\tag{7}
\]

A sharper version replaces \(\|L\|_2\beta\) by any verified bound on
\(\|LJ^{-1}\|_2\). For example, a right-inverse residual gives
\[
\|LJ^{-1}\|_2
\le
\frac{\|LR\|_2}{1-\|I-JR\|_2}
\quad\text{when }\|I-JR\|_2<1.
\tag{8}
\]

Equations (3)–(8) include **every matrix direction**. They do not assume
\[
V_E-\widehat V_E=\alpha\widehat V_E,
\]
positive definiteness, or a relative Loewner bound.

The factor \(1/(1-\kappa)\) cannot be uniformly improved from these inputs: equality already occurs for the scalar example \(R=1,\ J=1-\kappa\).

Kernel, reference-coefficient, and input uncertainties must be included in the suprema in (3) and (5), or added explicitly. Comparing two approximate Hessians is not a substitute for (5).

### 1.3 Interior masses and an independent boundary-Hessian route

For a fixed diagnostic radius \(r_j\), let
\[
O_j(x)=\phi_j(u(x)),\qquad
\phi_j(u)=\mathcal M(r_j;u).
\]
The radius/window is frozen when differentiating the mass operator. Differentiating a moving trapping radius defines a different observable.

Solve
\[
J^{\mathsf T}\lambda_j=\nabla_u\phi_j.
\]
Then
\[
\nabla_xO_j=B^{\mathsf T}\lambda_j,\qquad
\boxed{
V_j=U^{\mathsf T}H_jU,\quad
H_j=\phi_{j,uu}-\sum_i(\lambda_j)_i f_{i,uu}.
}
\tag{9}
\]

**Proof.** Differentiating \(F(u(x),x)=0\) twice gives
\[
D_x^2u[p,q]=-J^{-1}f_{uu}[Up,Uq].
\]
Substitute this into \(D_x^2\phi_j(u(x))[p,q]\) and use the adjoint equation.

This also computes the boundary Hessian by taking \(\phi_\infty\) to be the directly evaluated boundary charge. It is independent of the generating identity used in (2).

Here is an explicit error bound. Let
\[
\delta U_2\ge\|U-\widehat U\|_2,\qquad
\delta U_F\ge\|U-\widehat U\|_F,
\]
obtained from response residuals as in (6). If
\[
r_{\lambda,j}=
\sup_{\mathcal U}
\|J^{\mathsf T}\widehat\lambda_j-\nabla_u\phi_j\|_2,
\]
then \(\delta\lambda_j\le\beta r_{\lambda,j}\). Bound
\[
\epsilon_{H_j,F}\ge\|H_j-\widehat H_j\|_F.
\]
One explicit construction is
\[
\epsilon_{H_j,F}
\le
\epsilon_{\phi'',j}
+\|\widehat\lambda_j\|_2\,\epsilon_{f''}
+\beta r_{\lambda,j}\,K_{f''},
\tag{10}
\]
where \(K_{f''}\) bounds the contraction map
\(z\mapsto\sum_i z_i f_{i,uu}\) from Euclidean to Frobenius norm, and
\(\epsilon_{f''}\) bounds its derivative-assembly error over the root box.

Putting \(u_0=\|\widehat U\|_2\),
\[
\boxed{
\begin{aligned}
\|V_j-\widehat V_j\|_F
\le{}&
\epsilon_{{\rm asm},j}
+(u_0+\delta U_2)^2\epsilon_{H_j,F}\\
&+(2u_0+\delta U_2)\,
\|\widehat H_j\|_2\,\delta U_F .
\end{aligned}}
\tag{11}
\]
This follows by expanding
\(U^{\mathsf T}H_jU-\widehat U^{\mathsf T}\widehat H_j\widehat U\).
Again, no parallel-error assumption appears.

If the chosen rows have \(F_{ux}\ne0\), or the observable depends explicitly on \(x\), use the general formula
\[
D_x^2O=T^{\mathsf T}
\left(\phi_{zz}-\sum_i\lambda_iF_{i,zz}\right)T,
\quad
z=(x,u),\quad T=\binom{I}{U},
\tag{12}
\]
rather than omitting the mixed derivatives.

### 1.4 What reference bounds are actually required

`00_P3_current_nc.tex`, `eq:dm-reference`, defines
> “\(\mathcal W[u]=\Tr(P_\star h[u])+W_2[u]-DW_2[u_\star](u)\)”,

and `eq:dm-covariance` identifies the response as \(\mathcal W''\).

Thus the boundary-response certificate needs enclosed \(\mathcal W''\); the mass-Hessian certificate needs \(\mathcal W'''\), and usually a fourth-derivative bound to enclose its variation over the root box. The linear prepared terms contribute nothing at these orders.

These constants are explicit for a specified quadrature of the printed bulk term. For one quadrature contribution
\[
W_{2,z}=-\frac{w_z}{4N_z}F_z^{\mathsf T}\mathsf T_zF_z,
\]
suppose \(N_z\ge n_*>0\), \(\|F_z\|\le F_*\), and the linear maps from field variations to \(N_z,F_z\) have norms \(L_N,L_F\). For \(k\ge2\),
\[
\boxed{
\|D^kW_{2,z}\|
\le
\frac{|w_z|\|\mathsf T_z\|\,k!}{4}
\frac{L_N^{k-2}}{n_*^{k-1}}
\left(L_F+\frac{F_*L_N}{n_*}\right)^2.
}
\tag{13}
\]
Sum this bound over quadrature nodes. The numerical prefactors for \(k=2,3,4\) are \(1/2,3/2,6\).

For the **cubic bulk tensor actually printed** in Appendix `app:charge`, Cauchy–Schwarz gives the fallback bound
\[
0\preceq\mathsf T\preceq
\left(\int_p\frac{|q|^2}{\varepsilon^3}\right)I
\preceq\frac{1}{ma^3}I.
\tag{14}
\]
At \(ma=0.2\), this is \(\|\mathsf T\|\le5/a^2\). Interval quadrature should improve it substantially.

Equation (13) does not manufacture the missing finite-chain endpoint prescription. Its definition and error must be supplied separately.

---

## 2. Sharp propagation from a Hessian error to the charge operator

### 2.1 Keep the complete centered Wick–Taylor operator

Fix the expansion covariance \(C=\bar C\) and prepared ordering projector \(P=P_\star\). Write
\[
\Delta=C-P,\qquad
d_\mu=\operatorname{Tr}(A_\mu\Delta),\qquad
n_\mu=c^\dagger A_\mu c.
\]

For a real symmetric matrix \(K\), define the centered Hessian contribution
\[
\boxed{
\mathcal Q_K=
\frac12\sum_{\mu\nu}K_{\mu\nu}
\left[
:n_\mu n_\nu:_P
-d_\mu:n_\nu:_P-d_\nu:n_\mu:_P
+d_\mu d_\nu I
\right].
}
\tag{15}
\]

This includes the scalar and one-body counterterms. Bounding only the quartic term is insufficient.

For an observable \(O\) with first derivative represented by \(h_O\),
\[
\widehat O_2-OI=:c^\dagger h_Oc:_C+\mathcal Q_{V_O}.
\tag{16}
\]
Its constant is therefore
\[
\mathcal E_{H,O}
=
O-\operatorname{Tr}(h_O\Delta)+\tfrac12d^{\mathsf T}V_Od.
\tag{17}
\]

This is the centered Taylor convention consistent with the function printed in `00_P3_current_nc.tex`,
> “\(h_H[X]=\sum_{\mu\nu}V_{\mu\nu}A_\mu\Tr(A_\nu X)\)”,

and its line
> “\(\widehat E_2=\mathcal E_H+:c^\dagger(h[\bar C]-h_H[\bar\Delta_\star])c:_{P_\star}+\widehat V_\star\)”.

The implementations must emit and check (17). Any different scalar convention is an additional scalar error, not an innocuous change.

### 2.2 Exact mean and second moment

Define
\[
F_K=\sum_{\mu\nu}K_{\mu\nu}A_\mu\Delta A_\nu,\qquad
b_K=-\frac12\operatorname{Tr}(F_K\Delta),
\tag{18}
\]
and put \(Q=I-C\). Let
\[
X_\mu=Q^{1/2}A_\mu C^{1/2}.
\]
On the antisymmetric two-particle space define
\[
D_K=
\left.
P_{\wedge}\left(\sum_{\mu\nu}K_{\mu\nu}X_\mu\otimes X_\nu\right)
P_{\wedge}
\right|_{\wedge^2\mathcal H}.
\tag{19}
\]

Wick reordering from \(P\) to \(C\) gives
\[
\mathcal Q_K
=
b_KI-:c^\dagger F_Kc:_C
+\frac12\sum_{\mu\nu}K_{\mu\nu}:n_\mu n_\nu:_C.
\tag{20}
\]
The scalar, quadratic, and quartic terms are mutually orthogonal in the Gaussian state inner product. Hence
\[
\boxed{
\langle\mathcal Q_K\rangle_C=b_K,\qquad
\|\mathcal Q_K\|_C^2
=
b_K^2+\|Q^{1/2}F_KC^{1/2}\|_F^2+\|D_K\|_F^2,
}
\tag{21}
\]
where \(\|Y\|_C^2=\operatorname{Tr}(\rho_CY^\dagger Y)\).

For a pure Slater state, this is simply orthogonality of the vacuum, single particle-hole, and double particle-hole sectors. For a mixed Gaussian state, Gaussian purification gives the square-root factors in (19)–(21).

In particular, the quantity required for the branch is exactly
\[
\boxed{
q_\infty=
\frac4{E^2}\left[
b_{V_E}^2+
\|Q^{1/2}(h_E-F_{V_E})C^{1/2}\|_F^2+
\|D_{V_E}\|_F^2
\right].
}
\tag{22}
\]
The analogous formula holds for every interior mass. No many-body diagonalization of the branch projector is needed.

The double term can be evaluated by traces, not by storing a four-index Fock tensor:
\[
\begin{aligned}
\|D_K\|_F^2
=\frac12\sum_{\mu\nu\alpha\beta}
K_{\mu\nu}K_{\alpha\beta}
\big[
&\operatorname{Tr}(X_\mu^\dagger X_\alpha)
 \operatorname{Tr}(X_\nu^\dagger X_\beta)\\
&-\operatorname{Tr}(X_\mu^\dagger X_\alpha
                    X_\nu^\dagger X_\beta)
\big].
\end{aligned}
\tag{23}
\]
The total expression is real and nonnegative; cancellation must be enclosed numerically.

### 2.3 The optimal all-direction constant

Choose a Frobenius-orthonormal basis \(S_a\) of real symmetric \(d\times d\) matrices. Its off-diagonal basis elements have entries \(1/\sqrt2\), not \(1\). Define
\[
Z_a=\mathcal Q_{S_a},\qquad
G_{ab}=\frac12\langle Z_aZ_b+Z_bZ_a\rangle_C.
\tag{24}
\]
Then \(G\succeq0\). For \(K=\sum_a e_aS_a\),
\[
\|\mathcal Q_K\|_C^2=e^{\mathsf T}Ge,\qquad
\|K\|_F=\|e\|_2.
\]
Therefore
\[
\boxed{
\|\mathcal Q_{V-\widehat V}\|_C
\le L_*\|V-\widehat V\|_F,\qquad
L_*=\sqrt{\lambda_{\max}(G)}.
}
\tag{25}
\]

**This constant is sharp.** Equality is attained by a top eigenvector of \(G\). It covers errors orthogonal to the computed Hessian as well as parallel errors.

For a certified ellipsoidal error set
\[
e=Dz,\qquad \|z\|_2\le1,
\]
the sharp constant is instead
\[
\boxed{\epsilon_{\rm Hess}=\sqrt{\lambda_{\max}(D^{\mathsf T}GD)}.}
\tag{26}
\]
A componentwise interval box is not automatically this unit ellipsoid. A safe box bound is
\[
|e_a|\le r_a
\quad\Longrightarrow\quad
\|\mathcal Q_K\|_C^2\le r^{\mathsf T}|G|r.
\tag{27}
\]

### 2.4 A cheaper explicit constant

The full matrix \(G\) need not be assembled. Define
\[
\Xi_{\mu\nu}=\operatorname{Tr}(A_\mu\Delta A_\nu\Delta),\quad
S_{\mu\nu}=\operatorname{Tr}(X_\mu^\dagger X_\nu),\quad
A_2=\sum_\mu A_\mu^2,
\]
and
\[
\alpha_C=\|C^{1/2}A_2C^{1/2}\|_2,\qquad
\alpha_Q=\|Q^{1/2}A_2Q^{1/2}\|_2.
\]
Then
\[
\boxed{
L_*^2\le L_{\rm safe}^2:=
\frac14\|\Xi\|_F^2+
\alpha_Q\alpha_C\|\Delta\|_F^2+
\|S\|_2^2.
}
\tag{28}
\]

To prove the middle term, write \(\mathbb A=(A_1,\ldots,A_d)\) and
\[
Q^{1/2}F_KC^{1/2}
=
Q^{1/2}\mathbb A(K\otimes\Delta)\mathbb A^\dagger C^{1/2}.
\]
The first and third terms follow respectively from Cauchy–Schwarz and the Gram matrix of the tensors \(X_\mu\otimes X_\nu\). Antisymmetric compression cannot increase the Frobenius norm.

Thus a directly usable certificate is
\[
\boxed{\epsilon_{\rm Hess}=L_{\rm safe}\epsilon_V.}
\tag{29}
\]
Use (25) or (26) only when the cheaper bound misses the numerical gate.

For the mean correction, a sharper and inexpensive bound is
\[
\boxed{
|b_V-b_{\widehat V}|
\le \frac12\|\Xi\|_F\,\epsilon_V.
}
\tag{30}
\]

### 2.5 Errors that are not Hessian errors

Changing from the held projector to \(P_\star\) changes \(\Delta\), hence \(b_K\) and \(F_K\). It is not covered by a bound on \(K\) alone. At fixed \(C,V\), the double term \(D_V\) is unchanged by this ordering change, but the scalar and single terms change explicitly through (18).

Likewise, include errors in:

* the first derivative \(h_O\);
* the Taylor constant;
* the exact slice covariance;
* the diagnostic window;
* the source/reference prescription.

The clean implementation is to enclose the scalar, single, and double quantities in (22) directly. Alternatively, append scalar and one-body error coordinates to the Gram construction. Do not turn a source-value comparison into a derivative bound.

---

## 3. Propagation to branch weights and mass biases

### 3.1 Boundary branch weight

Let
\[
D_E=\widehat E_2-EI,\qquad E\ge E_->0.
\]
Suppose the computed centered operator \(\widetilde D_E\) satisfies
\[
\langle\widetilde D_E^2\rangle_C\le \widetilde s_E,
\qquad
\|D_E-\widetilde D_E\|_C\le\epsilon_E.
\]
The triangle inequality gives
\[
\boxed{
q_\infty\le q_\infty^{\rm up}
:=
\frac{4(\sqrt{\widetilde s_E}+\epsilon_E)^2}{E_-^2}.
}
\tag{31}
\]

For the paper’s actual projector
\[
\Pi=\mathbf1_{[E/2,\,3E/2]}(\widehat E_2),
\]
spectral calculus gives
\[
I-\Pi\le4(\widehat E_2/E-I)^2.
\]
Therefore
\[
\boxed{\operatorname{Tr}(\rho_C\Pi)\ge
\max\{0,\,1-q_\infty^{\rm up}\}.}
\tag{32}
\]

No spectral gap at either branch endpoint is required. Conversely, without such a gap, a small Hessian error generally does **not** imply a small projector-norm error. Equation (32) avoids that problem.

If \(\widetilde D_E=\widetilde E_2-\widetilde E I\), then \(\epsilon_E\) must enclose
\[
(\widehat E_2-\widetilde E_2)-(E-\widetilde E)I.
\]
Do not omit the scalar error, or count it twice when the centered representation already cancels it.

### 3.2 Interior masses: probabilities and unconditioned biases

For each fixed mass test, set
\[
D_j=\widehat M_j-M_jI,\qquad M_j\ge M_{j,-}>0.
\]
With corresponding bounds \(\widetilde s_j,\epsilon_j\), define
\[
r_j^{\rm up}=
\frac{\sqrt{\widetilde s_j}+\epsilon_j}{M_{j,-}}.
\]
Then
\[
\boxed{
q_{\rm int}\le4\sum_j(r_j^{\rm up})^2,\qquad
\Pr_C(\widehat M_j\le0)\le(r_j^{\rm up})^2.
}
\tag{33}
\]

For the signed Gaussian mean bias,
\[
\langle\widehat M_j\rangle_C-M_j=b_{V_j}.
\]
If \(\widetilde b_j\) is its computed value and \(\epsilon_{b,j}\) its enclosed error,
\[
\boxed{
\frac{\langle\widehat M_j\rangle_C-M_j}{M_j}
\in
\frac{[\widetilde b_j-\epsilon_{b,j},
       \widetilde b_j+\epsilon_{b,j}]}
     {[M_{j,-},M_{j,+}]}.
}
\tag{34}
\]
Use (30) for the Hessian part of \(\epsilon_{b,j}\), then include ordering, state, gradient, and scalar errors.

A certified \(q_{\rm int}\le0.080\) implies, individually,
\[
\Pr_C(\widehat M_j\le0)\le0.020,\qquad
\frac{|\langle\widehat M_j\rangle_C-M_j|}{M_j}
\le\sqrt{0.020}=0.141421\ldots .
\tag{35}
\]
The latter is only a coarse mean-bias consequence. Equation (34) should be much more informative.

The interior “test subspace” can be defined as the spectral subspace below one of
\(4\sum_j(\widehat M_j/M_j-I)^2\). It is not the boundary branch, and noncommuting mass tests must not be silently treated as a joint spectral event.

### 3.3 Bias after projection onto the boundary branch

If “mass bias” means the expectation **after** applying \(\Pi\), an additional bound is needed.

Let
\[
X_j=\widehat M_j/M_j-I,\quad
\mu_j=\langle X_j\rangle_C,\quad
s_j=\langle X_j^2\rangle_C,
\]
and let \(p=1-\langle\Pi\rangle_C\). For
\(\rho_+=\Pi\rho_C\Pi/(1-p)\),
\[
\boxed{
|\operatorname{Tr}(\rho_+X_j)|
\le
\frac{|\mu_j|+2\sqrt{p\,s_j}+p\|X_j\|_{\rm op}}{1-p}.
}
\tag{36}
\]
This follows by expanding
\[
\Pi X_j\Pi=X_j-(I-\Pi)X_j-X_j(I-\Pi)
 +(I-\Pi)X_j(I-\Pi)
\]
and applying Cauchy–Schwarz.

Thus a branch-conditioned mass-bias claim requires a bound on
\(\|X_j\|_{\rm op}\), or a sharper bound on the displayed off-branch blocks. The unconditioned mass variance alone does not suffice when
\([\Pi,\widehat M_j]\ne0\).

If they commute, the simpler bound is available:
\[
|\operatorname{Tr}(\rho_+X_j)|\le\sqrt{s_j/(1-p)}.
\tag{37}
\]
Commutation must be proved, not assumed.

### 3.4 Numerical margins supplied by the printed numbers

With exact normalization and all errors collected into the centered-operator error, (31) reads
\[
\sqrt{q_\infty^{\rm true}}
\le
\sqrt{q_\infty^{\rm nominal}}+2\epsilon_E/E.
\]

Therefore a nominal \(q_\infty\le10^{-3}\) can tolerate
\[
\boxed{
\epsilon_E/E
\le
\frac{\sqrt{0.080}-\sqrt{0.001}}2
=0.1256099679\ldots
}
\tag{38}
\]
and still establish branch weight at least \(0.92\).

The interior endpoint is much tighter. If the unrounded nominal value were exactly \(0.0798\), then
\[
\boxed{
\left[\sum_j(\epsilon_j/M_j)^2\right]^{1/2}
\le
\frac{\sqrt{0.080}-\sqrt{0.0798}}2
\simeq1.77\times10^{-4}.
}
\tag{39}
\]
Other errors and normalization uncertainty consume this margin. The printed rounded \(0.0798\) is not adequate input for this gate.

These are **state-norm error allowances**, not relative Hessian percentages. Convert them using (25) or (28).

### 3.5 Sharp refinement if the triangle bound fails

For a fixed state and normalization, the worst second moment over an ellipsoidal error set can be computed exactly.

Let \(Z(e)=\sum_a e_aZ_a\), \(e=Dz\), \(\|z\|\le1\), and define
\[
s_0=\langle\widetilde D_E^2\rangle,\quad
c_a=\operatorname{Re}\langle\widetilde D_EZ_a\rangle,\quad
H=D^{\mathsf T}GD,\quad g=D^{\mathsf T}c.
\]
Then
\[
\boxed{
\sup_{\|z\|\le1}
\langle(\widetilde D_E+Z(Dz))^2\rangle
=
s_0+
\inf_{\lambda>\lambda_{\max}(H)}
\left[\lambda+g^{\mathsf T}(\lambda I-H)^{-1}g\right],
}
\tag{40}
\]
including the limiting endpoint in the usual degenerate case.

Completing the square proves the upper bound; the maximizing vector, with a possible top-eigenspace component, attains it. This is the sharp worst-case propagation for that certified ellipsoid.

---

## 4. Two-route computation on the eighteen slices

### Target and inputs

First fix the target as the **declared-reference, finite-parent, ten-channel operator**, or explicitly label the result as held-reference. Do not combine the two.

For every slice, both routes need:

1. **An exact eighteen-row manifest:** history identifier, \(\sigma/a\), stored time, chain lengths, channel set, boundary conditions, and diagnostic radii/windows. The complete eighteen-row list is absent from the supplied files.
2. **Quantum inputs:** prepared fields \(u_\star\), prepared occupations and \(P_\star\), propagated sea plus dust, and a rigorous enclosure of the full slice covariance. Occupied-orbital representations are preferable when available.
3. **The complete finite-parent equations:** vertices \(A_\mu\), averaging and back-averaging maps, radial rows, origin treatment, outer normalization, and the reference functional with its endpoint terms.
4. **Copy bookkeeping:** the paper prints “the eight valleys' weight \(16k\)”. Implement these as independent copies in contractions. A connected trace carries the copy weight once; disconnected products carry products of weights. Multiplying one random variable by \(16k\) gives the wrong variance.
5. **Numerical provenance:** precision, rounding method, solve tolerances, all omitted-mode/tail bounds, and any stored Hessian or response matrix used merely as an initial approximation.

Use the history’s actual kernel. `00_P3_current_nc.tex` says:
> “Sources and fields use a Gaussian of width \(\sigma=4a\)”.

Substituting the compact-support kernel from the model declaration changes the calculation.

A fixed snapshot audit does not verify that the snapshot belongs to an evolution under \(D\mathcal W\). The paper explicitly says:
> “no stored history is verified under Eq.~\eqref{eq:dm-reference}'s source prescription”.

That trajectory question remains separate.

### Route A: response residual plus excitation-sector contractions

For each slice:

1. Reconstruct the vertices, preparation reference, and full target sources.
2. Re-solve the declared-reference fields from the stored covariance.
3. Validate the root and \(J^{-1}\) using (3)–(4).
4. Compute all response columns \(U\); enclose their residual using (5).
5. Obtain \(V_E\) from (2), with the discrete generating identity checked.
6. Obtain each \(V_j\) and mass gradient from (9).
7. Evaluate (18), (22), and (23), using interval arithmetic.
8. Propagate all-direction error using (28) first; use (25), (26), or (40) only if needed.

### Route B: independently differentiated charge plus generic Wick contractions

Reconstruct the same physical problem independently, without importing Route A’s fields, projectors, Jacobians, response matrices, Hessians, or contraction outputs.

1. Validate the target fields with an independently implemented solve/certificate.
2. Differentiate the directly evaluated boundary charge and masses using second-order interval jets or the adjoint formula (12). Do not use \(\nabla_xE=Lu\) as this route’s Hessian construction.
3. Evaluate the centered operator’s first and second moments by a separate CAR/Wick recursion, through products of eight fermionic operators.
4. Produce its own actual-error enclosure and final branch/mass intervals.

Both implementations should pass exact small-Fock-space tests of:

* scalar and centering constants;
* diagonal and off-diagonal Hessian perturbations;
* directions Frobenius-orthogonal to the nominal Hessian;
* the one-particle Hartree–Fock cancellation;
* the prepared-sea mean-zero property, **without incorrectly setting its variance to zero**;
* copy multiplicities.

These tests check implementation. The norm/residual proof, not sampling perturbation directions, establishes error inclusion.

### Independence and scope

`01_requirements_new.txt`, section 5, requires:
> “two independent routes agreeing within their stated errors before anything enters the paper”.

`00_P3_current_nc.tex` additionally warns:
> “two post-processing calculations on the same stored data not being two independent calculations”.

Accordingly, two scripts consuming a shared exported Hessian or shared Wick moments do not qualify. The specification above independently reconstructs the derivative and quantum calculations.

If both routes start from the same certified covariance snapshots, report them accurately as **two independent slice calculations conditional on those states**, not as two independent collapse evolutions. Independent evolution claims require independent reconstruction or propagation of the states as well.

### Acceptance gate

Each route must independently pass all of the following on every requested slice:

* The same target functional, ordering projector, channel/copy convention, and diagnostic windows.
* Positive normalization intervals \(E_->0\), \(M_{j,-}>0\).
* A regular target root box and a verified full-Jacobian inverse bound.
* Actual-error inclusion for the full Hessian, not just agreement between implementations.
* Enclosed state, reference, first-derivative, scalar, window, tail, and roundoff errors.
* \(q_\infty^{\rm up}\le0.080\) for the **boundary** claim.
* \(q_{\rm int}^{\rm up}\le0.080\) separately if that interior claim is retained.
* Signed mass-bias intervals from (34); branch-conditioned biases only with the additional ingredients in (36).
* Agreement of independently certified intervals and matching physical definitions.

A failed sufficient bound means **not certified**, not a physical failure. Tighten the root box, precision, derivative enclosure, or retained rank according to the dominant error term.

Output one row per slice containing at least
\[
\begin{gathered}
E,\ M_j,\ \eta,\ \kappa,\ \beta,\ r_U,\ 
\epsilon_{V_E},\epsilon_{V_j},\
L_{\rm safe},\epsilon_E,\epsilon_j,\\
q_\infty^{\rm up},\quad 1-q_\infty^{\rm up},\quad
q_{\rm int}^{\rm up},\quad
\Pr(\widehat M_j\le0)^{\rm up},\quad
\text{signed mass-bias intervals},
\end{gathered}
\]
with separate error contributions and immutable input identifiers.

The additional \(T/r_s=11.0\) branch check requested by the paper is a separately counted row, not silently part of the eighteen.

---

## 5. Cost and feasible implementation

The supplied paper specifies:
> “ten 2048-site chains”,

and a two-component operator per chain. Thus each channel has one-body dimension \(n=4096\).

* One complex dense matrix per channel, across ten channels, occupies about **2.5 GiB**.
* If there are two retained moments per radial site, \(d=4096\), one real dense Hessian occupies **128 MiB**.
* Then the symmetric-Hessian coordinate count is
  \(d(d+1)/2=8{,}390{,}656\). A dense Gram matrix (24) would require about **512 TiB**. **Do not assemble it.** Use (28), matrix-free actions, or certified low-rank compression.

For \(f\) field variables, \(d\) moments, and \(J_m\) mass tests, dense field work per slice scales as
\[
O\!\left(f^3+f^2d+
J_m(f^2d+fd^2)\right),
\tag{41}
\]
with one factorization reused for response and adjoint solves. Local/banded structure can reduce this, but Gaussian-kernel truncation requires an explicit tail bound.

For a rank-\(r\) representation
\[
\widetilde V=\sum_{\ell=1}^r\lambda_\ell z_\ell z_\ell^{\mathsf T},
\]
form composite vertices \(B_\ell=\sum_\mu(z_\ell)_\mu A_\mu\). The pair contractions can then be evaluated without a four-index tensor. A conservative dense cost is
\[
O(r^2 K n^3),\qquad K=10,
\tag{42}
\]
with streaming to limit memory. The discarded Hessian tail is included through
\[
\|\mathcal Q_{V-\widetilde V}\|_C
\le L_{\rm safe}\|V-\widetilde V\|_F.
\]
Do not assume a small useful rank before measuring it.

The requested job comprises **thirty-six independently certified slice calculations**. If each needs ten dense one-body decompositions, the cubic-work scale alone is
\[
36\times10\times4096^3
\simeq2.47\times10^{13}
\]
cubic operation units, before implementation-dependent constants and interval overhead.

No wall-clock estimate is justified by the attachments. Measure:

* the retained rank needed to meet the state-error margin;
* the effective Jacobian bandwidth or sparse-factor fill;
* one validated factorization/response cost;
* one nominal Wick-contraction cost;
* interval overhead and peak memory.

Those measurements determine whether the inexpensive bound (28) suffices or whether the sharper optimization is worth its cost.

---

## Bottom line

The mathematical gap can be closed:

\[
\boxed{
\text{validated response residual}
\ \Longrightarrow\
\|V-\widehat V\|_F
\ \Longrightarrow\
\|\Delta\widehat O_2\|_C
\ \Longrightarrow\
\text{branch-weight and mass-bias bounds}.
}
\]

The first implication is (7) or (11); the sharp second implication is (25); the final implications are (31)–(36). None assumes parallel Hessian errors.

The printed numbers also give a useful computational result: the nominal boundary endpoints below \(10^{-3}\) have a substantial \(0.12561\,E\) centered-operator error allowance for a \(0.92\) branch-weight claim, whereas retaining the interior endpoint \(0.0798\le0.080\) permits only about \(1.77\times10^{-4}\) in the stacked normalized mass-error norm.

**What remains to measure is now specific:** the full target-law response residuals, inverse bounds, Wick sensitivity constants, and non-Hessian error terms on the eighteen identified slices. Until those certificates exist, neither the supplied parallel-error statement nor the five nominal boundary readings proves the eighteen-slice boundary-branch bound.