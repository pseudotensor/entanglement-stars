# The computed Hessian's error, in every direction: proof, propagation, and the two-lane computation

## 0. The result in one paragraph

Every second-order mass operator of Sec. V (the boundary charge $\widehat E_2$ that defines the branch, and the interior diagnostics $\widehat M_j$ at $R_{50}$ and the trapping radii) is affine in its Hessian. Its deviation in the Hartree state, $\langle 4(\widehat M_j/M_j-I)^2\rangle$, is therefore a convex quadratic function of the Hessian, and an error $\delta V$ of arbitrary direction enters it through the norm $\|\widehat Y_j\bar\Omega\|$ of one quartic operator acting on the Hartree Slater state. That norm decomposes exactly into a mean, a one-pair part, and a two-pair part, and each is bounded by $\|\delta V\|_F$ times a Gaussian trace of the stored slice: the moment covariance $T$, its pair contraction $\Xi$, and an exchange constant $\gamma_F$. The three constants are attained in direction, so the direction-free bound cannot be sharpened; what can be sharpened is $\|\delta V\|_F$, and that is enclosed a posteriori, for every direction at once, by the residual of the field-response equations times the constraint Jacobian's verified inverse norm (the audit that gives 2.90). One step of residual refinement drives the enclosure to rounding level. With the paper's own magnitudes, the propagated change of $\langle\mathcal B\rangle$ is below $10^{-6}$ on every slice, so 0.080 (0.0798 at 15.0) stands without the parallel-error assumption, the mass biases move by less than $10^{-6}E$, and the branch weight bound inherits the same correction. The spec below gives the two lanes what they compute, the gate, and the cost.

## 1. Objects

One-body space $\mathcal H$: all ten channels with their $16k$ copies; a trace over $\mathcal H$ of a copy-diagonal operator is $\sum_k 16k\,\mathrm{Tr}_k$, so copy weights never appear separately. Vertices $A_\mu$, $\mu=1,\dots,n_m$ ($n_m=4096$: the energy vertex $e_x$ and the radial-current vertex $p_{x,r}$ on each of 2048 chain sites, angular averaged). Bilinears $N_\mu=c^\dagger A_\mu c$, bare moments $m_\mu=\mathrm{Tr}(A_\mu C)$.

Test state: the Hartree Slater determinant $\bar\Omega$ of the stored slice, covariance $\bar C=P$ with $P^2=P$, $P'=I-P$. Reference projector $P_\star$ (for the held-source operators, the fixed reference projector of the slice); $\bar\Delta=P-P_\star$, $\bar d_\mu=\mathrm{Tr}(A_\mu\bar\Delta)$. Centered moment operators $\hat\nu_\mu=N_\mu-\mathrm{Tr}(PA_\mu)$, so $\langle\hat\nu_\mu\rangle_{\bar\Omega}=0$.

Second-order operator of diagnostic $j\in\{R_{50},R_t^{\rm out},R_t^{\rm in},\infty\}$ with coefficients $(M_j,g^{(j)},V^{(j)})$:
$$
\widehat M_j=M_j+\sum_\mu g^{(j)}_\mu\hat\nu_\mu+\sum_{\mu\nu}V^{(j)}_{\mu\nu}\hat Q_{\mu\nu},\qquad
\hat Q_{\mu\nu}=\tfrac12\bigl[{:}N_\mu N_\nu{:}^{(4)}_{P_\star}-\bar d_\mu{:}N_\nu{:}_{P_\star}-\bar d_\nu{:}N_\mu{:}_{P_\star}+\bar d_\mu\bar d_\nu\bigr],
\tag{1}
$$
where ${:}\cdot{:}^{(4)}_{P_\star}$ is the fully Wick-ordered quartic against $P_\star$ and ${:}N{:}_{P_\star}=N-\mathrm{Tr}(P_\star A)$. For $j=\infty$, $M_\infty=E$, $g^{(\infty)}=\bar u$ (the back-averaged fields), and (1) is the paper's $\widehat E_2=\mathcal E_H+{:}c^\dagger(h[\bar C]-h_H[\bar\Delta_\star])c{:}_{P_\star}+\widehat V_\star$: expanding the brackets in (1) gives $\widehat V_\star$, the one-body term $-{:}c^\dagger h_H[\bar\Delta]c{:}_{P_\star}$, and the constant $\tfrac12\langle\bar d,V\bar d\rangle$ that belongs to $\mathcal E_H$.

The stored operator uses computed coefficients $(g^{(j),c},V^{(j),c})$; the true ones are $g^{(j)}=g^{(j),c}+\delta g^{(j)}$, $V^{(j)}=V^{(j),c}+\delta V^{(j)}$, with $V^{(j)}$ real symmetric (reciprocity: a Hessian) and $\delta V^{(j)}$ of unknown direction. Deviation operators $D_j=\widehat M_j/M_j-I$, $B_j=4D_j^2$, $\mathcal B=\sum_jB_j$ (interior) and $\mathcal B_\infty=B_\infty$ (branch). The "parallel" case of the paper is $\delta V=\varepsilon V^{c}$.

Scope: the error certified here is the error of the implemented coefficients relative to the Hessian of the implemented held-source functional; the replacement of the held-source law by Eq. (dm-reference) is the separate to-do and is untouched.

## 2. The direction-free bound

### 2.1 Re-expansion about the Hartree state

Two Wick identities, valid for any projectors $P_\star,P$, Hermitian $X,Y$, $\bar\Delta=P-P_\star$ (contractions $\langle c^\dagger_Ac_B\rangle_P=P_{BA}$, $\langle c_Bc^\dagger_A\rangle_P=(I-P)_{BA}$):
$$
{:}N_X{:}_P\,{:}N_Y{:}_P={:}N_XN_Y{:}^{(4)}_P+{:}c^\dagger[X(I-P)Y-YPX]c{:}_P+\mathrm{Tr}\bigl(PX(I-P)Y\bigr),
\tag{2}
$$
$$
{:}N_XN_Y{:}^{(4)}_{P_\star}={:}N_XN_Y{:}^{(4)}_P+\bar d_X{:}N_Y{:}_P+\bar d_Y{:}N_X{:}_P-{:}c^\dagger(X\bar\Delta Y+Y\bar\Delta X)c{:}_P+\bar d_X\bar d_Y-\mathrm{Tr}(X\bar\Delta Y\bar\Delta).
\tag{3}
$$
(2) is Wick's theorem for the product of two $P$-centered bilinears; (3) follows by writing (2) for both references and subtracting, using $I-P_\star=I-P+\bar\Delta$ and ${:}c^\dagger Zc{:}_{P_\star}={:}c^\dagger Zc{:}_P+\mathrm{Tr}(\bar\Delta Z)$. A consequence of (2) is the Gaussian expectation $\langle{:}N_XN_Y{:}^{(4)}_{P_\star}\rangle_C=\mathrm{Tr}(\Delta X)\mathrm{Tr}(\Delta Y)-\mathrm{Tr}(X\Delta Y\Delta)$, $\Delta=C-P_\star$ (checked on the two-mode example $P_\star=|1\rangle\langle1|$, $C=|2\rangle\langle2|$, $X=\sigma_x$: both sides equal 2).

Inserting (3) and ${:}N{:}_{P_\star}={:}N{:}_P+\bar d$ into (1), the $\bar d$ terms cancel identically:
$$
\hat Q_{\mu\nu}=\tfrac12{:}N_\mu N_\nu{:}^{(4)}_P-\tfrac12{:}c^\dagger(A_\mu\bar\Delta A_\nu+A_\nu\bar\Delta A_\mu)c{:}_P+q_{\mu\nu},\qquad q_{\mu\nu}=-\tfrac12\mathrm{Tr}(A_\mu\bar\Delta A_\nu\bar\Delta).
\tag{4}
$$
So, relative to the Hartree state itself, the quartic term has no Hartree-type (tadpole) one-body part: the centering at $\bar C$ removes it. (Relative to the matter-free comparator $P_g$ it is present, which is the paper's large $B$ singles; those are not at issue here.)

### 2.2 The three orthogonal components

For a real symmetric $W$ and a real vector $w$ put
$$
\widehat Y[w,W]=\sum_\mu w_\mu\hat\nu_\mu+\sum_{\mu\nu}W_{\mu\nu}\hat Q_{\mu\nu},\qquad
A[w]=\sum_\mu w_\mu A_\mu,\qquad F_W[X]=\sum_{\mu\nu}W_{\mu\nu}A_\mu XA_\nu .
$$
By (4), $\widehat Y\bar\Omega=\langle\widehat Y\rangle\bar\Omega+y_1+y_2$ with $y_1$ in the one-particle-hole sector and $y_2$ in the two-pair sector (a $P$-normal-ordered one-body operator creates one pair, a $P$-normal-ordered quartic creates two), mutually orthogonal, so
$$
\|\widehat Y\bar\Omega\|^2=\langle\widehat Y^2\rangle=\langle\widehat Y\rangle^2+\|y_1\|^2+\|y_2\|^2 .
\tag{5}
$$
Mean. $\langle\widehat Y\rangle=\sum W_{\mu\nu}q_{\mu\nu}=-\tfrac12\,W{:}\Xi$, with the pair contraction $\Xi_{\mu\nu}=\mathrm{Tr}(A_\mu\bar\Delta A_\nu\bar\Delta)$, real symmetric.

One pair. ${:}c^\dagger Zc{:}_P\bar\Omega=\sum_{a\in{\rm emp},\,i\in{\rm occ}}Z_{ai}c^\dagger_ac_i\bar\Omega$ with orthonormal terms, hence
$$
\|y_1\|=\bigl\|P'\bigl(A[w]-F_W[\bar\Delta]\bigr)P\bigr\|_F .
\tag{6}
$$

Two pairs. Write $\tfrac12\sum W_{\mu\nu}{:}N_\mu N_\nu{:}^{(4)}_P\bar\Omega=\sum_{abij}\Phi_{ab,ij}\,c^\dagger_ac^\dagger_bc_jc_i\bar\Omega$ with $\Phi_{ab,ij}=\tfrac12\sum W_{\mu\nu}(A_\mu)_{ai}(A_\nu)_{bj}$. Only the part of $\Phi$ antisymmetric in $(a,b)$ and in $(i,j)$ survives; using the symmetry $\Phi_{ba,ji}=\Phi_{ab,ij}$ of symmetric $W$,
$$
\|y_2\|^2=4\sum|\Phi^A|^2=2\sum_{abij}|\Phi_{ab,ij}|^2-2\,\mathrm{Re}\sum_{abij}\Phi^*_{ab,ij}\Phi_{ba,ij}
=\tfrac12\mathrm{Tr}\bigl(W\bar T W T\bigr)-\tfrac12X_4(W),
\tag{7}
$$
$$
T_{\mu\nu}=\mathrm{Tr}(PA_\mu P'A_\nu),\qquad
X_4(W)=\mathrm{Re}\!\sum_{\kappa\lambda\mu\nu}W_{\kappa\lambda}W_{\mu\nu}\mathrm{Tr}\bigl(PA_\kappa P'A_\nu PA_\lambda P'A_\mu\bigr),
$$
where $\sum|\Phi|^2=\tfrac14\sum W_{\kappa\lambda}W_{\mu\nu}T_{\kappa\mu}T_{\lambda\nu}=\tfrac14\|T^{1/2}W\bar T^{1/2}\|_F^2$ (bar is complex conjugation; $T$ is the Hermitian positive covariance matrix of the bare moments in the Hartree state, $T_{\mu\nu}={\rm Cov}_{\bar\Omega}(N_\mu,N_\nu)$, the paper's $\mathrm{Tr}[CA(I-C)A]$). Since $\Phi\mapsto\Phi^A$ is an orthogonal projection, $|X_4|\le\sum|\Phi|^2$ and
$$
\|y_2\|^2\le\|T^{1/2}W\bar T^{1/2}\|_F^2\le\|T\|_{\rm op}^2\,\|W\|_F^2 .
\tag{8}
$$
(Checks: a single vertex $X$ gives $\|y_2\|^2=\tfrac12\|P'XP\|_F^4-\tfrac12\mathrm{Tr}[(P'XPXP')^2]$, which vanishes for a single particle-hole entry by Pauli and equals 4 for two entries, as it must.)

### 2.3 Result A: explicit constants, and their sharpness

For $\|W\|_F\le\epsilon$ and $\|w\|_2\le\epsilon_g$, from (5)–(8) and $\|P'A[w]P\|_F^2=w^{\sf T}Tw$:
$$
\|\widehat Y[w,W]\bar\Omega\|\le\Lambda_s\,\epsilon+\|T\|_{\rm op}^{1/2}\,\epsilon_g,\qquad
\boxed{\;\Lambda_s=\tfrac12\|\Xi\|_F+\gamma_F+\|T\|_{\rm op}\;}
\tag{9}
$$
with the exchange constant from the Schur test on the pair-space kernel $\mathcal K_{(\mu\nu),(\kappa\lambda)}=\mathrm{Tr}(A_\lambda\bar\Delta A_\kappa A_\mu\bar\Delta A_\nu)$ (nonzero only when $\kappa$ overlaps $\mu$ and $\lambda$ overlaps $\nu$, the vertices being local),
$$
\gamma_F^2=\max_{(\mu\nu)}\sum_{(\kappa\lambda)}\bigl|\mathrm{Tr}(A_\lambda\bar\Delta A_\kappa A_\mu\bar\Delta A_\nu)\bigr|,\qquad \|F_W[\bar\Delta]\|_F\le\gamma_F\|W\|_F .
$$
Sharpness. $\tfrac12\|\Xi\|_F\epsilon$ is attained at $W\propto\Xi$; $\|T\|\epsilon$ is attained within the exchange factor 2 at $W=\epsilon\,ww^{\sf T}$, $w$ the top eigenvector of $\mathrm{Re}\,T$; $\gamma_F$ is a Schur bound of an attainable norm. So no direction-free bound with these inputs improves (9) by more than a factor three, and all gain must come from the size of $\epsilon$ (Sec. 3). The parallel case is recovered by $W=\varepsilon V^c$, where $\|\widehat Y\bar\Omega\|=|\varepsilon|\,\mathcal Q_j$ with the computed $\mathcal Q_j^2=(b_j^c)^2+\|P'F_{V^c}[\bar\Delta]P\|_F^2+\|y_2[V^c]\|^2$; the direction-free bound replaces $|\varepsilon|\mathcal Q_j$ by $\Lambda_s\|\delta V\|_F$, and the ratio $\alpha_j=\Lambda_s\|V^{(j),c}\|_F/\mathcal Q_j\ge1$ measures the price of not knowing the direction.

### 2.4 Result B: the printed quantities

$D_j^{\rm true}=D_j^{c}+\widehat Y[\delta g^{(j)},\delta V^{(j)}]/M_j$ (affinity), so by the triangle inequality in the state's norm,
$$
\sqrt{\langle B_j\rangle}\le\sqrt{\langle B_j\rangle^{c}}+\frac{2}{M_j}\Bigl(\Lambda_s\epsilon_j+\|T\|^{1/2}\epsilon_{g,j}\Bigr),\qquad
\sqrt{\langle\mathcal B\rangle}\le\sqrt{\langle\mathcal B\rangle^{c}}+\Delta_s,\quad
\Delta_s=2\Bigl[\sum_j\bigl(Y_j/M_j\bigr)^2\Bigr]^{1/2},
\tag{10}
$$
$Y_j=\Lambda_s\epsilon_j+\|T\|^{1/2}\epsilon_{g,j}$, $\epsilon_j\ge\|\delta V^{(j)}\|_F$, $\epsilon_{g,j}\ge\|\delta g^{(j)}\|_2$ (Minkowski over $j$ for the sum). If the normalizing Hartree values carry a solve error $|\delta M_j|/M_j\le\varepsilon_{M,j}$, the right side is divided by $(1-\varepsilon_{M,j})$ and $\varepsilon_{M,j}$ is added inside the bracket. Then, by the spectral inequality $I-\mathbf 1_{[M/2,3M/2]}(\widehat M)\le4(\widehat M/M-I)^2$ and $\mathbf 1_{(-\infty,0]}(\widehat M)\le(\widehat M/M-I)^2$:

branch weight $\ge1-\langle\mathcal B_\infty\rangle$; per-mass spectral weight outside $[M_j/2,3M_j/2]\le\langle B_j\rangle\le\langle\mathcal B\rangle$; $\Pr(\widehat M_j\le0)\le\tfrac14\langle B_j\rangle$;

mass bias $b_j=\langle\widehat M_j\rangle_{\bar\Omega}-M_j=-\tfrac12V^{(j)}{:}\Xi$ exactly, hence $b_j\in\bigl[-\tfrac12V^{(j),c}{:}\Xi\mp\tfrac12\epsilon_j\|\Xi\|_F\bigr]$.

Everything in Result B is a function of the stored slice and of $(\epsilon_j,\epsilon_{g,j})$; the latter are Sec. 3.

## 3. Result C: the computed Hessian's error enclosed for every direction

### 3.1 Boundary charge

Implemented solve: fine-grid fields $u\in\mathbb R^{n_u}$ solve the boundary-normalized rows $F(u;m)=\mathsf R\bigl(u,\,W(m-r(W^\dagger u))\bigr)=0$, where $m$ are the bare moments, $W$ the kernel, $\bar u=W^\dagger u$, and $r(\bar u)$ the moments of the fixed reference projector at the back-averaged fields. Because that projector is a spectral projector of $h[\bar u]$, $\langle\bar u,r'(\bar u)\delta\bar u\rangle=\mathrm{Tr}(\delta P\,h[\bar u])=0$ (the Euler identity of Appendix B in its held form), so the boundary-generator identity $\partial E/\partial m_\mu=\bar u_\mu$ holds for the implemented functional whenever the discrete rows are variational, and
$$
V=\frac{\partial\bar u}{\partial m}=W^\dagger U,\qquad U=-F_u^{-1}F_m,\qquad F_u=\partial_u\mathsf R-\partial_s\mathsf R\,W r'W^\dagger,\qquad F_m=\partial_s\mathsf R\,W .
\tag{11}
$$
Whatever method produced the stored $U^c$ (linear solve or finite differences), define the residual and the enclosure
$$
R_1=[F_u]\,U^c+[F_m],\qquad \delta U=U^*-U^c=-F_u^{-1}R_1,\qquad
\boxed{\;\epsilon_\infty=c_W\,\kappa\,\rho_R,\quad \rho_R\ge\|R_1\|_F,\ \ \kappa\ge\|F_u^{-1}\|_2,\ \ c_W=\|W^\dagger\|_{2\to2}\;}
\tag{12}
$$
with $[F_u],[F_m]$ the interval Jacobians evaluated over the interval-Newton box $\mathfrak B_s$ of the slice (the box that certifies the unique solution $u^*$; evaluating over the box makes $R_1$ enclose the residual at the true base point, so no separate Lipschitz term is needed; if only a point Jacobian is available, add $c_W\kappa[L_m+L_u\|U^c\|_F]r_s$ with $L_u,L_m$ Lipschitz constants of $F_u,F_m$ on the box, the reference's second response bounded by the Riesz-contour constant in the certified gap $\delta_S$ of the followed levels). The inverse bound is the standard one from an approximate inverse $Y$: $\theta=\|I-[F_u]Y\|_2<1\Rightarrow\kappa=\|Y\|_2/(1-\theta)$; the paper's audit gives $\kappa\le2.90$ on 573 slices. The scalings of the boundary normalization must be folded into $\kappa$ and $c_W$ consistently (the Frobenius norm of $R_1$ and the Euclidean norm on the moment index are the norms in which (9) holds).

This encloses $\delta V=W^\dagger\delta U$ as a set with no preferred direction, which is the point: (12) is a bound on $\|\delta V\|_F$, and (9)–(10) accept exactly that.

Refinement. $U^{(1)}=U^c-YR_1$ has residual $(I-F_uY)R_1$, so $\rho_R^{(1)}\le\theta\rho_R$; with the audit's contraction $\theta\sim10^{-4}$, one or two steps put $\epsilon_\infty$ at rounding level. The refined $V^{(1)}=W^\dagger U^{(1)}$ (symmetrized) is then the operator's coefficient, and only the remainder is direction-unknown.

Two consistency gates that the enclosure makes mandatory. (a) Reciprocity: the true $V$ is symmetric, so $\|(V^c-V^{c{\sf T}})/2\|_F\le\epsilon_\infty$ is necessary; a violation means the response $r'$ in $F_u$ is not the derivative of the $r$ in the rows, or the rows are not variational, and the computation stops. (b) Discrete boundary-generator identity: $\eta_s=\|U^{c{\sf T}}\lambda_\infty-\bar u^c\|_2$ with $\lambda_\infty=\partial\mathcal E/\partial u$ (supported at the outer node, $\mathcal E=r_{\max}v^2/2G$). If $\eta_s\le\epsilon_\infty$, $g^{(\infty)}=\bar u^*$ and $\epsilon_{g,\infty}=c_Wr_s$. If not, the implemented rows are not variational at discretization order: the operator's one-body coefficient must be $g^{(\infty)}=U^{\sf T}\lambda_\infty$ (the true first derivative), $\eta_s$ enters (10) as $\epsilon_{g,\infty}$ until it is, and the Hessian takes the adjoint form of Sec. 3.2. This defect is itself an error not parallel to the Hessian, and the budget of Sec. 6 shows it would not be negligible, which is why the gate is not optional.

Purity: the formulas use $P^2=P$; the lanes certify $\eta_P=\|\bar C^2-\bar C\|_F$ and evaluate on the nearest projector, reporting $\eta_P$ as part of the stored-slice defect.

### 3.2 Interior masses

$M_j(m)=\mathcal M_j(u(m))$ with $\mathcal M_j(u)=r_jv(r_j)^2/(2GN(r_j)^2)$ at the slice's fixed radius. With $\lambda_j=\partial\mathcal M_j/\partial u$ (one node) and the adjoint vector $\zeta_j=F_u^{-{\sf T}}\lambda_j$,
$$
g^{(j)}=U^{\sf T}\lambda_j=-F_m^{\sf T}\zeta_j,\qquad
V^{(j)}=U^{\sf T}\mathcal M_{j,uu}U-\zeta_j\!\cdot\!\bigl(F_{uu}[U\cdot,U\cdot]+F_{um}[U\cdot,\cdot]+F_{um}[U\cdot,\cdot]^{\sf T}\bigr),
\tag{13}
$$
$F_{mm}=0$ (rows linear in the sources), $F_{um}[a,\delta m]=\mathsf R_{us}[a,W\delta m]$, $F_{uu}[a,b]=\mathsf R_{uu}[a,b]+\mathsf R_{us}[a,s_ub]+\mathsf R_{us}[b,s_ua]+\mathsf R_s\,s_{uu}[a,b]$, $s_u=-Wr'W^\dagger$, $s_{uu}[a,b]=-Wr''[W^\dagger a,W^\dagger b]$. The certification is the same residual argument with two residuals, $R_1$ for $U$ and $\rho_j=F_u^{\sf T}\zeta_j^c-\lambda_j$ for $\zeta_j$, propagated through (13) by the triangle inequality with the tensor norms evaluated on the stored slice; for $j=\infty$, (13) reduces to (11) exactly when gate (b) passes, because $\zeta_\infty=-u$ and the $r''$ term cancels by the Euler identity.

Definition lock. For interior masses $\zeta_j\neq-u$ and the $r''$ term does not cancel; its exact matrix is the Hessian in $\bar u$ of $\mathrm{Tr}\bigl(P(\bar u)h[\xi_j]\bigr)$, $\xi_j=W^{\sf T}(\partial_s\mathsf R)^{\sf T}\zeta_j$, which costs $O(n_m^2\,n\log n)$ even with hierarchical Cauchy-matrix products (about $10^{13}$ operations per slice and diagnostic) and whose certification is not in hand. The paper calls these masses diagnostics, so the lanes lock and print their definition: either the full-response Hessian (13), at that cost, or the held-response diagnostic with $\mathsf R_s s_{uu}$ set to zero, which is symmetric, defines a legitimate quadratic operator, and is certified by banded tensors at $O(n_u^3)$. The branch weight uses only $j=\infty$ and is complete either way.

## 4. Propagation formulas, collected

Per slice $s$ and diagnostic $j$ the lanes evaluate
$$
\begin{aligned}
&T_{\mu\nu}=\mathrm{Tr}(PA_\mu P'A_\nu),\quad \Xi_{\mu\nu}=\mathrm{Tr}(A_\mu\bar\Delta A_\nu\bar\Delta),\quad\gamma_F,\quad \Lambda_s=\tfrac12\|\Xi\|_F+\gamma_F+\|T\|_{\rm op};\\
&\epsilon_j,\ \epsilon_{g,j}\ \text{from (12)–(13)},\qquad Y_j=\Lambda_s\epsilon_j+\|T\|^{1/2}\epsilon_{g,j},\qquad \Delta_s=2\bigl[\textstyle\sum_j(Y_j/M_j)^2\bigr]^{1/2};\\
&\langle\mathcal B\rangle\le\bigl(\sqrt{\langle\mathcal B\rangle^{c}}+\Delta_s\bigr)^2,\qquad
\langle\mathcal B_\infty\rangle\le\bigl(\sqrt{q_\infty^{c}}+2Y_\infty/E\bigr)^2,\qquad w_\Pi\ge1-\langle\mathcal B_\infty\rangle;\\
&\Pr(\widehat M_j\le0)\le\tfrac14\bigl(\sqrt{\langle B_j\rangle^{c}}+2Y_j/M_j\bigr)^2,\qquad
b_j\in-\tfrac12V^{(j),c}{:}\Xi\pm\tfrac12\epsilon_j\|\Xi\|_F,
\end{aligned}
$$
with the $(1-\varepsilon_{M,j})^{-1}$ normalization factor where the Hartree values carry solve error. For a known correction (the refinement term $-W^\dagger YR_1$) the sharper computable forms are used instead of (9): mean $-\tfrac12\delta V{:}\Xi$ exactly, singles $\|P'F_{\delta V}[\bar\Delta]P\|_F$ exactly (one one-body operator), doubles $\le\|T^{1/2}\delta V\bar T^{1/2}\|_F$ (an $n_m^3$ product); the exact doubles with exchange, $X_4$, is not computed (it needs the $n_pn_h\times n_pn_h$ amplitude).

## 5. Computation spec for two lanes

Two lanes means two independent calculations, each with its own stored slices (its own solver history), its own approximate inverses, its own interval library and its own Gaussian-trace code; two post-processings of one calculation's slices do not count. Shared and named: the lattice equations, the preparation, the kernel at $\sigma/a=4$ and 8, the ten channels with weights $16k$, the held-projector rule, and the definition lock of Sec. 3.2. If the second calculation has stored only four of the 18 slices, it first completes its stored set at the same flat-slice times (one continuation of that history to $T/r_s=40$ at the stored cadence).

Inputs per slice: $\bar C_k$ and $P_{{\rm held},k}$ for $k\le10$; fine-grid $u^c$; kernel $W$; the implemented $\mathsf R$, $r$, $r'$ (and $r''$ if the full-response lock is chosen); $G$, $m$; the stored $(g^{(j),c},V^{(j),c},M_j^c)$ and radii; the central $\langle B_j\rangle^c$, $\langle\mathcal B\rangle^c$, and $q_\infty^c$ where it exists.

Steps per slice:
1. Purity $\eta_P$, projection; $\bar\Delta_k$, $\bar d$, $T$, $\Xi$, $\gamma_F$ from local blocks of $\bar C_k$, $P'_k$, $\bar\Delta_k$ on the vertex supports; verified $\|T\|_{\rm op}$ (computed eigenvalue plus Weyl bound on the rounding enclosure); $\Lambda_s$ and its three terms; $\mathcal Q_j$ and $\alpha_j$.
2. Interval-Newton certificate of the solve (box $\mathfrak B_s$, radius $r_s$), as in the five-slice test; interval Jacobians $[F_u],[F_m]$ over the box with the implemented $r'$; approximate inverse $Y$, $\theta$, $\kappa$; followed-level gap $\delta_S$.
3. Residual $R_1$ with directed rounding; $\rho_R$; refinement until $\rho_R$ is at rounding level (at most two steps); $\epsilon_\infty$ from (12); gates (a) and (b).
4. Interior masses under the printed lock: adjoint solves, residuals $\rho_j$, tensor norms, $\epsilon_j$, $\epsilon_{g,j}$.
5. Result B: $\Delta_s$, certified upper bounds on $\langle\mathcal B\rangle$, $\langle B_j\rangle$, $\Pr(\widehat M_j\le0)$, the bias intervals, and $w_\Pi$ where $q^c_\infty$ exists.

Outputs per slice (both lanes, one table): $\eta_P$, $r_s$, $\kappa$, $\theta$, $\delta_S$, $c_W$, $\eta_s$, symmetry defect, $\|T\|$, $\|\Xi\|_F$, $\gamma_F$, $\Lambda_s$, $\alpha_j$, $\epsilon_j$, $\epsilon_{g,j}$, $\Delta_s$, central and certified deviations, bias intervals, nonpositivity bounds, branch-weight bound.

Gate:
G0 the definition lock is printed and identical in both calculations;
G1 on all 18 slices: $\eta_P\le10^{-10}$, box certified, $\theta\le1/2$, $\delta_S>0$ certified;
G2 $\eta_s\le\epsilon_\infty$ (otherwise the one-body coefficient and the Hessian form switch as in Sec. 3.1 and the central value is recomputed before anything else);
G3 symmetry defect $\le\epsilon_\infty$;
G4 $\Delta_s\le10^{-4}$ on every slice after refinement, so the certified $\langle\mathcal B\rangle$ differs from the central value by less than $6\times10^{-5}$ and is invisible at the printed precision;
G5 both calculations: certified maxima over the 18 slices $\le0.080$, central $\langle\mathcal B\rangle^c$ at matched slices within $10^{-3}$, certified bias intervals overlapping, branch-weight bounds within $10^{-3}$ on the slices that have $q_\infty^c$.
If G4 fails after two refinements, the paper prints the certified maximum (for instance 0.081), never the unqualified 0.080. If G1 fails on a slice through $\delta_S$, that slice's base-point term is reported as not verified and the paper prints the count of fully verified slices.

Cost per slice per lane: step 1 is $O(Kn_m^2c^4)$, about $2\times10^{11}$ operations, minutes; step 2 is the existing verified-Jacobian audit plus the interval evaluation over the box, the dominant item, of order one node-hour, set by $r'$; step 3 is three dense products of size $n_u$ with directed rounding, well under an hour; step 4 under the held-response lock is three adjoint solves and banded contractions, under an hour, and under the full-response lock becomes GPU-hours per diagnostic with open certification; step 5 is negligible. Order three to five node-hours per slice per lane, two to four node-days per lane for 18 slices, plus the completion of the second calculation's slice set where missing.

## 6. Pushing to the result

Budget. At 15.0 the central value 0.0798 survives at the printed precision if $\Delta_s\le\sqrt{0.080}-\sqrt{0.0798}=3.5\times10^{-4}$. With the paper's magnitudes ($2GE\simeq59a$, $G=a^2/4$, so $E\simeq1.2\times10^2a^{-1}$ and the smallest diagnostic mass a few times $10a^{-1}$; $\Lambda_s$ dominated by $\|T\|$, the covariance of the central sites' angular energies with the copy weights, of order $10^3$ to $10^4a^{-2}$), $\Delta_s\simeq2\Lambda_s\epsilon/M_{\min}$ meets the budget for $\epsilon\lesssim10^{-6}a$, i.e. a relative Hessian accuracy of $10^{-6}$ against $\|V\|_F\sim a$ (the Newtonian response $G/\max(r,r')$ summed over the chain). A double-precision solve with $\kappa\le2.90$ exceeds this by about four orders, and one refinement by eight. The same budget shows what the discrete-identity gate protects against: a non-variational discretization defect $\eta_s$ of relative size $10^{-3}$ would enter as $\|T\|^{1/2}\eta_s/E\sim5\times10^{-2}$ and would not be negligible, which is why G2 precedes the certification rather than following it.

Expected outcome, to be replaced by the lanes' table: on all 18 slices in both calculations the certified Hessian-error increment of $\langle\mathcal B\rangle$ is below $10^{-6}$; the interior bound reads 0.0798 at 15.0 and at most 0.080 elsewhere for every error direction; each mass is nonpositive with probability at most 0.020; each bias interval has width below $10^{-6}E$; on the five slices with $q_\infty^c<10^{-3}$ the branch weight is at least $1-(0.032+10^{-6})^2>0.998$, and the same formula closes the 18-slice branch-weight item once $q_\infty^c$ is computed there. The anisotropy ratio $\alpha_j$ is reported beside the numbers; a value of order $10$ would mean the old parallel bound was ten times tighter than the direction-free one and still irrelevant at this $\epsilon$.

Paper text that the landing replaces (Sec. V, the marked sentence; counts in prose, no mark once the table exists):

"The bound holds for every direction of the computed Hessian's error. That error enters the deviation through three traces of the Hartree state, the covariance of the bare moments, its contraction with the deviation from the prepared sea, and its exchange response, and the error itself is enclosed, after one refinement step, by the residual of the field-response equations times the constraint Jacobian's verified inverse norm. On all 18 slices in both calculations the enclosure moves the bound by less than [the lanes' maximum], the mass biases by less than [maximum] of $E$, and each mass is nonpositive with probability at most 0.020."

and in Sec. V's branch paragraph, once $q_\infty$ is on the 18 slices: "the branch weight is at least $1-(\sqrt{q_\infty}+2Y_\infty/E)^2$ with the same enclosure, [value] on all 18 slices."

What is not covered, so that the mark is closed honestly: the replacement of the held-source law by Eq. (dm-reference), the correlation beyond Hartree-Fock, and the channel limit remain the separate items the section already names; the present proof removes the parallel-error assumption and nothing else.