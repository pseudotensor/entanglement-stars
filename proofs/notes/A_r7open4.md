
## Result and status

The massless \(L^{7/4}\) estimate follows from the supplied fermion Hamiltonian. In particular, neither a gap nor a localization-commutator estimate for the evolving covariance is needed. The useful route is:

\[
\text{window energy}
\;\Longrightarrow\;
\|\Pi_A(C-P_0)\Pi_{A^c}\|_2^2=O(L^{3/2})
\;\Longrightarrow\;
|\Delta S_A|=O(L^{7/4}).
\]

The second implication requires an entropy estimate sensitive to the **flat projector’s boundary deficit**, rather than a dimension-only continuity estimate.

Below I give that proof, an explicit constant, the boundary-deficit majorant, the set construction for convexity, and constructive weighted bounds for the massive estimate.

**I have not produced or executed the required [111] numerical certificate. Consequently, I cannot honestly mark the entire five-part OPEN item resolved, publish invented support values, or describe the advertised decimal intervals as verified by this response.** Section 6 identifies the remaining numerical quantities exactly and derives the enclosure formulas they must enter. This is a substantive analytical resolution, but not the complete deliverable requested.

### Input anchors

The supplied source is `00_PAPER_flat_b39645f976f_reviewed.tex`; the component files were not supplied separately. The relevant attributable statements are:

- Appendix A.1: “**polynomial approximants to \(s\) … reduce the entropy through a closed-path expansion to summable boundary deficits**.”
- Appendix A.2: “**Convexity follows from Gaussian strong subadditivity … through a finite-volume limit**.”
- Appendix A.3: “**\(P=P_0+P_1+R_2\)**” and
  “**\(|S_A(P)-S_A(P_0)|\le C\varkappa A_{\rm lat}\epsilon^2(1+|\log\epsilon|)^4\)**.”
- The radiation section: “**At zero mass the window energy alone controls the cut**” and
  “**\(|S_A(C)-S_A(P_0)|\le C_*(\kappa_f)L^{7/4}\)**.”
- The coefficient paragraph: “**The convexity bound is rigorous, \(0.8164<\eta(0)<1.0203\) and \(0.7883<\eta(0.2)<0.9831\)**.”

No implementation of a numerical function computing these supports or their certificates is included in the supplied material.

---

# 1. The entropy estimate that makes the massless exponent work

Throughout, \(a=1\), and logarithms are natural. For a projector \(P\), put

\[
X_P=\Pi_A P\Pi_{A^c},\qquad
D_P=X_PX_P^\dagger=P_A-P_A^2.
\]

Define, for \(0\le x\le \tfrac12\),

\[
f(x)=s\!\left(\frac{1+\sqrt{1-4x^2}}2\right).
\]

Then

\[
S_A(P)=\operatorname{Tr}f(|X_P|).
\]

For two projectors \(P,C\), define

\[
B=\|X_C-X_P\|_2^2,\qquad
V_A(P)=\operatorname{Tr}\bigl[f'(|X_P|)^2\bigr],
\qquad M=4|A|.
\]

The following finite-dimensional inequality also holds when the complement is infinite:

\[
\boxed{
|S_A(C)-S_A(P)|
\le
\sqrt{V_A(P)\,B}
+
1024\,B\log\!\left(e+\frac MB\right).
}
\tag{1}
\]

The second term is interpreted as zero at \(B=0\).

This differs importantly from the dimension-only estimate \(O(\sqrt{MB})\). For the flat sea, \(V_A(P_0)=O(A_{\rm lat})\), not \(O(|A|)\).

## Proof, including the entropy endpoints

Writing \(z=\sqrt{1-4x^2}\), differentiation gives

\[
f'(x)=4x\frac{\operatorname{artanh}z}{z},
\qquad
f''(x)=4\frac{\operatorname{artanh}z-z}{z^3}.
\tag{2}
\]

Thus \(f\) is convex, \(0\le f'\le2\), and, for \(0<x\le\tfrac12\),

\[
f'(x)\le8x(1+|\log x|),\qquad
0\le f''(x)\le32(1+|\log x|).
\tag{3}
\]

For example, when \(z\ge\tfrac12\),

\[
\operatorname{artanh}z
=\log\frac{1+z}{2x}\le\log\frac1x;
\]

when \(z\le\tfrac12\), use the convergent power series for \(\operatorname{artanh}z\).

Extend \(f\) evenly. For \(0<\tau\le\tfrac14\), replace it on \([-\tau,\tau]\) by

\[
q_\tau(x)=f(\tau)+\frac{f'(\tau)}{2\tau}(x^2-\tau^2),
\]

and subtract \(q_\tau(0)\) from the resulting function. Call the result \(F_\tau\). Directly from (3),

\[
\begin{aligned}
\|F_\tau-f\|_\infty
&\le32\tau^2(1+|\log\tau|),\\
\operatorname{Lip}(F_\tau')
&\le32(1+|\log\tau|),\\
\|F_\tau'-f'\|_\infty
&\le16\tau(1+|\log\tau|).
\end{aligned}
\tag{4}
\]

Use the Hermitian dilations

\[
\mathcal D_j=
\begin{pmatrix}
0&X_j\\ X_j^\dagger&0
\end{pmatrix},
\qquad X_0=X_P,\quad X_1=X_C.
\]

They act on a space of dimension at most \(3M\): restrict the right-hand space to
\(\operatorname{Ran}X_0^\dagger+\operatorname{Ran}X_1^\dagger\). Moreover,

\[
S_A(P_j)=\tfrac12\operatorname{Tr}f(\mathcal D_j),
\qquad
\|\mathcal D_1-\mathcal D_0\|_2^2=2B.
\]

For a scalar function with Lipschitz derivative, the divided-difference formula for the trace Hessian gives

\[
\left|
\operatorname{Tr}F(H+K)-\operatorname{Tr}F(H)
-\operatorname{Tr}F'(H)K
\right|
\le \frac{\operatorname{Lip}(F')}{2}\|K\|_2^2.
\tag{5}
\]

This follows by integrating the second derivative along \(H+tK\); every divided difference of \(F'\) has absolute value at most its Lipschitz constant.

Apply (5) to \(F_\tau\), and then (4), with
\(\tau=\sqrt{B/M}\) when \(B/M\le1/16\). All regularization and quadratic terms are bounded by

\[
1024B\log(e+M/B).
\]

The unregularized linear term satisfies

\[
\frac12\left|\operatorname{Tr}f'(\mathcal D_0)
(\mathcal D_1-\mathcal D_0)\right|
\le \sqrt{V_A(P)B}.
\]

For \(B/M>1/16\), use
\(|\Delta S_A|\le M\log2\le16(\log2)B\). This proves (1).

A further consequence of (3) is

\[
f'(x)^2\le256x^{4/3},
\]

and hence

\[
\boxed{
V_A(P)\le256\,\operatorname{Tr}D_P^{2/3}.
}
\tag{6}
\]

Equations (1) and (6) are the endpoint estimates needed below.

---

# 2. An explicit, summable boundary-deficit majorant

Let \(P_0(z)\) be the Fourier coefficients of the flat projector. The following deliberately loose bound is uniform in \(\zeta\ge0\):

\[
\boxed{
\sum_{|z|\ge t}\|P_0(z)\|_{\mathrm F}^2
\le 2^{15}t^{-3},\qquad t>0.
}
\tag{7}
\]

Here the Frobenius norm includes the native four components. No valley factor is subsequently added.

## Derivation from the sine symbol

Write

\[
P_0(k)=\frac12\left(I-\frac{H_0(k)}{\varepsilon(k)}\right).
\]

Let \(r(k)\) be the torus distance to the nearest of the eight sine zeros. In its nearest-valley cube,

\[
\varepsilon(k)\ge |\sin k|\ge \frac2\pi r(k).
\tag{8}
\]

For a unit momentum direction \(e\), differentiating the normalized four-vector
\((\zeta,\sin k_1,\sin k_2,\sin k_3)/\varepsilon(k)\) gives

\[
\|\partial_e^2P_0(k)\|_{\mathrm F}
\le \frac{32}{r(k)^2}.
\tag{9}
\]

One explicit check is to bound the second derivative of a normalized vector by
\(2+6/\varepsilon^2\), and then use (8) and
\(r\le\sqrt3\pi/2\).

Consider the second difference

\[
\Delta_h^2P_0(k)=P_0(k+h)-2P_0(k)+P_0(k-h).
\]

On \(r(k)<2|h|\), its squared Frobenius norm is at most \(32\).
Outside that set, Taylor’s formula and (9) give

\[
\|\Delta_h^2P_0(k)\|_{\mathrm F}
\le128\,|h|^2r(k)^{-2}.
\]

Integrating over the eight nearest-valley cubes, with normalized Brillouin-zone measure, yields

\[
\int_{\rm BZ}\|\Delta_h^2P_0(k)\|_{\mathrm F}^2\frac{d^3k}{(2\pi)^3}
\le
\frac{32768+1024/3}{\pi^2}|h|^3
<4096|h|^3.
\tag{10}
\]

Now average over a centered Gaussian \(h\) with covariance \(t^{-2}I\). Parseval gives the Fourier multiplier

\[
16\,\mathbb E\sin^4(z\cdot h/2)
=
2\left(3-4e^{-|z|^2/(2t^2)}+e^{-2|z|^2/t^2}\right).
\]

It is at least one for \(|z|\ge t\), while
\(\mathbb E|h|^3<8t^{-3}\). Equation (7) follows.

## From the kernel tail to entropy density

For \(x\in A\), let \(d(x)=\operatorname{dist}(x,A^c)\), and define

\[
d_A(x)=\operatorname{tr}[D_{P_0}]_{xx}
=\sum_{y\notin A}\|P_0(x-y)\|_{\mathrm F}^2.
\]

Equation (7), together with the trivial bound \(d_A(x)\le2\), implies

\[
d_A(x)\le D_0(1+d(x))^{-3},
\qquad D_0=2^{18}.
\tag{11}
\]

For a positive operator \(D\), scalar Jensen applied to each diagonal matrix element gives

\[
\operatorname{tr}[D^{2/3}]_{xx}
\le4^{1/3}\bigl(\operatorname{tr}D_{xx}\bigr)^{2/3}.
\]

Therefore

\[
\boxed{
\operatorname{tr}[D_{P_0}^{2/3}]_{xx}
\le4^{1/3}D_0^{2/3}(1+d(x))^{-2}.
}
\tag{12}
\]

Also \(s(t)\le4[t(1-t)]^{2/3}\), so

\[
\boxed{
\operatorname{tr}[s(P_{0,A})]_{xx}
\le4^{4/3}D_0^{2/3}(1+d(x))^{-2}.
}
\tag{13}
\]

This is the requested **summable boundary-deficit majorant**.

For a \(\varkappa\)-regular cut, the definition in the supplied Appendix A gives

\[
\begin{aligned}
\sum_{x\in A}(1+d(x))^{-2}
&=
2\int_0^\infty
\frac{\#\{x\in A:d(x)\le t\}}{(1+t)^3}\,dt\\
&\le2\varkappa A_{\rm lat}.
\end{aligned}
\tag{14}
\]

The same majorant is integrable in the half-space depth, uniformly in its normal.

## Explicit polynomial approximation

Use the endpoint-preserving polynomials

\[
q_n(t)=\sum_{j=1}^n
\frac{1-(1-2t)^{2j}}{2j(2j-1)}.
\]

They increase to \(s(t)\). Setting \(u=t(1-t)\), one has, for \(0<\alpha<1\),

\[
\begin{aligned}
0\le s(t)-q_n(t)
&\le \frac{(4u)^\alpha}{2}
\sum_{j>n}j^{\alpha-2}\\
&\le
\frac{4^\alpha}{2(1-\alpha)}
n^{\alpha-1}u^\alpha.
\end{aligned}
\tag{15}
\]

At \(\alpha=2/3\), equations (12)–(15) bound the approximation error by \(O(n^{-1/3})A_{\rm lat}\), uniformly in ball radius, cut normal, and \(\zeta\).

That supplies the dominated-convergence step missing from the manuscript: polynomial approximation can be interchanged with the large-ball limit and angular integration.

For completeness, the fixed-polynomial path sums have an independent absolute majorant. A dyadic decomposition around the eight point nodes, integrating four times by parts on each annulus, gives

\[
\|P_0(z)\|\le K(1+|z|)^{-3}
\]

for a finite stencil-dependent \(K\). Convolution of \((1+|z|)^{-3}\) obeys

\[
f^{*j}(z)\le K_j(1+|z|)^{-3}
[\log(e+|z|)]^{j-1}.
\]

Consequently every fixed-length closed-path sum remains absolutely convergent after multiplication by its path diameter. The boundary-deficit count is at most
\(\varkappa A_{\rm lat}(1+\text{diameter})\).

For a fixed path with vertices \(0,s_1,\ldots,s_j\), its digital-ball deficit divided by \(4\pi R^2\) tends to

\[
\frac1{4\pi}\int_{S^2}
\max(0,n\cdot s_1,\ldots,n\cdot s_j)\,d\Omega.
\tag{16}
\]

One can check this by dividing the sphere into the finitely many patches on which a particular \(s_i\) maximizes \(n\cdot s_i\), then counting crossings on lattice lines parallel to that \(s_i\). Patch seams contribute \(O(R)\); the projected two-dimensional counts are Riemann sums. The path majorant permits summation of (16), and (15) then permits \(n\to\infty\).

Thus the majorant proves the stated sphere limit, not merely an area upper bound.

---

# 3. The explicit set construction for convexity

Here is a construction that genuinely uses submodularity. Simply observing that continuum boundaries exchange pieces under union and intersection is insufficient.

Work first with rational slopes \(p_1,p_2\in\mathbb Q^2\) and rational \(0<\lambda<1\). Set

\[
p=\lambda p_1+(1-\lambda)p_2.
\]

Choose a primitive \(a\in\mathbb Z^2\) parallel to \(p_1-p_2\). Construct an \(M\)-periodic continuous sawtooth \(w_M\) such that the graph

\[
g_M(x)=p\cdot x+w_M(a\cdot x)
\]

has slope \(p_1\) for fraction \(\lambda\) of each period and slope \(p_2\) for the remainder.

Choose \(u\in\mathbb Z^2\) with \(p\cdot u\in\mathbb Z\) and
\(g=a\cdot u\ne0\). Take periods \(M\to\infty\) coprime to \(g\).
For a finite horizontal square \(Q_R\), define the **actual lattice sets**

\[
A_j=
\left\{
(x,z):
x\in Q_R\cap\mathbb Z^2,\quad
p\cdot x-HM\le z
\le p\cdot x+w_M(a\cdot x+jg)
\right\},
\quad 0\le j<M,
\tag{17}
\]

with integer \(z\). Choose \(H\) so all slab thicknesses exceed \(M\).

At every horizontal lattice point, the \(M\) sampled sawtooth values are a permutation of the same multiset. Let its decreasing rearrangement be
\(h_1\ge\cdots\ge h_M\). The rank sets are exactly

\[
B_k=\left\{x\in Q_R,\quad p\cdot x-HM\le z\le p\cdot x+h_k\right\}.
\tag{18}
\]

These are planar slabs.

For a submodular set function \(S\),

\[
\boxed{\sum_{j=0}^{M-1}S(A_j)\ge\sum_{k=1}^M S(B_k).}
\tag{19}
\]

A finite proof is repeated uncrossing:
replace two incomparable sets by their union and intersection. Submodularity cannot increase the sum. Membership multiplicities are preserved. Repetition produces the nested rank sets (18).

Translation invariance makes the entropy per horizontal area of every \(A_j\) the same. The integer vertical translation needed here is precisely why \(p\cdot u\in\mathbb Z\) was imposed. Lateral-boundary differences vanish as \(R\to\infty\) at fixed \(M\).

Now take \(R\to\infty\), followed by \(M\to\infty\). The common lower face contributes
\(\Phi((-p,1))\). The upper sawtooth contributes

\[
\lambda\Phi((-p_1,1))+(1-\lambda)\Phi((-p_2,1)).
\]

Every slab on the right of (19) contributes twice
\(\Phi((-p,1))\). Edge and opposite-face interactions vanish using the majorant (13). Subtracting the common lower-face contribution gives

\[
\Phi((-p,1))
\le
\lambda\Phi((-p_1,1))
+(1-\lambda)\Phi((-p_2,1)).
\tag{20}
\]

Continuity extends this to real slopes and weights. Positive homogeneity makes \(\Phi\) convex on each coordinate half-space \(\{\xi_i>0\}\) and \(\{\xi_i<0\}\). These half-spaces cover every nonzero point. Restriction to any line therefore gives convexity unless the line passes through zero; on such a line, evenness and homogeneity give
\(\Phi(t\xi)=|t|\Phi(\xi)\), which is convex.

This proves the claimed global convexity with an explicit finite-set construction.

---

# 4. Constructive weighted bounds for the massive estimate

The supplied Appendix A.3 assumes potential flows, so the spin-curl term vanishes. Its operator has midpoint principal symbol

\[
H(x,k)=N(x)h_0(k)+v_i(x)q_i(k).
\]

The actual symmetric endpoint assignment differs from midpoint quantization by an \(O(\epsilon^2)\), finite-range operator.

## 4.1 Frozen band signs are not an additional assumption

The pathwise operator gap in Appendix A.3 must first be transferred to frozen symbols.

A normalized lattice Gaussian of width \(R\), multiplied by a plane wave and a frozen eigenvector, has residual bounded by

\[
B_{\rm fr}\bigl(\epsilon R+R^{-1}\bigr),
\]

where, for example, the deliberately generous choice

\[
B_{\rm fr}=100(1+C_1)(1+\zeta+n^*+v^*)
\]

bounds the finite-range stencil moments. Choose \(R=\lceil\epsilon^{-1/2}\rceil\).

For sufficiently small \(\epsilon\), every frozen eigenvalue has absolute value at least \(\delta/2\). Along the supplied continuous gapped path, its sign cannot change. At the flat endpoint the lower band is negative and the upper band positive. Hence the same is true throughout the path.

This establishes, rather than assumes, the frozen-band identification used in the following contour calculation.

## 4.2 First corrector

Let \(\operatorname{Op}^{W}\) denote midpoint lattice quantization. The first corrector is

\[
\boxed{
P_1=\operatorname{Op}^{W}\pi_1,\qquad
\pi_1(x,k)=
\frac{i}{2N(x)\varepsilon(k)}
\sum_{j,l}\partial_jv_l(x)\,q_l(k)
[P_0(k),\partial_{k_j}P_0(k)].
}
\tag{21}
\]

To check it, the first-order commutation equation is

\[
[H,\pi_1]
=
-\frac i2\sum_j
\{\partial_jH,\partial_{k_j}P_0\}.
\]

The clock term vanishes because

\[
\{h_0,\partial_{k_j}P_0\}=0.
\]

The two off-diagonal band blocks are then obtained by division by
\(\pm2N\varepsilon\), giving (21). Its diagonal band blocks vanish by the first-order projection equation.

The time reversal specified in the supplied Appendix A satisfies

\[
\Theta P_0\Theta^{-1}=P_0,\qquad
\Theta P_1\Theta^{-1}=-P_1.
\tag{22}
\]

The two momentum-odd factors in (21) cancel their signs, while complex conjugation reverses \(i\).

## 4.3 A weighted remainder certificate

For a kernel \(K\), use the Banach-algebra norm

\[
\|K\|_\mu
=
\sum_{r\in\mathbb Z^3}e^{\mu|r|_1}
\sup_x\|K_{x,x-r}\|.
\tag{23}
\]

The result needed for entropy transfer is

\[
\boxed{
\|P_1\|_\mu\le K_1\epsilon,\qquad
\|P-P_0-P_1\|_\mu\le K_2\epsilon^2.
}
\tag{24}
\]

Here is a constructive specification of \(K_1,K_2,\mu\).

Let

\[
B_H=n^*(\zeta+3)+\frac32v^*,\quad
\gamma=\min(1,\delta/2),\quad
K_k=3en^*+3e^2v^*,
\]

and choose

\[
\rho=\min\!\left(1,\frac{\gamma}{2K_k}\right),
\qquad \mu=\rho/2.
\]

Take a rectangular contour enclosing \([-B_H,-\delta/2]\), with right edge on the imaginary axis and the other edges at distance at least one from the frozen spectrum. Its length can be taken as \(6(B_H+1)\).

For \(z\) on that contour define

\[
a=z-H,\qquad r_0=(z-H)^{-1},\qquad
r_1=-\frac i2r_0\{a,r_0\}_{x,k}.
\]

The complex strip \(|\operatorname{Im}k_i|\le\rho\) has

\[
\|r_0\|\le A_0=2/\gamma.
\]

Set

\[
\begin{aligned}
D_1&=3C_1(\zeta+3e+\tfrac32e^2),\\
D_2&=9C_2(\zeta+3e+\tfrac32e^2),\\
K_{xk}&=9C_1(e+e^2),\\
A_1&=A_0^2D_1,\\
A_2&=2A_0^3D_1^2+A_0^2D_2,\\
B_1&=A_0^3D_1K_k,\\
B_2&=3A_0^4D_1^2K_k+A_0^3D_2K_k+A_0^3D_1K_{xk}.
\end{aligned}
\]

These bound, respectively, \(r_0\), its first and second spatial derivatives after division by \(\epsilon,\epsilon^2\), and \(r_1/\epsilon,\partial_xr_1/\epsilon^2\).

All Fourier-weighted moments are bounded using

\[
J_j(\mu)=\sum_r(1+|r|_1)^j e^{-\mu|r|_1},
\qquad
J_0(\mu)=\left(\frac{1+e^{-\mu}}{1-e^{-\mu}}\right)^3,
\qquad
J_j=(1-\partial_\mu)^jJ_0.
\]

Use the abbreviations

\[
\begin{array}{lll}
a_{0j}=4(B_H+1)J_j,&a_{1j}=D_1J_j,&a_{2j}=D_2J_j,\\
b_{0j}=A_0J_j,&b_{1j}=A_1J_j,&b_{2j}=A_2J_j,\\
c_{0j}=B_1J_j,&c_{1j}=B_2J_j.&
\end{array}
\]

The endpoint-to-midpoint assignment error is bounded by
\(d_h\epsilon^2\), where

\[
d_h=\frac{C_2}{8}(3e^\mu+6e^{2\mu}).
\]

Define

\[
\begin{aligned}
C_E={}&\frac18
\left(a_{20}b_{02}+2a_{11}b_{11}+a_{02}b_{20}\right)\\
&+\frac12\left(a_{10}c_{01}+a_{01}c_{10}\right)
+d_h(b_{00}+c_{00}).
\end{aligned}
\tag{25}
\]

Then

\[
(z-h)\operatorname{Op}^{W}(r_0+r_1)=I+E_2(z),
\qquad
\|E_2(z)\|_\mu\le C_E\epsilon^2.
\tag{26}
\]

To verify (25), the exact kernel product uses

\[
a(x+s/2,r)\,b(x-r/2,s).
\]

Taylor expansion in the displacement gives a second-order remainder bounded by

\[
\frac18\left(
M_{2,0}(a)M_{0,2}(b)
+2M_{1,1}(a)M_{1,1}(b)
+M_{0,2}(a)M_{2,0}(b)
\right).
\]

For the \(r_1\) product, only a first-order bound is used:

\[
\frac12\left(
M_{1,0}(a)M_{0,1}(r_1)
+M_{0,1}(a)M_{1,0}(r_1)
\right).
\]

Thus no third spatial derivative is required.

When \(C_E\epsilon^2\le1/2\), invert \(I+E_2\) by its convergent Neumann series in (23), and integrate the resolvent around the contour. Equation (24) follows with

\[
K_1=\frac{|\Gamma|}{2\pi}c_{00},
\qquad
K_2=\frac{|\Gamma|}{\pi}(b_{00}+c_{00})C_E.
\tag{27}
\]

These are explicit functions of the hypothesis tuple in the supplied Appendix A.3. No field-history measurement enters them.

## 4.4 Why the entropy remainder is \(\epsilon^2\log^4\epsilon\)

Restrict both sides of the cut to distance

\[
w=\left\lceil\frac4\mu|\log\epsilon|\right\rceil
\]

from the boundary. The omitted cross blocks have trace norm \(O(\varkappa A_{\rm lat}\epsilon^2)\), by (23)–(24) and the cut’s regularity. Their entropy cost is at most twice that trace norm.

The inside shell has at most \(\varkappa A_{\rm lat}(1+w)\) sites. Every outside-shell site is within \(O(w)\) lattice steps of an inside boundary site. A sufficient bound on the total one-particle shell dimension is

\[
4\varkappa A_{\rm lat}
\left[(1+w)+2(6w+5)^3\right].
\tag{28}
\]

Apply the regularized trace expansion from Section 1 with \(\tau=\epsilon\):

- the \(P_1\) linear term vanishes by (22);
- the \(R_2\) linear term costs \(O(\varkappa A_{\rm lat}\epsilon^2)\);
- the quadratic term costs
  \(O(\varkappa A_{\rm lat}\epsilon^2(1+|\log\epsilon|))\);
- the endpoint regularization costs the dimension (28) times
  \(O(\epsilon^2(1+|\log\epsilon|))\).

Therefore

\[
\boxed{
|S_A(P)-S_A(P_0)|
\le C\varkappa A_{\rm lat}
\epsilon^2(1+|\log\epsilon|)^4.
}
\tag{29}
\]

This proves the printed logarithmic power. The third power comes from the conservative outside-shell rank bound; the fourth comes from entropy regularization.

---

# 5. The massless window theorem, with an explicit constant

Take any multiplication window satisfying

\[
0\le\varphi\le1,\qquad
\varphi=1\ \text{on }|x|\le2L,\qquad
\varphi=0\ \text{on }|x|\ge3L.
\]

Smoothness is not needed for this implication. Let \(A=\{|x|<L\}\), \(L\ge1\), and let \(C\) be any orthogonal projector.

Put

\[
T=\varphi C-P_0\varphi,\qquad
E_\varphi(C)=\|\Omega_0^{1/2}T\|_2^2\le\kappa_fL.
\]

## 5.1 The window operator is a contraction

The exact decomposition

\[
T=(I-P_0)\varphi C-P_0\varphi(I-C)
\]

has orthogonal domain blocks and orthogonal range blocks. Hence

\[
\boxed{\|T\|\le\|\varphi\|\le1.}
\tag{30}
\]

This is the step that avoids a localization commutator.

## 5.2 Point-node counting

Let \(Q_\lambda=\mathbf1_{\{\Omega_0\le\lambda\}}\), \(0<\lambda\le1\).
By (8), each low-energy region is contained in a ball of radius
\(\pi\lambda/2\) around one of the eight nodes. Therefore

\[
\begin{aligned}
\operatorname{Tr}(\Pi_AQ_\lambda)
&\le4|A|\,
\frac{8(4\pi/3)(\pi\lambda/2)^3}{(2\pi)^3}\\
&=\frac{2\pi}{3}|A|\lambda^3
\le18\pi L^3\lambda^3.
\end{aligned}
\tag{31}
\]

The native four-component count and the eight nodes are both included exactly once.

Using (30) and the energy constraint,

\[
\|\Pi_AQ_\lambda T\|_2
\le\sqrt{18\pi}\,L^{3/2}\lambda^{3/2},
\]

and

\[
\|\Pi_A(I-Q_\lambda)T\|_2
\le\sqrt{\frac{\kappa_fL}{\lambda}}.
\tag{32}
\]

## 5.3 The only localization remainder

Because \(\varphi=1\) on \(A\),

\[
\Pi_A(C-P_0)=\Pi_AT-\Pi_AP_0(1-\varphi).
\tag{33}
\]

The second term connects \(A\) only to sites at distance at least \(L\).
Equation (7) gives

\[
\|\Pi_AP_0(1-\varphi)\|_2^2
\le |A|\,2^{15}L^{-3}
\le27\,2^{15}.
\tag{34}
\]

Combining (31)–(34),

\[
\boxed{
\sqrt B\le
\sqrt{18\pi}\,L^{3/2}\lambda^{3/2}
+\sqrt{\kappa_fL/\lambda}
+\sqrt{27\,2^{15}}.
}
\tag{35}
\]

## 5.4 Optimization

Ignoring the \(L\)-independent last term, the minimizing splitting energy is

\[
\lambda_{\rm opt}
=
\left(\frac{\kappa_f}{9(18\pi)}\right)^{1/4}L^{-1/2},
\tag{36}
\]

clipped at one when necessary. At the interior optimum, the first two terms sum to

\[
4\,3^{-3/4}(18\pi)^{1/8}\kappa_f^{3/8}L^{3/4}.
\]

Thus \(B=O(L^{3/2})\).

For a simple constant valid for every \(L\ge1\), choose
\(\lambda=L^{-1/2}\), and define

\[
b(\kappa)=
\left(\sqrt{18\pi}+\sqrt\kappa+\sqrt{27\,2^{15}}\right)^2.
\tag{37}
\]

Then \(B\le b(\kappa_f)L^{3/2}\).

## 5.5 Entropy and the explicit constant

Equations (6), (12), and (14), with the supplied digital-ball regularity
\(\varkappa=8\), imply

\[
V_A(P_0)\le v_0L^2,\qquad
v_0=\pi\,2^{80/3}.
\tag{38}
\]

Also \(M\le108L^3\). Insert (37)–(38) into (1). The function

\[
B\longmapsto B\log(e+M/B)
\]

is increasing. Finally,

\[
\log L\le\frac4eL^{1/4}\qquad(L\ge1).
\]

Consequently,

\[
\boxed{
|S_A(C)-S_A(P_0)|
\le C_*(\kappa_f)L^{7/4},
}
\tag{39}
\]

with the entirely explicit choice

\[
\boxed{
C_*(\kappa)=
\sqrt{v_0b(\kappa)}
+
1024\,b(\kappa)
\left[
\log\!\left(e+\frac{108}{b(\kappa)}\right)+\frac6e
\right].
}
\tag{40}
\]

The constant is intentionally loose. Its significance is that it is finite, explicit, independent of \(L\), and determined without any property of the evolving generator.

### Replacement text for the massless paragraph

> At zero mass, the window energy controls the cut without a spectral gap. Let \(C\) be a pure covariance, and let \(0\le\varphi\le1\) equal one within radius \(2L\) and zero beyond radius \(3L\). If
> \[
> \|\Omega_0^{1/2}(\varphi C-P_0\varphi)\|_2^2\le\kappa_fL,
> \]
> then the ball of radius \(L\ge1\) obeys
> \[
> |S_A(C)-S_A(P_0)|\le C_*(\kappa_f)L^{7/4}.
> \]
> An explicit constant is given in the analytical supplement. The sine dispersion gives a localized low-energy count proportional to \(L^3\lambda^3\); the window energy bounds the complementary sector by \(\kappa_fL/\lambda\). Splitting at \(\lambda\propto L^{-1/2}\) makes the squared cross-block difference \(O(L^{3/2})\). Its entropy contribution is bounded by its square root times the flat sea’s area-sized boundary susceptibility, plus an \(O(L^{3/2}\log L)\) endpoint term. The resulting \(L^{7/4}\) remainder is smaller than the leading area term.

This replaces the repository-only mechanism without changing the claimed exponent.

---

# 6. The remaining numerical enclosure: exact quantities and formulas

This part is **not completed** by the supplied decimal numbers or by the preceding proofs.

Let

\[
a=\eta_{100}(\zeta),\qquad
b=\sqrt3\,\eta_{111}(\zeta).
\]

Cubic symmetry and convexity give the exact bodies

\[
K_{\rm in}
=
\operatorname{conv}\left(
\{\pm ae_i\}
\cup
\{(\pm b/3,\pm b/3,\pm b/3)\}
\right),
\]

and

\[
K_{\rm out}
=
\{x:\ |x_i|\le a,\quad |x_1|+|x_2|+|x_3|\le b\}.
\tag{41}
\]

Their support functions give rigorous lower and upper bounds on the angular mean.

For the inner body,

\[
\overline h_{\rm in}(a,b)
=
\frac{12}{\pi}
\int_0^1\!\!du\int_0^u\!\!dv\,
\frac{\max\{a,(b/3)(1+u+v)\}}
{(1+u^2+v^2)^2}.
\tag{42}
\]

When \(b\ge3a/2\), set \(t=3a/b-1\). Direct integration gives

\[
\boxed{
\overline h_{\rm in}
=
\frac b2+
\frac{2b}{\pi}
\left[
\sqrt{2+t^2}\,
\arctan\frac{t}{\sqrt{2+t^2}}
-\arctan t
\right].
}
\tag{43}
\]

For \(2a\le b\le3a\), the outer support on the ordered octant is

\[
h_{\rm out}(n)=a(n_1+n_2)+(b-2a)n_3.
\]

Hence

\[
\boxed{
\overline h_{\rm out}
=
\frac32a+(b-3a)m_{\min},
\qquad
m_{\min}
=
\frac32-\frac{3\sqrt2}{\pi}\arctan\sqrt2.
}
\tag{44}
\]

The parameter ranges in (43)–(44) must themselves be checked against the certified support intervals; they are not numerical inputs to assume.

## How the fermion model determines the missing [111] input

Use layer coordinate \(l=x_1+x_2+x_3\). The exact layer symbol is

\[
h_{111}(p;q_1,q_2)
=
\zeta\beta_{\rm D}
+\alpha_1\sin(p+q_1)
+\alpha_2\sin(p+q_2)
+\alpha_3\sin p.
\tag{45}
\]

A momentum shift and a constant Clifford rotation reduce it to

\[
h_r(p)=
\zeta\beta_{\rm D}
+u(r)\alpha_1\sin p
+v(r)\alpha_2\cos p,
\]

where

\[
u(r)^2=\frac{3+r}{2},\qquad
v(r)^2=\frac{3-r}{2},\qquad
r=|1+e^{2iq_1}+e^{2iq_2}|.
\tag{46}
\]

Let \(\mathcal S_\zeta(r)\) be this four-component chain’s half-chain entropy, with one face counted. Then the required support is exactly

\[
\boxed{
b(\zeta)=\int_0^3p_3(r)\,\mathcal S_\zeta(r)\,dr,
}
\tag{47}
\]

where, with \(\mathrm K\) using the manuscript’s squared-modulus convention,

\[
p_3(r)=
\begin{cases}
\displaystyle
\frac{4r}{\pi^2\sqrt{(1+r)^3(3-r)}}
\mathrm K\!\left(
\frac{16r}{(1+r)^3(3-r)}
\right),&0<r<1,\\[2ex]
\displaystyle
\frac{\sqrt r}{\pi^2}
\mathrm K\!\left(
\frac{(1+r)^3(3-r)}{16r}
\right),&1<r<3.
\end{cases}
\tag{48}
\]

Equation (48) follows by first adding two unit planar vectors, whose radial density is
\(2/[\pi\sqrt{4-s^2}]\), and then integrating the angle of the third.

Thus the missing numerical inputs are **four outward-rounded scalar intervals**:

\[
a(0),\quad b(0),\quad a(0.2),\quad b(0.2).
\]

The supplied `eq:dr-area` gives only the approximation
“\(\eta_{100}(0)\simeq0.78658\),” not an outward-rounded interval. The supplied Appendix A.2 gives no numerical [111] support at either mass.

To substantiate the advertised intervals, the numerical certificate must establish, with outward rounding,

\[
\begin{array}{ll}
\overline h_{\rm in}(a_-(0),b_-(0))>0.8164,
&
\overline h_{\rm out}(a_+(0),b_+(0))<1.0203,\\
\overline h_{\rm in}(a_-(0.2),b_-(0.2))>0.7883,
&
\overline h_{\rm out}(a_+(0.2),b_+(0.2))<0.9831.
\end{array}
\tag{49}
\]

For (47), that certificate must separately bound:

1. the half-chain truncation error;
2. Fourier-coefficient error;
3. singular-value or entropy-evaluation error;
4. quadrature error, including \(r=1\);
5. the massless endpoint \(r\uparrow3\).

Neither the support values nor those five error bounds are supplied. Inferring \(b\) from the desired endpoints in (49) would use the target as an input and would not be a verification.

I therefore do **not** provide replacement paper text asserting those decimal enclosures, an unexecuted script described as a completed certificate, or an invented H100/32-core runtime.

---

# Adversarial self-check

1. **Purity is used in the right place.**  
   The identity \(P_A-P_A^2=X_PX_P^\dagger\), and therefore the cross-block entropy formula, requires projectors. The massless statement in the supplied manuscript explicitly concerns a “pure covariance.” No mixed-state extension has been silently asserted.

2. **The entropy derivative is not treated as Lipschitz at zero.**  
   \(f''(x)\) diverges logarithmically. Equations (4)–(5) regularize that endpoint and charge the finite rank for the regularization. Omitting that charge would invalidate the argument.

3. **The Jensen direction is correct.**  
   Since \(u^{2/3}\) is concave,
   \(\langle e,D^{2/3}e\rangle\le\langle e,De\rangle^{2/3}\).
   This produces an upper bound on the entropy density, as required.

4. **The Fourier-tail direction is correct.**  
   The averaged second-difference multiplier is bounded **below** on \(|z|\ge t\), while its total Fourier sum is bounded **above** by (10). Together they give an upper bound on the tail.

5. **Low-energy counting contains all species once.**  
   Equation (31) includes four components and eight nodes. Its enclosing balls are supersets of the low-energy regions because
   \(\varepsilon\ge2r/\pi\). The containment direction is correct.

6. **The high-energy inequality has the correct weight.**  
   On \(I-Q_\lambda\), \(\Omega_0\ge\lambda\), hence
   \(\|(I-Q_\lambda)T\|_2^2\le E_\varphi(C)/\lambda\).
   It is not \(E_\varphi/\lambda^2\).

7. **No localization commutator has been discarded.**  
   Equation (33) is exact. The entire localization remainder is
   \(\Pi_AP_0(1-\varphi)\), which is bounded in (34). The contraction property (30) follows from orthogonal blocks, not an assumed commutation.

8. **The exponents recompute correctly.**  
   At \(\lambda\sim L^{-1/2}\),
   \[
   L^3\lambda^3\sim L^{3/2},
   \qquad L/\lambda\sim L^{3/2}.
   \]
   Therefore \(B\sim L^{3/2}\), and
   \(\sqrt{L^2B}\sim L^{7/4}\).
   The remaining \(L^{3/2}\log L\) term is bounded above by a constant times \(L^{7/4}\).

9. **The massive first-order term really cancels.**  
   The clock contribution vanishes through
   \(\{h_0,\partial_kP_0\}=0\). The surviving corrector is odd under the stated time reversal. The regularized entropy gradient at \(P_0\) is even, so their real trace pairing is zero.

10. **The massive proof uses only two spatial derivatives.**  
    The \(r_0\) product is expanded to first order with a second-order remainder. The \(r_1\) product is controlled only to zeroth order with a first-order remainder. No derivative of order three has been hidden in the symbol expansion.

11. **Submodularity has the required direction.**  
    Uncrossing decreases the entropy sum:
    sawtooth sets have entropy sum at least that of the sorted slabs. After subtracting the common lower faces, this gives
    \(\Phi(\lambda\xi+(1-\lambda)\eta)\le\lambda\Phi(\xi)+(1-\lambda)\Phi(\eta)\), not the reverse.

12. **The numerical lower and upper bodies are not interchanged.**  
    The convex hull in (41) lies inside \(K\); its mean support is a lower bound. The half-space intersection contains \(K\); its mean support is an upper bound. Equation (49) uses lower support endpoints for the lower bound and upper endpoints for the upper bound.

13. **No numerical certification has been manufactured.**  
    The [111] supports and their outward-rounded errors have not been computed here. The analytical results above do not, by themselves, certify the manuscript’s four advertised decimal endpoints.