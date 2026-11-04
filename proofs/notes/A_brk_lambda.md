
## Result

**The fermions determine a static fourth-order vacuum polarization. They do not determine the independent charge-law coefficient \(\lambda\).** The band calculation and the zero-frequency determinant calculation give the same polarization, including its normalization. Computing that number does not remove the freedom exhibited in the supplied obstruction.

There are two distinct statements:

1. **At fixed Hamiltonian, vacuum, and field convention, the microscopic static response is fixed.** It is not arbitrarily regulator-dependent.
2. **Identifying that response with the charge-law coefficient is an additional matching law.** No such identification appears in the supplied model.

Below I derive both response formulas, give an explicit Brillouin-zone integrand and two quadrature formulas with rigorous discretization bounds, and derive the exact reference/law repartition freedom. **I have not executed those quadratures, so I do not report an invented numerical interval for the microscopic coefficient.** The obstruction to predicting \(\lambda\), however, is exact and does not depend on their numerical value.

---

# 1. The missing identification is explicit in the supplied equations

In `00_PAPER_flat_b39645f976f_reviewed.tex`, function `\mathcal W[u]`, equation `eq:dm-reference`, the reference is

> `\mathcal W[u]&=\Tr(P_\star h[u])+W_2[u]-DW_2[u_\star](u)`

The same file defines `W_2[u]` as

> “the second-gradient term of the vacuum energy of the prepared band”

and describes its tensor in `eq:dm-response` as

> “a tensor response and not a renormalization of \(G\).”

The charge-law functional is a different object. In `sec:model-gravity`, the paper states that the field equations are

> “the stationarity conditions for \(E_m-\int\mathcal A_0\,d^3y\) at fixed covariance.”

Consequently their variational form is
\[
D_u\!\left[\Tr(Ch[u])-\mathcal W[u]-\mathscr A_\lambda[u]\right]=0,
\tag{1}
\]
where \(\mathscr A_\lambda\) denotes the spatial integral of the law density.

Changing \(\lambda\) changes \(\mathscr A_\lambda\), but changes neither \(h[u]\) nor the declared \(\mathcal W[u]\). Therefore it changes neither of the external-field fermion calculations requested below.

That is the decisive obstruction:

\[
\boxed{
\text{Identical fermion response for every }\lambda
\quad\not\Rightarrow\quad
\text{a selected }\lambda .
}
\tag{2}
\]

The following calculations make precise what the common fermion response actually is.

---

# 2. Calculation one: the band/Moyal coefficient

## 2.1 Response and normalization

The response comparable to the quoted \(q^4v_T\) law is the **static, infinitesimal response about**
\[
N=1,\qquad v=0,\qquad P_\star=P_0.
\]
This lies below the sign-reversal threshold. At \(ma=0.2\), the unperturbed one-particle gap is \(2m=0.4/a\).

Use a capital \(Q\) for the external wave number, to distinguish it from the paper’s current symbol \(q_i(p)\). Define
\[
\mu=ma=\frac15,\qquad k=ap,\qquad z=aQ,
\]
\[
\mathbf Q=Qn,\qquad |n|=|t|=1,\qquad n\cdot t=0,
\]
with \(t\) the transverse polarization. Put
\[
s_i(k)=\sin k_i,\qquad c_i(k)=\cos k_i,
\]
\[
e(k)=\sqrt{\mu^2+\sum_i s_i(k)^2},
\qquad
H_0(k)=\mu\beta_{\rm D}+\sum_i\alpha_i s_i(k),
\]
\[
P_\pm(k)=\frac12\left(I\pm\frac{H_0(k)}{e(k)}\right),
\qquad
k_\pm=k\pm\frac z2n.
\tag{3}
\]

Define the static Hessian by
\[
W_{\rm band}^{(2)}
=\frac12\int\frac{d^3Q}{(2\pi)^3}\,
v_T(-Q)\,\Pi_T(Q)\,v_T(Q).
\tag{4}
\]

This convention fixes all factors of two below.

## 2.2 The exact transfer vertex

In `00_PAPER_flat_b39645f976f_reviewed.tex`, function `h[N,v]`, equation `eq:dm-hamiltonian`, the relevant printed terms are

> `\tfrac12\sum_i\{v^i,Q_i\}`

and

> `\tfrac18\sum_k\{(\nabla\times v)^k,\Sigma^{\rm lat}_k\}`.

Their matrix element between momenta \(p_-\) and \(p_+\), divided by the flow Fourier amplitude, is \(a^{-1}V\), where
\[
V(k,z;n,t)=R(k,z)I+i\boldsymbol\Sigma\cdot S(k,z),
\tag{5}
\]
\[
R(k,z)=\sum_jt_j\,s_j(k)c_j(k)\cos(zn_j),
\tag{6}
\]
and
\[
S_r(k,z)=\frac z4(n\times t)_r\,\overline C_r(k,z),
\tag{7}
\]
\[
\overline C_r(k,z)
=\frac12\left[
c_i(k_+)c_j(k_+)+c_i(k_-)c_j(k_-)
\right],
\qquad (i,j,r)\ {\rm cyclic}.
\tag{8}
\]

The factor \(1/4\) in (7) comes from the printed \(1/8\) anticommutator and its two endpoint symbols. The \(\cos(zn_j)\) in (6) is the exact two-step-current endpoint factor. Neither may be dropped in a fourth-gradient calculation.

## 2.3 Why this gives the Moyal coefficient

For the band projector, the Moyal construction is
\[
\pi\star\pi=\pi,\qquad [h,\pi]_\star=0,
\]
or equivalently the lower-band contour of the star resolvent. Fourier multiplication turns its star products into the exact momentum shifts \(k_\pm\). Expanding those shifts in \(z\) is precisely the spatial Moyal expansion.

At second order in flow amplitude, the contour residues give the occupied-to-empty-band energy shift
\[
\Delta W^{(2)}
=-\sum_{\substack{a\ {\rm occupied}\\b\ {\rm empty}}}
\frac{|\langle b|\delta h|a\rangle|^2}{E_b-E_a}.
\tag{9}
\]
Here \(E_b-E_a=(e_++e_-)/a\). Thus the exact quadratic band response is
\[
\boxed{
\Pi_T(Q)
=-\frac{2}{a^4}\,\mathcal I(z;n,t),
}
\tag{10}
\]
where
\[
\boxed{
\mathcal I(z;n,t)
=\int_{[-\pi,\pi]^3}\frac{d^3k}{(2\pi)^3}\,
\frac{
\operatorname{tr}\!\left[
P_-(k_-)\,V^\dagger P_+(k_+)\,V
\right]}
{e(k_-)+e(k_+)} .
}
\tag{11}
\]

The four-component trace is included. **There is no additional spin or valley multiplier.**

Since the band is gapped,
\[
\mathcal I(z;n,t)
=I_2(n,t)z^2+I_4(n,t)z^4+O(z^6).
\tag{12}
\]
Consequently
\[
\boxed{
\Pi_T(Q)
=-\frac{2I_2(n,t)}{a^2}Q^2
-2I_4(n,t)Q^4+O(a^2Q^6).
}
\tag{13}
\]

The requested microscopic fourth-order response coefficient, in convention (4), is therefore
\[
\boxed{\Pi^{\rm band}_{4,T}(n,t)=-2I_4(n,t).}
\tag{14}
\]

This calculation obtains the quadratic transverse part of the fourth-gradient functional. It does not purport to classify all nonlinear fourth-gradient terms.

## 2.4 An explicit scalar Brillouin-zone integrand

No matrix implementation is necessary. Write
\[
s_\pm=(\sin k_{\pm1},\sin k_{\pm2},\sin k_{\pm3}),
\qquad e_\pm=\sqrt{\mu^2+s_\pm^2}.
\]
Taking the trace in (11) gives
\[
\begin{aligned}
\mathcal T(k,z;n,t)
={}&R^2+S^2\\
&-\frac{1}{e_-e_+}
\Big[
R^2(\mu^2+s_-\cdot s_+)
+S^2(\mu^2-s_-\cdot s_+)\\
&\hspace{29mm}
+2(S\cdot s_-)(S\cdot s_+)
-2R\,S\cdot(s_-\times s_+)
\Big].
\end{aligned}
\tag{15}
\]

Thus a directly evaluable zone integral is
\[
\boxed{
I_4(n,t)
=\int_{[-\pi,\pi]^3}\frac{d^3k}{(2\pi)^3}\,
f_4(k;n,t),
\qquad
f_4=[z^4]\frac{\mathcal T(k,z;n,t)}{e_-+e_+}.
}
\tag{16}
\]

Here \([z^4]\) means the Taylor coefficient, equivalently \(1/4!\) times the fourth derivative at zero. Equation (16), with (6)–(8) and (15), includes the orbital vertex, spin vertex, interference, endpoint corrections, and shifted denominators.

A proportionality such as the one in `04_F_r7open9_val.md`, §2,

> “zone integral ∝ ∫_BZ q⁴/ε⁵”

does not specify these terms or their constants. It cannot be used as a numerical fourth-order coefficient.

## 2.5 Check against the declared second-order tensor

Let
\[
F^0_{ij}=\frac12(n_it_j+n_jt_i),
\qquad
\widehat{\mathsf T}=a^2\mathsf T,
\]
where \(\mathsf T\) is the printed tensor in `eq:dm-response`. Expanding (11) to second order gives
\[
\boxed{
I_2(n,t)=\frac14F^0_{ij}
\widehat{\mathsf T}_{ij,kl}F^0_{kl}.
}
\tag{17}
\]

Therefore (10) reproduces
\[
W_2=-\frac14\int F_{ij}\mathsf T_{ij,kl}F_{kl}\,d^3x
\]
at \(N=1\), exactly as printed.

This checks the spin normalization, trace count, sign, and Hessian factor. The quoted second-order number \(\widehat\gamma_{11}=0.0105\) does not determine the independent Taylor coefficient \(I_4\).

---

# 3. Calculation two: the static induced action

The fermion determinant can be constructed from the supplied Hamiltonian without postulating a gravitational action. For a static background, use the zero-temperature limit of the fermionic trace, or its coherent-state representation,
\[
\Gamma_E[v]=-\operatorname{Tr}\log(\partial_\tau+h[v]).
\tag{18}
\]

Its quadratic term is
\[
\Gamma_E^{(2)}
=\frac12\operatorname{Tr}(G_0\delta h\,G_0\delta h).
\tag{19}
\]
There is no quadratic seagull vertex: the supplied \(h[N,v]\) is linear in the coefficient fields.

In dimensionless frequency,
\[
G_0(i\nu,k)
=\sum_{\sigma=\pm}\frac{P_\sigma(k)}{i\nu-\sigma e(k)}.
\tag{20}
\]
The frequency integral is
\[
\int\frac{d\nu}{2\pi}
\frac{1}{(i\nu-E_a)(i\nu-E_b)}
=
\frac{\theta(-E_a)-\theta(-E_b)}{E_a-E_b}.
\tag{21}
\]

Same-band terms vanish because their occupations agree. Each interband ordering contributes \(-1/(e_++e_-)\). The two orderings give
\[
\Pi_T^{\rm det}(Q)
=-\frac{2}{a^4}
\int_{\rm BZ}\frac{d^3k}{(2\pi)^3}
\frac{\mathcal T(k,z;n,t)}{e_-+e_+}.
\tag{22}
\]

Hence
\[
\boxed{
\frac{\Gamma_E^{(2)}}{\beta}=W_{\rm band}^{(2)},
\qquad
\Pi^{\rm det}_{4,T}(n,t)
=\Pi^{\rm band}_{4,T}(n,t)
=-2I_4(n,t).
}
\tag{23}
\]

**They agree before numerical integration.** The two calculations are the spectral/Moyal expansion and the frequency-integrated loop expansion of the same static fermion response.

At \(\mu=1/5\), the denominators are bounded away from zero. There is no massless infrared ambiguity in exchanging the small-frequency and small-wavevector limits around this vacuum. A time-dependent response, a pocket-filled state, or a massless limit would be a different calculation.

---

# 4. Extracting an isotropic coefficient from the cubic response

The microscopic lattice response is cubic, not generally isotropic. An axial transverse coefficient alone cannot be called the isotropic coefficient.

A precise definition is the rotational projection
\[
I_4^{\rm iso}
=\frac{1}{4\pi}\int_{S^2}d\Omega_n\,
\frac12\sum_{\alpha=1}^2 I_4(n,t_\alpha).
\tag{24}
\]
This is a definition of the isotropic part of the computed tensor, not an additional field law.

At this order, the polarization-averaged integrand is a polynomial of degree at most six in \(n\). Cubic symmetry therefore reduces (24) exactly to
\[
\boxed{
I_4^{\rm iso}
=\frac27 I_{100}
+\frac8{35}\left(I_{110,\parallel}+I_{110,\perp}\right)
+\frac9{35}I_{111}.
}
\tag{25}
\]
The four integrals use
\[
\begin{array}{c|c|c}
&n&t\\ \hline
100&(1,0,0)&(0,1,0)\\
110,\parallel&(1,1,0)/\sqrt2&(1,-1,0)/\sqrt2\\
110,\perp&(1,1,0)/\sqrt2&(0,0,1)\\
111&(1,1,1)/\sqrt3&(1,-1,0)/\sqrt2 .
\end{array}
\tag{26}
\]

For completeness, the angular rule underlying (25) has weights per point \(1/21\) on the six axes, \(4/105\) on the twelve edge directions, and \(9/280\) on the eight corner directions. It reproduces the spherical moments through degree six, including
\[
\langle n_x^4\rangle=\frac15,\quad
\langle n_x^6\rangle=\frac17,\quad
\langle n_x^2n_y^2n_z^2\rangle=\frac1{105}.
\]

In the normalization of the question,
\[
\boxed{
c_{4,\rm band}^{\rm iso}=-I_4^{\rm iso},
\qquad
\Pi_{4,T}^{\rm iso}=-2I_4^{\rm iso},
}
\tag{27}
\]
where the rotationally projected transverse band energy contains
\[
c_{4,\rm band}^{\rm iso}\int d^3x\,|\nabla\times\nabla\times v|^2.
\]

These are microscopic response coefficients, not yet \(\lambda\).

---

# 5. CPU evaluation and two rigorous quadrature bounds

## 5.1 Evaluate Taylor coefficients, not small differences

Equation (16) can be evaluated using degree-four Taylor polynomials. For example,
\[
\begin{aligned}
\sin(k_i\pm zn_i/2)
={}&s_i\pm \frac{n_ic_i}{2}z
-\frac{n_i^2s_i}{8}z^2
\mp\frac{n_i^3c_i}{48}z^3
+\frac{n_i^4s_i}{384}z^4,\\
\cos(k_i\pm zn_i/2)
={}&c_i\mp \frac{n_is_i}{2}z
-\frac{n_i^2c_i}{8}z^2
\pm\frac{n_i^3s_i}{48}z^3
+\frac{n_i^4c_i}{384}z^4.
\end{aligned}
\tag{28}
\]
All products are truncated after degree four. For the square roots use
\[
(1+y)^{1/2}
=1+\frac y2-\frac{y^2}{8}+\frac{y^3}{16}
-\frac{5y^4}{128}+O(y^5),
\tag{29}
\]
and polynomial reciprocals for divisions.

This avoids subtracting nearly equal finite-\(Q\) responses and dividing the result by \(Q^4\).

Let \(f_{\rm iso}(k)\) be the corresponding positive-weight combination of the four \(f_4\) integrands in (25). The scalar expression is \(\pi\)-periodic in each \(k_i\), so
\[
I_4^{\rm iso}
=\frac1{\pi^3}\int_{[-\pi/2,\pi/2]^3} f_{\rm iso}(k)\,d^3k.
\tag{30}
\]
This domain reduction does not introduce a valley multiplier.

## 5.2 An explicit analytic majorant

The following deliberately loose bound suffices to certify quadrature truncation.

Take
\[
|\operatorname{Im}k_i|\le\delta=\frac2{25}=0.08,
\qquad |z|\le r=\frac1{25}=0.04.
\]
Then
\[
|\operatorname{Im}(k_i\pm zn_i/2)|\le0.1,
\]
and
\[
\operatorname{Re}(e_\pm^2)
\ge \frac1{25}-3\sinh^2(0.1)>0.009.
\tag{31}
\]
The principal square roots are analytic there. Elementary bounds give
\[
\|s_\pm\|<1.75,\qquad |R|<0.9,\qquad \|S\|<0.011,
\]
\[
|e_-e_+|>0.009,\qquad |e_-+e_+|>0.18.
\tag{32}
\]
Substitution in (15) bounds
\[
\left|\frac{\mathcal T}{e_-+e_+}\right|<2000.
\]
Cauchy’s coefficient bound therefore gives
\[
|f_4(k;n,t)|
\le \frac{2000}{r^4}
=781\,250\,000
< M,\qquad M=8\times10^8.
\tag{33}
\]
The same bound holds for \(f_{\rm iso}\).

For complex arguments, (15) is analytically continued using its displayed bilinear products, not complex conjugation.

## 5.3 Quadrature A: periodic trapezoidal rule

Define the exact finite sum
\[
\mathrm{Trap}_N
=\frac1{N^3}
\sum_{j_1,j_2,j_3=0}^{N-1}
f_{\rm iso}\!\left(
-\frac\pi2+\frac{\pi j_1}{N},
-\frac\pi2+\frac{\pi j_2}{N},
-\frac\pi2+\frac{\pi j_3}{N}
\right).
\tag{34}
\]
Analyticity and \(\pi\)-periodicity bound its omitted Fourier coefficients:
\[
\boxed{
|I_4^{\rm iso}-\mathrm{Trap}_N|
\le
M\left[
\left(\frac{1+\xi}{1-\xi}\right)^3-1
\right],
\qquad
\xi=e^{-2\delta N}.
}
\tag{35}
\]
For \(N=288\),
\[
\boxed{
|I_4^{\rm iso}-\mathrm{Trap}_{288}|<5\times10^{-11}.
}
\tag{36}
\]

## 5.4 Quadrature B: tensor Gauss–Legendre

Let \(x_j\) be the roots of the degree-\(n\) Legendre polynomial and let
\[
\widehat w_j
=\frac{1}{(1-x_j^2)[P_n'(x_j)]^2},
\qquad \sum_j\widehat w_j=1.
\]
Define
\[
\mathrm{GL}_n
=\sum_{i,j,k=1}^{n}
\widehat w_i\widehat w_j\widehat w_k\,
f_{\rm iso}\!\left(\frac\pi2x_i,\frac\pi2x_j,\frac\pi2x_k\right).
\tag{37}
\]
Set
\[
\rho=\frac{2\delta}{\pi}
+\sqrt{1+\left(\frac{2\delta}{\pi}\right)^2}.
\]
The Bernstein ellipse lies in the analytic strip. Polynomial approximation, exactness through degree \(2n-1\), and positivity of the normalized weights give
\[
\boxed{
|I_4^{\rm iso}-\mathrm{GL}_n|
\le
\frac{12M}{\rho-1}\rho^{-(2n-1)}.
}
\tag{38}
\]
For \(n=512\),
\[
\boxed{
|I_4^{\rm iso}-\mathrm{GL}_{512}|<5\times10^{-12}.
}
\tag{39}
\]

These are two different quadratures with proved discretization bounds, not a convergence claim inferred from their difference.

### What remains to be measured

To obtain a publishable decimal, evaluate (34) and (37) with outward-rounded arithmetic, including certified node and weight intervals, and report
\[
\begin{aligned}
I_4^{\rm iso}\in{}&
[\mathrm{Trap}_{288}^{\rm lo}-5{\times}10^{-11},
 \mathrm{Trap}_{288}^{\rm hi}+5{\times}10^{-11}]\\
&\cap
[\mathrm{GL}_{512}^{\rm lo}-5{\times}10^{-12},
 \mathrm{GL}_{512}^{\rm hi}+5{\times}10^{-12}].
\end{aligned}
\tag{40}
\]
Then multiply the interval by \(-2\) for the response coefficient in (27).

**Neither sum has been evaluated here.** Thus (40) is an explicit certified evaluation prescription, not an executed numerical result. An evaluated interval for \(I_4^{\rm iso}\), and independently enclosed arithmetic errors for both sums, are exactly the missing numerical readouts.

---

# 6. Why their agreement still does not select \(\lambda\)

The transverse charge law is
\[
\chi_T^{\rm law}(Q)
=\frac{Q^2}{16\pi G}
+\frac{\lambda a^2}{8\pi G}Q^4+\cdots.
\tag{41}
\]
The fermion response is
\[
\Pi_T^{\rm band}(Q)
=-\frac{2I_2}{a^2}Q^2-2I_4Q^4+\cdots.
\tag{42}
\]

These are not two formulas for the same quantity:

- Equation (41) assigns the source required by a trial field.
- Equation (42) gives the change of a specified fermion state’s moments in an externally imposed field.

Indeed, their leading terms are already different. The band energy is concave under this perturbation, \(I_2\ge0\), and its response is the polarization that the declared reference subtracts. The law’s transverse coefficient is \(+1/(16\pi G)\).

For flat preparation, subtraction through \(W_2\) removes the \(Q^2\) vacuum term. The residual static sea contribution is
\[
J^{\rm sea,sub}_{4,T}
=-2I_4^{\rm iso}Q^4v_T.
\tag{43}
\]
Moving this source contribution to the law side produces an **effective shift**
\[
\boxed{
\lambda_{\rm eff}
=\lambda+16\pi\frac{G}{a^2}I_4^{\rm iso}.
}
\tag{44}
\]
It does not determine the \(\lambda\) already present.

One could instead impose
\[
\mathscr A_4=W_{{\rm band},4}^{\rm iso}.
\]
That would give
\[
\lambda=-16\pi\frac{G}{a^2}I_4^{\rm iso}.
\tag{45}
\]
But (45) follows from the newly imposed equality, **not from the supplied declarations**. Choosing the opposite sign, choosing an induced-only effective law, or projecting out inadmissible pieces likewise requires a stated matching prescription.

The exact obstruction survives any numerical precision: replace \(\lambda\) by \(\lambda+\Delta\lambda\) while retaining the same \(h\), reference, and mass. Equations (11) and (22) remain unchanged, while (41) changes by
\[
\Delta J_{4,T}
=\frac{\Delta\lambda\,a^2}{8\pi G}Q^4v_T.
\tag{46}
\]

---

# 7. Exact scheme freedom, with signs and constants

## 7.1 Reference/law repartition

Define
\[
B=\nabla\times\nabla\times v,
\qquad
S_4[N,v]=\int d^3x\,\frac{B^2}{N}.
\]
For any real dimensionless \(c\), consider
\[
\mathcal W_c=\mathcal W+cS_4,
\qquad
\mathscr A_c=\mathscr A-cS_4.
\tag{47}
\]
Their sum is unchanged, so (1) and the coupled matter evolution are unchanged.

The added reference derivatives are
\[
\Delta r_N=-c\frac{B^2}{N^2},
\qquad
\Delta r_v=2c\,\nabla\times\nabla\times\left(\frac BN\right).
\tag{48}
\]
Since the sources are state minus reference, their shifts have the opposite signs. The law must therefore shift by \(-cS_4\), as in (47).

Using the question’s normalization,
\[
\boxed{
\lambda_c=\lambda-\frac{16\pi G}{a^2}c.
}
\tag{49}
\]

At homogeneous preparation, \(B=0\) and \(DS_4=0\), so this addition also leaves the preparation moments unchanged.

**Important distinction:** the paper fixes the reference through second order. Adding \(cS_4\) changes that reference prescription. It is an equivalent repartition only if the law is changed simultaneously. With the declared reference held fixed, different \(\lambda\) are genuinely different admissible laws, not merely different names for one law.

## 7.2 Averaging convention

In `00_PAPER_flat_b39645f976f_reviewed.tex`, `eq:dm-average` and its following text specify the kernel’s

> “per-axis variance \(\sigma^2=\ell^2/15+a^2/12\)”.

Its transfer function satisfies
\[
K(Q)=1-\frac12\sigma^2Q^2+O(Q^4).
\]
A raw response expressed in coarse fields has
\[
\Pi_\ell(Q)=|K(Q)|^2\Pi_{\rm point}(Q).
\]
Thus
\[
\boxed{
\Pi_{4,\ell}=\Pi_{4,\rm point}-\sigma^2\Pi_{2,\rm point}.
}
\tag{50}
\]

This is an exact fourth-order convention shift. However, if the same back-averaging is used for both the band response and the declared \(W_2\) subtraction, the \(W_2\)-generated shift cancels in their difference:
\[
|K|^2(\Pi_{\rm band}-\Pi_2Q^2)
=\Pi_4Q^4+O(Q^6).
\tag{51}
\]
It would be incorrect to count that canceled averaging term as an intrinsic ambiguity of the residual fourth-order polarization.

## 7.3 Regulator and limits

- **Lattice regulator:** the sine hopping, two-step current, and spin vertex are fixed inputs. Replacing them changes the matter model; it is not an unfixed regulator within this calculation.
- **Static limits at \(ma=0.2\):** the gap makes the response analytic near zero frequency and momentum. No infrared order-of-limits freedom is needed here.
- **Finite local subtraction:** equations (47)–(49) give its exact freedom.
- **Independent charge-law term:** even after fixing every preceding convention, the admissible bare \(\lambda\) remains free.

Thus the answer is not simply “the integral is scheme-dependent.” **The integral is fixed in a fixed convention; the missing charge-law matching remains independent.**

---

# 8. Replacement Outlook sentence

The following states the exact freedom without presenting an unevaluated loop coefficient as a prediction:

> **“With \(B=\nabla\times\nabla\times v\), the simultaneous change \(\mathcal W\mapsto\mathcal W+c\int |B|^2/N\,d^3x\), \(\lambda\mapsto\lambda-16\pi Gc/a^2\) leaves the coupled equations unchanged; at the stated second-order subtraction, the fermion polarization supplies a source correction and \(\lambda\) remains an independent fourth-order transverse-response datum.”**

---

## Adversarial checks

1. **Missing spin or valley factor?**  
   The \(1/8\) spin anticommutator gives (7); the four-component trace is taken in (15); the full zone already counts all valleys.

2. **Wrong energy/Hessian factor?**  
   The band energy shift is \(-\mathcal I/a^4\) per quadratic Fourier product. Convention (4) therefore gives \(\Pi=-2\mathcal I/a^4\).

3. **Is this actually fourth gradient order?**  
   Every external spatial derivative contributes \(Q\). Extracting \([z^4]\), including the endpoint vertices, gives exactly the quadratic fourth-gradient term.

4. **Could agreement of the two routes fix an extra parameter?**  
   No. Their equality is the frequency-integral identity (21). Both are independent of the bare \(\lambda\).

5. **Is the reference-shift sign correct?**  
   Increasing the reference decreases the source. The law must decrease by the same functional derivative, giving \(\lambda_c=\lambda-16\pi Gc/a^2\), not the opposite sign.

6. **Was an arbitrary numerical value supplied?**  
   No. The exact integrands and two proved quadrature bounds are supplied; an executed microscopic interval is not claimed. No possible result of those quadratures eliminates the independent term (46).

**Conclusion:** the two static fermion calculations agree and determine a vacuum-polarization correction. Turning that correction into a prediction for \(\lambda\) requires an additional transverse matching law. The supplied model does not contain it.