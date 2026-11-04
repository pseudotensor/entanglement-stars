# R7E4_A2_TEXT_v1: App. A.2 replacement with the certified [111] enclosure (edit 4 / TODO[50])

Agreement on the text and rules: reviewer B A_r7e4_a2text 18791399, reviewer A F_r7e4_a2text_val 8b3b7f31 (corrected text and rules), reviewer B A_r7e4_a2text_accept d441cb3e. Numbers: R7E4_COMPARE_v1 (6c4e055d), routes 1 (769ca57d) and 2 (8b1dde32) agree; the hull is printed outward at 5 dp.

Replace the whole printed paragraph '\paragraph{A.2 Convexity of the planar coefficient.}' with the block below.

Main text: NO numerical change. The coefficient intervals stay eta(0) in [0.8164, 1.0203] and eta(0.2) in [0.7883, 0.9831] (Sec. results-vacuum), and the spacing endpoints stay 1.807 to 2.021 l_P (Sec. results-firstlaw) and 2.55 to 2.86 (Sec. page). 2.56 is NOT supported, so the printed 2.55 stands. These printed values are now backed by two agreeing certified routes, so any 'pending certification' wording or todo mark on them can go. The Page figure band needs no regeneration.

```latex
\paragraph{A.2 Convexity of the planar coefficient.}
\(\Phi(\boldsymbol\xi)=|\boldsymbol\xi|\eta_{\boldsymbol\xi/|\boldsymbol\xi|}\),
\(\Phi(0)=0\), is even and positively homogeneous. Strong
subadditivity~\cite{LiebRuskai1973} compares translated sawtooth
cuts with their uncrossed nested slabs, and a finite-volume limit
makes \(\Phi\) convex on each open coordinate half-space, which
homogeneity extends to all of \(\mathbb R^3\) through the local
convexity criterion on planes through the origin
(Ref.~\cite{McKinneyCode2026}). Thus \(\Phi\) is the support
function \(\max_{x\in\mathsf K}x\cdot\boldsymbol\xi\) of a compact
cubic-symmetric convex body \(\mathsf K\)~\cite{Schneider2014}.
Set \(a_\zeta=\eta_{100}(\zeta)\) and
\(b_\zeta=\sqrt3\,\eta_{111}(\zeta)\). Cubic symmetry places
\(\mathsf K\) between \({\rm conv}\{\pm a_\zeta e_i\}\) and
\([-a_\zeta,a_\zeta]^3\), giving
\begin{equation}
a_\zeta\max_i|n_i|\le\eta_{\boldsymbol n}(\zeta)
\le a_\zeta\sum_i|n_i|.
\label{eq:B-axial}
\end{equation}
Certified chain integrals give
\(a_0\in[0.78658,0.78659]\), \(a_{0.2}\in[0.75678,0.75679]\),
\(b_0\in[1.59934,1.59935]\), \(b_{0.2}\in[1.54565,1.54568]\).
Adding the [111] support, the convex hull of \(\pm a_\zeta e_i\) and
\((\pm b_\zeta/3,\pm b_\zeta/3,\pm b_\zeta/3)\) at the lower interval
endpoints lies inside \(\mathsf K\), and the intersection of the
half-spaces \(|x_i|\le a_\zeta\) and \(\sum_i|x_i|\le b_\zeta\) at
the upper endpoints contains it. The angular means of their support
functions, rounded outward, bound \(\eta(\zeta)\) and give the
interval of Sec.~\ref{sec:results-vacuum}
(Ref.~\cite{McKinneyCode2026}).
```
