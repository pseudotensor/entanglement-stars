# FABLE_write_R6_appendices_BC_v1.1 (write lane, reviewer A max; supersedes v1 per reviewer B's read)

Base: the live worktree <internal>/.wt-fermion-minimal (branch fix-reaudit-kinginputs, PR3) at HEAD a7acc6a4e0b, its two appendix files unchanged since v1 (fermion_new/A2_local_area.tex md5 f4148aa09b06eaaefa84d60233a7f4c4, 14,524 comment-stripped characters by the ceiling's command, sed 's/\([^\\]\)%.*$/\1/; s/^%.*$//' FILE | wc -c; fermion_new/charge_conservation.tex md5 978a9747e5886ec966386841a1ca8009, 5,699). The job answers reviewer B's read of v1 (ASTRA_appendices_condensed_read_v1.md, eight items) under the rule "restore what a sketch's conclusion rests on in the shortest form; plain-language fixes no longer than the original". No manuscript file was edited; the checks and the five combined builds were made under <internal>/r6_app11 on a copy of the live inputs (fermion_new/ byte-identical to the worktree's; the main file the 06:04 snapshot, which the live file has since outrun in the abstract only, #467's D0 clause at 07:02, net zero words; a confirmation build with the live main file is in the RECORD).

Delivered: two complete replacement files, applied as whole files. A2_local_area.tex: 12,485 characters against the binding ceiling 12,500 (15 under; v1 12,421; the live file 14,524), md5 f173620b26cee2f439cc636bf9ca193a. charge_conservation.tex: 5,252 against the ceiling 5,300 (48 under; v1 5,202; the live file 5,699), md5 45e26f095a79d520444e7262fb86d2fc. Together 17,737 against the ceilings' 17,800. Neither file carries a comment, so raw and stripped counts agree. The restorations reviewer B asked for cost 64 characters in B and 50 in C net of what paid for them in B: v1's own FOR THE LEAD levers (a), (b), (c) and its item 4, namely the three citations that doubled one fact and were cited nowhere else (HornJohnson2013 beside Koteljanskii1950, Edmonds1970 beside Lovasz1983, McCarthy1967 beside Rotfeld1969; the bibliography drops from 85 to 82 numbered entries), B.3's idea sentence, B.1's three named symmetry reasons, the words "on the commutators" after the Schur test, and the "[110] value as a third" clause of B.2. Each of these is recorded with its source in the held supplement FABLE_write_R6_supplement_appendices_v1.1.tex beside this file (supersedes v1's; six pages, 0 errors, 0 overfull; no number left the sketches in v1.1, so the supplement's change is bookkeeping: three restored items leave it, five v1.1 removals join it). Copies of the two files and the check scripts are in FABLE_write_R6_appendices_BC_v1.1_scripts/ beside this file; the build PDFs are in deliveries/claude_work/ (FABLE_write_R6_appendices_BC_v1.1_build_marks_on.pdf, _off.pdf, _on_live03.pdf, _off_live03.pdf, _off_livemain.pdf).

Three edits were made in this pass to the stalled v1.1 files beyond what they already carried: "positive-charge" in the first law (item 1, 16 characters); the split of B.1's path-expansion sentence at "They vanish" (item 7, two characters); and the split of the first law's sentence at "is simple." (not an reviewer B item; the restored conditions had made it an 88-word sentence, now 54 at most in C; four characters shorter, no content moved).

### NEW FILE A2_local_area.tex

The complete replacement for fermion_new/A2_local_area.tex (12,485 characters, md5 f173620b26cee2f439cc636bf9ca193a):

```latex
\section{Local area estimates}\label{app:local}

Section~\ref{sec:results-vacuum} rests on six results about the flat
sea at fixed \(ma\), sketched here with none relying on a later one
and the estimates a specialist would fill in named.

\paragraph{Conventions.}
Here \(a=1\), \(\zeta=ma\); \(\bar N\), \(\bar v\) are the back-averaged
site fields of Sec.~\ref{sec:model-dictionary} and \(h=h[\bar N,\bar v]\)
of Eq.~\eqref{eq:dm-hamiltonian} acts on
\(\ell^2(\mathbb Z^3)\otimes\mathbb C^4\); \(\hat h_0(k)\),
\(\varepsilon(k)\) and \(q_i(k)\) are the symbol
of \(h_0\), the dispersion and the current symbol of
Secs.~\ref{sec:model-matter} and \ref{sec:model-dictionary} at
\(k=ap\); \(P_0\) is the flat sea of Sec.~\ref{sec:model-sources}. Time
reversal \(\Theta=(1\otimes\sigma_y)\mathsf C\), \(\mathsf C\) complex
conjugation in the site basis (\(\alpha_i=\sigma_x\otimes\sigma_i\),
\(\beta_{\rm D}=\sigma_z\otimes1\)), fixes \(h_0\) and \(P_0\), reverses
\(Q_i\) and sends \(\bar v\) to \(-\bar v\). For fields of Lipschitz
constant at most \(C_1\epsilon\), \(h\) is within \(3C_1\epsilon\) in
norm of \(\bar N(X)h_0+\sum_i\bar v_i(X)Q_i\) (the Schur
test). For a cut \(A\) with site projection \(\Pi_A\),
\(S_A(C)=\Tr s(C_A)\) is the cut entropy of Eq.~\eqref{eq:dr-cut};
\(A_{\rm lat}\) is the lattice area, \(4\pi r^2\) for the digital ball
\(A_r(x_0)=\{x\in\mathbb Z^3:|x-x_0|<r\}\), and \(A\) is
\(\varkappa\)-regular if at most \(\varkappa A_{\rm lat}(1+d)\) of its
sites lie within distance \(d\) of \(A^c\) for every \(d\ge0\), as
digital balls are with \(\varkappa=8\).

\subsection*{B.1 The sphere coefficient}

Shown: the planar coefficient
\(\eta_{\boldsymbol n}=\int_0^\infty\mathsf e(\boldsymbol n,t)\,dt\),
\(\mathsf e(\boldsymbol n,t)=\tr[s(C_{\mathsf H})]_{00}\) the entropy
density of \(C=P_0\) at the origin at depth \(t\) below the cut
\(\mathsf H=\{\boldsymbol n\cdot x\le t\}\), is finite, continuous,
even, positive and cubic-invariant, and large balls have the entropy
\(\eta(\zeta)A_{\rm lat}+o(A_{\rm lat})\),
\(\eta(\zeta)=(4\pi)^{-1}\int_{S^2}\eta_{\boldsymbol n}d^2\boldsymbol n\).
The step is a path expansion: polynomials vanishing at \(0\) and \(1\)
approximate \(s\) with error small against \([u(1-u)]^{1/2}\)
(Weierstrass; the operator Jensen inequality~\cite{HansenPedersen1982}
controls the weight). They vanish at \(P_0\), so their trace on \(C_A\)
is the deficit sum \(\sum_Yw_Y[\#\{x:x+Y\subset A\}-\#A]\) over
closed lattice paths \(Y\) through the origin, \(w_Y\) the traced
kernel product, with \(\sum_Y|w_Y|\,{\rm diam}\,Y<\infty\). For the
half-space the deficit's depth integral is
\(\max_{y\in Y}\boldsymbol n\cdot y\), Lipschitz in \(\boldsymbol n\).
For the ball the deficit is a shell count, \(r^2\) times the sphere
integral of that maximum up to \(O_Y(r^{3/2})\) uniformly in \(x_0\),
and dominated convergence gives the ball limit.

At a rational normal \(\mathbf b/|\mathbf b|\), \(\mathbf b\) primitive,
the Fourier transform along the lattice layers \(\mathbf b\cdot x=j\),
of cell area \(|\mathbf b|\), reduces the cut to half-chains of the
layer chain \(h_{\mathbf b}(k_\perp)\), so
\(\eta_{\boldsymbol n}=|\mathbf b|^{-1}\int_{\mathbb T^2}\frac{d^2k_\perp}{(2\pi)^2}S_{\rm half}[h_{\mathbf b}(k_\perp)]\),
\(S_{\rm half}\) the half-chain entropy; for \(\mathbf b=(1,0,0)\) this
is Eq.~\eqref{eq:dr-area-chain}, with \(S_{\rm half}=s_4(M_\perp)\) by
Ref.~\cite{Eisler2025}.

\subsection*{B.2 Convexity of the planar coefficient}

Shown: \(\Phi(\boldsymbol\xi)=|\boldsymbol\xi|\eta_{\boldsymbol\xi/|\boldsymbol\xi|}\),
\(\Phi(0)=0\), is convex, even and positively homogeneous, hence the
support function \(\max_{x\in\mathsf K}x\cdot\boldsymbol\xi\) of a compact
convex body \(\mathsf K\)~\cite{Schneider2014} with the cubic symmetry
of \(\eta_{\boldsymbol n}\). Why: the Gaussian entropy is strongly
subadditive on finite lattice regions (Koteljanskii's
inequality~\cite{Koteljanskii1950}), so on a cube
\(S_A+S_{A^c}-S_{\rm cube}\) is a symmetric submodular set function
of \(A\), whose Lov\'asz extension~\cite{Lovasz1983} is
convex, even and positively homogeneous; on \(\boldsymbol\xi\cdot x\),
divided by the cube's volume, it converges to \(\Phi(\boldsymbol\xi)\) by
the path expansion of B.1, the cube's outer faces cancelling.

The bounds. \(\mathsf K\subset[-\eta_{100},\eta_{100}]^3\) and, by the
reflections of \(x_2\) and \(x_3\), \(\eta_{100}e_1\in\mathsf K\); the
support functions of the cube and of \({\rm conv}\{\pm\eta_{100}e_i\}\)
give
\begin{equation}
 \eta_{100}\max_i|n_i|\le\eta_{\boldsymbol n}\le\eta_{100}\sum_i|n_i| ,
\label{eq:B-axial}
\end{equation}
whose angular means \(M_1=\frac{3\sqrt2}\pi\arctan\frac1{\sqrt2}=0.8312\)
and \(\tfrac32\) give Eq.~\eqref{eq:dr-area-fixed-mass}. With the
exact \(\eta_{100}\) and \(\eta_{111}\) alone the same inclusions put
\(\mathsf K\) between the convex hull of the cubic orbits of
\(\eta_{100}e_1\) and \(\eta_{111}(1,1,1)/\sqrt3\) and the polytope the
planes through them cut from the cube; the angular means of their
support functions, 0.8164 and 1.0203 at \(\zeta=0\), are the certified
enclosure of \(\eta(0)\) in Sec.~\ref{sec:results-vacuum}, each end
attained, and the same anchors at \(\zeta=0.2\) give the enclosure
printed there.

\subsection*{B.3 The massive estimate}

Hypotheses: (A1) \(\bar N\), \(\bar v_i\) are restrictions of \(C^2\)
functions with \(n_*\le\bar N\le n^*\), \(|\bar v|\le v^*\), gradients at
most \(C_1\epsilon\) and second gradients at most \(C_2\epsilon^2\),
\(\epsilon=a/L\); (A2) a supremum-norm continuous path from the flat
sea to \((\bar N,\bar v)\) within (A1); (A3) a gap \(\delta>0\)
of \(h\) about zero along it, so that \(P=\mathbb 1_{(-\infty,0)}(h)\) is
the continued reference of Eq.~\eqref{eq:dm-reference}. Shown: for
\(\zeta>0\) and \(\epsilon\le\epsilon_0\), \(P=P_0+P_1+R_2\) with a
corrector \(P_1\) linear in the field gradients and odd under
\(\Theta\), a remainder \(R_2\) of order \(\epsilon^2\), and, for every
\(\varkappa\)-regular finite cut,
\begin{equation}
 |S_A(P)-S_A(P_0)|\le C\varkappa A_{\rm lat}\epsilon^2(1+|\log\epsilon|)^4 ,
\label{eq:B-massive}
\end{equation}
\(\epsilon_0,C\) depending only on \((\zeta,\delta,n_*,n^*,v^*,C_1,C_2)\).
The steps:
Taylor's formula on commutator kernels fixes \(P_1\) from \([h,P_0]\);
Combes-Thomas bounds~\cite{CombesThomas1973} along the gapped path
control \(R_2\) in an exponentially weighted Schur norm; the
second-order trace Taylor formula~\cite{Bhatia1997} for the cross
block's entropy, with the block Schatten
bound~\cite{Rotfeld1969} for the area factor, gives the
bound, the logarithms from regularizing \(g\) at scale \(\epsilon\).

\subsection*{B.4 The massless finite-parent estimate}

At \(\zeta=0\) the estimate is made on a finite parent
\(\Lambda_L=(\mathbb Z/L\mathbb Z)^3\), \(L\) even, with antiperiodic
shifts, so that the flat sea \(P_0^{(L)}\) has no zero mode;
\(\epsilon=1/L\). Hypotheses: \(n_*\le\bar N\le n^*\) and
\(|\bar v|\le v^*\) with the margin \(n_*-v^*\ge c_*>0\); the moment
bound
\(\sum_{\mathbf q}|\mathbf q|(|\widehat N(\mathbf q)|+\sum_i|\widehat v_i(\mathbf q)|)\le\mathcal F_1\epsilon\)
on the fields' Fourier coefficients; a nonwrapping ball
\(A=A_r(x_0)\), \(c_rL\le r\le0.45L\); and \(P=\mathbb 1_{(-\infty,0)}(h)\)
plus a projection onto part of \(\ker h\). Then for
\(0\le\zeta\le\zeta_*\), uniformly over the fields,
\(|S_A(P)-S_A(P_0^{(L)})|\le CA_{\rm lat}\epsilon^{1/10}\), \(C\) depending
only on \((\zeta_*,\mathcal F_1,c_*,n^*,v^*,c_r)\). The idea: the moment
bound and the margin, which makes \(h\) coercive on both bands, split
\(P-P_0\) into a part of small rank, the modes of low energy counted
on the momentum grid, and a part of small norm, bounded by the
cross-block bound of Sec.~\ref{sec:results-vacuum} in H\"older form
and Mirsky's inequality~\cite{Mirsky1960}; the flat kernel's decay
bounds \(|S_A(P_0^{(L)})-S_A(P_0)|\) by \(Cr^{3/2}(1+\log r)\), so by
the ball limit of B.1 \(S_A(P)/A_{\rm lat}\to\eta(\zeta)\) as
\(L\to\infty\) at fixed \(r/L\).

\subsection*{B.5 The gap at the trapping root}

The gap premise (A3) is certified from the symbol of \(h\) frozen at
each site. For \(0<n_*\le\bar N\le n^*\) and \(|\bar v_i|\le v^*\) with
Lipschitz constants at most \(C_1\epsilon\), the symbol frozen at the
site \(x\), \(\bar N(x)\hat h_0(k)+\sum_i\bar v_i(x)q_i(k)\), has the
frozen gap \(\mathsf g(x)=\min_k[\bar N(x)\varepsilon(k)-|\bar v(x)\cdot q(k)|]\),
and \(\mathsf g_*=\inf_x\mathsf g(x)\). For \(\mathsf g_*>0\),
\(\epsilon<\min\{1/16,(\mathsf g_*/2c_1)^2\}\) and every \(\psi\),
\begin{equation}
 \|h\psi\|\ge\Bigl(\bigl[\mathsf g_*(\mathsf g_*-c_3\epsilon^{1/2})\bigr]_+^{1/2}-3C_1\epsilon\Bigr)\|\psi\|,
\label{eq:B-gapcert}
\end{equation}
\(c_1=(\sqrt3+1)(\sqrt{\zeta^2+3}+\tfrac32)C_1\),
\(c_3=4c_1+27\pi(n^*+v^*)\), so \({\rm spec}\,h\) avoids
\((-\delta,\delta)\), \(\delta\) the right side, positive for
\(\epsilon\) small. The step: a cosine partition of unity on cells of
side \(\epsilon^{-1/2}\) and the commutator bound of the Conventions
compare \(h\), at a cost of order \(\epsilon^{1/2}\) per cell, with the
symbol frozen at each cell's centre, bounded below by \(\mathsf g_*\)
by Plancherel's identity.

At and across the root. With \(t_i=\sin^2k_i\) and \(S=\sum_it_i\),
Cauchy's inequality gives \(|q|^2\le S-S^2/3\) while
\(\varepsilon^2=\zeta^2+S\), so where \(|\bar v(x)|\le s\bar N(x)\) the
frozen gap is at least \(\bar N(x)\mathfrak g_s(\zeta)\),
\(\mathfrak g_s(\zeta)=\min_{0\le S\le3}[\sqrt{\zeta^2+S}-s\sqrt{S-S^2/3}]\);
at \(s=1\), the root included, this is positive for \(\zeta>0\),
\(\tfrac23\zeta^{3/2}+O(\zeta^{5/2})\) at small \(\zeta\) and 0.058 at
\(\zeta=0.2\), while where \(|\bar v(x)|>\bar N(x)\) the frozen gap is
positive exactly below the pocket threshold of
Eq.~\eqref{eq:dr-flow-threshold} in the flow's direction, at least
\(w_c^{(3)}(\zeta)=1/\nu_3>1\) by the bound \(|q|/\varepsilon\le\nu_3\) of
Appendix~\ref{app:charge}. Along the linear path from the flat sea
the frozen gap is at least \(\min\{\zeta,\mathsf g_*\}\), so a massive
configuration with \(\mathsf g_*>0\) meets (A2) and (A3) of the
massive estimate, Eq.~\eqref{eq:B-gapcert} holding along it with
\(\mathsf g_*\to\min\{\zeta,\mathsf g_*\}\), \(n^*\to\max\{1,n^*\}\);
for \(\zeta=0\) the frozen gap vanishes at the valleys for every
flow, which leaves the massless estimate.

\subsection*{B.6 Local gap and local area}

Where a pocket is open inside the cut, the massive bound is
transferred from a clipped comparison. For \(\zeta>0\), fields obeying (A1)
and a finite pocket region \(\mathcal U=\{x:|\bar v_x|>s_0\bar N_x\}\),
\(0<s_0<s_c\) with \(\mathfrak g_{s_c}(\zeta)>0\), let \(h^{\rm cmp}\) have
the flow clipped smoothly to \(|\bar v^{\rm cmp}_x|<s_c\bar N_x\) and
unchanged where \(|\bar v_x|\le s_0\bar N_x\): \(h-h^{\rm cmp}\) is
supported within two sites of \(\mathcal U\), and the clipped fields
obey (A1) and have a frozen gap by the Cauchy step of B.5, hence a gap
\(\delta^{\rm cmp}\) by Eq.~\eqref{eq:B-gapcert} for \(\epsilon\) small.
For a projection \(P\) commuting with \(h\), let \(E_D\) be the largest
energy modulus of a mode of \(h\) filled by \(P\) at positive or empty
at negative energy, and for \(0<\delta_1<\delta^{\rm cmp}\) let \(r_b\)
count such modes outside the window \((-\delta_1,\delta_1)\), so that
\(r_b=0\) for \(\delta_1>E_D\). There is a nonincreasing \(\mathcal E(d)\le C_{\mathcal E}e^{-\vartheta d}\),
\(\vartheta\) the Combes-Thomas rate of \(h^{\rm cmp}\) below its gap,
with \(d_0=\min\{d:\mathcal E(d)\le\log2\}\), such that for \(\epsilon\)
small and every \(\varkappa\)-regular finite cut \(A\) whose complement
lies at distance at least \(d\) from \(\mathcal U\),
\begin{equation}
\begin{aligned}
 |S_A(P)-S_A(P_0)|&\le C^{\rm cmp}_{\rm loc}\varkappa A_{\rm lat}\epsilon^2(1+|\log\epsilon|)^4\\
 &\quad+r_b\log2+\mathcal E(d),
\end{aligned}
\label{eq:B-transfer}
\end{equation}
the last two terms at most \((r_b+1)\log2\) for \(d\ge d_0\). The
steps: spectral functions of \(h\) and \(h^{\rm cmp}\) differ at distance
\(d\) from \(\mathcal U\) by exponentially small terms (Helffer-Sj\"ostrand
and Combes-Thomas bounds), so on the cut \(P\) differs from the
comparison's negative projection by the window's modes, of cut
entropy at most \(\mathcal E(d)\) by the cross-block bound, and by
\(r_b\) modes, at most \(r_b\log2\) by rank; the massive estimate for
the comparison gives the first term.
```

### NEW FILE charge_conservation.tex

The complete replacement for fermion_new/charge_conservation.tex (5,252 characters, md5 45e26f095a79d520444e7262fb86d2fc):

```latex
\section{Conservation of the boundary charge}\label{app:charge}

The boundary charge \(E\) generates the matter's Hartree evolution
and is conserved by it, with no premise on the reference's level
structure; the first law of a vacuum exterior and the rate of the
mass function at a trapping root follow from the rows, and the
back-averaging error is bounded.

\paragraph{Traces of projector derivatives.}
For a dif\-fer\-en\-ti\-able family of orthog\-o\-nal projections \(P(t)\)
commuting with a bounded Hermitian \(h(t)\), \(\dot P\) trace class,
\(P^2=P\) gives \(\dot P=P\dot P+\dot PP\), so
\(P\dot PP=(1-P)\dot P(1-P)=0\), while \([h,P]=0\) gives
\(h=PhP+(1-P)h(1-P)\); hence, by cyclicity,
\(\Tr(h\dot P)=\Tr(hP\dot PP)+\Tr(h(1-P)\dot P(1-P))=0\), degenerate
blocks of \(h\) included, since a subprojector of a spectral block
commutes with \(h\).

\paragraph{The Hartree flow conserves the boundary charge.}
On a closed parent at fixed kernel, let the radial fields \(N(r)\),
\(v(r)\hat r\) solve the rows \eqref{eq:dr-radial} with the angularly
averaged sources of \(\Delta=C-P\), a regular centre, \(\mathcal M>0\)
for \(r>0\) and \(N(\infty)=1\), differentiably in \(C\), and let
\(E[C]=\mathcal M(\infty)\). Varying the rows at fixed kernel and
\(r\), the \(\delta\mathcal M\) terms cancel between
\(N'\delta\mathcal M\) and \(N\delta\mathcal M'\), leaving
\(\partial_r(N\delta\mathcal M)=4\pi r^2(N\delta\rho_\ell+v\,\delta J_{\ell r})\),
which \(\delta\mathcal M(0)=0\) and \(N(\infty)=1\) integrate to
\(\delta E=\int(N\delta\rho_\ell+v^i\delta J_{\ell i})\,d^3y\). Radial
symmetry, the adjointness of the back-averaging to the source
averaging (Eq.~\eqref{eq:dm-average}) and the allocations of
Sec.~\ref{sec:model-dictionary},
\(\sum_xN_xe_x+\sum_{x,i}v_x^ip_{x,i}=\bar h\), make this
\(\Tr(\bar h\,\delta\Delta)\), \(\bar h=h[W_\ell^\dagger N,W_\ell^\dagger v]\),
and \(\Tr(\bar h\,\delta P)=0\), trivially for \(P\) held fixed and by
the trace identity for \(P\) varied as a differentiable projection
commuting with \(\bar h\). Hence, for every variation of \(C\) and
along every history with \(i\dot C=[\bar h,C]\) and \(P\) its analytic
lift,
\begin{equation}
\begin{aligned}
 \delta E&=\Tr(\bar h\,\delta C),\\
 \frac{dE}{dt}&=\Tr(\bar h\dot C)-\Tr(\bar h\dot P)=-i\Tr(\bar h[\bar h,C])=0 ,
\end{aligned}
\label{eq:C-generator}
\end{equation}
Eq.~\eqref{eq:dr-boundary-variation} with its conservation.

\paragraph{The first law of a vacuum exterior.}
For two positive-charge configurations solving the rows
\eqref{eq:dr-radial} with \(N>0\), \(v<0\), \(N(\infty)=1\) and vacuum
beyond a radius inside both roots, each is the exterior
\eqref{eq:dr-exterior} at its charge, so its root \(R=2GE\) is simple.
Its slope \(\kappa_R=\partial_r(N+v)|_R=1/(4GE)\) and area
\(A=4\pi(2GE)^2\) give \(\delta A=8\pi G\,\delta E/\kappa_R\), and the
work term of Eq.~\eqref{eq:page-first-law-rows} vanishes; hence, with
no stationarity premise,
\begin{equation}
 \delta E=\frac{\kappa_R}{8\pi G}\,\delta A
 =T_H\,\delta\!\left(\frac{A}{4G}\right),\qquad T_H=\frac1{8\pi GE}.
\label{eq:C-firstlaw}
\end{equation}

\paragraph{The rate of the mass function at a trapping root.}
No static filling sits at a trapping surface, because the rate the
evolution rows give the mass function at a root \(N=|v|\) is
positive for every positive band-diagonal source of any mass
\(m\ge0\). With \(t_i=\sin^2(ap_i)\),
\(a^2|q|^2=\sum_it_i(1-t_i)<\sum_it_i=a^2(\varepsilon^2-m^2)\) except at
the eight valleys, where both sides vanish, so \(q_r=n\cdot q<\varepsilon\)
for every unit \(n\) wherever \(\varepsilon(p)>0\); hence, for a
band-diagonal occupation \(f\ge0\) with \(\int_pf>0\), \(\rho=2\int_pf\varepsilon\),
\(J=2\int_pfq_r\) and the radial kinetic stress \(p_r=2\int_pfq_r^2/\varepsilon\),
the rate at a root \(v=-N\) is
\begin{equation}
\begin{aligned}
 \partial_t\mathcal M&=4\pi Nr^2(\rho+p_r-2J)\\
 &=8\pi Nr^2\!\int_pf\frac{(\varepsilon-q_r)^2}{\varepsilon}>0
 \quad\text{for every }m\ge0,
\end{aligned}
\label{eq:C-rate}
\end{equation}
the valleys being a null set. For \(m>0\) there is a margin: with
\(S=\sum_it_i\), Cauchy's inequality gives
\(|q|^2/\varepsilon^2\le(S-S^2/3)/((ma)^2+S)\), maximal over \([0,3]\) at
\(S=ma(\sqrt{(ma)^2+3}-ma)\), so
\(|q|/\varepsilon\le\nu_3=\sqrt{1+(ma)^2/3}-ma/\sqrt3<1\),
\(q_r\le\nu_3\varepsilon\) and \(\rho+p_r-2J\ge(1-\nu_3)(\rho-J)\); at
\(m=0\) only the strict inequality stands.

\paragraph{Back-averaging error.}
For an even probability kernel \(k\) on \(\mathbb R^3\) with per-axis
variance \(\sigma^2\) and vanishing mixed second moments, and
\(F\in C^2(\mathbb R^3)\) with bounded Hessian,
\begin{equation}
 |(k*F)(x)-F(x)|\le\tfrac32\sigma^2\|\nabla^2F\|_\infty ,
\label{eq:C-average}
\end{equation}
by Taylor's formula with integral remainder integrated against
\(k\), the linear term vanishing by evenness and \(\int|z|^2k=3\sigma^2\);
for \(F\in C^4\) the next order is
\((k*F)(x)=F(x)+\tfrac12\sigma^2\,\mathrm{tr}\,\nabla^2F(x)+R(x)\),
\(|R|\le\tfrac1{24}\sup_{y,|e|=1}|\partial_e^4F(y)|\int|z|^4k\). The
kernel \(k_{\ell,a}=K_\ell*B_a\) of Eq.~\eqref{eq:dm-average} has
vanishing mixed moments and \(\sigma^2=\ell^2/15+a^2/12\), the radial
second moment of \(K_\ell\) being \(\ell^2/5\).
```

### ASTRA ITEMS

1. The first law's branch and normalization (Appendix C): the sentence now opens "For two positive-charge configurations solving the rows (dr-radial) with N > 0, v < 0, N(infinity) = 1 and vacuum beyond a radius inside both roots", so the inward branch v < 0, the lapse normalization N(infinity) = 1 and the positive charges that give both roots are stated with the conclusion kappa_R = 1/(4GE), T_H = 1/(8 pi G E); "positive-charge" is this pass's addition to the stalled file (16 characters); the interval [R_0, infinity) and R_0 < 2G min{E, E + delta E} remain folded into "a radius inside both roots" (supplement, table F).
2. B.6's holes and the modulus of E_D: "let E_D be the largest energy modulus of a mode of h filled by P at positive or empty at negative energy, and for 0 < delta_1 < delta^cmp let r_b count such modes outside the window (-delta_1, delta_1), so that r_b = 0 for delta_1 > E_D": empty negative-energy modes are counted, E_D is a modulus, and the window is named, which item 8 needs.
3. B.5's path-dependent gap condition: "meets (A2) and (A3) of the massive estimate, Eq. (B-gapcert) holding along it with g_* to min{zeta, g_*}, n^* to max{1, n^*}", the old sentence's certificate along the whole path (its threshold with the path's g_*), shortened only by "along the path" to "along it"; the linear path's parameters stay in the supplement (table D).
4. B.6's clipped comparison and B.3's regularity: "the clipped fields obey (A1) and have a frozen gap by the Cauchy step of B.5, hence a gap delta^cmp by Eq. (B-gapcert) for epsilon small", reviewer B's words; the explicit gap value N_x g_{s_c}(zeta) stays in the supplement (table E).
5. The (A1) references: B.4 states its hypotheses in their own terms, "n_* <= N <= n^* and |v| <= v^* with the margin n_* - v^* >= c_* > 0; the moment bound sum_q |q| (|N(q)| + sum_i |v_i(q)|) <= F_1 epsilon on the fields' Fourier coefficients; a nonwrapping ball A = A_r(x_0), c_r L <= r <= 0.45 L; and P = 1_{(-infinity, 0)}(h) plus a projection onto part of ker h", the absolute-value sum written out; B.5 opens "For 0 < n_* <= N <= n^* and |v_i| <= v^* with Lipschitz constants at most C_1 epsilon", the componentwise flow bound and the Lipschitz hypothesis alone; (A1) is now named only where B.3 defines it and where B.6 applies B.3.
6. B.4's small-rank term: "split P - P_0 into a part of small rank, the modes of low energy counted on the momentum grid, and a part of small norm, bounded by the cross-block bound of Sec. III C in Holder form and Mirsky's inequality", the low-energy modes and not the zero crossings, with the moment bound and the margin named as the premises of the split and "priced" gone; the rank count C L^3 (rho + epsilon)^3, the norm C epsilon/rho at rho = epsilon^{2/5} and p = 5/6 stay in the supplement (table C).
7. B.1's path expansion: two sentences now, "polynomials vanishing at 0 and 1 approximate s with error small against [u(1-u)]^{1/2} (Weierstrass; the operator Jensen inequality controls the weight). They vanish at P_0, so their trace on C_A is the deficit sum sum_Y w_Y [#{x : x + Y in A} - #A] over closed lattice paths Y through the origin, w_Y the traced kernel product, with sum_Y |w_Y| diam Y < infinity": the error's relative sense, the vanishing at P_0 as the reason, the deficit sum named, and the convergence condition as a formula; the split at "They vanish" is this pass's (two characters).
8. B.6's window and "priced": the window (-delta_1, delta_1) is defined with r_b (item 2), and the steps read "P differs from the comparison's negative projection by the window's modes, of cut entropy at most E(d) by the cross-block bound, and by r_b modes, at most r_b log 2 by rank; the massive estimate for the comparison gives the first term"; "priced" occurs nowhere in either file.

### LABELS

- Appendix B defines, each once and in the live order: app:local (the section), eq:B-axial (B.2), eq:B-massive (B.3), eq:B-gapcert (B.5), eq:B-transfer (B.6). Cited outside, on the same results (verify.py on the live set): app:local by 03_model.tex, 04_results.tex and gauss_law_premises.tex (and once by the Sec. II v5.2 block); eq:B-axial, eq:B-massive, eq:B-gapcert and eq:B-transfer by 04_results.tex; eq:B-transfer also by 07_page_curve.tex.
- Appendix C defines, each once and in the live order: app:charge (the section), eq:C-generator, eq:C-firstlaw, eq:C-rate, eq:C-average. Cited outside: app:charge by 03_model.tex (three times in the v5.2 block as well), 04_results.tex, 05_star.tex and 06_scaling.tex; eq:C-firstlaw by 04_results.tex; eq:C-rate by 05_star.tex. eq:C-trace stays retired as in v1 (cited nowhere outside Appendix C in the live files, the condensed deliveries, the v5.2 block or the letter).
- References from the two files, every one resolving in the live files and in the Sec. II v5.2 block (its six Sec. II labels defined once in each 03): Appendix B cites sec:results-vacuum, sec:model-dictionary, sec:model-matter, sec:model-sources, eq:dm-hamiltonian, eq:dm-reference, eq:dr-cut, eq:dr-area-chain, eq:dr-area-fixed-mass, eq:dr-flow-threshold, app:charge and its own eq:B-gapcert; Appendix C cites eq:dr-radial, eq:dr-boundary-variation, eq:dm-average, sec:model-dictionary, eq:dr-exterior and eq:page-first-law-rows. Appendix B's nine cite keys (HansenPedersen1982, Eisler2025, Schneider2014, Koteljanskii1950, Lovasz1983, CombesThomas1973, Bhatia1997, Rotfeld1969, Mirsky1960) are all in ms_fermion_new.bib; Appendix C cites nothing. No label in the set is defined twice; no reference anywhere in it is unresolved.
- Mutual citation: Appendix C cites no appendix; Appendix D (app:rows) cites app:local and not app:charge; Appendix B cites Appendix C once (the nu_3 bound). No cycle.
- The live letter (review_output/paper_lane_20261013/letter/response_letter_declared.md) points at app:local three times and app:charge five times, both kept, and carries no eq:B- or eq:C- label and no "mutual information"; it needs no change for this delivery.

### RECORD

- Counts (comment-stripped, the ceiling's command; neither file has a comment): A2_local_area.tex 12,485 (ceiling 12,500; v1 12,421; live 14,524); charge_conservation.tex 5,252 (ceiling 5,300; v1 5,202; live 5,699). By part, heading included: B's section line 49, lead 190, Conventions 1,348, B.1 1,853, B.2 1,834, B.3 1,411, B.4 1,374, B.5 2,352, B.6 2,074; C's section line 65, lead 295, traces 495, the Hartree flow 1,570, the first law 699, the rate 1,313, back-averaging 819. Form: B has four equation environments (one with aligned), 226 \( and 226 \), braces balanced (128/128), no $; C has four equation environments (two with aligned), 80 \( and 80 \), braces balanced (51/51), no $. Sentences (each math span counted as one word): B 35, longest 73; C 12, longest 54 (v1's first-law sentence had grown to 88 with the restored conditions before the split).
- Checks (FABLE_write_R6_appendices_BC_v1.1_scripts/verify.py, run against the live fermion_new/ and ms_fermion_new.bib): labels each once across the set; every \ref and \eqref in both files resolves; every B or C label cited outside the appendices is defined on the same result; every \cite key is in the bib; braces, \( \) and environments balanced; no Theorem, Lemma, Proposition, Corollary, em-dash, en-dash or "mutual information"; no lone digit in prose; Appendix C cites not Appendix D and Appendix D cites not Appendix C.
- Combined builds (read-only copies under <internal>/r6_app11; the manuscript untouched): fermion_new/ identical to the live worktree's (02 7bed5a8e, 04 254d39ec, 05 19cdf081, 06 8de7f411, 07 e4a9bdf8, 08 db1dc5f3, A1 76922514, D a1dcf179), the figures live, ms_fermion_new.bib live (e413c578), ms_fermion_new.tex the 06:04 snapshot (0ee85b12; the live file a2c91046 differs in the abstract only), A2_local_area.tex and charge_conservation.tex this delivery; 03_model.tex = the Sec. II v5.2 block (the NEW SECTION II fence of FABLE_write_secII_order_v5.2.md, lines 18 to 662, md5 9943552ae55307eb8b43ecf55552e3fe, 35,202 characters, identical to FABLE_write_secII_v5.2_checks/03_model_v52.tex) in on and off, the live 03 (ffb13ad4, 35,444) in on_live03 and off_live03. Compiled one after the other with the paper's command (rm -f ms_fermion_new.{aux,bbl,blg,out,log,toc,fdb_latexmk,fls} ms_fermion_newNotes.bib && latexmk -pdf -g ms_fermion_new), each under a 600 s timeout; none hung; each ran three pdflatex passes and two bibtex passes.
- Marks on, Sec. II v5.2 (31 \todo in the sources): 30 pages; page 30 holds 45 nonblank lines (pdftotext -layout), the bibliography's tail to entry [82], its left column 44 lines ending 213 pt (about 20 lines) above the foot and its right column 41 lines ending 255 pt above; 0 errors, 0 undefined references or citations, 2 overfull boxes, both in 03_model.tex at the v5.2 block's lines 116 and 348 (164 pt and 71 pt), none in Appendices B and C. Appendix A begins on page 24, B on 25, C on 26, D on 27, the bibliography on 28. PDF md5 c9e357682bebb8d4ea28bd6ac0b28623.
- Marks off, Sec. II v5.2 (\newcommand{\todo}[1]{}): 29 pages; page 29 holds 60 nonblank lines, its left column 59 lines ending 53 pt (five lines at the 10.5 pt pitch) above the foot and its right column 60 lines ending 56 pt (five lines) above, the bibliography's tail to entry [82]; 0 errors, 0 undefined, the same 2 overfull boxes. Appendix A begins on page 23, B on 24, C and D on 26, the bibliography on 28. PDF md5 df342264cdcc820cc6986339be52af19.
- Without the Sec. II v5.2 substitution (the live 03): marks on 30 pages, page 30 with the same 45 lines (44 and 41 per column, 212 and 254 pt above the foot), 0 errors, 0 undefined, 0 overfull, PDF md5 2fbd445c4eb3e12de0c599238397954d; marks off 29 pages, page 29 with the same 60 lines (59 and 60 per column, 54 and 57 pt above the foot), 0 overfull, PDF md5 c9da89a3a34d8aa4f602146b667b03e4. The appendix pages are as above (B on 25 with the marks on, 24 off).
- Confirmation with the live main file (a2c91046, #467's abstract) in place of the snapshot, marks off, Sec. II v5.2: 29 pages with the identical page-29 fill (59 and 60 lines, 53 and 56 pt); PDF md5 8788f70b5f58c31cefb63ef563f6713d. The abstract drift changes nothing.
- Against v1's builds (same block, same 03): v1 was 30 on (page 30 holding 47 lines) and 29 off with page 29's left column at its foot and its right column about five lines short; v1.1 is 30 on (45 lines) and 29 off with both columns five lines short, the three dropped bibliography entries (85 to 82 numbered) having paid for the restorations with about five lines to spare.
- Supplement: FABLE_write_R6_supplement_appendices_v1.1.tex (md5 21a1619d22d5e5954050de3bd4044192), the v1 tables with the three restored items removed from the record and the five v1.1 removals added (B.1's symmetry reasons, B.3's idea sentence with the one-line reason the oddness of P_1 gives, the three doubled citations, the Schur test's object, the ma = 0.2 clause), the table columns narrowed to the text width; pdflatex twice, 0 errors, 0 overfull, six pages (v1 compiled with six overfull boxes).

### FOR THE LEAD

1. MARGIN: 15 and 48 characters under the ceilings; the marks-off build is at 29 pages with about ten bibliography lines of room on page 29 (five in each column), twice v1's. A landing result that grows the body by more than that opens page 30; the paper lane's content-free lever (Fig. 1 at column width, inbox #465) is still the larger one.
2. What v1.1 paid with, in case the lead wants any back (each costs the characters named and only another cut could pay): B.3's idea sentence (135; the statement keeps "odd under Theta", and the supplement's row gives the one-line reason that S_A is Theta-invariant while P_1 is odd), B.1's three symmetry reasons (95), the Schur test's object "on the commutators" (19), the "[110] value as a third" clause (28; B.2 now accounts for the body's two-anchor enclosure 0.7883 < eta(0.2) < 0.9831 and is silent on the three-anchor one, which the body states "with the measured [110] interval as a further premise"; the supplement says how it is built), and the three doubled citations (back at nine bibliography lines, about 0.07 page). None was a printed result.
3. The two overfull boxes belong to the Sec. II v5.2 block (its lines 116 and 348); the builds with the live 03 have none.
4. Two edits here were not reviewer B's: the first law's sentence is split at "is simple." (four characters shorter, nothing moved) because the restored conditions had made it 88 words, and B.1's path-expansion sentence is split at "They vanish" (two characters) for item 7. reviewer B's replacement for item 2 said "orthogonal projection"; the file keeps v1's "a projection P commuting with h" (the paper's usage); "orthogonal " costs 11 characters within the 15 of room if the lead wants it.
5. The prepared tree's main file is the 06:04 snapshot; the live file took #467's D0 clause at 07:02 (abstract only, net zero words); the confirmation build shows the same 29 pages and fill, so the RECORD's numbers stand for the live manuscript.
6. The retired label eq:C-trace and the letter: as in v1, nothing to change.
