# Direct converter: balances and conservation obstruction

Supports Section V A, “The radiating species and the light field”
(`fermion_new/07_page_curve.tex`), and the interacting-extension
statements in Section II, “The model” (`fermion_new/03_model.tex`).

The argument below preserves the complete passage moved from Section V A.
Its paper reference `Appendix~\ref{app:local}` denotes Appendix B,
“Local area estimates” (`fermion_new/A2_local_area.tex`, whose heading is
`\section{Local area estimates}\label{app:local}`).
The existing pointer to `proofs/interaction/conversion_balance.md`
is retained.

The direct on-site converter, absent from the histories, is
\begin{equation}
\begin{gathered}
H=H_0+UV,\qquad V=\sum_xW_x,\qquad U=\lambda_X/a^3\in\mathbb R,\\
W_x=\sum_{\mu=0}^3\eta_{\mu\mu}
 \Phi_{L,x}^{\mu\dagger}\Phi_{H,x}^{\mu}+\mathrm{h.c.},\\
\Phi_{A,x}^\mu=c_{A,x}^{T}M^\mu c_{A,x},\qquad
\eta=\operatorname{diag}(1,-1,-1,-1),\\
M^0=\tau_z\otimes\sigma_y,\qquad M^1=\tau_y\otimes\sigma_z,\\
M^2=i\tau_y\otimes I_2,\qquad M^3=-\tau_y\otimes\sigma_x .
\end{gathered}
\end{equation}
The basis is \(\alpha_i=\tau_x\otimes\sigma_i\),
\(\beta_{\rm D}=\tau_z\otimes I_2\) (Appendix~\ref{app:local});
\(\Phi\) has no half, \(\lambda_X\) is an area, and the site term
includes every crystal-momentum-conserving valley quartet.
The clock weights
\(\mathcal H_x=\mathcal H_{0,x}+UW_x\), with the free allocations
summed over species. Equal splitting of each free hopping bond
\(T_{xy}\) gives the flow-coupled energy-dipole current
\begin{equation}
\begin{aligned}
\mathcal Q_i&=\mathcal Q_{0,i}+U\mathcal Q_{1,i},\\
\mathcal Q_{1,i}&=\frac i2\sum_{\langle x,y\rangle}
 (y_i-x_i)[T_{xy},W_y-W_x].
\end{aligned}
\end{equation}
Disjoint on-site terms commute, eliminating the quadratic term.
With \(j^E_{xy}=i[\mathcal H_x,\mathcal H_y]\), allocate
\(\mathcal P_{x,i}=\mathcal P^0_{x,i}
+\tfrac12\sum_y(y_i-x_i)(j^E_{xy}-j^{E,0}_{xy})\).
It sums to \(\mathcal Q_i\) and, under unweighted \(H\), obeys
\begin{equation}
\begin{aligned}
\dot{\mathcal P}_{x,i}+\sum_y(J^0_{xy,i}+J^I_{xy,i})&=s^I_{x,i},\\
\sum_xs^I_{x,i}&=i[H,\mathcal Q_i],\\
J^0_{xy,i}&=\frac i2\sum_Ac_A^\dagger
 \Bigl(\{\Pi_xh_{0,A}\Pi_y,Q_i\}\\
&\qquad-\{\Pi_yh_{0,A}\Pi_x,Q_i\}\Bigr)c_A,\\
J^I_{xy,i}&=i([\mathcal H_x,\mathcal P_{y,i}]
 -[\mathcal H_y,\mathcal P_{x,i}])\\
&\quad-i([\mathcal H_{0,x},\mathcal P^0_{y,i}]
 -[\mathcal H_{0,y},\mathcal P^0_{x,i}]),\\
s^I_{x,i}&=i[\mathcal H_x,\mathcal Q_i]
 -i[\mathcal H_{0,x},\mathcal Q_{0,i}].
\end{aligned}
\end{equation}
The finite-range antisymmetric flux uses local displacement lifts
on a periodic parent; its allocation is not unique.
At back-averaged fields,
\(\mathcal G=\sum_xN_x\mathcal H_x+\sum_{x,j}v_{x,j}\mathcal P_{x,j}\)
adds \(i[\mathcal G-H,\mathcal P_{x,i}]\) to the right side.
Reference subtraction additionally contributes minus the time
derivative of the reference momentum.
For the normalized positive-band spinors
\(u_m(d)=\sqrt{(E+m)/(2E)}\,(I_2;\,\boldsymbol\sigma\cdot d/(E+m))\),
\(E=\sqrt{m^2+|d|^2}\), the element of \(UV\) between a heavy pair and
a light pair on a parent of \(V_s\) sites is \(U\mathcal A/V_s\) times
the crystal-momentum delta,
\begin{equation}
\begin{aligned}
\mathcal A_{t_1t_2;s_1s_2}
={}&4\sum_\mu\eta_{\mu\mu}
 \bigl[(u_{L1}^TM^\mu u_{L2})_{t_1t_2}\bigr]^*\\
&\qquad\times(u_{H1}^TM^\mu u_{H2})_{s_1s_2},
\end{aligned}
\end{equation}
in each pair's basis \((\uparrow\uparrow,\uparrow\downarrow,
\downarrow\uparrow,\downarrow\downarrow)\).
At \(ma=3/5\), opposite axial heavy and light momenta with
\(\sin ah=9/20\), \(\sin ak=3/4\) give energy \(3/(4a)\) per leg
and \(\mathcal A=-(24/5)\operatorname{diag}(1,0,0,1)\).
A common on-shell boost gives
\(\partial_{P_z}(q_{LL,z}-q_{HH,z})=-18/25\).
First-order conservation would require
\((E_f-E_i)(\mathcal Q_1)_{fi}=(q_f-q_i)V_{fi}\);
the resonant family violates this at small nonzero \(P_z\).

More generally, a translation-invariant finite-range one-body
leading symbol scalar on each free band is \(f_A+g_Ah_{0,A}\),
with trigonometric polynomials \(f_A,g_A\). Local terms have range
and coefficients independent of parent volume. Require a
finite-range correction differentiable at zero coupling, with the
corrected charge conserved through first order on all sufficiently
large periodic parents. Differentiating the resonant constraint
at zero pair momentum on the connected same-valley shell makes
the gradient of the odd part of each band eigenvalue constant
on equal-energy surfaces. Their nonparallel normals make it a
constant vector; analyticity and periodicity force it to vanish.
The polynomial identity then removes the odd parts of \(f_A\)
and \(g_A\). At this test parameter no leading symbol in this
class with a nonzero odd part of a band eigenvalue under
\(p\mapsto-p\) has such a conserved correction. One valley
suffices; intervalley scattering is unnecessary. The operator, routing, resonant family, proof and
quadratic-force witness are collected in
\nolinkurl{proofs/interaction/conversion_balance.md}. Neither
balance defines a mechanical stress, which needs a deformation
rule, nor a metric-derived gravitational stress; first-order
interaction expectations of energy, current and routed flux vanish
on number-conserving species-product states, the quadratic stress
and force need not, and their referenced contribution to the
sources, the interacting sea and its response included, must be
carried or bounded and is not computed here.
