# Axial-pair conversion amplitudes

**PASS for the named amplitude cut. [R,N]** The exact controlled readouts give a zero same-valley threshold tensor, collinear entries \(-32/5\), spin averages \(512/25\) and \(256/25\), the collinear slope \(-16\), and the selected intervalley entry \(4\). The 31 raw objects and 40 decisions specified for this cut are complete. The filtered threshold and first nonzero terms retain their completed numerical evidence at the kinematics stated below.

Scope: this result answers inbox hreduce #100 and its continuation instruction #101 for the axial-pair converter defined here. It concerns normalized positive-band two-particle matrix elements at first order in the interaction coefficient. The general momentum formulas are re-derived analytically; the numerical checks cover the stated finite configurations and the specified derivative. The broader D plan remains incomplete. The result supplies no scattering rate, interacting phase, current identity or Hartree upgrade. The manuscript is unchanged. Tags: [R] re-derived; [N] numerical with controls. Inherited [N] entries are identified explicitly.

SOURCES: `<internal>/hreduce.md` (#100 and #101); `<internal>/ReviewA_vgate_v1.md`; `<internal>/ReviewB_vgate_v1.md`; `<internal>/frozen_spec.tex`; `<internal>/VERTEX_gate_v1.md`; `<internal>/VERTEX_gate_filtered_v1.md`; the exact source and box-only record paths listed below.

## Operator and normalization

Let \(H,L\) label the heavy and massless four-component fields. The unchanged on-site interaction is

\[
H_X=U\sum_x\left[\sum_{\mu=0}^3\eta_{\mu\mu}
\Phi_{L,x}^{\mu\dagger}\Phi_{H,x}^{\mu}+\mathrm{h.c.}\right],
\qquad \Phi_s^\mu=c_s^T M^\mu c_s,
\qquad U=\lambda_X/a^3\in\mathbb R,
\]

with no factor \(1/2\) in \(\Phi\). The component convention is

\[
\alpha_i=\tau_x\otimes\sigma_i,\qquad \beta=\tau_z\otimes I_2,
\qquad \eta=\operatorname{diag}(1,-1,-1,-1),
\]

\[
M^0=\tau_z\otimes\sigma_y,\quad
M^1=\tau_y\otimes\sigma_z,\quad
M^2=i\tau_y\otimes I_2,\quad
M^3=-\tau_y\otimes\sigma_x.
\]

Here \(\tau_i\) and \(\sigma_i\) are Pauli matrices on the two component factors. For sine momentum \(\mathbf d=a^{-1}\sin(a\mathbf p)\),

\[
u_m(\mathbf d)=\sqrt{\frac{E+m}{2E}}
\begin{pmatrix}I_2\\\boldsymbol\sigma\cdot\mathbf d/(E+m)\end{pmatrix},
\qquad E=\sqrt{m^2+|\mathbf d|^2},\qquad u_m^\dagger u_m=I_2.
\]

Plane waves are normalized on \(V_s\) lattice sites. Strip the common factor \(U/V_s\) and the crystal-momentum Kronecker delta from the matrix element. In the ordered incoming and outgoing spin-pair basis
\((\uparrow\uparrow,\uparrow\downarrow,\downarrow\uparrow,\downarrow\downarrow)\), the dimensionless tensor is

\[
\mathcal A_{t_1t_2;s_1s_2}
=4\sum_\mu\eta_{\mu\mu}
\left(u_{L1}^TM^\mu u_{L2}\right)^*_{t_1t_2}
\left(u_{H1}^TM^\mu u_{H2}\right)_{s_1s_2}.
\tag{1}
\]

The factor four follows from the canonical anticommutation relations:
\(\Phi\,a^\dagger(u)a^\dagger(v)|0\rangle=-2u^TMv|0\rangle\).
The reported spin average is
\(\overline{|\mathcal A|^2}=\tfrac14\operatorname{Tr}(\mathcal A^\dagger\mathcal A)\), averaging over the incoming spins and summing the outgoing spins. No flux, phase-space or occupation factor is included.

The separate ordered-wedge contraction uses component-pair order
\((01,02,03,12,13,23)\),

\[
K_{ab;cd}=4\sum_\mu\eta_{\mu\mu}(M^\mu_{ab})^*M^\mu_{cd},\qquad
W(u,v)_{ab;s_1s_2}=u_{a s_1}v_{b s_2}-u_{b s_1}v_{a s_2},
\]

so \(\mathcal A=W_L^\dagger K W_H\). The current and wedge readouts use the same retained primary matrices and spinors. Their agreement checks the two contractions and their normalization; it is not a second state evolution.

## Unfiltered targets

Define the pair currents by \(J_s^\mu=u_{s1}^TM^\mu u_{s2}\). For opposite heavy sine momenta, direct substitution gives

\[
J_H^0=\frac mE\sigma_y,\qquad
J_H^i=-\frac{i}{E}\sigma_y(\mathbf d\times\boldsymbol\sigma)_i,
\qquad J_L^0=0
\]

for opposite massless sine directions. At the same-valley heavy rest threshold, the heavy spatial currents vanish. Every term in (1) therefore vanishes. This is the stated centre-of-momentum, same-valley threshold cancellation. [R]

For \(\mathbf d=d\hat z\) and outgoing light directions \(\pm\hat z\),

\[
\mathcal A_\parallel=-\frac{8d}{E}D,
\qquad D=\operatorname{diag}(1,0,0,1),
\qquad \overline{|\mathcal A_\parallel|^2}=\frac{32d^2}{E^2}.
\tag{2}
\]

Writing \(r=d/(E+m)\) gives \(d/E=2r/(1+r^2)\), hence

\[
\mathcal A_\parallel=-\frac{16r}{1+r^2}D,
\qquad \left.\frac{\partial\mathcal A_\parallel}{\partial r}\right|_{r=0}=-16D.
\tag{3}
\]

For a general unit outgoing sine direction \(\mathbf n\), the analytic spin sum is
\(16[|\mathbf d|^2+(\mathbf n\cdot\mathbf d)^2]/E^2\). The finite checks below use \(\mathbf n=\hat z\) and \(\hat x\). [R]

| Named target | Exact controlled result | Verdict and scope |
|---|---|---|
| Same-valley rest threshold | \(\mathcal A=0\), \(\overline{\lvert\mathcal A\rvert^2}=0\) | **PASS [R,N]**; retained tensor and residuals, with their missing Boolean decisions completed |
| Collinear tensor, \((ma,da,Ea)=(3/5,4/5,1)\) | \(\mathcal A_\parallel=-(32/5)D\) | **PASS [R,N]**; current and ordered-wedge contractions agree |
| Collinear spin average at that point | \(512/25\) | **PASS [R,N]** |
| Perpendicular light directions at that point | \(\overline{\lvert\mathcal A_\perp\rvert^2}=256/25\) | **PASS [R,N]**; the full signed tensor and light-spinor controls also pass |
| First nonzero collinear coefficient | \(\partial_r\mathcal A_\parallel(r=0)=-16D\) | **PASS [R,N]**; retained current jets and normalized-tensor readouts agree |
| Selected \(c=001,\mathbf n=\hat z\) intervalley entry | \(\mathcal A_{\uparrow\downarrow;\uparrow\downarrow}=4\) | **PASS [R,N]**; distinct crystal modes and exact energy/momentum conservation |

At the collinear finite point, the incoming signed \((\cos ap_z,\sin ap_z)\) pairs are \((3/5,\pm4/5)\); the outgoing pairs are \((0,\pm1)\). Both pair phases are one, and both particles have energy \(1/a\). The normalized returned tensor is \((E/d)\mathcal A_\parallel=-8D\). In the derivative check, \(r\) is the dimensionless variable in (3).

For perpendicular outgoing light momenta, the new normalized columns are
\(2^{-1/2}(I_2;\pm\sigma_x)\), with Hamiltonians \(\pm\alpha_x/a\). Their eigen and normalization residuals are exactly zero. Their pair currents are
\((J_L^0,J_L^1,J_L^2,J_L^3)=(0,0,-\sigma_x,-iI_2)\). The full returned tensor is

\[
\mathcal A_\perp=\frac{16}{5}
\begin{pmatrix}
0&0&0&0\\
-1&0&0&1\\
-1&0&0&1\\
0&0&0&0
\end{pmatrix}.
\tag{4}
\]

The heavy incoming momenta remain on the z axis; the outgoing light momenta lie on the x axis. The saved kinematics retains those axis placements and reuses the already checked energy and pair-phase values through explicit record references.

The intervalley label \(c\) is the bitwise difference of the two incoming valley-corner labels. For \(c=001\), take incoming momenta
\(0,\pi\hat z/a\) and outgoing momenta
\(q\hat z,(\pi/a-q)\hat z\), with
\((\cos aq,\sin aq)=(4/5,3/5)\) and \(ma=3/5\).
The four particle energies are \(3/(5a)\). Both outgoing sine directions are positive z, although their crystal momenta belong to different valleys. The modes are distinct; equality of their normalized spinor matrices does not merge the occupied modes. Both pair phases are minus one. The retained rest-heavy spatial currents vanish, and the new temporal light current is \(\sigma_y\), giving

\[
\mathcal A_{001,z}=4
\begin{pmatrix}
0&0&0&0\\
0&1&-1&0\\
0&-1&1&0\\
0&0&0&0
\end{pmatrix}.
\tag{5}
\]

Its zero-based entry \((1,1)\) is four. This nonzero example excludes a universal unfiltered lattice threshold-cancellation claim. Its domain is distinct from the same-valley cancellation and from the filtered pair-to-other-valley-pair channel below.

Raw phase products and conservation defects retain exact unexpanded expression trees. The separate comparison residuals simplify to exact zero. The raw records preserve those distinct representations.

## Filtered targets: inherited numerical completion

The accepted filtered operator averages each axial pair with the actual normalized spatial kernel and sums a common valley-filter index. Its stripped amplitude is

\[
\mathcal A_F=\kappa(\mathbf P)S\mathcal A,\qquad
S=\sum_{b\in\{0,1\}^3}\prod_{\ell=1}^4 f_b(\mathbf p_\ell),\qquad
f_b(\mathbf p)=\prod_{i=1}^3\frac{1+(-1)^{b_i}\cos(ap_i)}2.
\tag{6}
\]

Here \(\mathbf P\) is the conserved pair momentum. Retain the actual lattice overlap symbol of the kernel-averaged operator:

\[
B_a(z)=a^{-3}\mathbf1_{[-a/2,a/2]^3}(z),\qquad
K_\ell(z)=\frac{9009}{1024\pi\ell^3}(1-|z|^2/\ell^2)_+^5,
\qquad k_{\ell,a}=K_\ell*B_a,
\]

\[
(W_\ell F)(y)=a^3\sum_xk_{\ell,a}(y-x)F_x,\qquad
g_r=a^3\int d^3y\,k_{\ell,a}(y)k_{\ell,a}(y-ar),\qquad
\kappa(\mathbf P)=\sum_{r\in\mathbb Z^3}g_r e^{ia\mathbf P\cdot r}.
\]

The cell partition and normalized compact kernel give \(\sum_r g_r=1\). Both retained filtered configurations have \(\mathbf P=0\), hence \(\kappa(0)=1\). This retains the lattice Gram coefficients of the map \(W_\ell\) from the lattice to the continuum, as defined in `VERTEX_gate_filtered_v1.md`. No kernel width or quadrature is selected for these zero-pair-momentum checks.

The same channel has the heavy pair and light pair near the zero valley. In the cross channel the heavy pair is near zero and the light pair is near the z-corner valley. These paired configurations have total crystal momentum zero modulo the reciprocal lattice. The completed filtered records give:

| Named filtered target | Same channel | Cross channel | Verdict |
|---|---|---|---|
| Threshold at \(ma=3/5\) | \(S=81/100,\ \mathcal A_F=0\) | \(S=1/100,\ \mathcal A_F=0\) | **PASS [R,N inherited]** |
| First derivative with respect to \(r=d/(E+m)\) at that threshold | \(-324D/25\) | \(-4D/25\) | **PASS [R,N inherited]** |
| Finite point \((ma,da,Ea)=(3/10,2/5,1/2)\) | \(\mathcal A_{F,+}=-(32/5)S_+D\) | \(\mathcal A_{F,-}=-(32/5)S_-D\) | **PASS [R,N inherited]** |

At the finite filtered point,

\[
S_\pm=\frac{161\pm60\sqrt7}{400},\qquad
\overline{|\mathcal A_{F,\pm}|^2}=\frac{512}{25}S_\pm^2.
\tag{7}
\]

The threshold weights multiply the unfiltered slope \(-16D\); the zero threshold tensor removes a derivative-of-filter contribution at that order. The cross-channel derivative is nonzero, so finite-momentum leakage remains even though the exact corner filters separate valleys. These inherited targets were read with their completed decisions and exact source records. They were not recalculated for this result.

The finite filtered point differs from the new unfiltered point. Multiplying \((m,d,E)\) by a common positive factor leaves the normalized heavy spinors and their pair currents unchanged; massless spinors depend only on direction. This permits reuse of the retained unfiltered currents and wedges. Finite lattice filter factors depend on \(a\mathbf p\) and retain the labels in (7).

## Controls, completion and provenance

The useful prefix contains eight new raw objects and 24 decisions: eleven decisions on retained threshold comparison residuals, eight on retained leg eigen/normalization residuals, and five new collinear comparisons. The finish contains 23 new raw objects and sixteen decisions. Every new comparison retains its actual value, expected value, residual and Boolean separately. All 40 decisions are true, with no missing raw, comparison or decision record.

The numerical method uses exact rational/complex sparse data and restricted square-root expression trees. The known answers were independently derived before the producer read the retained numerical operands. Cached matrices, seven original leg records, pair currents, wedge factors and current jets retain path, hash, payload selector and original record identifiers. The eight newly evaluated leg-residual Booleans had no completed predecessor decisions in the accepted filtered admission. All accepted primary products remain reused.

The requested prefix measurement has 47 complete unit receipts. Its metadata-only budget binds the entire terminal prefix and reviewed source pair; the finish has 87 complete unit receipts and uses only the absent plan. The measured cost was posted and received under inbox #101. The complete measurements, limits and custody observations remain in the box records. Both children used one CPU and one thread, nice value ten, with GPU visibility empty. Both are terminal and reaped, with return code zero, empty logs, unchanged input/source identities and no operational or scientific error.

The accepted A/B/C decision lists, of sizes 74, 80 and 18, remain inherited. The completed filtered result retains its own 45-decision scope. The original broad-D record still has its original 430 missing decisions. This delivery closes the expressly named amplitude cut without claiming completion of that larger plan. The earlier symmetry, current and Hartree dispositions keep their original scope.

The complete author derivation, source/output assessments and sole independent comparison are stored at:

- `<internal>/VERTEX_amplitudes_author_v1.tex`
- `<internal>/redteam_VERTEX_amplitudes_v1.tex`

The exact executed source files are:

- `<internal>/vertex_amplitudes_v1.py`; SHA256 `2b1ca6623fef16d27d42cd66fe7a835d14b651038bf28769c99896d4e0f7143a`.
- `<internal>/run_vertex_amplitudes_v1.py`; SHA256 `d414b993be77ed39a1f53f1af8ff7f5fd4204acc2754bd17119d33562b7e577d`.
- `<internal>/frozen_spec.tex`; SHA256 `55468d2ad78b5a9df0a391cf240242716e504bec45aec23b8ffe33abadd517cb`.

The proposal input hashes are `49a7e54761d97173731d7d6c147bc8e6d56ee53d94110e71a6d82f938f055e50` for `ReviewA_vgate_v1.md` and `4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b` for `ReviewB_vgate_v1.md`, at the full paths in SOURCES. The numerical records remain on the box:

| Full box path | SHA256 |
|---|---|
| `<internal>/admission.jsonl` | `cbb90c9dc5931f20d965581c51141622760693869e86e58c2b7e48117c5d5b7b` |
| `<internal>/amplitudes.jsonl` | `e4cb67fa2a6977f000bbb102eaee36a3d7b87d87a53a2a5a703a70bd5d19b8fc` |
| `<internal>/metrics.jsonl` | `7bc2d8654b64c7ba0fe71329ad8ceb60bd9b3119b9d2c1ca57437faaaaa26879` |
| `<internal>/vertex_amplitudes_result.json` | `730f188f164549acce968d5114ab24ab393206bec386cb6bc0099f2a2d37d232` |
| `<internal>/execution.json` | `4a3c801b915818a45ecd87e33bfac0eeefa67bfd8c387551330946cce71c1b27` |
| `<internal>/measured_budget.json` | `41ee6117499d885be74429faa87e5c6c981e2d793bd8fb72f54e6b4c3c94e745` |
| `<internal>/admission.jsonl` | `a0bad63aa794c13eead4473d95aea0c800b328ea2c471f289268ee091c85b297` |
| `<internal>/amplitudes.jsonl` | `3f35167b88ebe6df9db36d030c23b7b1715c838696f841dc233a75d230deac74` |
| `<internal>/metrics.jsonl` | `879df402c7c5a177358786975283b254c2e24fa133021c2df2d43b9908012c7a` |
| `<internal>/vertex_amplitudes_result.json` | `f4297f536a5abbfcb5491daae05296eb55961531b267dbe56e56e6552e12d865` |
| `<internal>/execution.json` | `9fbbbaefa57dd4b7eedb41133e01e204e001523a074536897b557b32ab3c04c6` |

The empty log paths are `<internal>/stdout.log`, `<internal>/stderr.log`, `<internal>/stdout.log` and `<internal>/stderr.log`; each has SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The immutable numerical predecessors are `<internal>/spinor_algebra.json`, `<internal>/on_site_completion.json`, `<internal>/bond_current.json`, `<internal>/amplitudes.json`, `<internal>/vertex_gate_missing_result.json`, `<internal>/execution.json`, `<internal>/amplitudes.jsonl`, `<internal>/vertex_gate_filtered_result.json` and `<internal>/execution.json`. Their complete content/descriptor pins and selected cache pointers are in both admission journals and the frozen specification. All new record paths and hashes are indexed in `<internal>/LARGE_FILES.tsv`.
