# PAGE exterior excess and thermal reference: bounded second-route handoff v1

Scope: task 2 of `<internal>/GOAL_codex.md`, for the abstract's D0 diagnostic and Sec. V's exterior-cut entropy, collar ledger, stationary thermal reference, and conditional heavy-curve decomposition. This note reports this lane's saved outputs. It makes no comparison with first-route numbers and no judgment on a manuscript sentence. That comparison belongs to hreduce.

The independent analytical definitions and derivations carry **[R]**. Executed analytic/stored-state controls and the independent finite-domain stationary scattering calculation carry **[N]** within their stated scope. The light entropy, window, and energy-ledger results retain **[S]** dependence on supplied primitive eigenspectra and integrated columns. They are independent arithmetic on those exports, **not a regeneration of the light occupied columns, covariance, or restriction eigenvalues**. No covariance-level **[X]** is claimed.

The sole independent review is complete in `solve/redteam_PAGE_secondroute_v1.md`. Its requested implementation corrections are included here: checkpoint-wide invalid-spectrum flags, matched beta channels and explicit unmatched labels, separate endpoint/side/sector routes, explicit ledger weighting, missing-vacuum diagnostics, and the corrected doubled outer endpoint in the thermal refinement. This is the bounded numerical handoff; missing inputs and proofs are listed below.

## 1. Conventions, input scope, and recorded cuts

Entropies are in nats. Saved energies are in \(1/a\), saved times in \(a\), and the displayed clock and cut labels use the **nominal** \(r_s/a=64\). The instantaneous charge radius \(r_E(T)=2GE(T)\) is a separate quantity. The B2 history has \(g=G/a^2=0.25\), \(\ell/a=4\), and record interval \(\tau/a=8\). Its applied-field file has 4096 radial sites and 801 records; the light parent appends the vacuum exterior to 8192 sites, with two radial components. The recorded production settings are massless-sector reduction, linear field interpolation, eight midpoint substeps per record, and Chebyshev tolerance \(10^{-16}\). This lane did not repeat that evolution.

For light spectra this calculation preserves the producer's named exterior cut \(X\) and interior complement \(X^c\). It reads the actual `cut2`, `cut16`, `cut24`, `cut30`, and `cut48` arrays; it does not construct a new geometric mask from their labels. At channel one, the recorded plus-sector interior spectra have lengths 128 at cut 2 and 1920 at cut 30, while the exterior occupied-Gram spectra have length 4096. Gram dimension is not exterior geometric dimension; the extra zero modes contribute zero entropy. The exact coordinate mask and boundary-site assignment remain part of the supplied export convention, not an independently regenerated cut. No movement of the cut to \(2GE(T)\) is made.

This differs from the separate heavy identity control, which explicitly selects stored coordinates \(R<2r_s\), giving 127 sites and both radial components. That independently declared heavy mask is not used to reinterpret a production light cut.

For a physical Gaussian covariance, use
\[
s(x)=-x\log x-(1-x)\log(1-x),\qquad s(0)=s(1)=0,\qquad
S_X(C)=\operatorname{Tr}s(C_X).
\]
The channel weight is \(g_\kappa=16\kappa\), applied **once after** the two sector entropies are added. A displayed channel-one result therefore already has weight 16. The continuum scattering weights in section 7 are a separate port-count convention.

At checkpoints with both exterior sectors, the direct route is
\[
\begin{split}
x_\kappa^{\rm dir}
={}&\sum_j s(\lambda^{X,+}_{C,j})+\sum_j s(\lambda^{X,-}_{C,j})\\
&-\sum_j s(\lambda^{X^c,+}_{P,j})-\sum_j s(\lambda^{X^c,-}_{P,j}),\\
\Delta S_{\mathcal K}^{\rm dir}
={}&\sum_{\kappa\in\mathcal K}16\kappa\,x_\kappa^{\rm dir}.
\end{split}
\]
The held comparator is the supplied same-parent Killing spectral projector. Its complement entropy equals its exterior entropy for the stated pure projector, including its specified zero-mode filling. This is a reference-definition dependency; this lane did not independently reconstruct that projector.

The distinct all-clock route is
\[
x_\kappa^{\rm comp}
=2\left[\sum_j s(\lambda^{X^c,+}_{C,j})
       -\sum_j s(\lambda^{X^c,+}_{P,j})\right].
\]
It uses both physical-state purity and equality of sector entropies. Available side/sector diagnostics are reported below; they do not establish these properties at every intermediate checkpoint. Exported `sv_block` arrays are cross blocks of \(C-P\), not of \(C\), and are not used as the physical Gaussian cross block.

The entropy consumer records each spectrum's extrema, number of eigenvalues outside \([0,1]\), total clipping displacement, and endpoint sensitivity. Values within \(10^{-8}\) of an endpoint can be clipped with those diagnostics; values beyond that tolerance produce a null entropy for that spectrum. Another cut or sector does not repair it. Known failures on any saved cut are propagated to all rows at the same \((\kappa,T)\). The full scan reports minimum \(-2.3080277455603186\times10^{-12}\), maximum \(2697.2699104220155\), and 14,113,923 endpoint clips within tolerance. These counts describe the exported arrays, not a covariance physicality certificate.

## 2. Independent identities and executed known answers [R, N]

With \(C=\left(\begin{smallmatrix}C_X&K\\K^\dagger&C_Y\end{smallmatrix}\right)\), block multiplication gives the exact general identity
\[
\boxed{C_X-C_X^2=KK^\dagger+\Pi_X(C-C^2)\Pi_X.}
\]
The impurity term is retained. For a pure covariance it vanishes; for a physical mixed covariance it is positive and may contribute local entropy even when \(K=0\). If \(u_j\) are the eigenvalues of \(KK^\dagger+\Pi_X(C-C^2)\Pi_X\), the equivalent entropy is \(\sum_j s((1-\sqrt{1-4u_j})/2)\); both quadratic roots have the same entropy. Using \(KK^\dagger\) alone requires purity.

The two-mode fixture uses
\[
h=\begin{pmatrix}0&-1\\-1&0\end{pmatrix},\qquad
P=\tfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix},\qquad X=\{\text{second mode}\}.
\]
The state is formed from its occupied column and the comparator from its explicit projector. For \(C=fP\), the cut has \(C_X=f/2\), \(KK^\dagger=f^2/4\), and impurity \(f(1-f)/2\).

| Executed control | Saved value |
|---|---:|
| Pure fixture cut entropy | 0.6931471805599453 |
| Vacuum excess | 0 |
| Scalar entropy-identity error | 0 |
| Largest displayed two-mode block-identity residual | \(1.3877787807814457\times10^{-17}\) |
| \(f=0.37\) fixture entropy, both routes | 0.47889113905134384 |
| Empty fixture entropy | 0 |
| Product fixture \(C=\operatorname{diag}(0.37,0.61)\), selected second-mode cross block | 0 |
| Product fixture physical impurity | 0.2379 |
| Product fixture local entropy | 0.6687480868518094 |
| Product fixture displayed identity residual | 0 |

The actual stored-state control uses heavy channel one at \(T/r_s=12\), with
\[
C=VV^\dagger+WFW^\dagger=ZZ^\dagger,\qquad
Z=[V,\;WF^{1/2}].
\]
It independently forms the cut covariance and cross block, and the general impurity from the stored columns. The selected dust occupations are all one, so the factored physical impurity \(W_XF(1-F)W_X^\dagger\) is zero. The fractional-occupation fixture above supplies the nonzero physical-mixedness control.

| Heavy stored-state control, \(R<2r_s\) | Saved value |
|---|---:|
| Selected sites | 127 |
| Occupation range | [1, 1] |
| Sea Gram operator defect | \(8.19781572279854\times10^{-13}\) |
| Dust Gram operator defect | \(3.854594427215421\times10^{-14}\) |
| Sea/dust overlap operator norm | \(1.07656276040178\times10^{-12}\) |
| General impurity operator norm | \(3.7719220061998594\times10^{-13}\) |
| Factored physical impurity operator norm | 0 |
| Block-identity residual, operator norm | \(7.025246185526992\times10^{-16}\) |
| Block-identity residual, Frobenius norm | \(2.4842653898596207\times10^{-15}\) |

This is an algebraic control on an actual heavy snapshot. The saved output explicitly withholds a covariance physicality certificate and makes no light-state inference.

## 3. Light exterior excess and invalid exports [S, N arithmetic]

The nine production spectra outside the declared \(10^{-8}\) tolerance are:

| \(\kappa\) | \(T/r_s\) | Recorded cut/\(r_s\) | Spectrum | Maximum |
|---:|---:|---:|---|---:|
| 3 | 12 | 16 | exterior plus | 1.0002625318053762 |
| 4 | 50 | 48 | exterior plus | 1.0000000156806088 |
| 5 | 50 | 48 | exterior plus | 2697.2699104220155 |
| 7 | 12 | 16 | exterior plus | 1.3368424646747914 |
| 10 | 12 | 16 | exterior plus | 1.090953544230913 |
| 12 | 12 | 16 | exterior plus | 1.1472800375028105 |
| 13 | 12 | 16 | exterior plus | 1143.2176701969593 |
| 17 | 12 | 16 | exterior plus | 1.0000255620746135 |
| 18 | 12 | 16 | exterior plus | 1.026562165651616 |

These arrays cannot be restrictions of a physical Gaussian covariance at the declared tolerance. The absent light occupied columns prevent distinguishing a failed/corrupt eigenspectrum export from a nonphysical saved covariance. No conclusion about the exact evolved state follows from that distinction being unresolved.

In particular, the two \(T/r_s=50\) failures apply to the common checkpoint underlying cuts 2 and 30 in the 18-channel sum. Valid scalar arrays at those cuts remain arithmetic inputs, but **there is no 18-channel physical-state clearance at 50**. The checkpoint at 100 is assessed independently; absence of a known saved-spectrum failure there is still not a covariance-level certificate.

| \(T/r_s\) | Cut/\(r_s\) | Channels | Direct excess/nats | Doubled-plus complement excess/nats | No known checkpoint spectrum failure |
|---:|---:|---|---:|---:|---|
| 50 | 2 | \(\kappa=1\), weighted | 5.130862796 | 5.130862751 | yes |
| 50 | 2 | \(\kappa=1,\ldots,18\) | 6.178697071 | 6.178694132 | **no** |
| 50 | 30 | \(\kappa=1\), weighted | 0.01397890610 | 0.01397887910 | yes |
| 50 | 30 | \(\kappa=1,\ldots,18\) | 0.009649536572 | 0.009648057601 | **no** |
| 100 | 2 | \(\kappa=1\), weighted | 8.238650530 | 8.238650514 | yes |
| 100 | 2 | \(\kappa=1,\ldots,18\) | 9.504489933 | 9.504486449 | yes |
| 100 | 30 | \(\kappa=1\), weighted | 7.169958601 | 7.169958587 | yes |
| 100 | 30 | \(\kappa=1,\ldots,18\) | 8.258671975 | 8.258669323 | yes |
| 60 | 30 | \(\kappa=1\), weighted | 3.592982194 | 3.592982175 | yes |
| 60 | 30 | \(\kappa=1,\ldots,18\) | 6.756002172 | 6.756000359 | yes |
| 70 | 30 | \(\kappa=1\), weighted | 3.490763935 | 3.490763919 | yes |
| 70 | 30 | \(\kappa=1,\ldots,18\) | 4.960320003 | 4.960317793 | yes |

The weighted 18-channel diagnostics use the same inputs and are not independent state generations. Each sector gap below is the plus-sector entropy minus the minus-sector entropy on the interior:

| \(T/r_s\) | Cut/\(r_s\) | \(S_{C,X}-S_{C,X^c}\) | Interior state sector gap | Interior vacuum sector gap | Direct minus doubled-plus excess |
|---:|---:|---:|---:|---:|---:|
| 50 | 2 | \(2.693114810\times10^{-6}\) | \(-2.447108134\times10^{-7}\) | \(9.768275078\times10^{-10}\) | \(2.938802407\times10^{-6}\) |
| 50 | 30 | \(1.574987582\times10^{-6}\) | \(1.508229097\times10^{-7}\) | \(5.480606191\times10^{-8}\) | \(1.478970788\times10^{-6}\) |
| 100 | 2 | \(3.014512941\times10^{-6}\) | \(-4.679102119\times10^{-7}\) | \(1.450645826\times10^{-9}\) | \(3.483873805\times10^{-6}\) |
| 100 | 30 | \(1.595521514\times10^{-6}\) | \(-1.006609534\times10^{-6}\) | \(4.982469903\times10^{-8}\) | \(2.651955725\times10^{-6}\) |

The corresponding channel-one direct-minus-complement differences are \(4.478722104\times10^{-8}\), \(2.700767965\times10^{-8}\), \(1.603648059\times10^{-8}\), and \(1.337969024\times10^{-8}\) nats, in the table's row order. Complete per-spectrum diagnostics, source keys, and channel contributions are in `spectral_entropy.json`; complete aggregate gap rows are in `tables.json`.

At \(T/r_s=100\), the supplied charge gives the following normalization. These are arithmetic normalizations using the stated \(\eta=0.9192\), not a new determination of \(\eta\).

| Quantity | Saved value |
|---|---:|
| \(E a\) | 106.0533079 |
| \(r_E/a=2gEa\) | 53.02665393 |
| \(A/a^2=4\pi(r_E/a)^2\) | 35334.44796 |
| \(S_{\rm BH}=A/(4G)\) | 35334.44796 |
| \(S_\eta=0.9192\,A/a^2\) | 32479.42456 |
| Direct cut-2 \(\Delta S_{18}/S_{\rm BH}\) | 0.0002689865127 |
| Direct cut-2 \(\Delta S_{18}/S_\eta\) | 0.000292631106 |

## 4. Collar ledger and its second term [R, S]

For the exterior projector, define
\[
Q_X=\tfrac12\{h,\Pi_X\},\qquad
J_X=\tfrac{i}{2}[h^2,\Pi_X],\qquad D=C-P.
\]
Positive \(J_X\) brings energy into the exterior. The saved-window identity is reconstructed as
\[
\underbrace{E_D(t_1)-E_D(t_0)}_{\text{second term}}
=F_{\rm current}+W_C-\Delta E_P+\epsilon_{\rm tel}.
\]
The second term is formed from the two `inventory_D` endpoints. Current, jumps, and vacuum change are read separately; \(W_C-\Delta E_P\) is formed before rounding. The saved producer `second_term_D`, which may use the current/jump balance, is a cross-check and can differ from the endpoint definition by the telescoping residual. The shell between cuts 16 and 30 has the difference of those exterior inventories, with that sign.

Every row below applies \(16\kappa\) once. These **14-\(r_s\) saved ledger windows** are distinct from the six-\(r_s\) beta windows in section 5. The full saved ledger also contains individual unweighted channels, weighted \(\kappa=1,2,3\) aggregates, and shell rows.

| Channels | Cut/\(r_s\) | Window | Second term | Current | Jumps minus vacuum | Balance residual |
|---|---:|---|---:|---:|---:|---:|
| 1 | 16 | 36–50 | 0.1808966908 | 0.1808214871 | \(7.520337601\times10^{-5}\) | \(2.670512700\times10^{-10}\) |
| 1 | 30 | 36–50 | \(-4.712404916\times10^{-6}\) | \(-4.430214176\times10^{-6}\) | \(-2.823493964\times10^{-7}\) | \(1.586562608\times10^{-10}\) |
| 1 | 16 | 86–100 | 0.003277882279 | 0.003170289929 | 0.0001075917955 | \(5.545999215\times10^{-10}\) |
| 1 | 30 | 86–100 | 0.01480908607 | 0.01471705973 | \(9.202551646\times10^{-5}\) | \(8.245457650\times10^{-10}\) |
| 1, 2 | 16 | 26–40 | 0.0008000072412 | 0.0008028720913 | \(-2.861199293\times10^{-6}\) | \(-3.650809793\times10^{-9}\) |
| 1, 2 | 30 | 26–40 | \(-8.926726878\times10^{-7}\) | \(-9.551513358\times10^{-7}\) | \(6.530789800\times10^{-8}\) | \(-2.829249966\times10^{-9}\) |
| 1, 2 | 16 | 36–50 | 0.3123197631 | 0.3121524539 | 0.0001673087575 | \(4.762599204\times10^{-10}\) |
| 1, 2 | 30 | 36–50 | \(-2.091095666\times10^{-6}\) | \(-1.517313649\times10^{-6}\) | \(-5.741736315\times10^{-7}\) | \(3.916140189\times10^{-10}\) |
| 1, 2 | 16 | 46–60 | 0.06179236456 | 0.06152975999 | 0.0002626011600 | \(3.415287608\times10^{-9}\) |
| 1, 2 | 30 | 46–60 | 0.1845092533 | 0.1845090012 | \(2.499074782\times10^{-7}\) | \(2.182625303\times10^{-9}\) |
| 1, 2 | 16 | 56–70 | 0.04256753488 | 0.04228848526 | 0.0002790463996 | \(3.222289990\times10^{-9}\) |
| 1, 2 | 30 | 56–70 | 0.3354821409 | 0.3354687443 | \(1.339501225\times10^{-5}\) | \(1.582772802\times10^{-9}\) |
| 1, 2 | 16 | 66–80 | 0.01953306289 | 0.01928438144 | 0.0002486781834 | \(3.264865711\times10^{-9}\) |
| 1, 2 | 30 | 66–80 | 0.05605629973 | 0.05598170432 | \(7.459256079\times10^{-5}\) | \(2.841321844\times10^{-9}\) |
| 1, 2 | 16 | 76–90 | 0.003723180474 | 0.003540591904 | 0.0001825880689 | \(5.013056636\times10^{-10}\) |
| 1, 2 | 30 | 76–90 | 0.03621927854 | 0.03615069526 | \(6.858169287\times10^{-5}\) | \(1.585462428\times10^{-9}\) |
| 1, 2 | 16 | 86–100 | 0.003774437631 | 0.003660764443 | 0.0001136730461 | \(1.419593332\times10^{-10}\) |
| 1, 2 | 30 | 86–100 | 0.01576961682 | 0.01567594356 | \(9.367220067\times10^{-5}\) | \(1.053340526\times10^{-9}\) |
| 1–18 | 16 | 26–40 | 0.001966976342 | 0.001911800355 | \(5.483473589\times10^{-5}\) | \(3.412517631\times10^{-7}\) |
| 1–18 | 30 | 26–40 | \(-7.517592167\times10^{-7}\) | \(-1.272125519\times10^{-5}\) | \(1.168169777\times10^{-5}\) | \(2.877982021\times10^{-7}\) |
| 1–18 | 16 | 36–50 | 0.3436007943 | 0.3401405589 | 0.003459947836 | \(2.875486367\times10^{-7}\) |
| 1–18 | 30 | 36–50 | \(8.725546650\times10^{-7}\) | \(-3.382410796\times10^{-6}\) | \(4.002320971\times10^{-6}\) | \(2.526444897\times10^{-7}\) |
| 1–18 | 16 | 46–60 | 0.06706369265 | 0.06543713286 | 0.001626312471 | \(2.473237650\times10^{-7}\) |
| 1–18 | 30 | 46–60 | 0.1976227049 | 0.1976197350 | \(2.736902326\times10^{-6}\) | \(2.329907770\times10^{-7}\) |
| 1–18 | 16 | 56–70 | 0.04422159273 | 0.04341803603 | 0.0008033157867 | \(2.409140407\times10^{-7}\) |
| 1–18 | 30 | 56–70 | 0.3670085460 | 0.3671064365 | \(-9.811922923\times10^{-5}\) | \(2.287283203\times10^{-7}\) |
| 1–18 | 16 | 66–80 | 0.02009502069 | 0.01973087068 | 0.0003638518829 | \(2.981250518\times10^{-7}\) |
| 1–18 | 30 | 66–80 | 0.05909470355 | 0.05811327403 | 0.0009811928036 | \(2.367227232\times10^{-7}\) |
| 1–18 | 16 | 76–90 | 0.004284628274 | 0.004080172148 | 0.0002041892032 | \(2.669228607\times10^{-7}\) |
| 1–18 | 30 | 76–90 | 0.03694928354 | 0.03752802988 | \(-0.0005789634551\) | \(2.171099887\times10^{-7}\) |
| 1–18 | 16 | 86–100 | 0.004175110138 | 0.004114645897 | \(6.017621341\times10^{-5}\) | \(2.880275467\times10^{-7}\) |
| 1–18 | 30 | 86–100 | 0.01636743236 | 0.01638524383 | \(-1.804449913\times10^{-5}\) | \(2.330349247\times10^{-7}\) |

Across the saved rows, the maximum **unweighted single-channel** telescoping and balance residual is \(2.266920263\times10^{-10}\). The maximum across **weighted aggregates** is \(3.412517631\times10^{-7}\), attained in the displayed 18-channel ledger. The maximum unweighted difference from the saved balance-defined `second_term_D` is \(2.266920263\times10^{-10}\). These quantities have different weights and must not share a single error label.

The accumulated collar flow equals the independently summed supplied per-record integrated flow at displayed precision. Endpoint traces, currents, jump traces, and the held-vacuum trace changes have not been regenerated from the missing light covariance. The ledger arithmetic therefore retains [S], including at checkpoints whose entropy exports fail.

## 5. Matched entropy/energy chords [S]

The reported finite-window statistic is
\[
b_{\mathcal K,X}[A,B]
=\frac{T_H(A)+T_H(B)}2
\frac{\Delta S_{\mathcal K,X}(u_B)-\Delta S_{\mathcal K,X}(u_A)}
{\mathcal E_{{\rm collar},\mathcal K}(B)-\mathcal E_{{\rm collar},\mathcal K}(A)},
\qquad T_H(T)=\frac1{8\pi g E(T)}.
\]
The entropy and energy channels are summed with \(16\kappa\) before taking the ratio. The energy is the signed accumulated collar flow. This is a chord with an endpoint-mean temperature, not an instantaneous derivative or the distinct statistic \(\Delta S/\int dE/T_H\).

For cut 16, \(u_A=A,u_B=B\). For cut 2, the same collar-arrival endpoints are mapped backward from \(r=16r_s\) to \(2r_s\) using \(d(r/r_s)/d(T/r_s)=N-|v|\) on the saved applied fields, with linear interpolation in space and time. The entropy series is then linearly interpolated at those emission clocks. All entropy rows in this section use the same doubled-plus complement route; they do not switch to the direct route at selected endpoints.

The fine ray calculation uses DOP853, maximum step \(0.03125r_s\), relative tolerance \(10^{-11}\), absolute tolerance \(10^{-12}\). Its coarse control uses step \(0.0625r_s\), relative tolerance \(10^{-9}\), absolute tolerance \(10^{-10}\). The maximum saved emission-clock change is \(1.172530503\times10^{-6}r_s\). The displayed temperature-formula residual and unweighted accumulated-flow residual are zero.

### Broad matched windows

The table gives both entropy increments and their common matched energy denominator. All these endpoints/interpolation brackets have no known invalid input under the saved flags; that is not a certificate for every intervening state.

| Channels | Arrival window | \(\delta E\) | \(\delta S_{2}\) | \(b_2\) | \(\delta S_{16}\) | \(b_{16}\) |
|---|---|---:|---:|---:|---:|---:|
| 1 | 40–46 | 0.1721561863 | 2.447182339 | 0.02132063907 | 2.617789717 | 0.02280702538 |
| 1 | 46–100 | 0.09224559163 | 4.767774112 | 0.07757795230 | 4.908260572 | 0.07986385168 |
| 1 | 80–100 | 0.003865523817 | 0.4058195032 | 0.1575870579 | 0.5235911636 | 0.2033199252 |
| 1, 2 | 40–46 | 0.2984336835 | 3.905762115 | 0.01962973654 | 4.301567615 | 0.02161899177 |
| 1, 2 | 46–100 | 0.1054920784 | 4.110065793 | 0.05847862958 | 4.099699385 | 0.05833113479 |
| 1, 2 | 80–100 | 0.004592940561 | 0.4692935109 | 0.1533732817 | 0.5840191727 | 0.1908676234 |
| 1–18 | 40–46 | 0.3253087443 | 4.599131253 | 0.02120491756 | 4.683247978 | 0.02159274912 |
| 1–18 | 46–100 | 0.1109364327 | 3.814729457 | 0.05161284607 | 3.962569076 | 0.05361309892 |
| 1–18 | 80–100 | 0.005383684664 | 0.4984306606 | 0.1389700205 | 0.5945054884 | 0.1657571382 |

The saved `K18_L12_unmatched` rows instead divide an 18-channel entropy increment by the channels-one-and-two energy increment. They are explicitly unmatched diagnostics in the data and are not substituted for any matched row above.

### Consecutive six-\(r_s\) windows

Each pair is \((b_2,b_{16})\). An asterisk identifies a cut-2 row whose interpolation input carries a known checkpoint failure; its number is arithmetic only. All other displayed six-window rows have no known invalid endpoint/bracket input, while retaining the all-clock purity and sector dependency.

| Arrival window | Channels 1: \(b_2\) | Channels 1: \(b_{16}\) | Channels 1,2: \(b_2\) | Channels 1,2: \(b_{16}\) | Channels 1–18: \(b_2\) | Channels 1–18: \(b_{16}\) |
|---|---:|---:|---:|---:|---:|---:|
| 46–52 | 0.02569048149 | 0.04121083679 | -0.0003805019009 | 0.01004165271 | -0.008113135929 | 0.01312997943 |
| 52–58 | 0.05103839824 | 0.02923309239 | 0.02572676223 | 0.003547344227 | 0.01705627577 | -0.003722214083 |
| 58–64 | 0.1005847352 | 0.1426612377 | 0.09297848904 | 0.1306663249 | 0.09037551146 | 0.1212315073 |
| 64–70 | 0.02653066944 | -0.001323882598 | 0.01711970701 | -0.01197757977 | 0.01237430036* | -0.01574251431 |
| 70–76 | 0.2357931112 | 0.2523908625 | 0.2301046158 | 0.2463133580 | 0.2258194820* | 0.2423970800 |
| 76–82 | 0.1823933658 | 0.05738370769 | 0.1758862986 | 0.05736440164 | 0.1391499875 | 0.06676601814 |
| 82–88 | 0.1829436793 | 0.1045964480 | 0.1906643130 | 0.1181878557 | 0.2062438245 | 0.08305121017 |
| 88–94 | 0.1208657736 | 0.2219310560 | 0.1213005307 | 0.1849144090 | 0.1074040522 | 0.1960017193 |
| 94–100 | 0.1780091433 | 0.1938879228 | 0.1763380539 | 0.1986289367 | 0.1563566775 | 0.1617801431 |

The matched denominators for the same row order are:

| Arrival window | \(\delta E_1\) | \(\delta E_{1,2}\) | \(\delta E_{1\ldots18}\) |
|---|---:|---:|---:|
| 46–52 | 0.024728569813305512 | 0.03188104598666541 | 0.03226371115306392 |
| 52–58 | 0.016404558220528997 | 0.01878874383692164 | 0.0220359875675058 |
| 58–64 | 0.01600877159216907 | 0.0169281776616117 | 0.017186515586217865 |
| 64–70 | 0.018328100704820827 | 0.019655912154121324 | 0.020047671438078396 |
| 70–76 | 0.011356214483384974 | 0.01200415329367388 | 0.01215223416511435 |
| 76–82 | 0.0017219034052686766 | 0.0019940212805522606 | 0.0024959322036011145 |
| 82–88 | 0.001152884477050975 | 0.00123035759315826 | 0.0013027049863012508 |
| 88–94 | 0.0011551245261216536 | 0.0016121989028761163 | 0.0017878263461170718 |
| 94–100 | 0.001389464407111518 | 0.0013974676985526457 | 0.0016638492670608684 |

### Extrema and readout sensitivities

The exhaustive grid uses arrival endpoints \(40,42,\ldots,100\), widths at least \(6r_s\), and signed nonzero denominators: 406 arithmetic windows per channel-set/cut. This grid starts at 40; the named broad D0 chord starts at 46. “Filtered” removes known invalid endpoint/interpolation inputs, not every window containing an invalid intermediate checkpoint.

| Channels | Cut | Selection/count | Minimum, window | Maximum, window |
|---|---:|---|---|---|
| 1 | 2 | all = filtered, 406 | -0.01048250462, 44–50 | 0.3138274968, 72–78 |
| 1 | 16 | all = filtered, 406 | -0.01484938224, 44–50 | 0.3792106198, 80–86 |
| 1,2 | 2 | all = filtered, 406 | -0.03300127873, 44–50 | 0.2961203695, 72–78 |
| 1,2 | 16 | all = filtered, 406 | -0.03962958496, 44–50 | 0.3514262171, 72–78 |
| 1–18 | 2 | all, 406 | -0.03904583995, 44–50 | 0.2891480979*, 72–78 |
| 1–18 | 2 | filtered, 354 | -0.03904583995, 44–50 | 0.2062438245, 82–88 |
| 1–18 | 16 | all, 406 | -0.04457330951*, 44–50 | 0.3433178232, 72–78 |
| 1–18 | 16 | filtered, 380 | -0.02253527235, 44–52 | 0.3433178232, 72–78 |

The channel-one cut-2 maximum at 72–78 has \(\delta E=0.005484428693307564\), \(\delta S=1.1462750427530555\), fine-ray \(b=0.3138274968231505\), coarse-ray \(b=0.3138274533839258\), and nearest-checkpoint \(b=0.2161533663432205\). The channel-one cut-16 maximum at 80–86 has denominator \(0.0006952338881092146\). For channels 1,2, the 72–78 denominator is \(0.006031291384715587\); for 1–18 it is \(0.006121817175881006\). The filtered 18-channel cut-2 maximum at 82–88 has denominator \(0.0013027049863012508\). Thus the numerical readout choice and denominator remain visible with the extremum.

For the 18-channel cut-2 broad 46–100 chord, fine-ray, coarse-ray, and nearest-checkpoint values are respectively \(0.05161284607205016\), \(0.05161284880562579\), and \(0.04799223457011141\). The two emission endpoints are \(24.667192687340425\) and \(79.51095103197667\). All raw increments, endpoint temperatures, emission clocks, nearest/coarse sensitivities, and their separate validity flags are saved in `beta.json` and selected in `tables.json`.

No six-\(r_s\) chord is by itself a resolved thermal spectrum. These saved diagnostics do not supply the missing physical-state purity/sector controls or a covariance-generated \(S_{\rm cut}\) series.

## 6. History and numerical-control sensitivities [S]

Every difference below is **control minus production**, on the stated matched channels, clock, cut, and entropy route. Histories are not averaged. The full flags for both inputs, including availability, spectral completeness, and checkpoint failures, are retained in `tables.json`.

The piecewise history uses full-chain `eig_X`, `eig_Xc`, `eig_X_vac`, and `eig_Xc_vac`, with no sector factor of two. Its missing own-vacuum keys at other clocks remain missing; no production vacuum is borrowed. A common vacuum could cancel only after establishing equality of the actual checkpoint Hamiltonian and cut under a cut-preserving basis identification, with the same zero-mode occupation. Matching nominal settings or a fields-file path alone would not establish that.

| Piecewise minus linear production; direct excess, channels 1–8 | Cut 16 | Cut 24 | Cut 30 | Cut 48 |
|---|---:|---:|---:|---:|
| \(T/r_s=50\), production checkpoint failure | 0.4494947409 | 1.989940457 | 0.001657190745 | unavailable |
| \(T/r_s=100\), no known checkpoint failure in either input | 0.3138119181 | 0.3125336649 | 0.3250997923 | 0.5712747962 |

The cut-2 piecewise/interpolated spread is unmeasured. The displayed differences compare the named histories and their conventions; they are not an isolated error estimate for one interpolation operation.

The substep control compares 16 with eight substeps using **only \(\kappa=18\)**, already weighted by \(16\kappa\), on the doubled-plus complement route:

| \(T/r_s\) | Cut 2 | Cut 16 | Cut 24 | Cut 30 | Cut 48 |
|---:|---:|---:|---:|---:|---:|
| 50 | \(1.184886122\times10^{-5}\) | \(2.559506693\times10^{-5}\) | \(7.857340762\times10^{-5}\) | \(1.536985451\times10^{-6}\) | \(2.880944123\times10^{-7}\) |
| 100 | \(6.843911166\times10^{-6}\) | \(2.419410867\times10^{-5}\) | \(1.936282066\times10^{-5}\) | \(2.004268649\times10^{-5}\) | \(5.392895666\times10^{-5}\) |

The saved conditional \(18|\delta x_{18}^{\rm weighted}|\) proxies at cut 2 are \(0.00021327950196337042\) and \(0.00012319040098418554\) for 50 and 100. At cut 30 they are \(2.7665738116411376\times10^{-5}\) and \(0.00036076835675658003\). Such a proxy requires every retained channel's weighted change to be no larger than the measured channel-18 change. It is not a measured 18-channel refinement and says nothing about \(\kappa>18\).

The two alternative completions cover channels one and two through clock 40, using direct excess:

| Clock | Control | Cut 2 | Cut 16 | Cut 30 |
|---:|---|---:|---:|---:|
| 20 | held sources (`completion_fs`) | -1.661073959 | 0.0004984704863 | 0.00008802892474 |
| 20 | paired averaging (`completion_pa`) | -1.442089630 | 0.01345747468 | 0.01208357454 |
| 30 | held sources | -4.014550695 | -0.001608960593 | -0.0001690050719 |
| 30 | paired averaging | -2.723959777 | -0.002356224715 | 0.007353441771 |
| 40 | held sources | -5.562346642 | -0.06828188265 | -0.002081218480 |
| 40 | paired averaging | -4.790143463 | -0.02723450335 | -0.003465360495 |

The ranges among production and those two completions at cut 2 are \(1.6610739585977115\), \(4.0145506952377445\), and \(5.56234664192128\) nats at clocks 20, 30, and 40. At cut 30 they are \(0.012083574540589836\), \(0.00752244684323955\), and \(0.0034653604952481487\). Cut-24 and cut-48 values are also saved. Completion spreads after 40 and for channels 3–18 are unmeasured.

The scale-history (`scale_s8`) comparison is a distinct history sensitivity, using direct excess through all 18 channels:

| Clock | Cut 2 | Cut 16 | Cut 30 |
|---:|---:|---:|---:|
| 20 | -0.5381917318 | 0.01119186444 | 0.01541654263 |
| 30 | -4.312619699 | -0.004315772439 | 0.003182298159 |
| 40 | -5.854595505 | -0.09024431365 | -0.006371298955 |

These selected rows have no known checkpoint failures in either input. The same scale-history scan contains an invalid exported eigenvalue as large as \(1358.8254913177825\) at another checkpoint; that failure neither certifies the history globally nor erases the separately flagged arithmetic rows above.

A high-channel common-mode proxy within the finite 18-channel sum and any angular remainder above 18 require separate premises or bounds. No genuine light-entropy angular tail certificate is supplied here.

## 7. Independently computed stationary thermal reference [R, N]

This comparator is computed from the continuum radial scattering equation, independently of the collapse exports. It is restricted to
\[
1\le|\kappa|\le8,\qquad 10^{-3}\le q=\omega r_s\le3.
\]
Set \(f_H(q)=(1+e^{4\pi q})^{-1}\), \(n_\kappa(q)=\Gamma_\kappa(q)f_H(q)\), and \(w_\kappa=64\kappa\) on positive \(\kappa\). This weight counts eight valleys, particles and antiparticles, both signed angular labels, and \(2\kappa\) magnetic labels. The rates are
\[
(I_E,I_N,I_S)
=\frac1{2\pi}\sum_{\kappa=1}^{8}64\kappa
  \int_{10^{-3}}^3(qn_\kappa,n_\kappa,s(n_\kappa))\,dq,\qquad
\mathcal R=\frac{I_S}{4\pi I_E}.
\]
Thus \(L=I_E/r_s^2\), \(\dot n=I_N/r_s\), and \(\dot S_{\rm marg}=I_S/r_s\). The entropy integrand is \(s(\Gamma f_H)\).

With \(x=r/r_s\), \(z=x+\log(x-1)\), and \(W=\kappa\sqrt{1-1/x}/x\), the own code solves
\[
-Z_\pm''+(W^2\pm W')Z_\pm=q^2Z_\pm,\qquad
W'=\frac{\kappa\sqrt{1-1/x}(3-2x)}{2x^3}.
\]
It imposes a unit inward horizon wave and matches to asymptotic Jost waves. Transmission is obtained as \(|A_{\rm in}|^{-2}\), including for tiny transmissions, rather than by subtracting reflection from one.

| Setting | Base | Refined, corrected endpoint |
|---|---|---|
| Inner \(y=\log(x-1)\) | -60 | -70 |
| Outer \(x\) | \(\max(200,40\kappa/q)\) | \(\max(400,80\kappa/q)\) |
| Jost order | 8 | 12 |
| ODE method | DOP853 | DOP853 |
| Relative / absolute ODE tolerance | \(10^{-10}/10^{-12}\) | \(3\times10^{-12}/3\times10^{-14}\) |
| Summed-rate absolute / relative quadrature targets | \(10^{-10}/10^{-6}\) | \(10^{-11}/10^{-7}\) |

Quadrature panels are \(0.001,0.003,0.01,0.03,0.1,0.2,0.4,0.8,1.5,3\). The code allocates the absolute budget across channels, panels, and weights, and saves every scattering sample. The final refined run doubles the 200 floor as well as the \(\kappa/q\) term. Earlier samples are reused only when their actual endpoint and other settings match this corrected run.

| Finite-domain quantity | Base | Refined |
|---|---:|---:|
| \(I_E\) | 0.0052373154737797205 | 0.005237315471366045 |
| \(I_N\) | 0.015541820008058442 | 0.015541820002075492 |
| \(I_S\) | 0.10787652997408312 | 0.10787652993789794 |
| \(\mathcal R\) | 1.6391110173649306 | 1.6391110175705235 |

This pair records a combined boundary/order/ODE/quadrature sensitivity. It is not an isolated error budget for each setting or a certified total error after adding omitted domains.

The refined weighted channel contributions are:

| \(\kappa\) | \(I_{E,\kappa}\) | \(I_{N,\kappa}\) | \(I_{S,\kappa}\) |
|---:|---:|---:|---:|
| 1 | 0.005041211944906845 | 0.01525058563720862 | 0.10443276322646987 |
| 2 | 0.00019226224059381455 | 0.00028751349386342144 | 0.003381643239593047 |
| 3 | \(3.7839150020684634\times10^{-6}\) | \(3.679916767155274\times10^{-6}\) | \(6.124066628028888\times10^{-5}\) |
| 4 | \(5.6631674320347\times10^{-8}\) | \(4.053763983572945\times10^{-8}\) | \(8.718036442581493\times10^{-7}\) |
| 5 | \(7.30508179851166\times10^{-10}\) | \(4.1256022177896504\times10^{-10}\) | \(1.0875750931825035\times10^{-8}\) |
| 6 | \(8.586233397481833\times10^{-12}\) | \(3.998759481524991\times10^{-12}\) | \(1.2480792747373746\times10^{-10}\) |
| 7 | \(9.399371206823699\times10^{-14}\) | \(3.726619571463026\times10^{-14}\) | \(1.3429121555518388\times10^{-12}\) |
| 8 | \(5.902641752507187\times10^{-16}\) | \(2.1257631392371064\times10^{-16}\) | \(8.711458878334411\times10^{-15}\) |

For channel 7 the reported energy-quadrature error estimate is \(6.302840779417345\times10^{-14}\); its individual ratio is not resolved to all digits stored in the JSON. Small channel ratios do not inherit aggregate precision.

The full-frequency, fully transmitted fermionic known answer is \(\mathcal R=2\). The saved quadrature residuals against \(\pi^2/12,\log2,\pi^2/6\) are \(1.1102230246251565\times10^{-16}\), \(1.1102230246251565\times10^{-16}\), and \(2.220446049250313\times10^{-16}\); the reported ratio is 2.0. Three free-scattering samples have maximum \(|\Gamma-1|=2.927924569462448\times10^{-11}\). Fifteen partner-potential samples for \(\kappa=1,2,8\) have relative transmission differences below \(1.02\times10^{-11}\).

| All-sample scattering control | Base | Refined |
|---|---:|---:|
| Saved samples | 1722 | 1880 |
| Maximum absolute unitarity residual | \(1.971646402\times10^{-9}\) | \(6.668754438\times10^{-11}\) |
| Maximum relative Jost-equation residual | \(2.816182506\times10^{-14}\) | \(1.036621444\times10^{-19}\) |

The refined count includes the additional partner and threshold samples. A local matching residual is not a rigorous global ODE error bound.

The derived threshold law is
\[
\Gamma_\kappa(q)=
\frac{4q^{2\kappa}}{16^\kappa[(2\kappa-1)!!]^2}[1+o(1)].
\]
For \(\kappa=1\), the saved transmissions at \(q=0.001,0.0005,0.00025\) are \(2.5079543758849505\times10^{-7}\), \(6.259882483379114\times10^{-8}\), and \(1.5637313913822193\times10^{-8}\). These are threshold controls, not a uniform envelope on the omitted interval.

For \(q>3\) **within the retained eight angular channels**, unitarity and the Fermi envelope give genuine bounds:

| Omitted high-frequency rate | Absolute bound | Bound / retained refined rate |
|---|---:|---:|
| Number | \(1.2375891586635574\times10^{-15}\) | \(7.9629615997244\times10^{-14}\) |
| Energy | \(3.8112516920497946\times10^{-15}\) | \(7.277109261198865\times10^{-13}\) |
| Entropy | \(4.913119042556477\times10^{-14}\) | \(4.554391066698982\times10^{-13}\) |

No certified low-frequency remainder below \(q=10^{-3}\), and no certified infinite-angular remainder above \(|\kappa|=8\), is supplied. The finite-domain result and the finite-angular high-frequency bound therefore do not establish a statement that all omitted tails are below \(10^{-7}\) of every full rate.

## 8. Conditional heavy curve and remaining inputs

The existing X10 analytical decomposition is retained conditionally if a heavy-field curve prints. For physical evolved sea \(V\), dust \(W\) with occupations \(F\), and the actual cut,
\[
G=\begin{pmatrix}
V^\dagger\Pi_XV & V^\dagger\Pi_XWF^{1/2}\\
F^{1/2}W^\dagger\Pi_XV & F^{1/2}W^\dagger\Pi_XWF^{1/2}
\end{pmatrix}.
\]
Define \(S_H=\operatorname{Tr}s(G)\), \(S_Q\) and \(S_d\) from its diagonal blocks, \(S_P\) from the held physical comparator, and \(K_{sd}=S_Q+S_d-S_H\). Then
\[
S_H=S_P+(S_Q-S_P)+S_d-K_{sd},\qquad
0\le K_{sd}\le2\min(S_Q,S_d).
\]
The off-diagonal overlap remains part of the calculation; fractional occupations require the general impurity term. The actual physical panel/slice specification, same-parent comparator, complete required slices/channels, and physical copy convention were not supplied for a new heavy-curve numerical decomposition. The T12 stored identity control does not fill that request. No heavy-panel numbers or curve clearance are delivered here.

Remaining requirements are explicit:

1. **Light occupied columns or equivalent full covariance at 50 and 100.** These are required to regenerate the cut spectra, diagnose the nine invalid exports, construct the held reference independently, and establish state physicality. The named older pod5 location is absent; production sidecars are not occupied-column payloads.
2. **All-clock state controls.** The complement and sector shortcut used in beta needs a physical covariance/purity or impurity allowance and its actual sector relation at every relevant endpoint/bracket. The direct checkpoints do not supply that globally.
3. **Control coverage.** Cut-2 piecewise/interpolated history, all-channel substep refinement, completion changes after 40 and for channels 3–18, and the entropy angular remainder above 18 remain unmeasured.
4. **Primitive energy regeneration.** The endpoint traces, current integrals, jump terms, and reference changes remain supplied inputs despite the independent ledger arithmetic.
5. **Thermal omitted domains.** A uniform low-frequency envelope and a genuine infinite-angular tail certificate remain missing.
6. **Conditional heavy panel.** Its physical slice/cut/channel/copy inputs are needed if that curve is to print.

## SOURCES and reproducibility

The numerical programs were implemented separately from this analytical specification and executed on CPU at `192.168.1.93`, using `<internal>/python`, `nice -n 10`, and the recorded thread limits. The individual result files report one thread per process; the runtime check excludes GPU use and enforces the bulk-output location. Runtime metadata is an admission record, not a measurement of later peak BLAS thread use. This author ran no numerical code when preparing either the specification or this handoff.

### Governing and format sources

- `<internal>/GOAL_codex.md`
- `<internal>/COMMS.md`
- `<internal>/requirements_new.txt`
- `<internal>/PAGE_VERIFY_PLAN.md`
- `<internal>/FABLE_RULING_O1_D13_protocol_v1.md`
- `<internal>/FABLE_RULING_quasifrozen.md`
- `<internal>/FABLE_RULING_oneobject_MI.md`
- `<internal>/FABLE_REORIENT_5.md`
- `<internal>/FABLE_LIGHTSPECIES.md`
- `<internal>/FABLE_RULINGS_1.md`
- `<internal>/route2-busi.md`, entries 4 and 5, for the export and conditional heavy task.
- `<internal>/EXPORT_codexbusi_A2gen_B2_state_format_v1.md`
- `<internal>/PAGE_solved_history_v4.1.md`
- `<internal>/PAGE_v4.1_numbers_for_paper.md`
- `<internal>/PAGE_VERIFY_A4_beta_eff_v1.1.md`
- `<internal>/WEB40I_heavy_entropy_decomposition_v1.tex`

The delivery sources above supplied definitions, settings, file formats, and missing-input context. Their paper-facing numerical summaries were not inputs to this lane's recomputation or a basis for a cross-route comparison in this note.

### Remote primitive numerical inputs

- B2 history: `<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json`.
- Applied fields: `<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz`.
- Corresponding source copies: `<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json` and `<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz`.
- Raw production aliases: `<internal>/v6o_B2_k<K>_d13.npz` and `.json`, for \(K=1,\ldots,18\). Each entropy row records its exact resolved path and NPZ key. The ledger source list records every JSON path. The resolved families are:

| \(K\) | Directory | Filename stem before `_k<K>_d13.npz` or `.json` |
|---|---|---|
| 1–6 | `<internal>/v6o_B2` | `v6o_B2` |
| 7–9 | `<internal>/v6o_B2_k7to9` | `v6o_B2_k7to9` |
| 10–12 | `<internal>/v6o_B2_k10to12` | `v6o_B2_k10to12` |
| 13–16 | `<internal>/v6o_B2_k13to16` | `v6o_B2_k13to16` |
| 17–18 | `<internal>/v6o_B2_k17to18` | `v6o_B2_k17to18` |

- Piecewise control: `<internal>/v6n_B2_k<K>_d13.npz`, \(K=1,\ldots,8\).
- Substep control: `<internal>/v6o_B2r2c_sub16_k18_d13.npz`.
- Held-source completion: `<internal>/v6o_W3fs_k1to1_k1_d13.npz` and `<internal>/v6o_W3fs_k2to2_k2_d13.npz`.
- Paired-averaging completion: `<internal>/v6o_W3pa_k1to1_k1_d13.npz` and `<internal>/v6o_W3pa_k2to2_k2_d13.npz`.
- Scale-history aliases: `<internal>/v6o_s8_k<K>_d13.npz`, \(K=1,\ldots,18\); exact resolved per-channel paths are retained in `scale_s8/spectral_entropy.json`.
- Actual heavy identity input: `<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0012.00.npz`.
- Available but not substituted for the missing light state or a heavy panel: `<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_T0100.00.npz`.
- Missing older light-state location named in the export: `<internal>/v6mA2_B2_gensplit`.

### Own specification, programs, review, and outputs

- Analytical specification: `<internal>/PAGE_secondroute_spec_v1.md`.
- Sole independent review: `<internal>/redteam_PAGE_secondroute_v1.md`.
- Own entropy program: `<internal>/run_PAGE_spectral_secondroute_v1.py`.
- Own Gaussian controls: `<internal>/run_PAGE_gaussian_controls_v1.py`.
- Own ledger program: `<internal>/run_PAGE_ledger_secondroute_v1.py`.
- Own window program: `<internal>/run_PAGE_beta_secondroute_v1.py`.
- Own stationary scattering: `<internal>/run_PAGE_thermal_R_v1.py`.
- Own compact-table program: `<internal>/summarize_PAGE_secondroute_v1.py`.
- Runtime contract: `<internal>/cpu_runtime.py`.
- Analytic controls: `<internal>/analytic_controls.json`.
- Stored-state identity: `<internal>/stored_identity.json`.
- Thermal known answers: `<internal>/thermal_known_answers.json`.
- Production spectra: `<internal>/spectral_entropy.json`.
- Production failures: `<internal>/invalid_spectra.json`.
- Ledger: `<internal>/ledger.json`.
- Windows and ray diagnostics: `<internal>/beta.json`.
- Piecewise control: `<internal>/spectral_entropy.json`.
- Substep control: `<internal>/spectral_entropy.json`.
- Held-source completion: `<internal>/spectral_entropy.json`.
- Paired-averaging completion: `<internal>/spectral_entropy.json`.
- Scale-history control: `<internal>/spectral_entropy.json`.
- Final thermal rates and per-channel controls: `<internal>/thermal_R.json`.
- Compact complete tables: `<internal>/tables.json` and `<internal>/tables.md`. These are the numerical transcription source for this note, including full channel sets, availability flags, all six-window increments and extrema, additional ledger rows, control ranges, and scattering residuals.

The authoritative remote output root is `<internal>/PAGE_secondroute_v1`. Its corresponding result JSONs, `ray_map.json`, raw control matrices, and per-sample scattering JSONL files retain the larger audit record. Base scattering samples are under `thermal_R/scattering_k<K>_base.jsonl`; final corrected refined samples are under `thermal_R_endpoint/scattering_k<K>_refined.jsonl`, with reuse provenance written per sample. The local compact copies above contain the values used in this handoff.
