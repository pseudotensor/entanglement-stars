# Reply to the checker, item by item, then the printable text

I accept the checker's verdict. Every item it marks WRONG is conceded below with the corrected expression and its derivation from the paper's rows and symbols; every item it marks UNSUPPORTED is conceded as an identification the paper does not supply. I rebut nothing. Where I add a derivation it either reproduces the checker's correction from the paper's equations or sharpens it (the first-order cancellation is Möbius invariance of the chiral vacuum, Sec. 10 below). The notation is the checker's: \(\epsilon_u=e^{-(u-t_c)/(2r_s)}\), \(\vartheta=e^{-3/2}\epsilon_u\), \(s=\sqrt{R_b/r_s}\), \(\delta_d=(2\chi q)^{2/3}\), \(\alpha_d=\delta_d^{1/2}=(2a\omega)^{1/3}\).

## 1. Eq. (1), \(g=N+v\cos(ap)\): conceded as a received redshift

On the frozen axial massless symbol \(\omega=N\varepsilon+vq\), with \(\varepsilon=\sin z/a\), \(q=\sin z\cos z/a\), \(z=ap\), the identity \(\omega/\varepsilon=N+v\cos z\) is exact. It is a band-energy ratio, nothing more. The paper's received redshift is the operational one of Sec. star, \(\ln(1/g_D)=-\ln(d\tau_{\rm em}/dT_D)+\ln\alpha_D\), \(\alpha_D=\sqrt{N_D^2-v_D^2}\). Deriving it along the ray: neighbouring outgoing fronts, \(dr/dT=\bar V_+=\bar N+\bar v\), separate at the rate \(\partial_r\bar V_+\); a normal-comoving emitter (\(dr_e/dT=\bar v\)) launches two fronts a lattice time \(dT_e\) apart with separation \((\bar V_+-\bar v)_edT_e=\bar N_edT_e=d\tau_{\rm em}\); the receiver reads their arrival spacing \(\delta r_D/\bar V_{+,D}\) on its clock \(\alpha_D\). Hence
\[
g_D=\frac{\bar V_{+,D}}{\alpha_D}\exp\Big[-\!\int_{T_e}^{T_D}\partial_r\bar V_+(T,r(T))\,dT\Big],
\]
which on stationary fields gives \(g_D=(N+v)_{\rm em}/\alpha_D\) and \(g_\infty=(N+v)_{\rm em}\), as the paper states. No finite-wavelength emitter or detector prescription exists in the paper (the model has no interaction, so no emission process defines one), and the sampled-phase frequency \((\omega-vp)/N\) differs from \(\varepsilon(p)\) at \(O((ak)^2)\). The band-energy ratio enters the printed text only as the local scale at the dispersive cutoff, where it and the continuum \(g_\infty\) both give \(\alpha_d^2/2\) at \(N_R=1\).

## 2. Eq. (2): conceded qualification

\(N+v=N(1-\sqrt{2G\mathcal M(r)/r})\) follows from the rows and is the stationary continuum \(g_\infty\) only; evolving fields need the flight factor above. The marginally bound, normal-comoving, \(N=1\) dust is a comparison-solution specification, not a consequence of pressurelessness; the computed cloud is released at \(4r_s\), a bound preparation whose approach constant differs while the exponent \(\kappa_H\) does not.

## 3. Eq. (3): identity kept, bound conceded

From Eq. sc-clock-exact, \((\ln N)'=\tfrac{\lambda-1}{2}(\ln\mathcal M)'=-\tfrac12\,\frac{wJ}{\rho+wJ}(\ln\mathcal M)'\), so \(\ln N_{\rm sh}=\tfrac12\int_{R_e}^\infty\frac{wJ}{\rho+wJ}\,d\ln\mathcal M\). The bound \(|\ln N_{\rm sh}|\le\tfrac12\ln[E/\mathcal M(R_e)]\) needs \(|wJ/(\rho+wJ)|\le1\) and \(\mathcal M'\ge0\) on the interval; without monotonicity only the total-variation form holds. The paper imposes neither on its signed reference-subtracted sources.

## 4. Eq. (4): conceded, the power is \(N_R^{1/3}\)

At a held root \(v=-N_R\), so \(a\omega=N_R\sin z(1-\cos z)=N_R(z^3/2-z^5/8+\dots)\). With \(\xi=(2a\omega/N_R)^{1/3}\), \(z=\xi+\xi^3/12+O(\xi^5)\), and
\[
\frac{\omega}{\varepsilon}\Big|_R=N_R(1-\cos z)=\frac{N_R\xi^2}{2}\Big[1+\frac{\xi^2}{12}+O(\xi^4)\Big]
=\frac{N_R^{1/3}\alpha_d^2}{2}\big[1+O(\alpha_d^2N_R^{-2/3})\big],
\]
\(v_{g,R}=N_R(\cos z-\cos2z)=N_R(\tfrac32\xi^2-\tfrac38\xi^4+O(\xi^6))\), the paper's Eq. dr-crossing-ray at \(N_R=1\). The word "floor" is withdrawn: the paper already says the cutoff is neither a stopping surface nor a maximal redshift.

## 5. Eq. (5): conceded on both counts

Root motion: for \(x=r-R(T)\), \(\dot x=\kappa_Rx+B_Rx^2-\dot R+O(x^3)\), and a normal-comoving emitter has \(\dot x_e=-N_R-\dot R+O(x_e)\); the \(-\dot R\) terms were omitted.

First correction: on stationary fields \(u=t_e-r^*(R_e)\), \(dr^*/dr=1/(N+v)\), and \(\dot R_e=v_e\) give \(du/dt_e=N_e/(N+v)_e\), hence
\[
\frac{d\ln(1/g_\infty)}{du}=-\frac{dt_e}{du}\,\frac{v_e\,\partial_r(N+v)}{(N+v)_e}=-w_e\,\partial_r(N+v)\big|_{R_e}.
\]
On the constant-charge exterior \(-w_e=1/s\) and \(\partial_r(N+v)=\tfrac12\sqrt{r_s}\,R_e^{-3/2}=1/(2r_ss^3)\), so the slope is \(\kappa_H/s^4=\kappa_H[1-2(t_c-t_e)/r_s+\dots]\), coefficient \(4\kappa_H(t_c-t_e)\), not \(\kappa_H(t_c-t_e)\).

## 6. Eqs. (6)-(15), (19): confirmed independently

I re-derived the map. The dust interior in conformal time gives \(ds/d\eta=-1/(2\chi_b)\), so a ray crossing the ball (\(\Delta\eta=2\chi_b\)) enters at \(s+1\) and leaves at \(s\). From \(dt/ds=-2r_ss^2\), \(dr^*/ds=2r_ss^2/(s-1)\) and \(dr^*_{\rm in}/ds'=2r_ss'^2/(s'+1)\): \(du/ds=-2r_ss^3/(s-1)\), \(dU/ds=-2r_s(s+1)^3/(s+2)\), hence \(dU/du=(s+1)^3(s-1)/[s^3(s+2)]\) and \(\kappa_u/\kappa_H=6/[s^4(s+1)(s+2)]\). Expanding \(u-t_c=r_s[-3-2\ln(d/2)-\tfrac52d-\tfrac3{16}d^2-\tfrac1{12}d^3+\dots]\), \(d=\delta/r_s\), gives \(d=2\vartheta-5\vartheta^2+18\vartheta^3\), \(g_\infty=\vartheta-4\vartheta^2+19\vartheta^3\), \(\ln(1/g_\infty)=(u-t_c)/(2r_s)+\tfrac32+4\vartheta-11\vartheta^2\); with \(s=1+\vartheta-3\vartheta^2+12\vartheta^3\), \(\kappa_u/\kappa_H=1-\tfrac{29}{6}\vartheta+\tfrac{1021}{36}\vartheta^2\), and \(\kappa_u=\kappa_HD-\partial_u\ln D\) gives \(d_1=29/12\), \(d_2=4925/432\). All agree with the checker.

## 7. \(C_c\): conceded, factor of two

With \(u(s)-t_c=-\tfrac23r_s(s^3-1)-r_s(s^2+2s)-2r_s\ln(s-1)\), the end \(s_d=(1-\delta_d)^{-1/2}\) gives \(u_d-t_c=\tfrac43r_s\ln[1/(2\chi q)]-(3-2\ln2)r_s+O(r_s\delta_d)\), so the segment duration minus its logarithm is
\[
C_c=2\ln2-3-\frac{u(s_c)-t_c}{r_s}=2\ln2-3-2\kappa_H[u(s_c)-t_c],
\]
\(C_{0.5}=-1.3767\), \(C_{0.05}=-7.6313\), the paper's rounded \(-1.4\), \(-7.6\) unchanged; the size-64, \(q=1\), 50-percent duration is \(2.76r_s\) as printed.

## 8. After (9), the embedded-root slope: conceded

From the paper's \(\kappa_R=(N_R/2R)[1-8\pi GR^2(\rho_R-J_R)]\), with \(\mathcal M(R)=R/2G\) and \(\mathcal M'_R=4\pi R^2(\rho_R+wJ_R)\) at \(w=-1\):
\[
\frac{\kappa_R}{\kappa_H}=N_R\beta_R\frac{r_s}{R}=\frac{N_R\beta_R}{1-f_{\rm ext}},\qquad \beta_R=1-2G\mathcal M'_R,\quad f_{\rm ext}=1-\frac{\mathcal M(R)}{E}.
\]
\(\kappa_H/(1-f_{\rm ext})\) omitted \(N_R\beta_R\).

## 9. M1 to M5

M1: correct, each history's own \(2GE\). M2: conceded; no material shell is tracked, and my later reading makes it worse (Sec. 13). M4: conceded; the exact condition is \(\delta_d<1-s_c^{-2}\) with \(6/[s_c^4(s_c+1)(s_c+2)]=1-c\); \(\delta_c\simeq12c/29\) only at small \(c\). At \(q=1\): \(\delta_c=0.0211\) at \(c=0.05\) and \(0.0881\) at \(c=0.2\), so \(r_s/a>2/\delta_c^{3/2}\) gives 658 and 76.6; at the stored sizes the 5-percent segment is empty everywhere, the 20-percent segment is nonempty from 128 upward, the 50-percent segment at 64 and above. M5: conceded; on stationary fields \(d\tau_{\rm em}/dT_D=(N+v)_e\), so \(\ln(1/g_D)=\ln(1/g_\infty)+\ln\alpha_D\) and \(\ln(1/g_\infty)=\ln(1/g_D)-\tfrac12\ln(1-r_s/r_D)\), \(+0.144\) at \(r_D=4r_s\).

## 10. Eqs. (16)-(17), \(\nu=\nu_HD\), \(\dot S=\dot S_HD\): conceded, and the concession is stronger than stated

Writing \(X=e^{-\kappa_Hu}\), \(\vartheta=\beta X\), the dust map is \(U_H-U=CX[1-d_1\beta X+\tfrac12(d_2+d_1^2)\beta^2X^2+\dots]\). A fractional-linear map \(CX/(1-\gamma X)\) with \(\gamma=-d_1\beta\) reproduces it through first order, and the chiral vacuum two-point function \(\sqrt{U'_1U'_2}/(U_1-U_2)\) is invariant under fractional-linear maps of \(U\). So the \(O(\vartheta)\) term of \(D\) changes no outgoing observable of the companion: no two-point function, no stress, no occupation of any \(u\)-wavepacket. The Schwarzian confirms it: \(L_{\rm ch}/L_{{\rm ch},H}=1-6(d_2-d_1^2)\vartheta^2=1-\tfrac{1201}{36}\vartheta^2\), and \(d_2=d_1^2\) would be the pure fractional-linear case. Thermality in \(\tilde u\) is a statement about \(\tilde u\)-positive-frequency modes; the receiver's \(u\)-packets have, at first order, exactly thermal occupation, not \(D\) times it. At second order the particle rate needs the output basis and is not fixed by the stress coefficient. The symbol \(\nu\) also collides with the paper's trace norm; the printable text uses none.

## 11. Greybody reintegration, solar-mass interpretation, late occupations, turnover: conceded

The spectral integrals behind "\(p_L\) exceeds \(p_\nu\) by one" were not supplied, and a nonstationary flux needs the two-frequency covariance, not \(\Gamma_k(\omega)\). The solar-mass arithmetic stands as a comparison budget (\(114r_s\), about nine retarded-time thermal periods, 1.8 quanta by the reported but unverified number rate, 12 nats, \(3.5\times10^{-77}\) of \(E\)); "within five percent from \(7.5r_s\)" and "deficit \(4\times10^{-25}\)" are withdrawn. Eq. (20) needs the coherence term \(2\sqrt{F(1-F)}\,{\rm Re}(e^{i\phi}C_{+-})\) and, on evolving fields, \(C_{\rm out}=SC_{\rm in}S^\dagger\); Eq. (21) holds only for \(n_+=0\), and \(n_+=1\), \(n_-=0\) exceeds the Hawking comparison. \(N_R\to1\), \(\beta_R\to1\), \(R\to2GE\) are matching conditions, and the dimensional slip is corrected to \(2R\kappa_R\to1\), \(\kappa_R\to1/(2r_s)\). The turnover formulas are the paper's conditional algebra; a uniform occupation factor \(\eta\) raises \(\mathcal R\) (\(s(\eta p)>\eta s(p)\) by concavity), so \(t_P=(t_*/\eta)[1-(\mathcal R_\eta/(\mathcal R_\eta+f))^{3/2}]\).

## 12. Stored-field check and my later finding

On the stored histories at \(r_s/a=64\) and 128 the trapping radius is clear of heavy matter only near \(T/r_s=15\), where 4 to 5 percent of \(E\) lies outside it (the paper's \(R_t/2GE=0.95\) to 0.96), and the outgoing fountain re-embeds it afterwards. Consequences: the slope lookup \(|N_R-1|,|\beta_R-1|,|f_{\rm ext}|\) can be attempted only on the slices near 15, and even there \(f_{\rm ext}\) is already 0.04 to 0.05 at the kernel's own root shift of 0.9 percent; after 15 the identity of Sec. 8 must be read with the fountain's \(\beta_R<1\); the vacuum-shell premise of the dust constants and of \(\kappa_R=\kappa_H\) is met on no stored slice; and a material-shell crossing time (M2) is not even unique, since the dust crosses the root outward again. The Jacobian audit must be run through the fountain's back-averaged fields, not against the dust constants. The paper's todo on the complete null-map audit stands.

---

## Printable text

**Sec. scaling-rays, after the sentence ending "no packet resolving its own frequency fits inside it."**

> On the dust comparison solution the exact map expands in the received exponential. With \(\epsilon_u=e^{-(u-t_c)/(2r_s)}\), \(\vartheta=e^{-3/2}\epsilon_u\), and \(\tilde u\) defined by \(U_H-U=C_Ue^{-\kappa_H\tilde u}\),
> \[
> \frac{d\tilde u}{du}=1-\frac{29}{12}\vartheta+\frac{4925}{432}\vartheta^2+O(\vartheta^3),\qquad
> \frac{\kappa_u}{\kappa_H}=1-\frac{29}{6}\vartheta+\frac{1021}{36}\vartheta^2+O(\vartheta^3),
> \]
> the surface at \((R_b-r_s)/r_s=2\vartheta-5\vartheta^2+18\vartheta^3+O(\vartheta^4)\), and its received factor \(g_\infty=\vartheta-4\vartheta^2+19\vartheta^3+O(\vartheta^4)\), so that \(\ln(1/g_\infty)=(u-t_c)/(2r_s)+\tfrac32+4\vartheta-11\vartheta^2+O(\vartheta^3)\). The first-order term of the map is a fractional-linear deformation of the exponential, which leaves the chiral vacuum two-point function unchanged, so for the dust solution's chiral conformal companion (the massless channel without its centrifugal term) no outgoing quantity changes at first order in \(\epsilon_u\); its stress flux is \(L_{\rm ch}/L_{{\rm ch},H}=1-\tfrac{1201}{36}e^{-3}\epsilon_u^2+O(\epsilon_u^3)=1-1.661\,\epsilon_u^2+\dots\), a conformal stress result and not a particle rate or a greybody-filtered luminosity. The segment on which \(\kappa_u\) lies within a fraction \(c\) of \(\kappa_H\) is nonempty when \(\delta_d<1-s_c^{-2}\), \(6/[s_c^4(s_c+1)(s_c+2)]=1-c\): at \(q=1\) the 5 percent segment needs \(r_s/a>658\) and the 20 percent segment \(r_s/a>76.6\), so at the stored sizes the 5 percent segment is empty, the 20 percent segment exists from size 128 upward, and the 50 percent segment at size 64 and above. The segment ends at \(u_d-t_c=\tfrac43r_s\ln[1/(2\chi q)]-(3-2\ln2)r_s+O(r_s\delta_d)\), where \(1/g_\infty=[1+\sqrt{1-\delta_d}]/\delta_d\simeq2/\alpha_d^2\), and subtracting the tolerance-dependent start gives \(C_{0.5}=-1.38\) and \(C_{0.05}=-7.63\). At one solar mass the end lies \(114r_s\) after \(t_c\), about nine thermal periods \(4\pi r_s\) of retarded time (4.6 in Schwarzschild time); the comparison rates of Eq.~\eqref{eq:sc-luminosity} integrated over it give \(3.5\times10^{-77}\) of \(E\), about 1.8 quanta by the reported but unverified number rate, and 12 nats of marginal entropy. These coefficients describe the comparison solution's propagation and its companion's stress. They fix neither the occupation of the outgoing branch nor its supply, which need the evolving incident covariance at the trapping radius and a once-counted output basis; \(d\tilde u/du\) is a reparametrization of the received clock, not a rate multiplier.

**Sec. scaling-far, after the sentence ending "Eq.~\eqref{eq:star-crossing-delay} is a delay, not a release law."**

> On evolving fields the received factor is the \(g_D\) of Sec.~\ref{sec:star}, equal along the ray to \((\bar V_{+,D}/\alpha_D)\exp[-\int_{T_e}^{T_D}\partial_r\bar V_+\,dT]\), \(\bar V_+=\bar N+\bar v\), which on stationary fields reduces to \((N+v)_{\rm em}/\alpha_D\); a static receiver at \(r_D\) on the held exterior converts by \(\ln(1/g_\infty)=\ln(1/g_D)-\tfrac12\ln(1-r_s/r_D)\). At a held simple trapping radius with clock \(N_R\) the axial massless symbol \(a\omega=N_R\sin z(1-\cos z)\), \(z=ak\), gives with \(\xi=(2a\omega/N_R)^{1/3}\) the carrier \(ak_R=\xi+\xi^3/12+O(\xi^5)\), group velocity \(N_R(\tfrac32\xi^2-\tfrac38\xi^4+O(\xi^6))\), and band-energy ratio \(\tfrac12N_R\xi^2[1+\xi^2/12+O(\xi^4)]\): local kinematics at the dispersive cutoff, not a received redshift and not a redshift floor. The slope at a simple trapping radius is \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\), \(\beta_R=1-2G\mathcal M'_R\), \(f_{\rm ext}=1-\mathcal M(R)/E\) the charge outside it, so \(\kappa_R=\kappa_H\) only on a vacuum shell with unit clock, the shell on which the dust constants above rest. On the stored histories at \(r_s/a=64\) and 128 the trapping radius is clear of heavy matter only near \(T/r_s=15\), with 4 to 5 percent of \(E\) outside it, and is embedded in the outgoing fountain afterwards, so no stored slice realizes that shell; on the histories the slope and the received map are to be read from \(N_R\), \(\beta_R\), \(f_{\rm ext}\) and the complete flight through the fountain's fields, not from the comparison constants.

**Sec. page, "Emission and held-field controls", after the sentence ending "so finite late flux alone proves no sustained supply."**

> The Hawking comparison case below is a supplied exterior, its inner port fed thermally at \(T_H\), so its rates are a comparison and not a result about the object's supply. For a stationary matched two-port channel at a held trapping radius, with incident occupations \(n_\pm\) on the two branches, local conversion probability \(F=[e^{2\pi\omega/\kappa_R}+1]^{-1}\) and exterior transmission \(\Gamma_k\), the outgoing occupation is \(n_{\rm out}=\Gamma_k[(1-F)n_++Fn_-+2\sqrt{F(1-F)}\,{\rm Re}(e^{i\phi}C_{+-})]\), \(C_{+-}\) the incident coherence: with an empty outgoing branch, \(n_+=0\), it is at most \(\Gamma_kF\), the comparison case at the same \(\kappa_R\), and with \(n_+=1\) it exceeds it, so the comparison bounds the supply only given the incident state, which the computed collapse does not fix once formation light has passed. On evolving fields the scattering map mixes frequencies, \(C_{\rm out}=SC_{\rm in}S^\dagger\), and the once-counted output needs its basis. The exponential segment of Sec.~\ref{sec:scaling-rays} fixes the temperature of what is converted, not the amount.