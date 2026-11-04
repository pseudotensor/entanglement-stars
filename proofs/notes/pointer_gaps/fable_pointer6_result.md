# Thin-shell first correction to the received redshift law
Backs Sec. VIII B (06_scaling.tex:44): "...a thin-shell expansion about a simple root of the back-averaged \(N+v\) gives that first correction as the shell integral of the root's curvature."

**1. What is corrected.** Sec. VIII B prints the leading law \(\ln(1/g_\infty)=\kappa_H(u-t_c)+\tfrac32+O(\delta/r_s)\) and the exact vacuum law \(dg_\infty/du=-\kappa_H g_\infty(1-g_\infty)^4\). "That first correction" is the \(O(g)\) term of the rate, \(-\kappa_H(1-g_\infty)^4=-\kappa_H(1-4g_\infty)+O(g_\infty^2)\), i.e. \(+4g_\infty=2\delta/r_s+O(\delta^2/r_s^2)\) in \(\ln(1/g_\infty)\). It is derived here on supplied fields.

**2. Assumptions.**
(A1) Spherical back-averaged fields \(\bar N>0\), \(\bar v<0\); \(\bar f\equiv\bar N+\bar v\), \(\bar w\equiv\bar v/\bar N\). Simple outer root: \(\bar f(R)=0\), \(\kappa_R\equiv\bar f'(R)>0\), \(\bar f>0\) on \((R,r_D]\).
(A2) Leading dispersion: outgoing long waves move at \(\dot r=\bar f\) (Sec. V A). The emitter is a massive valley ray comoving with the flow, \(\dot R_e=\bar v(R_e)\) (Eq. dr-comoving), proper time \(\bar N\,dt\). The receiver is static at \(r_D\), lapse \(N_D=\bar N(r_D)\) (\(N_D=1\) at infinity); \(u\) is its coordinate (retarded) time, \(d\tau_D=N_D\,du\).
(A3) Held fields over the flight. A ray at \(x=r-R\) moves relative to the root at \(\kappa_Rx-\dot R\), so the held law needs \(|\dot R|\ll\bar f(R_e)=N_Dg_D\), and the fields' drift over the shell crossing small on the same scale; as \(g_D\to0\) this fails for every moving root (the non-robust late redshift of Sec. VI).
(A4) Thin shell: \(\delta_e\equiv R_e-R\) with \(|\bar f''_R|\delta_e\ll\kappa_R\) and \(|\bar f'''_R|\delta_e\ll|\bar f''_R|\); and above the dispersive cutoff of Sec. VIII A, \(2g_\infty\gg(2\chi q)^{2/3}\).
(A5) Kernel. The expansion is of \(\bar f=W_\ell^\dagger f\) about its own root, so the kernel enters only through \(\bar f\)'s Taylor coefficients at \(R\). With \(\bar f-f=\tfrac12\sigma^2\mathcal Df+O(\sigma^4)\) (App. B; \(\mathcal D=\partial_r^2+2r^{-1}\partial_r\) for the cubic kernel, \(\partial_r^2\) for a radial chain kernel) and a fine-grid \(f\) smooth on \(L_f\gg\sigma\): the root moves by \(O(\sigma^2/L_f)\), \(\kappa_R\) by \(O(\kappa_R\sigma^2/L_f^2)\), \(\bar f''_R\) by \(O(\sigma^2/L_f^4)\); the \(O(g)\) term below exceeds the kernel's shift of \(\kappa_R\) only for \(g_D\gg\sigma^2/L_f^2\) (the paper's \(\delta/r_s\gg\sigma^2/r_s^2\)). Where \(f\) varies on scales \(\lesssim\sigma\), \(\bar f''_R\) is kernel-set, (A4) reads \(\delta_e\ll\sigma\), and the result is a statement about the lattice equations, not a geometry.

**3. Exact held-field law.** Two fronts emitted \(\Delta t_e\) apart are received \(\Delta u=(1-\bar v_e/\bar f_e)\Delta t_e=(\bar N_e/\bar f_e)\Delta t_e\) apart; proper periods \(\bar N_e\Delta t_e\) and \(N_D\Delta u\) give
\[
g_D(u)=\frac{\omega_D}{\omega_e}=\frac{\bar f(R_e(u))}{N_D},\qquad
\frac{d\ln g_D}{du}=\frac{\bar f'(R_e)}{\bar f(R_e)}\,\bar v(R_e)\,\frac{\bar f(R_e)}{\bar N(R_e)}=\bar f'(R_e)\,\bar w(R_e).
\]
Vacuum check (\(\bar N=1\), \(\bar v=-\sqrt{r_s/r}\)): \(\bar f'(R_e)=\kappa_H(1-g)^3\), \(\bar w(R_e)=-(1-g)\); product \(-\kappa_H(1-g)^4\), the printed exact law.

**4. Thin-shell expansion.** With \(\bar f'=\kappa_R+\bar f''_Rx\), \(\bar w=-1+\bar w'_Rx\), \(\bar w'_R=\kappa_R/N_R\) at a root (\(\bar w=\bar f/\bar N-1\), \(\bar f(R)=0\)), and \(N_Dg_D=\kappa_Rx+\tfrac12\bar f''_Rx^2\):
\[
\frac{d\ln g_D}{du}=-\kappa_R+c_1\,N_Dg_D+O(g_D^2),\qquad
c_1=\frac{\kappa_R}{N_R}-\frac{\bar f''(R)}{\kappa_R},\qquad
\ln\frac1{g_D}=\kappa_R(u-u_*)+\frac{c_1}{\kappa_R}N_Dg_D+O(g_D^2),
\]
\(u_*\) fixed by the complete flight, not by the shell (\(t_c-3r_s\) in vacuum, the printed \(\tfrac32\)). Writing \(u=t_e-\int^{R_e}dr/\bar f\), with regular part \(\int_R^{R_e}[\bar f^{-1}-(\kappa_R(r-R))^{-1}]dr=-\tfrac{1}{2\kappa_R^2}\int_R^{R_e}\bar f''dr\), and the emitter's fall \(t_c-t_e=\int_R^{R_e}dr/|\bar v|\), the same result in shell form is
\[
\ln\frac1{g_D}=\kappa_R(u-u_*)\;+\;\kappa_R\!\int_R^{R_e}\!\frac{dr}{|\bar v|}\;-\;\frac1{\kappa_R}\!\int_R^{R_e}\!\bar f''\,dr\;+\;O(\delta_e^2).
\]
The last term is the shell integral of the root's curvature, \(=-[\bar f'(R_e)-\kappa_R]/\kappa_R\) (half from the ray's crossing time, half from \(\bar f(R_e)\) against \(\kappa_R\delta_e\)). The middle term is the emitter's shell transit, \(\kappa_R\delta_e/N_R=N_Dg_D/N_R\): fixed by the root's slope and clock, no curvature. Vacuum: \(3g_\infty+g_\infty=4g_\infty\), \(c_1=2/r_s\).
*Rows form.* For fields obeying Eqs. (dr-charge), (dr-radial) at \(R\) (the solved fine-grid fields; the back-averaged ones to \(O(\sigma^2/L_f^2)\)), with \(\beta_R=1-2G\mathcal M'_R\): \(\kappa_R=N_R\beta_R/2R\), \(N'_R=4\pi GRN_RJ_R\), \(\bar w''_R=-G\mathcal M''_R/R-\beta_R/R^2+\beta_R^2/4R^2\), \(\bar f''_R=2N'_R\kappa_R/N_R+N_R\bar w''_R\), hence
\[
c_1=\frac2R-8\pi GR\,J_{\ell r}(R)+\frac{2G\,\mathcal M''(R)}{\beta_R}\qquad(=2/R\ \text{in vacuum}).
\]

**5. Evaluation on a stored record.** (i) From the stored back-averaged \(\bar N_n,\bar v_n\) (the fields in the chain operator) form \(\bar f_n\); take the outer sign change \(\bar f_n<0<\bar f_{n+1}\). (ii) Fit a cubic to \(\bar f\) on the sites within about \(\sigma\) of it; read \(R,\kappa_R,\bar f''_R,\bar f'''_R\), \(N_R=\bar N(R)\); check (A4) at the launch radii. (iii) Prediction for a launch at \(R_e\) received at \(r_D\): \(d\ln g_D/d\tau_D=[-\kappa_R+c_1\bar f(R_e)]/N_D\), against the stored comoving-emitter \(g_D(\tau_D)\) of Sec. VI and against the exact held law \(\bar f'(R_e)\bar w(R_e)/N_D\), valid at any \(R_e\). (iv) \(|\dot R|/\bar f(R_e)\) from consecutive slices measures the (A3) error.

**6. Verdict.** The printed sentence overclaims by one term. Only the local slope \(\bar f'(R_e)=\kappa_R+\int_R^{R_e}\bar f''dr\) is the pure shell integral of the curvature; the redshift rate carries the extra factor \(|\bar w(R_e)|\), whose first correction is the emitter's shell transit \(N_Dg_D/N_R\), vanishing nowhere (one quarter of the \(4g_\infty\) in vacuum). The sentence also needs "held" fields. Tight replacement:
"On held fields a thin-shell expansion about a simple root \(R\) of the back-averaged \(N+v\) gives that first correction as \([\kappa_R/N_R-\partial_r^2(N+v)|_R/\kappa_R]\,g_\infty\) in \(d\ln g_\infty/du\), the emitter's shell transit plus the shell integral of the root's curvature (derivation in Ref.)."
Otherwise drop the pointer and the clause after "gives that first correction".

Side checks, outside this pointer: (a) by \((1-g)^4\ge0.9\) the leading law is within 10 percent from \(u-t_c=4.1r_s\), by \(|g_{\rm lead}-g|\le0.1g\) from \(4.3r_s\); the printed \(4.4r_s\) should name its criterion. (b) App. B's second moment gives \(W_\ell^\dagger v=v[1-\sigma^2/(8r^2)]\) for \(v\propto r^{-1/2}\), hence \(r_{H,m}/r_s=1-\tfrac14\sigma^2/r_s^2\); Sec. V D prints \(9/8\) and \(9/4\), a factor 9 that needs the variance convention stated.