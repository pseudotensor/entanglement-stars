# Mass-loss closure, round two: verdicts, final forms, records, paper text

**Summary verdict.** All four of reviewer B's items are accepted; two are sharpened below with content reviewer B did not supply (the explicit constraint-residual terms of the defect, and the two conditions hidden in the root form). The only partial rebuttal is a wording point: the interior-defect statement keeps its mathematical content under reviewer B's qualifier. reviewer B's replacement sentence is accepted in content and rewritten for the paper's rules (plain words, symbols defined, no meta). Printable: the conditional balance, the root form with its factor and defect, and the conditions with their orders; not printable: a certified closure or the bare partner equality.

## 1. Item by item

**1a. Both constraint rows are used: accept.** Without the rows, the transverse evolution row with $K_t=w/r$, $K_r=w'+wN'/N$ and $\mathcal M=rw^2/2G$ gives exactly
$$\dot w=-\frac{2GN}{r}\mathcal M'+N'(1-w^2)+4\pi GNr(\rho_\ell-p_r)+rR_t .$$
Inserting the rows with their residuals, $\epsilon_{\mathcal M}\equiv\mathcal M'-4\pi r^2(\rho_\ell+wJ_{\ell r})$ and $\epsilon_N\equiv(\ln N)'+4\pi GrJ_{\ell r}/w$, and multiplying by $rw/G$:
$$\partial_t\mathcal M=-4\pi r^2\mathcal F+\mathcal D,\qquad
\mathcal D=\frac{r^2vR_t}{GN}-2v\,\epsilon_{\mathcal M}+\frac{rv(1-w^2)}{G}\,\epsilon_N .$$
The momentum-row residual drops out at a root ($w^2=1$); the charge-row residual enters there as $2N_R\epsilon_{\mathcal M}$. On the stored histories the rows are solved to $10^{-7}$ (charge) and $10^{-6}$ (fields) with the held-projector sources, so with those sources the extra terms sit at the solve tolerance and the defect is $R_t$; with Eq. (dm-reference)'s sources applied to the stored fields, and on every frozen-field history, they are not small. "Exact bookkeeping on every history" is true only for $\mathcal D$ defined as $\partial_t\mathcal M+4\pi r^2\mathcal F$; its evolution-row form needs the rows.

**1b. Outer boundary term: accept.** $N_c\mathcal D_c=N_b\mathcal D_b-4\pi\int_{r_c}^br^2N(N\epsilon_E+v\epsilon_M)\,dr$, with $\mathcal D_b=\partial_t\mathcal M(b)+L_b$, equal to $\dot E$ beyond all sources: zero on the Hartree flow, the step's charge deficit on stored histories, the end reflection on a finite parent once the front arrives. The anchoring identity itself also presumes the rows.

**1c. "Never reaches the collar balance": accept the rewording, keep the content.** Under the anchoring an interior defect contributes no term to $\mathcal D_c$; it changes the state, hence $L_c$ and $\mathcal M(r_c)$, and so the history. The integral form, $\dot E=4\pi\int_0^\infty r^2N(N\epsilon_E+v\epsilon_M)dr$, holds only on exact-row histories.

**1d. Kodama naming: accept.** With $K=N^{-1}\partial_t$ and $j_K^a=-T^a{}_bK^b$, $\mathcal F=-T^r{}_t=Nj_K^r$ (verified from the paper's $T$: $T^r{}_t=-N[w\rho_\ell+(1+w^2)J_{\ell r}+wp_r]$). Since $\sqrt{-g}=Nr^2\sin\theta$, $L_c=4\pi r_c^2\mathcal F$ is the Kodama current's flux per unit $t$, the conserved flux whose charge is $\mathcal M$; the $\partial_t$ current's weighted flux is $NL_c$ and is conserved only where $\partial_t$ is Killing. In the paper the name is avoided: "the charge's flux through the collar, the Killing-energy flux where the collar is in vacuum."

**2. Drift: accept.** $E_R=\mathcal M(r_c)-E_{\rm st}$ gives $\dot E_R=-L_c-\dot E_{\rm st}+\mathcal D_c$ with no drift term; $E_{\rm out}=E-\mathcal M(r_c)$ gives $\dot E_{\rm out}=\dot E+L_c-\mathcal D_c$; the sum is $\dot E$. My closeF wrote $\dot E_{\rm out}$ without $\dot E$ and the root equation with it, which double counts; the drift enters the root balance only through the anchoring of $\mathcal D_c$, as $N_b\dot E/N_c$. The storage equation and its sweep term stand.

**3a. Root form: accept, with two sharpenings.** $\alpha_R=1-2G\mathcal M'_R=2R\kappa_R/N_R=1-8\pi GR^2(\rho_\ell-J_{\ell r})_R$ is the paper's margin of Sec. VI.B. (i) For the massless field's own stationary state the root sources scale as $R^{-4}$, so $\alpha_R-1=O(g\chi^2)$ automatically, the same order as $\epsilon_S$: the condition $\alpha_R=1+O(\epsilon)$ is the root's clearance of heavy matter. (ii) $\mathcal D_R=\mathcal D_c+$ the shell's own defect, and the shell is where lattice transport is largest (the crossing branch has wavelength $(a^2/\omega)^{1/3}$ at the root) and where the reference responds to the moving root; condition (C) beyond the collar does not cover it. The bare equality $4\pi R^2N_RT_{ll}(R)=-L$ therefore needs four extra conditions: $\alpha_R=1+O(\epsilon)$, $|\mathcal D_R|\le\epsilon L$, $du/dt=1+O(\epsilon)$, and the heavy $T_{ll}(R)$ below $\epsilon L/(4\pi R^2N_R)$ if the source is the light field's. Retracted: "it is (S) read at the root."

**3b. Window bound: accept.** From $\Delta E_R+\Delta\mathcal S=-Q+\int\mathcal D_cdt-\Delta\delta_{\rm st}$, $|\Delta\mathcal S|\le s|\Delta E_R|$ by the mean value theorem on $\mathcal S(R)$, and $|\Delta E_R|\le|\Delta E_R+Q|+Q$, one gets $(1-s)|\Delta E_R+Q|\le(s+c)Q+2\delta_0$, reviewer B's bound. My boxed form mixed the normalizations by $Q$ and by $|\Delta E_R|$ and the denominators $1-s$ and $1-c$. The differential form needs each term below the claimed $\epsilon$. $\epsilon_S$ and $\epsilon_C$ as I gave them are dimensional order estimates to be measured, not bounds. Agreed that a vacuum shell is unnecessary; controlled net storage and defect suffice.

**4. Records and wording: accept.** The partition sum is a tautology by the definition of $E_{\rm out}$ and tests nothing; the first-law diagnostic is the charge row's variation between records, a different identity, not a flux balance; errors are normalized to $Q$ on common windows; collar contamination is the heavy flux over $L_c$, not inventory; $1/\Gamma$ of one $k=1$ pole is a lower bound on $t_{\rm rel}$, not an upper one. On frozen fields $\mathcal M(r,t)$ is constant for every $r$, so $\dot E_R=\dot E_{\rm st}=\dot E_{\rm out}=0$ and the identity reads $\mathcal D_c=L_c$: the whole collar flux is defect by construction, the geometric storage does not drain, and what drains is the light field's excitation inventory, which on a light-off-background history is not in $\mathcal M$ at all. Both proposed assertions are deleted.

## 2. The final identity and balance

Exact where both constraint rows hold at each time, with the general-history defect of 1a otherwise:
$$\boxed{\;\partial_t\mathcal M(r,t)=-4\pi r^2\mathcal F+\mathcal D,\qquad
\mathcal F=N\big[w\rho_\ell+(1+w^2)J_{\ell r}+wp_r\big]=-T^r{}_t,\qquad
\mathcal D=\frac{r^2vR_t}{GN}\;}$$
$$N_c\mathcal D_c=N_b\mathcal D_b-4\pi\!\int_{r_c}^{b}r^2N(N\epsilon_E+v\epsilon_M)\,dr,\qquad \mathcal D_b=\dot E\ \text{beyond all sources}.$$
With $E_R=R/2G$, $E_{\rm st}=\mathcal M(r_c)-E_R$, $E_{\rm out}=E-\mathcal M(r_c)$, $L_c=4\pi r_c^2\mathcal F(r_c)=L^L_c+L^H_c$:
$$\boxed{\;\dot E_R=-L_c-\dot E_{\rm st}+\mathcal D_c,\qquad
\dot E_{\rm out}=\dot E+L_c-\mathcal D_c,\qquad
\dot E_R+\dot E_{\rm st}+\dot E_{\rm out}=\dot E\;}$$
$$\dot E_{\rm st}=4\pi R^2\mathcal F_R-L_c+(\mathcal D_c-\mathcal D_R)-4\pi R^2\dot R\,(\rho_\ell-J_{\ell r})_R .$$
Double debiting (root loss $2L$ at collar flux $L$) is excluded unless $\dot E_{\rm st}=+L$, so it would show as storage growth.

## 3. The root form

$$\boxed{\;\alpha_R\,\frac{dE_R}{dt}=4\pi R^2N_R\,T_{ll}(R)+\mathcal D_R,\qquad
\alpha_R\equiv1-2G\mathcal M'_R=\frac{2R\kappa_R}{N_R}\ne0,\qquad T_{ll}=\rho_\ell+p_r-2J_{\ell r}\;}$$
(the sweep of the moving root; an outgoing null component has $\mathcal F\propto(1+w)^2=0$ at the root, so $T_{ll}(R)$ is carried by the ingoing part). Combined with the closure,
$$4\pi R^2N_RT_{ll}(R)+\mathcal D_R=-\alpha_R\,\frac{du}{dt}\,L\,[1+O(\epsilon)],$$
and $=-L$ only under the four conditions of 3a.

## 4. Relaxation conditions with explicit accuracy

Conditions on a window: (V) $L^H_c\le\epsilon_VL_c$, collar in the vacuum exterior, $|du/dt-1|$ and $|L(u)/L_c(t)-1|$ at most $\epsilon_u\simeq\ell_c/t_{\rm em}$ (in-transit change), $r_c\ll t_{\rm em}=E_R/L=R^3/(2G\mathcal L)$; (C) $|\mathcal D_c|\le\epsilon_CL_c$, drift included; (S) $E_{\rm st}=\mathcal S(R)+\delta_{\rm st}$ with $|2G\mathcal S'|\le\epsilon_S$ and $|\dot\delta_{\rm st}|\le\epsilon_\delta L_c$.

Exactly, $(1+2G\mathcal S')\dot E_R=-L_c+\mathcal D_c-\dot\delta_{\rm st}$, hence
$$\Big|\frac{dE_R}{du}+L\Big|\le\Big[\frac{\epsilon_S+\epsilon_C+\epsilon_\delta}{1-\epsilon_S}+\epsilon_V+\epsilon_u\Big]L,$$
and over any window, with $Q=\int L_cdt>0$, $s=\sup|2G\mathcal S'|<1$, $\int|\mathcal D_c|dt\le cQ$, $|\delta_{\rm st}|\le\delta_0$,
$$\boxed{\;\Big|\frac{\Delta E_R}{Q}+1\Big|\le\frac{s+c+2\delta_0/Q}{1-s}\;}$$
Order estimates, to be measured, not bounds: $\epsilon_S\simeq2\ell_c/t_{\rm em}=4g\mathcal L\chi^2(\ell_c/R)$, $2.6\times10^{-5}$ at size 64 and $r_c=16r_s$, about $10^{-77}$ at one solar mass; $\epsilon_C\simeq2\chi^2q^2\sqrt{R/r_c}$ for the lattice weight on the exterior gradient alone, $1.2\times10^{-4}$ and $6\times10^{-77}$. The term with content at every size is $\epsilon_\delta$ (equivalently $2\delta_0/Q$), the transient's decay, which no dimensional argument supplies. Measurements needed: $\mathcal S'(R)$ on the coupled history, $\dot\delta_{\rm st}$, the exterior and shell defects with the constraint residuals, the heavy collar flux over $L_c$, and the collar-to-infinity arrival matching.

## 5. What the stored records certify now

Two floors first, both in units of $Q$. The paper's fixed-radius diagnostic (mass rate at fixed outer-root radii minus the row's rate, median 0.17, 0.04 and 0.06 of the row's largest term at $\sigma/a=2,4,8$, stress provenance unclosed) is the measured $\mathcal D$ at those radii; the largest term is the heavy flux through the root sphere, so this defect floor is three or more orders of magnitude above $L$. The collar drift per record (0.7 percent of $E$ near formation on held steps, $5\times10^{-4}$ on the conserving step) against the thermal debit per record ($6\times10^{-7}E$ per $r_s$) gives $c\sim10^4$–$10^5$ under the anchoring: the window bound is vacuous on every stored history.

| Condition or term | Readout | Records | Verdict |
|---|---|---|---|
| Identity at fixed radius | $\partial_t\mathcal M+4\pi r^2\mathcal F$ against $r^2vR_t/(GN)$ plus constraint residuals | 4–17 percent of the heavy flux at fixed outer-root radii; provenance unclosed | tested at the percent level of the heavy flux, $\gtrsim10^3L$; not certified |
| Anchoring, $\dot E$ | collar drift per record | $7\times10^{-3}E$ to $5\times10^{-4}E$ | drift is $10^4$–$10^5Q$ per record; $c\gg1$ |
| (V) heavy collar flux | heavy current through $16r_s$ over $L_c$ | charge beyond $16r_s<2.6\times10^{-3}$ through 40; outward current 48–56, 50.7/640 beyond at 56; ratio unread | clear in inventory through about 40; flux ratio not read; fails from 48 |
| $\alpha_R\to1$, root clear | heavy charge at the root; $\kappa_R$ | 29/640 outside at 15, 4–5 percent of $E$; $\kappa_R/[1/4GE]$ median 0.74–1.21 | fails on every rooted record |
| (S) relaxed storage | $E_{\rm st}$ against $R$; $|\Delta E_{\rm st}|/|\Delta E_R|$ | order one or larger; episodic roots; light inventory absent from $\mathcal M$ on light-off-background histories | fails; untestable for the light field without light gravity |
| time-scale separation | longest relaxation time | one $k=1$ pole, $1/\Gamma=2.8\times10^3r_s$ (lower bound on $t_{\rm rel}$); broad-block drain 0.147 on route two; first return near 1000 | $t_{\rm rel}$ unbounded above; not certified |
| (C) exterior defect | $\Delta\mathcal M(16r_s)+\int L_c-\int\mathcal D_c$ with light gravity on | size 32 only, $k=1,2$, one route; charge difference $3.4\times10^{-5}$, comparable to the light energy inside the cut | not certified; parameter estimate only |
| held fields | $\mathcal D_c=L_c$ identically | collar flow $0.9$–$1.5L_H$; $E_{\rm inv}$ to seven digits | certifies the inventory drain $\int L_c\le E_{\rm inv}$, nothing about the closure |
| root source $T_{ll}(R)$ | light's $\rho_\ell+p_r-2J_{\ell r}$ at the dense root against $-\alpha_RL/(4\pi R^2N_R)$ | two-field 32, single records; formation light; heavy matter at the root | readable, not certifying |

No condition of the closure is certified at the thermal rate; a drain at the Hawking rate is indistinguishable at the collar from emission, and only $\dot E_R$ on a coupled history separates them. The test remains: a persistent root clear of heavy matter with light gravity on, run beyond $t_{\rm rel}$ and then until $|\Delta E_R|\gg\delta_0$, about $0.2\,t_{\rm em}\approx3\times10^5r_s$ at size 64.

## 6. Paper text

**Sec. VI.E**, replacing "With exterior storage and apparatus work, $dE_R/du=-L$ remains a separate root-source requirement.":

> Between the root and a collar radius $r_c$ the rows give, on histories that solve both constraint rows, the exact balance $dE_R/dt=-L_c-dE_{\rm st}/dt+\mathcal D_c$ (Appendix~\ref{app:charge}), where $L_c$ is the charge's flux through the collar, the Killing-energy flux where the collar is in vacuum, $E_{\rm st}=\mathcal M(r_c)-E_R$ the energy stored between root and collar, and $\mathcal D_c$ the source defect, which is the weighted defect beyond the collar when the charge is conserved there. The law $dE_R/du=-L$ therefore holds to relative accuracy $\epsilon$ exactly when the storage rate, the defect, the charge drift and the mismatch between the collar clock and $u$ each stay below $\epsilon L$ over the window; a quasi-stationary stored state of the massless field puts the first at $2\ell_c/t_{\rm em}$, with $\ell_c$ the collar's crossing time and $t_{\rm em}=E_R/L$, of order $\chi^2$, and the stored histories measure none of the four below the thermal rate. At the root the law reads $4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R=-(2R\kappa_R/N_R)L$, an inward flux of negative null energy, equal to $-L$ at a root clear of heavy matter with negligible root defect \todo{the light field's null source at a dense root clear of heavy matter on a coupled two-field history, long after formation light, with $\mathcal D_R$ and $2R\kappa_R/N_R$ measured, against $-L/(4\pi R^2N_R)$}.

Under the short-sentence rule, split the second sentence after "over the window" and begin the next "A quasi-stationary stored state of the massless field puts the first at $2\ell_c/t_{\rm em}$, …".

**Appendix C**, after "The rate of the mass function at a trapping root":

> \paragraph{The mass balance between root and collar.} At fixed $r$ the transverse evolution row, with the charge and momentum rows eliminating $w'$ and $N'$, gives $\partial_t\mathcal M=-4\pi r^2\mathcal F+\mathcal D$, $\mathcal F=N[w\rho_\ell+(1+w^2)J_{\ell r}+wp_r]=-T^r{}_t$, so that $4\pi r^2\mathcal F$ is the flux through the sphere per unit $t$ of the conserved current whose charge is $\mathcal M$, the Killing-energy flux where $N=1$, and $\mathcal D=r^2vR_t/(GN)$; where the rows carry residuals $\epsilon_{\mathcal M}=\mathcal M'-4\pi r^2(\rho_\ell+wJ_{\ell r})$ and $\epsilon_N=(\ln N)'+4\pi GrJ_{\ell r}/w$, the defect is $\mathcal D=r^2vR_t/(GN)-2v\epsilon_{\mathcal M}+rv(1-w^2)\epsilon_N/G$. By $(N\mathcal D)'=4\pi r^2N(N\epsilon_E+v\epsilon_M)$, $N\mathcal D$ at any radius is $N_b\mathcal D_b$ at an outer boundary $b$ minus the weighted defect between them, and beyond all sources $\mathcal D_b=\dot E$, zero on the Hartree flow and the step's charge drift on stored histories. For the root $R(t)$, a fixed collar $r_c$, $E_{\rm st}=\mathcal M(r_c)-E_R$ and $E_{\rm out}=E-\mathcal M(r_c)$ this gives $\dot E_R=-L_c-\dot E_{\rm st}+\mathcal D_c$ and $\dot E_{\rm out}=\dot E+L_c-\mathcal D_c$ with $L_c=4\pi r_c^2\mathcal F(r_c)$, and at the root $(2R\kappa_R/N_R)\dot E_R=4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R$, Eq.~\eqref{eq:C-rate} completed by the sweep of the moving root, $2R\kappa_R/N_R=1-2G\mathcal M'_R$ being the root-gradient factor.

## 7. Printable

Printable now: the identity with its general-history defect, the balance, the root form with $\alpha_R$ and $\mathcal D_R$, the conditions with their stated accuracy and order estimates, and the statement that the records test none of them at the thermal rate. Not printable: a certified closure, the bare partner equality without its four conditions, and any sentence attributing the computed collar flux to storage or to emission.