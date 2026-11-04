
**F** = `00_FableAPI_closeF_v1.md`; **A** = `01_AstraAPI_emisA2_v1.md`.

1. **Identity and defect location.** F §1, “exact bookkeeping on every history,” needs qualification: the fixed-radius algebra uses **both constraint rows exactly**. Defining \(R_t\) alone does not absorb their residuals on numerical or held-field histories.

   F §1, “the cumulative defect *outside*,” requires the stated zero outer boundary term. Generally, at an outer boundary \(b\),
   \[
   N_cD_c=N_bD_b-4\pi\!\int_{r_c}^{b}r^2N(N\epsilon_E+v\epsilon_M)\,dr,
   \qquad D_b=\dot E+L_b.
   \]
   Delete “never reaches the collar balance”: interior defects have no **direct integral contribution** under that anchoring, but can change the collar state. Also, “in general the Kodama flux” should read **lapse-weighted Kodama-current flux**, \(\mathcal F=N j_K^r\), with \(K=N^{-1}\partial_t\); the \(\partial_t\)-current’s surface flux per unit \(t\) is \(N\mathcal F\).

2. **Storage, sweep and numerical drift.** F §1, “the balance with every term,” counts drift inconsistently. Keeping \(D=r^2vR_t/(GN)\) as defined and writing \(e=\dot E\),
   \[
   \dot E_{\rm out}=e+L_c-D_c,\qquad
   \dot E_R=-L_c-\dot E_{\rm st}+D_c.
   \]
   Together these give the stated total balance \(\dot E_R+\dot E_{\rm st}+\dot E_{\rm out}=e\). **Do not additionally put \(+e\) in the root equation:** its boundary contribution is already in \(D_c\). A separate numerical term requires an explicitly redefined defect. The displayed shell-storage equation retains the correct sweep term, subject to exact constraints.

3. **Root form and relaxation accuracy.** F §2, “forces \(4\pi R^2N_RT_{ll}=-L\),” drops both the root-gradient factor and root defect. For a simple root, define
   \[
   \alpha_R=1-2G\mathcal M'_R=\frac{2R\kappa_R}{N_R}\ne0.
   \]
   Closure instead implies
   \[
   4\pi R^2N_RT_{ll}+D_R
   =-\alpha_R\frac{du}{dt}L[1+O(\epsilon)].
   \]
   The claimed bare partner equality additionally needs \(\alpha_R=1+O(\epsilon)\), \(D_R/L=O(\epsilon)\), clock matching, and negligible heavy null source if attributed to the light field.

   F §2’s boxed window bound has the wrong transient normalization. With \(Q=\int L_c\,dt>0\), \(s=\sup|2G\mathcal S'|<1\), \(\int|D_c|dt\le cQ\), and \(|\delta_{\rm st}|\le\delta_0\), a valid bound is
   \[
   \left|\frac{\Delta E_R}{Q}+1\right|
   \le\frac{s+c+2\delta_0/Q}{1-s}.
   \]
   Here \(D_c\) includes drift. Differential closure needs \(|\dot\delta_{\rm st}|/L_c\) below the **claimed accuracy**, not merely small. Measure \(\mathcal S'\), transient derivatives, exterior defect and arrival-time/flux matching; dimensional scaling and one resonance width do not establish these bounds. A §3(8), “a vacuum root-to-collar region,” is also unnecessarily strong: controlled net storage and defect suffice.

4. **Records and paper wording.** F §3, “certifiable,” exceeds the supplied evidence: no underlying records are attached. The partition sum is bookkeeping, and the quoted “first-law diagnostic” is not an independent flux-balance test. Measure, on common windows, \(\Delta\mathcal M_c+\int L_cdt-\int D_cdt\), constraint residuals, drift, storage change and heavy/reference flux contributions; normalize errors to \(Q\), **not \(E\)**. Heavy inventory fractions cannot certify negligible luminosity contamination; the reported \(\Gamma\) does not bound the longest relaxation time.

   F §3, “\(\dot E_R=0\) exactly, so \(L_c=-\dot E_{\rm st}\),” omits \(D_c\). Frozen geometry need not satisfy the coupled constraints; its geometric \(E_{\rm st}\) is itself fixed. Delete the proposed assertions that no emitted energy reaches a clear collar and that the change is “entirely storage.”

   **Replacement paper sentence:**
   > On constraint-satisfying coupled histories, \(\dot E_R=-L_c-\dot E_{\rm st}+D_c\), where \(D_c\) is exterior-anchored only when the outer boundary term vanishes. The reduction \(dE_R/du=-L\) requires storage, defect, numerical and clock corrections below the stated accuracy; the supplied record summaries do not certify that regime.

**Printable: not as submitted. The corrected conditional balance is printable; the claimed certified closure and bare partner equality are not.**