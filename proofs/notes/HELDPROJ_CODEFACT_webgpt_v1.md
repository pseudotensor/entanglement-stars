# #549 code fact: the held reference projector in route 2's new-law histories

Answer from mark3_r2, transcribed by main (af7c14de) on 2026-10-31. These are code facts only; no new computation was run.

## Answer

Under the new law the reference projector is FROZEN for the whole history, not just within each step. This is the engine's PHYSICS, not something the tangent code leaves out. The only part of the reference that depends on u is the W_2 polarization term t, and its derivative is T_W.

## Engine

The engine is m3r2_engine_v18.py (md5 89eb406a). Versions v19-v29, including v27 (fbd27122) and v29 (9da41114), inherit this unchanged and do not redefine _sources_held or t_moments.

- Header, lines 3-9: "sources s = Tr(C A) - r_new, r_new = Tr(P_star A) + D W_2[u] - D W_2[u_star] ... P_star = the prepared reference bands (T = 0) ... set by the driver (PBstar, ustar) and checkpointed ... no continued projector is solved ... the driver turns the tracking stop off".
- Lines 31-32: the class attributes are PBstar = None and ustar = None.
- Lines 49-64, History._sources_held, the source map for each stage, used by every Yoshida substep:
  - Line 61: r_, j_ = cpar.chain_sources(c, CB - PBs, V), where PBs = self.PBstar[c] (line 64). P_star is a constant band and is never recomputed from u.
  - Line 62: tN, tv = self.t_moments(c, Nb, vb), evaluated at the stage's own solved fields (f1 = fs.solve(x), line 51).
- Lines 35-43, t_moments: t = AD.dW2(c, Nb, vb) - AD.dW2(c, u_star), i.e. line 41 minus the value cached at line 39.

## Driver

The driver is m3r2_adapt_v7r.py (md5 828d4939).

- Where P_star is set:
  - Line 568: H.PBstar = [c.PB.copy() for c in chains].
  - Line 569: H.ustar = the T = 0 fields, at a fresh start from the prepared state.
  - Line 571: on resume, both are restored verbatim from engine_state.pkl.
  - Line 285: both are checkpointed.
  - They are never updated, so P_star is constant in time across every step and segment.
- exact_rates_newlaw, lines 742-775:
  - Line 747: zb = zeros is passed as PdotB to record_v3.chain_rates (m3r2_record_v3.py, md5 d7075e0d, line 80). The projector-rate term Tr(Pdot A) is therefore exactly 0, because P_star is constant.
  - Line 755: the reference rate is replaced by r_p = tdot V.
  - Lines 767-768: tdot = d/deps t_moments(u + eps udot), a central difference along udot, iterated self-consistently.

## Tangent

- singq_tierR_tan2.py (md5 a8528e92): tan_pass (line 8) and maj_pass (line 39) contain the field map B only, with no projector term.
- singq_vinv_v3.py (md5 cd348eba), lines 61-70, the reference Jacobian:
  - The per-chain AD._coef(c) coefficients and the Hessian blocks HNN, HNv and Hvv (lines 66-67) are scaled by weight/V into Tm (line 70).
  - This is the analytic derivative of the same AD.dW2 that t_moments uses, i.e. T_W.
- Clock pass 3 checked T_W udot against the engine's stored tdot (sP_f), per chain, at 318 records. They agree to at most 5.8e-4 relative.

## Conclusion

For these histories T_held = 0 is exact physics. The old-law level-following projector (record_v4 exact_rates with Pdots, used by engines before v18) does not appear anywhere in the new-law evolution.

## Open item

Two questions remain open: whether the implemented W_2 tensor equals the paper's, and a line-by-line audit of every update to the total subtracted reference. Both are being checked independently by vref_r1 under #551 (REFLAW_AUDIT_webgpt_v1.md, still to come).
