# TODO[8] paper replacement at T/r_s = 15.0 (#1382 form), v2, emitted by t8_paper_v1a.py from t8_audit_v1.json

Target: fermion_new/05_star.tex (md5 d18cb410), the unmarked sentence of the line-136 paragraph "Thus \(\mathcal M(R_t)=R_t/(2G)\) leaves 4 to 5 percent of E outside" and the TODO[8] mark of line 136 (v1's replacement for the kappa_R sentences stands; this file fills the 15.0 exterior clause from the audit).

## Replacement for the exterior clause
> At \(T/r_s=15.0\) on the reference collapse, the outer trapping radius, the outermost zero of \(N+v\) with no farther zero to the chain's end and beyond, lies in \([51.750,51.771]\,a\), the asymptotic charge in \([108.36,109.29]\), and \(4.4\) to \(5.3\) percent of \(E\) lies outside the trapping radius: of it, 4.7 percent of \(E\) is the heavy dust, 0.4 percent the propagated sea's departure from the prepared band, -0.2 percent (signed) the reference, and the light part is zero on this history, the four parts summing to \(E-R_t/(2G)\) within the enclosed quadrature, source and cut errors. At the same clock the step controls give 4.3 percent at \(\tau=4a\) and 4.1 percent at \(\tau=2a\), with trapping radii 54.15\,a and 55.43\,a, so the fraction spans 4.1 to 5.3 percent over \(\tau=8a,4a,2a\). \todo{completion restarted on this history through its clock} \todo{the implemented fixed-point contraction ball of the stored row solve, with the level-following gap per channel}

## Revised line-136 mark (the 15.0 part closed except the two marks above)
> \todo{TODO[8]: certified at \(T/r_s=15.0\) on the reference collapse (cut, asymptotic charge, exterior fraction and its four parts, step spread); open: the completion restarted on this history through 15.0 (two of three completions are stored and both end at 14.0, the point-sampled one at 12.5), the implemented fixed-point contraction ball, and the listed-slice range of \(\kappa_R/\kappa_H\).}

## Conditions carried (not bounds), and provenance
- 5-point Gauss-Legendre cell-defect quadrature (L1); sub-grid sup in the Krawczyk derivative range (L2); kernel-sum roundoff.
- The parts are printed as computed shares of the central E; their common error lines are the closure lines of the audit md (window quadrature 3.724e-02, complex64 sources 4.555e-01, cut width 2.080e-02, E 4.655e-01; closure residual 1.121e-03).
- Regime/size label: angular-channel lattice-equation reference collapse, nominal r_s/a = 64, sigma/a = 4, tau = 8a, ten 2048-site chains, fixed reference projector, no light; record-only.
