#!/usr/bin/env python
"""pt3_options.py (busi, PAGE design v3 sec. 9): the six marked conflicts as run-time options. Every side v3 says it runs is built; the ruling
picks one at run time (--c1 ... --c6) without new code. The defaults are v3's "v3 does" column.

C1 which clause is the declaration: both (default: R-loc and R-star on one propagation, neither claimed) | rloc (reviewer A) | rstar (reviewer B).
   Only the readers' label changes; both arms always run (they are subtractions on the same C).
C2 parent regime for the reference: both (default: physical readouts on the long unitary parent; eigenstate-reference ledgers on the dilute
   Hermitian 512 r_s chain read before the first return; plus the one absorber run for R-Q and R-star; E_emit printed on both) | reviewb (the long
   parent only, the absorber as a convergence control; R-an/R-loc read there with their character loss printed) | reviewa (the dilute chain and the
   absorber arm carry the ledgers; the long parent only for S(b|R)).
C3 R-Q: diagnostic (default: a labelled diagnostic, the t_0 shell charge and its R_Q scan printed) | reference (F3: read as the diabatic reference).
C4 the depleted arm's priority: side (default: one run at 30 L_H in stage 3 after stage 2, beside the native arm, no priority) | headline (F5:
   the depleted arm x record P is the headline criterion) | optional (A2/A12: only after the native audit is certified).
C5 adiabatic regime: diabatic_plus_Bad (default: the diabatic grid from the pilot plus the B-ad tuned sub-arm, 1024 r_s, 0.6 L_H, lambda ~ 3) |
   span (A5: finite-parent controls spanning lambda 0.1 to 10 from the pilot) | diabatic (F4: the diabatic regime only).
C6 Test B's late-interval verdict: both (default: "unresolved" and "dispersion effect against the spliced Unruh value" printed) | unresolved
   (A7) | dispersion (F11)."""
import argparse

CHOICES = {"c1": ("both", "rloc", "rstar"), "c2": ("both", "reviewb", "reviewa"), "c3": ("diagnostic", "reference"),
           "c4": ("side", "headline", "optional"), "c5": ("diabatic_plus_Bad", "span", "diabatic"), "c6": ("both", "unresolved", "dispersion")}
DEFAULTS = {k: v[0] for k, v in CHOICES.items()}


def add_args(ap):
    for k, ch in CHOICES.items():
        ap.add_argument("--" + k, choices=ch, default=DEFAULTS[k])
    return ap


def from_args(a):
    return {k: getattr(a, k) for k in CHOICES}


def arms_for(opt):
    """the arms and parents each option set runs (every side's arms are built; the option decides which run)."""
    out = {"reference_arms": ["R-an", "R-an(overlap)", "R-code(c_LZ=0.5,1,2)", "R-loc(R_Q scan, delta_c scan)", "R-star", "R-Q"],
           "declaration_label": {"both": None, "rloc": "R-loc", "rstar": "R-star"}[opt["c1"]]}
    out["parents"] = {"both": ["long unitary parent (C, flux, S(b|R))", "dilute Hermitian 512 r_s (eigenstate ledgers, before first return)",
                               "absorber run (R-Q, R-star)"],
                      "reviewb": ["long unitary parent (all readouts)", "absorber (convergence control only)"],
                      "reviewa": ["dilute Hermitian 512 r_s (ledgers)", "absorber run (complete drain; R-Q, R-star)", "long parent (S(b|R) only)"]}[opt["c2"]]
    out["RQ_status"] = opt["c3"]
    out["depleted"] = {"side": "one run, 30 L_H, stage 3, after stage 2", "headline": "headline criterion (depleted x record P)",
                       "optional": "after the native audit is certified"}[opt["c4"]]
    out["regime_grid"] = {"diabatic_plus_Bad": ["finite-parent diabatic grid from the pilot", "B-ad: 1024 r_s, 0.6 L_H, lambda ~ 3"],
                          "span": ["finite-parent controls spanning lambda 0.1 to 10 from the pilot"], "diabatic": ["diabatic grid only"]}[opt["c5"]]
    out["testB_late_verdicts"] = {"both": ["unresolved", "dispersion effect against the spliced Unruh value"], "unresolved": ["unresolved"],
                                  "dispersion": ["dispersion effect against the spliced Unruh value"]}[opt["c6"]]
    return out


def parser():
    return add_args(argparse.ArgumentParser())
