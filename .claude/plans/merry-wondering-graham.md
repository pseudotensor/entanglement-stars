# Plan: Eliminate bare shorthand tags; replace with descriptive names

## Context
The paper uses cryptic shorthand tags (IC1, IC2, IC3, B1, BJ, H0, H1, VGd, MI-iso, SCB, SmReg, Camp) at both definition sites and in running prose. The user wants **all** of these bare tags eliminated from the rendered PDF — both from definition-site headings and from every prose reference — and replaced with meaningful descriptive text.

## Tag-to-name mapping

| Tag | Parent | Descriptive name for references |
|-----|--------|-------------------------------|
| B1 | ass:backbone | "the MI mixing/isoperimetry clause of \assref{backbone}" or just \assref{backbone} |
| BJ | ass:backbone | "the vanishing-jump-range clause of \assref{backbone}" |
| IC1 | ass:isoreg | "the ellipticity input of \assref{isoreg}" / "the ellipticity condition" |
| IC2 | ass:isoreg | "the isotropy input of \assref{isoreg}" / "the isotropy condition" |
| IC3 | ass:isoreg | "the MI-regularity inputs (\assref{MSG} and \assref{CGVD})" |
| IC1--IC2 | ass:isoreg | "the nondegeneracy/isotropy inputs of \assref{isoreg}" |
| IC1--IC3 | ass:isoreg | "\assref{isoreg}" (the whole assumption) |
| H0 | def:MI-stationarity | "the bounded-degree condition" |
| H1 | def:MI-stationarity | "the stationarity/quasi-transitivity condition" |
| VGd | prop:MI-analytic-criterion | "uniform volume growth" / "the volume-growth hypothesis" |
| MI-iso | prop:MI-analytic-criterion | "ballwise MI isoperimetry" / "the MI-isoperimetry hypothesis" |
| SCB | prop:MSG_CGVD_implies_SCB | "the score-to-Campanato bridge (Proposition~\ref{...})" |
| Camp | ass:isoreg | "the Campanato estimate" |
| SmReg | ass:isoreg | "the derived regularity consequence" |

## Approach

### A. Definition-site heading changes (drop bare tags)

**sections/02_primitives.tex:**
- Line 208: `\textbf{(B1) MI mixing and ...}` → `\textbf{MI mixing and finite-observable MI isoperimetry.}`
- Line 210: `enumerate[label=(B1.\arabic*)]` → `enumerate[label=(\roman*)]`
- Line 220: `\textbf{(BJ) Vanishing jump range.}` → `\textbf{Vanishing jump range.}`
- Line 293: `enumerate[label=(IC\arabic*)]` → `enumerate[label=(\roman*)]`
- Line 314: `\textbf{(IC3) MI-based regularity inputs (MSG+CGVD).}` → `\textbf{MI-based regularity inputs (MSG+CGVD).}`
- Line 322: `\textbf{(Campanato estimate) Degree-$1$ Campanato (derived from \condref{IC3}).}` → `\textbf{Degree-$1$ Campanato estimate (derived from \assref{MSG} and \assref{CGVD}).}`
- Line 338: `\textbf{(SmReg) Derived consequence.}` → `\textbf{Derived regularity consequence.}`

**results/def_MI_stationarity.tex:**
- Line 24: `\item[\textbf{(H0)}]` → `\item[\textbf{(i)}]`
- Line 30: back-ref `\textbf{(H1)}` → `\textbf{(ii)}`
- Line 32: `\item[\textbf{(H1)}]` → `\item[\textbf{(ii)}]`

**results/prop_MI_analytic_criterion.tex:**
- Line 7: change `enumerate[label=(\textbf{\roman*})]` (keep roman numerals)
- Line 8: `\textbf{(VGd)} (Uniform polynomial volume growth.)` → `\textbf{Uniform polynomial volume growth.}`
- Line 12: `\textbf{(MI-iso)} (Ballwise MI isoperimetry.)` → `\textbf{Ballwise MI isoperimetry.}`

**appendices/appE_ic3_derivation.tex:**
- Line 3: section heading: `from IC3 (MSG+CGVD)` → `from the MI-regularity inputs (MSG+CGVD)`
- Line 143: `score-to-Campanato bridge (SCB)` → `score-to-Campanato bridge` (drop abbreviation)

**appendices/appG_toy_model.tex:**
- Line 242 (lemma title): `\textbf{(VGd)} and \textbf{(MI-iso)} on $\mathbb{Z}^d$` → `Volume growth and MI isoperimetry on $\mathbb{Z}^d$`
- Line 300: `\subsection{IC3 beyond...}` → `\subsection{MI-regularity inputs beyond...}`

### B. Remove `\condref` macro from ms.tex
Delete the `\condref` line added previously. It's no longer needed.

### C. Reference-site replacements (file by file)

**sections/01_intro.tex:**
- L14: `the isotropy input \condref{IC2}` → `the isotropy input (\assref{isoreg})`
- L19: `the full regularity selection \condref{IC1}--\condref{IC3}` → `the full regularity selection (\assref{isoreg})`
- L30: `from \condref{IC3} to the discrete Campanato estimate` → `from the MI-regularity inputs (\assref{MSG} and \assref{CGVD}) to the discrete Campanato estimate`

**sections/02_primitives.tex:**
- L212: `\condref{VGd} and \condref{MI-iso} of Proposition~\ref{...}` → `the volume-growth and MI-isoperimetry conditions of Proposition~\ref{prop:MI-analytic-criterion}`
- L315: `In addition to \condref{IC1}--\condref{IC2} we assume:` → `In addition to the ellipticity and isotropy conditions above, we assume:`
- L322: `(derived from \condref{IC3})` → `(derived from \assref{MSG} and \assref{CGVD})`
- L324: `\assref{backbone} together with \condref{IC1}--\condref{IC3}` → `\assref{backbone} together with \assref{isoreg}`
- L334 (display): `\condref{IC3}+Backbone → \condref{SCB} → Campanato` → `\text{MI-regularity}+\text{Backbone} → \text{score-to-Campanato bridge} → \text{Campanato estimate}`
- L339: `Under \condref{IC1}--\condref{IC2} together with` → `Under the ellipticity and isotropy conditions together with`
- L345: `\hyperref[cond:IC2]{IC2} separates` → `The isotropy condition separates`
- L346: `\hyperref[cond:IC2]{IC2} supplies` → `the isotropy condition supplies`
- L347: `from \condref{IC3}` → `from the MI-regularity inputs (\assref{MSG} and \assref{CGVD})`
- L403: `\condref{IC1}--\condref{IC2} in \assref{isoreg}` → `the nondegeneracy/isotropy inputs of \assref{isoreg}`

**sections/11_discussion.tex:**
- L36: rewrite the chain: `\condref{IC3} yields via chain \condref{IC3}+Backbone⇒\condref{SCB}⇒Campanato. packages \condref{IC1}--\condref{IC2} together with \condref{IC3}` → use descriptive: `the MI-regularity inputs yield it via the chain (MI-regularity)+Backbone⇒(score-to-Campanato bridge)⇒Campanato (Appendix~\ref{...}). \assref{isoreg} packages the nondegeneracy/isotropy conditions together with the MI-regularity inputs.`
- L61: `\condref{IC1}--\condref{IC2}` → `the nondegeneracy/isotropy inputs (\assref{isoreg})`

**results/thm_MI_backbone_package.tex:**
- L3: `\condref{B1} and ... \condref{BJ} clause` → `the MI mixing/isoperimetry and vanishing-jump-range clauses`
- L14: `the \hyperref[cond:BJ]{BJ} clause` → `the vanishing-jump-range clause`
- L18: `(Lindeberg/\hyperref[cond:BJ]{BJ})` → `(Lindeberg/vanishing-jump-range)`
- L19: `The \hyperref[cond:BJ]{BJ}/Lindeberg clause` → `The vanishing-jump-range/Lindeberg clause`
- L22: `input \condref{B1}` → `the MI mixing/isoperimetry input`

**results/prop_MI_analytic_criterion.tex:**
- L24: `\condref{BJ}` → `the vanishing-jump-range clause (\assref{backbone})`

**results/def_MI_stationarity.tex:**
- L30: `by \condref{H1}` → `by condition~(ii) below`

**appendices/appD_smoothness_selection.tex:**
- L3: `\condref{IC1}--\condref{IC2} and \condref{IC3} = MSG+CGVD` → `the nondegeneracy/isotropy inputs and the MI-regularity inputs (MSG+CGVD)`
- L19: `\condref{IC1}--\condref{IC2} ... \condref{IC3}` → `the nondegeneracy/isotropy conditions of \assref{isoreg} ... the MI-regularity inputs via Appendix~\ref{...}`

**appendices/appD0_isocamp.tex:**
- L26: `Assume \hyperref[cond:IC2]{IC2}` → `Assume the isotropy condition (\assref{isoreg})`
- L30: `the \hyperref[cond:IC2]{IC2} constants` → `the isotropy constants`
- L34: `\hyperref[cond:IC2]{IC2} controls` → `The isotropy condition controls`
- L61: `\hyperref[cond:IC1]{IC1}--\hyperref[cond:IC2]{IC2} and Mosco` → `The ellipticity/isotropy inputs and Mosco`

**appendices/appE_ic3_derivation.tex:**
- L8-9 (display): `\condref{IC3}+Backbone → \condref{SCB} → Campanato` → `\text{MI-regularity inputs}+\text{Backbone} → \text{score-to-Campanato bridge} → \text{Campanato estimate}`

**appendices/appG_toy_model.tex:**
- L10: `comprise \condref{IC3}` → `comprise the MI-regularity inputs`
- L93: `\condref{BJ} holds` → `the vanishing-jump-range condition (\assref{backbone}) holds`
- L116: `(IsoCamp and \condref{IC3})` → `(IsoCamp and MI-regularity)`
- L117: `the isotropy clause \condref{IC2}` → `the isotropy clause`
- L121: `This verifies \condref{IC1}` → `This verifies the ellipticity condition`
- L242-248: `\condref{VGd}` → `the volume-growth condition`, `\condref{MI-iso}` → `the MI-isoperimetry condition`
- L276: `\condref{IC2}` → `the isotropy condition`, etc.
- L281,285,286: `\hyperref[cond:IC2]{IC2}` → `the isotropy condition`
- L295: `\condref{IC2}` and `\condref{IC1}` → descriptive names

### D. Clean up labels
- Remove all `\label{cond:...}` and `\phantomsection\label{cond:...}` that are no longer targeted by any hyperref. (The definition-site bold headings no longer need hyperlink targets since references now point to the parent assumption/definition/proposition labels.)

## Files modified
1. `ms.tex` — remove `\condref` macro
2. `sections/01_intro.tex` — 3 reference replacements
3. `sections/02_primitives.tex` — ~7 heading changes + ~10 reference replacements
4. `sections/11_discussion.tex` — 2 reference replacements
5. `results/def_MI_stationarity.tex` — 2 heading changes + 1 reference replacement
6. `results/prop_MI_analytic_criterion.tex` — 2 heading changes + 1 reference replacement
7. `results/thm_MI_backbone_package.tex` — 5 reference replacements
8. `appendices/appD_smoothness_selection.tex` — 2 reference replacements
9. `appendices/appD0_isocamp.tex` — 4 reference replacements
10. `appendices/appE_ic3_derivation.tex` — 1 heading change + 1 reference replacement
11. `appendices/appG_toy_model.tex` — ~3 heading changes + ~12 reference replacements

## Verification
- `pdflatex ms.tex` twice
- `grep -c 'IC1\|IC2\|IC3\|(B1)\|(BJ)\|(H0)\|(H1)\|VGd\|MI-iso\|(SCB)\|(SmReg)' ms.log` should show zero undefined refs
- Grep the .tex source files for any surviving bare tags
- Spot-check PDF: no cryptic shorthand visible; all references use descriptive text
