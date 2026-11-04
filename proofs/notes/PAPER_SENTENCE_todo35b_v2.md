# TODO[35](b): paper-ready text for the derived two-window identity (I), v2 (reviewer A red-team fixes applied)

Target: fermion_new/06_scaling.tex (md5 2b2561ec, commit cf9ee964bd6). Text by reviewer B's merge of two independent drafts: todo35b/round2/astra_t35merge_result.md (9026f6e6); drafts reviewer A 842fa49b and reviewer B 4070692d. Every quoted sentence below was checked against the live tex. The merge's second proposed replacement ("The model fixes its temperature and area normalization but no laws of emission, storage, and release ...") is NOT in any live section file; it came from the old P3 input copy, so it is DROPPED.

## 1. Replace BY VERBATIM TEXT, not by line range (the sentence pair begins with "Shell" at the end of a line and runs two more lines)
> Shell storage includes exterior heavy matter and the fountain, reference response, field work, and moving-trapping-radius sweep. Its control through turnover therefore includes physical shedding.

with
> The shell remainder combines changes in heavy and light storage, including the fountain, with field work and reference transport; the moving-boundary sweep is accounted for by Eq.~\eqref{eq:page-two-window}. Control of the remainder through turnover therefore includes physical shedding.

## 2. Add, unmarked (a derived result: definition, one condition, zero remainder within the differentiable Hartree equations). Insert after line 276's "and net shell transport is not substituted for them.":
```latex
On each differentiable interval of a finite fixed-site, fixed-kernel Hartree history, consider any regular inward branch with a simple trapping radius \(R(T)<c\), \(c\) fixed, using the full degree-one reference of Eq.~\eqref{eq:dm-reference} consistently in both windows.
With \(u=(N,v,b)\), \(s_{\rm tot}\) the copy-weighted site sources and dots denoting \(d/dT\), set \(E_R=\mathcal M(R(T),T)=R/(2G)\) and \(\mathcal U=\int_R^cN\mathcal M'\,dr\).
Let \(W_{\rm sh}=\langle W_\ell^\dagger(\mathbf1_{R<r<c}\partial_Tu),s_{\rm tot}\rangle\), differentiating the fields, not the cutoff.
Subtracting the windowed balances and applying Leibniz's rule cancels \(\beta_R\), giving
\begin{equation}
\boxed{\begin{aligned}
N_R\dot E_R&=-L_K(c,T)-\dot{\mathcal U}+W_{\rm sh},\\
L_K(c,T)&=-N_c\partial_T\mathcal M(c,T)=\Xi_c^{\rm ref}-\Xi_c .
\end{aligned}}\label{eq:page-two-window}
\end{equation}
Euler's identity gives \(\Xi_c^{\rm ref}=-\langle\bar u_{>c},\dot r_{\rm tot}\rangle\), with \(\bar u_{>c}=W_\ell^\dagger(\mathbf1_{r>c}u)\); this exterior pairing stays in \(L_K\) beside the propagated sea's contribution to \(-\Xi_c\), which it partly cancels.
The identity has zero remainder within the differentiable Hartree equations; finite-step increment, source and solve defects remain in \(e_{r,j}\). Heavy matter is already included; neither received-light matching nor family-uniform smallness follows.
```

## 3. Replace the TODO[35] comment (line 278) with the visible mark:
```latex
\todo{(i) Bound the heavy-matter part of \(\mathfrak d_c\), of order
\(E_0\) on the computed object and without a family bound: measure
heavy shell storage, signed transport and the total variation of
\(N_R^{-1}\) on matched-clock mass, size and preparation ladders,
with width and step errors; derive thin-shell continuation giving
\(\sup|\mathfrak d_c|=o(E_0)\) through turnover while retaining the
combined work, reference and output terms.
(ii) Derive the assumed \(L=\mathcal L/R^2\) through the
two-field re-evolution under Eq.~\eqref{eq:dm-reference} and the
post-window supply calculation, independently measuring
\(du/dT\) and once-counted received-light energy.
(iii) Establish C1/C2 release matching, outside energy bookkeeping:
two independently re-propagated pure covariances and partner
accounting for no-return conditional and residual retained entropies,
with return and omitted-mode errors; realize an energy-funded,
charge-matched channel giving complete release and coherent
heavy-input recovery with uniform error against the minimum law.}
```


Regime label: a finite history of the fixed-site, fixed-kernel Hartree equations; any regular inward branch with a simple trapping radius; c fixed. The one condition: differentiability on the interval, i.e. the full degree-one reference used consistently in both windows. Family-uniform smallness and received-light matching are NOT claimed; they are the mark's items (i)-(iii).

## Red-team fixes applied in v2 (reviewer A t35redF, ee5fcf32: physics PASSES all five checks; integration and wording defects fixed)
1. Apply section 1 by verbatim text. The sentence pair starts with "Shell" at a line end; a line-range edit would delete "Its control through ... not substituted for them." The insertion anchor is the line ending "... not substituted for them."; the TODO comment is two lines below it.
2. The replacement now also covers the following sentence, so its antecedent stays unambiguous.
3. \mathcal S is renamed \mathcal U throughout, because \mathcal S[u] is already the spin term of eq:dr-dirac-schwinger. \tag{I} is removed so the equation takes the paper's sequential number (the label is kept). "cancellation partner" becomes "which it partly cancels", because no exact cancellation is established.
4. In the mark, (i) names the heavy-matter part of the shell remainder instead of the undefined eps_H, and (ii) uses L = calL/R^2 and drops the plan name "PAGE VERIFY".
5. After landing: confirm the compiled PDF stays at or under 29 pages. If the later live mark "a trapping radius clear of heavy matter with the shell remainder bounded, ..." still stands, drop its "clear of heavy matter" clause (superseded by the identity, which includes heavy matter) and keep its other items.
