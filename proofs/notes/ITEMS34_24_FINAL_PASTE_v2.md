# Items 34 and 24: FINAL agreed paste set v2, filled (for paperapi via the lead)

Assembled 2026-11-04T00:50Z by the route2 lane manager, with no new wording. Sources:
- reviewer A final34F (recvnull/api/FableAPI_final34F_v1.md 52dda448): all blocks except the held-comparison paragraph.
- reviewer B final34A2 (recvnull/api/AstraAPI_final34A2_v1.md 440eef81): the held-comparison paragraph, verbatim. reviewer B accepted reviewer A's three declines and confirmed the rest as final.
- reviewer A fluc2F (recvnull/api/FableAPI_fluc2F_v1.md 737cedf7): accepts that the checkpoint outcome prints as a defined fluctuation with no threshold, which makes reviewer B's held paragraph the agreed one. reviewer A flucF had called it "theirs minus the checkpoint sentence". The paragraph fluc2F returned is the hold-drain paragraph of Sec. page, not this one, and is NOT used.

FILLED in v2 from ITEM24_PASTE_FILL_v1.json 37140074 (script recvnull_A_pastefill_v1.py 106645b7 on A v1.5 data json 6a05eb7d, ITEM24_A_RESULT_v1.5 0e52ea13): x = 1.460638, 7.827908, 5.896494 pct (print 1.5, 7.8, 5.9), e = 0.525937 pp (print 0.6, rounded up), s = 5.86181e-5 (print 6e-5), f_140 = +23.935 pct (print 24), f_300 = +38.522 pct (print 39). Check x + e < 10: 2.06, 8.43, 6.50. The remaining bracketed slots ([x], [y], [u_1], [u_2], [t_1], [t_2], [z], [value], [bands...]) belong to item 34 and the coupled PASS and FAIL templates, which print only when their calculations land. This version is ready for paperapi.

## Paste set

### Item 34. NOW (prints)

Replaces "The received-time map and once-counted output term are \todo{received-output matching}. Net shell transport is not substituted for them." in the shell accounting paragraph of Sec.~\ref{sec:page-effective}.

The once-counted shell output is the light field's outgoing-branch Killing-energy current through the fixed shell \(c\), integrated over the slice clock with each crossing's energy counted once. It is a surface current, not the energy held in a set of output modes and not the net shell transport. The received-time map assigns shell crossings to receiver intervals in retarded time \(u\), with lattice dispersion retained. Inward crossings, repeated crossings, current outside any reported band, and cross terms between branches stay inside this matching rather than in \(\mathfrak d_c\). Formation light is hard (Sec.~\ref{sec:page}), so matching uses the full band and reports the Hawking-scale share separately. Agreement with the independently read received energy \(\int L\,du\) is \todo{two independent light-field re-propagations on the same stored back-averaged fields and preparation, with separately constructed outgoing-branch currents through \(c\), full-band matching on every declared receiver interval, each interval declared to carry received energy above its combined error bound, at a tolerance of 10 percent of that energy, with separate bounds on propagation, clock conversion, branch leakage, repeated crossings and cross terms between branches, and the spread between the two constructions reported beside those bounds}. Net shell transport is not substituted for the once-counted output or the received-time map.

### Item 34. PASS (held)

Replaces the marked sentence when both constructions land.

The mapped once-counted shell output agrees with the independently read received energy \(\int L\,du\) within [x] percent of that energy on every declared receiver interval, \(u/r_s=[u_1]\) to \([u_2]\), in both independent full-band constructions. This bound meets the 10 percent tolerance and includes branch leakage bounded by [y] percent of the same energy. The result identifies the received term on these intervals at this size, formation light included, and establishes neither sustained emission nor escape to infinity.

### Item 34. FAIL (held)

In both independent full-band constructions, the mapped once-counted shell output differs from \(\int L\,du\) by [x] percent of that energy on the declared receiver interval \(u/r_s=[u_1]\) to \([u_2]\), the mismatch lying in [bands and receiver subintervals]. Net of the combined [y] percent error bound, the difference exceeds the 10 percent tolerance. The once-counted shell output therefore does not identify the received term on this interval at this size. The measured difference remains a contribution to \(\mathfrak d_c\), and net shell transport is not substituted for the received term.

### Item 24(a). Assumed-laws clause, inside the "Third" list

Selection follows the coupled outcome, not the held one.

**NOW (prints):** \(dE_R/du=-L\) at the trapping radius, assumed for the coupled object and tested by the shell balance below

**PASS (held):** \(dE_R/du=-L\) at the trapping radius, verified at Hartree order on the declared intervals at this size and still assumed through turnover (the shell balance below)

**FAIL (held):** \(dE_R/du=-L\) at the trapping radius, assumed although not realized on the declared coupled intervals at this size (the shell balance below)

### Item 24(b). Analytical paragraph (prints)

New paragraph after "Net shell transport is not substituted for the once-counted output or the received-time map."

On stationary fields on which the light field's source tensor \(T_L\) is covariantly conserved, the light field's generator energy inside a fixed sphere, the interior sum of \(N\rho+v\cdot J\), changes by minus its net outward Killing-energy current. At the trapping radius, \(\partial_t=N_R(n+\hat r)\), with \(n\) the slice normal and \(\hat r\) the outward radial unit vector. If this net current through \(R\) equals the received luminosity \(L\) per unit of the same Killing time, then \(T_L(\partial_t,\partial_t)|_R=-L/(4\pi R^2)\). Equation~\eqref{eq:C-rate} then gives the light field's contribution to \(\Xi_R\) as \(4\pi N_R^2R^2(\rho+p_r-2J)_L=-L\), \(p_r\) its radial kinetic stress. Setting the heavy and reference tangents to zero gives \(N_R\beta_R\dot E_R=-L\). On the vacuum background of Eq.~\eqref{eq:dr-exterior}, \(N_R=\beta_R=1\) and a static receiver's retarded time advances with \(t\), so this light-only tangent is \(dE_R/du=-L\). This is a tangent comparison on held fields, not motion of the held trapping radius and not an emission law.

### Held comparison, PASS form (prints now, one calculation, mark kept, reviewer B final34A2 text, agreed)

New paragraph immediately after the analytical paragraph.

The held test of this tangent uses the \(k=1\) light chain of Sec.~\ref{sec:page}, its fields held from \(T/r_s=100\), and reads the balance at the held fields' fine-grid outer trapping radius \(R\). Let \(\Xi_{R,L}\) be the light contribution to the windowed state tangent \(\Xi_R\) defined above, \(L_R\) the outward-positive net light current through \(R\), both per unit of the slice clock \(T\), and an overbar an interval mean over one declared epoch. At the three declared epochs, \(T/r_s=140\) to 167.7, 300 to 327.7 and 470 to 497.7, \(|\overline{\Xi_{R,L}}+\overline{L_R}|\) is 1.5, 7.8 and 5.9 percent of that epoch's \(|\overline{L_R}|\). Halving the spacing of the sampled instants within each epoch moves the three values by at most 0.6 percentage points, and each error-inclusive bound lies within the declared tolerance of 10 percent. The same balance read at the \(16r_s\) shell, scaled by the magnitude of that shell's mean current, closes to \(6\times10^{-5}\) or better. The three-instant checkpoint means of \(\Xi_{R,L}+L_R\) at \(10r_s\) spacing differ from the corresponding full-epoch means by 24 and 39 percent of the respective \(|\overline{L_R}|\) in the first two epochs. These readings come from one calculation \todo{a second, independently built calculation of the three epoch means of the held light tangent and of the net current through \(R\) on the same held history, read at two sampling spacings}. This verifies the held window-to-current identification, not coupled depletion and not a finite massless reference.

### Coupled NOW (prints) and consequential mark (prints)

After the five-slice paragraph's last sentence, replacing its closing mark.

The computed trapping radius forms inside matter (Sec.~\ref{sec:star}), and the constant-coefficient massless-chain prescription supplies no finite endpoint (Sec.~\ref{sec:results-lattice}). The coupled depletion law is \todo{derive and validate a finite massless reference consistent with Eq.~\eqref{eq:dm-reference}, then independently re-evolve both fields with full copy weights from one preparation that keeps heavy-particle support clear of the outer trapping radius's averaging region on the declared intervals, bounding the magnitude of the full heavy state tangent below 10 percent of the magnitude of the net light current through \(R\) per unit of the same slice clock. Measure \(N_R\), \(\beta_R\), \(\dot E_R\), the reference tangent and the received-time map. Close the trapping-radius and shell ledgers and test \(\dot E_R/(du/dT)=-L\) directly against the independently read full-band receiver luminosity on declared intervals whose luminosity is resolved above its own error bound, with storage, field work, branch, lattice-transport, trapping-radius motion and between-slice errors in one budget of 10 percent of that luminosity}.

\todo{the coupled re-evolution's shell ledger and its matched thin-shell continuation through turnover, bounding \(\mathfrak d_c\) and other-clock reference terms, and the charge-matched partner-accounting channel for energy-funded complete release and uniform decoding and response-error bounds}

### Coupled PASS (held)

Replaces the coupled paragraph's second sentence when both re-evolutions land. The consequential mark stays.

Two independent coupled re-evolutions from one preparation with heavy matter clear of the outer trapping radius's averaging region, with full copy weights and the validated finite massless reference consistent with Eq.~\eqref{eq:dm-reference}, give \(\dot E_R/(du/dT)=-L\) within [x] percent of the received luminosity on every declared interval, \(T/r_s=[t_1]\) to \([t_2]\). The bound meets the 10 percent tolerance, and the measured \(N_R\beta_R\), the clock conversion \(du/dT\), the heavy and reference tangents, and the trapping-radius and shell remainders close the ledger inside it. This establishes the depletion law at Hartree order at this size on these intervals, not a continuum null-source measurement and not a scaling-uniform depletion law.

### Coupled FAIL (held)

Two independent coupled re-evolutions from one preparation with heavy matter clear of the outer trapping radius's averaging region, with full copy weights and the validated finite massless reference consistent with Eq.~\eqref{eq:dm-reference}, give a discrepancy \(\dot E_R/(du/dT)+L\) of [x] percent of the received luminosity over \(T/r_s=[t_1]\) to \([t_2]\). Net of the combined [y] percent error bound, it exceeds the 10 percent tolerance. The corresponding change of \(\mathfrak d_c\) is [value], with its measured contributions and its unassigned residual beside it. The identity \(N_R\beta_R\dot E_R=\Xi_R-\Xi_R^{\rm ref}\) stands, but the assumed depletion law is not realized on these intervals at this size.

### Consequential edits (prints)

The subsection's last sentence ends "and the computed checks above test the budget's pieces on held fields and stored slices while establishing neither the store nor the release."

In Sec.~\ref{sec:page}, "the held vacuum at stored slice 800" reads "the held vacuum at \(T/r_s=100\)", the same record under rule (C).

## Decision rule as applied, not paper text

Scales: item 34 uses each interval's received energy, the held check uses \(|\overline{L_R}|\) at that epoch, the coupled check uses the independently read received luminosity.

Held criterion: \(|\overline{\Xi_{R,L}}+\overline{L_R}|\le0.1|\overline{L_R}|\) at each epoch separately, no pooling, the spacing stability its error, the checkpoint means and the reading at \(c\) controls and not criteria. Applied to v1.4 the error-inclusive bounds are 3.5, 9.6 and 7.6 percent, all within tolerance, and the reading at \(c\) closes at \(6\times10^{-5}\), so the outcome is PASS under this rule and under the straddle rule alike. The checkpoint fluctuation is recorded and not printed, i24resF stands, and the pending ruling changes nothing unless it adds a criterion, which the rule fixed in advance excludes.

PASS for the remaining checks: every declared interval has an error-inclusive upper discrepancy bound within tolerance. A pass in one calculation prints the PASS text in its one-calculation form with a mark naming the second calculation, two agreeing calculations close the mark, and a second calculation beyond tolerance returns the item to the lead with both tables in the repository. FAIL: a discrepancy beyond tolerance after subtracting the combined error bound in both calculations. Unresolved: every other case, including any enclosure straddling the tolerance, keeping the NOW mark and the interval in the declared set. The 10 percent is the total agreement bound. FAIL texts carry no mark. Held PASS does not establish coupled PASS. The sign of \(\overline{L_R}\) is a separate reading and not part of any criterion here.

## Repository and status file

With the v1.4 table: \(R=36.307a\), \(c=1024a\), record 800, the main and half sampling spacings, the checkpoint means at 140 and 300 with the reading that a three-instant mean does not represent the interval mean at \(10r_s\), the product form, the \(R\pm2\sigma\) readings, and the signed epoch means of \(L_R\). Status file: the held PASS mark, its sentence, its vehicle (the second calculation at two sampling spacings), its lane. Brief for the second calculation: read interval means of the held light tangent and of the net current through \(R\) at two spacings on the same held history.