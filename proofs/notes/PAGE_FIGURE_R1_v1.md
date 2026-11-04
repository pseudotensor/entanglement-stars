# PAGE FIGURE route 1

[R] Re-derived analytic constructions. [X] Independent first-order/scalar scattering routes and independent author/peer derivations. [N] Numerical comparator results with controls. These tags apply to the stated comparator and finite-LP results; physical identification remains UNDECIDED.

The signed comparator calculation passes the required numerical tests after the prescribed positive-quadrature refinement. The 960/1920-node envelope change is 7.077866614357831e-6; the exact finite-certificate comparison bound is approximately 7.077866614405038e-6, below 1e-5. The same author and sole independent peer have accepted the numerical evidence under the explicit review coverage below. Final vector/CSV checks and visual review also pass. Physical identification remains UNDECIDED.

The governing work order is the signed FIGURE SPEC, §3 A–J of WRITE_PAGE_SIGNED_v1.md, with owner #215. The calculation implements that comparator without a circuit, sampling, a fit, smoothing, or a full evolution simulation. The entropy panel is the signed leading law. The correlation panel additionally assumes independent pure fixed-number pairs in an ideally decoded representation; ideal charge- and number-matched encoding, decoding and release; occupation-preserving release with particle-hole relabelling of the second half; the specified comparator spectrum throughout the formal evolution; untouched escaped modes; fractional continuum inventories with negligible transfer delay; and the product of marginals as comparison state with the specified exchange probes. The admissible range is not a physical error band, and the open endpoint is a formal end, not a certified completion time.

## Comparator and scattering evidence

The dimensionless frequency is q=omega R in [0.001,3]. Channels k=1,…,8 carry multiplicity 64k for one massless four-component field with eight valleys. This comparator is not identified with the physical moving-light output. For rho>1, x=rho+ln(rho−1), V=k sqrt(1−1/rho)/rho and

    d psi/dx = (i q sigma_3 − V sigma_1) psi.

The outgoing exterior Jost column O approaches exp(iqx)(1,0), and the horizon incoming column H approaches exp(−iqx)(0,1). With H_+=sigma_1 conjugate(H), A=det(O,H), B=det(H_+,O), the analytically normalized current relation is |A|²−|B|²=1 and Gamma=|A|⁻². The independent scalar route evolves phi=psi_1+psi_2 through phi_xx+[q²−V²+V_x]phi=0; its Wronskian is −2iq A. Its initial scalar values and derivatives are reconstructed from the same asymptotic columns, but its differential equation is independently integrated.

The finite endpoints are controlled by asymptotic series with explicit residual-to-solution estimates, not by a local potential ratio. At the horizon the factored variable is t=sqrt(rho−1), with analytic current normalization, and the perturbation growth is bounded by exp(2k asinh(t_H)). The outer series uses y=1/rho and a diagonalizing boost theta=atanh(V/q); its integrated tail residual is bounded with the finite boost factor. Exact recurrence cancellations are justified before interval evaluation: outer residual powers through N vanish; the two horizon residual rows begin respectively at powers 2N+2 and 2N+1. The retained coefficient residual and binomial tails are bounded separately. Directed interval arithmetic also includes actual represented chart launch, scalar launch, coordinate displacement and normalization errors, then converts both column errors into a relative transmission bound. The proof is an endpoint uncertainty certificate. Differential-equation and quadrature convergence are reported as numerical diagnostics, not global interval proofs.

Explicitly, if R_H(t) is the exact factored horizon-polynomial residual including its binomial-tail majorant, the analytic column error is bounded by exp(2k asinh(t_H)) times the integral from 0 to t_H of ||R_H(t)|| dt. If the exterior factored residual is bounded by B rho^(-N-1), its column error is at most exp(theta_b) B rho_b^(-N)/[N(1−1/rho_b)], with rho_b>3/2 and V(rho_b)/q<1. These integrals/majorants are evaluated outward, not inferred from the first omitted coefficient alone. Let epsilon_H and epsilon_O include those analytic errors and the separately enclosed actual launch/coordinate/normalization differences. Conservative normalized-basis factors give delta_H=2 exp(2k asinh(t_H)) epsilon_H and delta_O=2 exp(theta_b) epsilon_O, enlarged by the explicit exp(K|delta u|) coordinate-displacement factor where needed. Then E=delta_H+delta_O+delta_H delta_O bounds |A_end/A−1|. For E<1 the relative-Gamma gate uses max(2E+E²,(1−E)^(-2)−1), covering either relative normalization. The actual stable-B calculation also includes its represented horizon-current defect and final matching-rounding bound multiplicatively; the scalar Wronskian has its separately bounded launch conversion. Every used nonzero transmission meets the stronger saved 1e−10 endpoint gate.

Tiny transmissions use canonical current-preserving variables and a stable B-based logGamma, with an independent A-based consistency diagnostic. Matching is evaluated at high precision from the actual stored coordinates. The twenty specified frequencies were checked with both differential forms, separate horizon and exterior extensions and joint extension, plus high-precision base and refined anchors. Raw high-precision currents, rather than subtraction of enormous double-precision intensities, provide the independent current check. No Gamma or occupation is clipped.

The final primary quadrature comparison has 960 and 1920 positive-weight nodes per channel; the independently propagated scalar quadrature has 400 and 800. The respective total node counts are 7,680, 15,360, 3,200 and 6,400. The earlier 480-node primary rule remains preserved as intermediate evidence. The two panel subdivisions differ. All nodes lie strictly inside the prescribed integration interval. The stable entropy and logarithmic mean are evaluated even for tiny positive occupations. All four spectra retain positive weights, n, s and W; no positive node is discarded.

The primary fine results, printed to twelve significant digits, are:

| Quantity | Numerical estimate |
|---|---:|
| C0 | 0.00523731547127 |
| D0 | 0.107876529936 |
| calR=D0/(4 pi C0) | 1.63911101758 |
| T_W | 0.00263960826448 |

The defining finite-band measure is dnu=sum_k [64k/(2 pi)]dq, with n=Gamma/(exp(4 pi q)+1), s(n)=−n ln n−(1−n)ln(1−n), C0=<qn>_nu, D0=<s>_nu and T_W=<W>_nu. No out-of-band extrapolation is included.

The unrounded saved values, not these display decimals, enter every marker, inventory and plot. Relative differences from the SPEC reproduction values are respectively +5.1774722567626915e−11, +4.052134172471447e−12 and −1.3407592686208896e−11 for C0, D0 and calR. All are below 1e−6. The primary doubled-quadrature relative changes in C0,D0,T_W,calR are 1.2002690220590762e−16, 5.574469539154976e−16, 1.5573298582732805e−15 and 4.3742005170958993e−16. The scalar changes are 1.5315830328537853e−13, 2.9840598168712824e−13, 3.8848729437204557e−13 and 1.4524767840177192e−13. The largest plotted-marker time changes are below 1.216910e−16 for the primary comparison and 4.040815e−14 for the scalar comparison, below 1e−6. These are convergence diagnostics; the number of displayed digits is not a claim of twelve rigorously certified significant digits.

The maximum saved fine endpoint-relative uncertainty is 1.2573952451634428e−12 for the primary spectrum and 1.2876380939051403e−12 for the scalar spectrum. Both are below the requested 1e−8. The twenty-anchor maximum independent endpoint-extension change is 6.982303624170072e−12. The largest raw high-precision current residual is 3.503739422237126454124358740784762994557632524946478213084460309362488155810611896436811749747911394e−110. The largest double-versus-high-precision anchor discrepancy is 1.0062252164007759e−11. Conservation alone does not certify a transmission; these checks are used together with endpoint bounds, independent differential forms and refinement.

Nine scalar spectral values exceed one by roundoff, with maximum 1.0000000000000036. Their complete records are preserved and no clipping is applied. The signed independent-route/current tests still pass. Scalar zero placeholders for diagnostics defined only for the primary canonical chart are not additional measured zero residuals.

## Pair observable and inventory construction

The probe coefficients K are held fixed in the comparison-state variation. The product state is rho_d tensor rho_X; the complete pure pair is not substituted for it. On its four-dimensional Fock space the spectral logarithmic mean yields

    W(n)=2 n(1−n) Lambda(n²,(1−n)²)
        =n(1−n)(2n−1)/ln[n/(1−n)].

At n=1/2 the value is 1/8; at n=0 or 1 the spectral limit is zero. Direct four-state integration at n=1/2,0.1,0.001,1e−12, with 64 and 128 positive quadrature nodes, agrees with this expression below the required 1e−10 relative tolerance. The saved maximum 64/128 discrepancy is 8.94852238856e−80; the 128-node versus closed-form differences are below 8e−180 in the chosen arithmetic. These high-precision numerical controls are not a claim about microscopic physical errors.

| Occupation n | Four-state weight (display) |
|---:|---:|
| 1/2 | 0.125 |
| 0.1 | 0.0327686121585661461701 |
| 0.001 | 0.000144351729857576272225 |
| 1e−12 | 3.61912068251637218205698e−14 |

These are displays of the unrounded saved integration results; the numerical comparisons use the unrounded values.

The complete pure pair has zero centred canonical covariance for this probe. In the product state its mean is zero and its ordinary variance is n(1−n)[n²+(1−n)²], which generally differs from W. The number covariance is −n(1−n); dephasing preserves that number covariance but eliminates the cross block and the response. Particle-hole symmetry, singular endpoints and the limit W/s→1/(8 ln2) at half occupation were separately checked. No general interacting-state factorization is asserted.

For the correlation panel f=1, P=calR/(calR+1), D=calR−P, beta=1/calR and r0=(1+beta)/2. Every positive spectral label obeys the inventory law I_z=1−2theta, I>=0, I(0)=0 and I(calR)=0, with entropy-weighted mean F_f. Before P, theta=0. The uniform admissible history has I=F_f and theta=r0 after P. Its finite-interval reference is r0 times the post-Page overlap fraction, with the complementary retained share.

For each averaging interval, the two-inventory polygon and the two entropy-weighted equalities are precisely SPEC G. In normalized coordinates u=A/calR, v=B/calR, p_i=w_i s_i/D0 and t_i=w_i W_i/T_W, the objective is the corresponding constant plus sum_i t_i(u_i−v_i). The initial numerical primal and dual programs are independent. The final refined calculation constructs feasible primal inventories and independently verifies original-polygon dual supports exactly, as detailed below. A finite solver vector, including one exposed with a nonoptimal status, is only a candidate: exact rational polygon projection, an affine repair of both means, independent full-support dual support evaluation and the unchanged normalized residual/gap checks determine acceptance. No status is promoted to an optimality certificate. Any numerical restriction of the primal tiny-weight variables retains their explicitly fixed contribution; the dual certificate includes every positive node, including nodes outside the primal working set. If its full gap does not pass, the prescribed restriction is relaxed. There is no silent deletion of small occupations or response weights.

For sign sigma=±1 the exact signed normalized objective is J_sigma=sum_i sigma t_i(u_i−v_i). The independently evaluated original dual is lambda_A m_A+lambda_B m_B+sum_i max_(u,v in G)[(sigma t_i−p_i lambda_A)u+(−sigma t_i−p_i lambda_B)v], where G is the full original seven-inequality polygon and m_A=(P−beta a)/calR, m_B=(P−beta b)/calR. The physical unnormalized objective is O=calR T_W J_+. Accordingly an exact signed dual gap g gives the requested normalized gap calR*g/(2 Delta z); the entropy-mean residual is normalized by D0 calR and inventory violation by calR. The seven fine report-centre certificates have zero exact primal entropy residual, zero exact inventory violation and zero exact normalized primal-dual gap for each sign (the first centre is the analytic zero-overlap branch). No optimization residual is inferred merely from solver status.

The repaired inventories produce a constructive admissible history through (0,P),(a,A),(b,B),(D,0). Exact checks merge coincident knots, enforce nonnegativity and slopes in [−1,1], and verify the required entropy-weighted mean on each segment. Different envelope ordinates may use different histories. For the whole post-Page interval both extrema equal r0, independently of spectral allocation. The uniform reference itself satisfies every inventory constraint; this is stronger than merely checking that a line lies between two plotted bounds.

## Evidence and limitations retained in the final report

Pair, anchor and spectral evidence have been accepted by the author and the same independent peer. The anchor review includes every projected solution, launch and aggregate certificate and all high-precision backend operands; lengthy recurrence coefficient arrays remain immutable on the box with length/hash references, without a literal-digit-read claim. The authorized spectral review comprises every validation, all constants/settings and channel extrema, and all 291 complete boundary, requested-frequency, diagnostic-maximizer and exceptional witnesses; it is not a claim to have literally read all 21,120 raw node records. The source and exact saved projections were wholly reviewed before use. Reversible lossless views preserve ordered JSON and every serialized decimal string.

Numerical LP attempts and their exposed status, solution, model and option records are preserved, including unsuccessful attempts. SciPy discarded some vectors for a nonoptimal status; later native returns included vectors inconsistent with their saved row values and independently computed original residuals. Those candidates were rejected. No internal solver cause is inferred. The implementation added explicitly identified candidate paths with unchanged physical constraints and exact acceptance gates: native interior-point recovery, redundant exact coordinate bounds, reversible binary objective scaling, and an exact all-positive-label feasible allocation. Accepted cases were reused; scattering was not repeated.

For the remaining near-face centres, three exact feasible candidates—uniform, fixed A and fixed B—were checked against every original polygon constraint, both entropy-weighted means and the complete reconstructed histories. Their best objective was compared with independently constructed ORIGINAL dual certificates: zero multipliers and same-level, same-sign exact-face multipliers lifted by coordinate labels, with every support function recomputed over the target polygon. These are original finite-LP certificates, not substituted one-dimensional outer bounds. The staged 24 cases, 48 signs, all pass; their largest displayed normalized gap is 7.488526180468392e-13. Those intermediate records remain preserved; their partially completed bulk literal review is not represented as complete acceptance. The superseding final evidence contract covers the successful 960/1920 calculation, including exact joins to previously validated retained records. Long repeated weight, rank and epigraph arrays remain immutable and hash-referenced; no literal reading of every array digit is claimed.

Only after complete exact replacement coverage was accepted were the superseded owned numerical calls interrupted. Their process identities and extant attempts were retained. The interruption has no returned vector or solver verdict; no PID-to-current-case mapping was invented. Promotion preserves the staged certificate and NPZ bytes. That intermediate inventory assembly was explicitly cache-only, with a guard that would reject any missing case before calculation. It completed assembly and produced the retained failed 480/960 convergence comparison. The subsequent 960/1920 calculation uses the exact original-LP chain construction described below; its complete final certificate projection has passed and both scientific reviewers have accepted the selected evidence; final visual acceptance is reported separately.

Physical identification of the comparator response is UNDECIDED: no bounds are supplied here on the full coherent encoding/decoding, charge matching, compression, failure records and purifiers, cross blocks, comparison-state errors or integrated numerator/denominator errors. The SPEC ratio bound (eta_N+eta_T)/(1−eta_T) requires eta_T<1 and a positive T0. The computed T_W is positive for the comparator; it is not such a microscopic matching certificate. Consequently the admissible shading is not an error band and the formal endpoint is not a completion claim.



## Twenty independent scattering controls

The following are the saved refined high-precision first-order and reconstructed scalar results, displayed to 16 significant figures. Their independent values agree at this displayed precision; the unrounded values and their differences are retained in the anchor evidence. The two extension columns are the largest relative change among horizon-only, outer-only and simultaneous endpoint extensions for the corresponding double-precision differential route. These are numerical diagnostics, not a claim of 100 certified digits. The separately bounded endpoint-vector perturbations, actual launch-chart conversions and current tests are described above.

| k | q | Gamma, first order | Gamma, scalar | Largest first-order endpoint change | Largest scalar endpoint change | First-order raw current maximum | Scalar raw current maximum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .01 | 2.587563099824363e-5 | 2.587563099824363e-5 | 1.151302e-13 | 7.158719e-13 | 2.513017097416537218620258561275399778001e-116 | 1.374985857474533831831464831069307276724e-115 |
| 1 | .1 | .004344437615828307 | .004344437615828307 | 6.805668e-14 | 8.164581e-13 | 2.394612613339096409264454596899466825853e-116 | 1.467140160789345065905181933948066209466e-115 |
| 1 | .3 | .2117496882560008 | .2117496882560008 | 1.376677e-13 | 1.154632e-13 | 1.828832569704851056754246330637066803893e-116 | 1.046121956048836839556378508434776084698e-116 |
| 1 | 1 | .9998904979787673 | .9998904979787673 | 0 | 1.332268e-15 | 7.637653028711221988369799637028156260127e-116 | 2.461425646572050212165002932913303671578e-116 |
| 1 | 3 | .9999999999999996 | .9999999999999996 | 0 | 2.220447e-15 | 9.96562634198882530954060326933951566566e-118 | 3.599729301536018297868103397387908285726e-118 |
| 2 | .01 | 1.796051503120057e-11 | 1.796051503120057e-11 | 7.089885e-13 | 6.750156e-14 | 2.682574044863510633227152610138546239549e-115 | 8.853777218252646262783556102321976187509e-115 |
| 2 | .1 | 2.958567267182998e-7 | 2.958567267182998e-7 | 1.849632e-13 | 4.138912e-13 | 2.903604575551951587491165540152580264668e-115 | 1.598413707389554774019939932319839739397e-116 |
| 2 | .3 | 1.495524381037809e-4 | 1.495524381037809e-4 | 9.229285e-13 | 6.394885e-14 | 2.903875215773874120644239608178028963742e-115 | 8.536032012018512630358182711149222061452e-117 |
| 2 | 1 | .9754510822580519 | .9754510822580519 | 2.375878e-14 | 3.996803e-15 | 7.614563080673758269674415378728789336176e-117 | 4.450091825903115893279498999213827509762e-117 |
| 2 | 3 | .9999999999999782 | .9999999999999782 | 0 | 1.332268e-15 | 1.809124986404010315634619697615902601995e-117 | 2.080079238851586141178782061396236698008e-118 |
| 4 | .01 | 5.725777028098586e-25 | 5.725777028098586e-25 | 3.895773e-13 | 7.327472e-15 | 2.79505845385703785804536951288249536713e-110 | 3.503739422237126454124358740784762994558e-110 |
| 4 | .1 | 9.306557242750001e-17 | 9.306557242750001e-17 | 2.733370e-13 | 1.421086e-14 | 2.713844832632435480041852357166448193612e-110 | 8.305786664356571598638568179163685180248e-112 |
| 4 | .3 | 3.578830277911743e-12 | 3.578830277911743e-12 | 1.265655e-13 | 2.486900e-14 | 1.072082272258762441845438285394189737326e-113 | 7.541813313721410465900119219623252193512e-113 |
| 4 | 1 | 8.594981420361017e-5 | 8.594981420361017e-5 | 1.728951e-12 | 2.153833e-14 | 1.414065267450340566197093619075779781712e-116 | 3.163559395325056891949966102399500263086e-117 |
| 4 | 3 | .9999999997115840 | .9999999997115840 | 0 | 1.776357e-15 | 5.850820342461298495457334053081880838144e-118 | 2.77386421809576108795900140880136477689e-118 |
| 8 | .01 | 2.343988157166125e-54 | 2.343988157166125e-54 | 1.563195e-13 | 2.557954e-13 | 3.529047377195017179236573400692066141949e-111 | 2.910593785079554681656129299436077710458e-112 |
| 8 | .1 | 3.779128859664263e-38 | 3.779128859664263e-38 | 1.398882e-14 | 1.563195e-13 | 2.825501157199079605651584592688184665693e-112 | 6.580660424875714573745701812367332440864e-114 |
| 8 | .3 | 9.134538313205943e-30 | 9.134538313205943e-30 | 1.406653e-13 | 1.421086e-13 | 1.379897486139146668400965008419402903115e-114 | 3.751060446908974588839404092940620016092e-113 |
| 8 | 1 | 2.579657265634780e-18 | 2.579657265634780e-18 | 6.982304e-12 | 6.994406e-15 | 1.137173713782412454165379101293825516723e-114 | 8.250446993222530805184894290894556440168e-114 |
| 8 | 3 | .2167090300658148 | .2167090300658148 | 1.680434e-12 | 1.776357e-15 | 8.703820332993646946856023741707011368517e-118 | 2.178644593706076014200720480049119707803e-118 |

Nonzero displayed extension values have been rounded upward; zero is the saved floating-point comparison result, not proof of exact endpoint independence. All forty route-specific extension maxima lie below 1e-8. The largest saved raw high-precision current residual is 3.503739422237127e-110, including the very small high-k transmissions; this avoids subtracting double-precision exponentially large currents and calling their cancellation a measurement.


## Reproduction and source package

The delivered single Python script embeds eleven byte-verified source bodies, including the original producer and its exact certificate-first recovery, the primary-1920 extension, exact-chain core and paper adapter, refined inventory, saved validators and final rendering/artifact checks. It requires the declared NumPy, SciPy/packaged HiGHS, mpmath, SymPy, Matplotlib, Boost.Multiprecision and Poppler environment. Execution is restricted to the declared CPU box; all data, cache and temporary paths are under <internal>/page_fig_r1, with nice level at least 10, at most sixteen scientific workers and a 100 GiB /data reserve.

Use `python page_curve_amps_R1.py --stage STAGE`. The wrapper supports `unpack`, `pairs`, `spectra`, `anchors`, `original-inventory`, `recovery`, `admit-recovery`, `primary1920`, `exact-controls`, `exact-crosschecks`, `chain-inventory`, `spectral-evidence`, `inventory-evidence`, `checks`, `render` and `plot-check`. It materializes hash-verified embedded bodies and refuses unequal existing sources. No external source file is an unstated prerequisite.

This is a source-complete reproduction package with declared retained baseline data: the original primary spectra, original inventory and immutable failed-convergence validation. The historical stages reproduce that provenance; the final refinement proceeds through primary1920, exact controls/crosschecks, chain-inventory, saved evidence, checks, render and plot-check. Existing recognized-source records are reused. The exact manufactured control is pinned and reused rather than replayed. Neither the packaging nor this report promises an empty-data run that silently bypasses the documented numerical recovery history.

Where historical unresolved cases require recovery, the supported recovery stage writes separate exact candidates and original dual certificates. It does not signal processes. After complete replacement coverage, quiesce only the verified superseded owned process tree, preserving its interruption evidence. Admit-recovery validates the complete staged batch and promotes unchanged case and NPZ bytes. The embedded original inventory supports `PAGE_R1_SAVED_ONLY=1` and fails closed if a cached case is missing. No unsuccessful solver status is relabelled successful.

The final source-and-aggregate validation checks every unique case identity and scalar certificate, all shared positive weights/order/prefix bindings, every new exact chain support/partition/mean/objective/history proof, all paired finite-optimum bounds, all required grid insertions and selected history arrays. Both reviewers read all fifteen selected cases and twenty-six full proofs, with exact literal fragment sharing and ordered JSON reconstruction. Exhaustive saved-data checks cover the remaining repeated arrays and proofs; this is not a literal-read claim for every raw digit. All 87 new spectral witnesses were wholly read in addition to the earlier accepted 291 witnesses and independent anchors. Raw records, failed attempts and logs remain on the box.

## Signed acceptance ledger

| Signed requirement | Disposition |
|---|---|
| A: vector two-panel 7.0 by 2.4 inch figure, lettering at least 8 pt | PASS: one 504 by 172.8 pt vector page, embedded fonts, minimum base lettering 8 pt; final visual review passes at the delivered canvas size. Manuscript scaling remains a lead check. |
| B, I.1–3: comparator and independent differential forms | PASS: constants, twenty two-form controls, explicit endpoint perturbation bounds, independent endpoint extensions and raw high-precision current residuals. |
| C: four entropy inputs and Page coordinates | PASS: the four rows below, nonnegativity, endpoints, maxima and capacity ordering. |
| D, I.7: product-state pair response | PASS: four-state integration, fixed-probe interpretation, variance, number covariance, dephasing, symmetry and singular limits. |
| E–F: admissible uniform history and averaged crossing | PASS: explicit inventory constraints and saved r0, zx, taux; reference is independently feasible. |
| G, I.6: attainable finite-LP envelope | PASS: original-polygon primal and independent full-support dual certificates, exact new gaps, seven-centre table, crossing/terminal histories and whole-post-Page equality. |
| H: references, complementary shading, guides, crossing and inset | PASS: all CSV ordinates and complementary shares match accepted data, and the final curves, bands, guides, dot and inset were inspected. |
| I.4–5: positive quadrature and refinement | PASS: 960/1920 primary and 400/800 scalar nodes, all positive weights retained, integral/marker checks, finite-LP change below 1e-5 and path refinement 0.0220543493 pt below 0.1 pt. |
| J: physical identification and error discipline | UNDECIDED physical matching; comparator assumptions explicit, no physical covariance error band or certified completion claim. |
| J: final geometry, clipping, overlaps and CSV/vector correspondence | PASS: exact canvas/vector/CSV checks and final image inspection; the dot/guide separation is resolved in the inset without moving either coordinate. |
| J: subsection compilation, occupied area and manuscript page count | DEFERRED to the lead as explicitly assigned by owner #215. |

## Convergence refinement and exact original-LP construction

The 480/960 comparison fails the envelope criterion even though its independently certified LP intervals are much narrower than the discrepancy. At normalized centre z/calR=0.97625, the fine maximum is approximately 0.865309240664 and the coarse maximum 0.865282231176; their normalized primal-dual gaps are respectively 6.27e-11 and 3.30e-11. Thus a more accurate LP solve on those same nodes cannot remove the 2.70e-5 quadrature difference. The refined primary rule doubled each of the ten positive quadrature panels to 192 nodes, or 1920 per channel, retaining the existing 960-node rule as comparison. Its unrounded constants define the refined markers, geometry and envelope. The completed driver reports 3575 common normalized centres at each level, plus the seven report centres, and a final path-refinement change 0.02205434930255083 pt; final source-and-aggregate evidence review is complete.

For the refinement, an exact solution of the original finite inventory LP is available. In normalized coordinates d=u-v and u, the polygon has lower boundary max(l,d) and upper boundary min(U,V+d), where l=max(0,(P-a)/calR), U=(P+a)/calR and V=(D-b)/calR, with the original difference bounds. Each boundary has at most one kink. The two mean constraints determine either an endpoint mixture or barycentric masses on a three-vertex boundary chain. Sorting the exact signed response-to-entropy ratios allocates these masses to the labels, including exact fractional mixtures when a label straddles a cut. Original polygon constraints and mean equations remain satisfied by convexity and exact rational contractions.

The dual is constructed separately from the exact quantile boundary scores, then checked against every vertex of the ORIGINAL polygon. Its full-support objective must equal the feasible primal objective exactly. Grouped verification is exhaustive: a vertex-support difference is affine in the sorted score, so both actual score endpoints certify each whole segment; split labels are checked directly. Complete label coverage, all positive p/t weights, group masses, original constraints and reconstructed histories are retained. This solves the specified two-mean finite LP; it does not replace it with the unconstrained fractional knapsack. Independent rational lower-chain, upper-chain, endpoint, tie and degenerate controls, plus comparison to previously certified actual cases, precede its use. The source, manufactured controls and actual old-case crosschecks are accepted, and the refined inventory execution is complete. Final exhaustive saved-certificate validation and the agreed actual evidence review are complete, with no discrepancy.


## Refined marker and validation values

The plotted variables are z=S_marg/S0, tau=1−(1−z/calR)^(3/2) and F_f(z)=min(z,f(1−z/calR)). Thus z_P=f calR/(calR+f), tau_P=1−[calR/(calR+f)]^(3/2), and the peak height is z_P. The capacity-interval shading in panel (a) is the given capacity-input interval, not a newly estimated physical error.

The author wholly read all 285 refined validation entries and their auxiliary fields through EOF. There are no numerical failures. The numerical check record's rendering placeholder is superseded by the completed artifact checker and image inspection below; it does not itself substitute for the independent final saved-proof validation.

| Capacity f | Page coordinate z_P | Page time tau_P | Peak F_f |
|---:|---:|---:|---:|
| 1 | 0.621084526820049 | 0.510530015901316 | 0.621084526820049 |
| 0.887 | 0.575545359041925 | 0.477322380220808 | 0.575545359041925 |
| 0.7883924135 | 0.532342271734257 | 0.445153596107954 | 0.532342271734257 |
| 0.9830837141 | 0.614517040826964 | 0.505785886626489 | 0.614517040826964 |

The actual saved markers are r0=0.805043401354722122235824772358389699923059511752252660246924696232909585382605924087062897919496466, z_x=0.6259279078928504647178664752359480582063809087397476719400617553168395811356288035427209404107338211 and tau_x=0.5140189246813519833091157607021618156694131690417300224629558398437505708575474965894404914859514572. The retained late share is 1-r0=0.194956598645277877764175227641610300076940488247747339753075303767090414617394075912937102080503534. These numerical marker values use the unrounded fine constants; displayed decimals are not independent interval guarantees.

For the complete post-Page interval, the saved minimum, maximum and reference are all 0.8050434013547221, with their equality verified as exact fractions. Every fine exact-chain case reports zero entropy-mean residual, zero inventory violation and zero normalized primal-dual gap. The independent final validator confirms all complete source/proof joins; its selected full evidence has been read and accepted by both scientific reviewers. The prescribed 960/1920 envelope change is 7.077866614357831e-6; its exact finite-certificate comparison bound has numerical rendering 7.077866614405038e-6. The accepted comparison is between the finite quadrature LP optima, not a rigorous continuum quadrature-error radius.


## Seven required fine-grid centres — actual saved scalar results

The table copies the actual report-centre ordinates. The first row is the exact zero-overlap branch. The other rows have independent original-polygon primal and full-support dual certificates; their complete selected proof intake has been accepted by both scientific reviewers. Display values are numerical renderings of those exact finite-LP quantities.

| Centre | z | Minimum released share | Reference | Maximum released share |
|---|---:|---:|---:|---:|
| P-w/2 | 0.6010845268200485 | 0 | 0 | 0 |
| P | 0.6210845268200486 | 0.3840020845721358 | 0.4025217006773610 | 0.4393949464913498 |
| z_x | 0.6259279078928505 | 0.4769955059887944 | 0.5 | 0.5458028048574010 |
| P+w/2 | 0.6410845268200486 | 0.7680041691442716 | 0.8050434013547221 | 0.8787898929826995 |
| P+0.1 | 0.7210845268200485 | 0.7680041691442716 | 0.8050434013547221 | 0.8787898929826995 |
| P+0.5 | 1.1210845268200484 | 0.7680041691442716 | 0.8050434013547221 | 0.8787898929826995 |
| calR-w/2 | 1.6191110175780234 | 0.7712431342091663 | 0.8050434013547221 | 0.8476028923892230 |

The final saved validator reports 7,150 unique level/centre cases, 6,828 new exact sign proofs and 2,256 retained prior sign certificates. Its maximum normalized gap across both levels is 6.511954197496235e-10, attained in a retained 960-node certificate; each new exact-chain gap is zero as a Fraction. The validator passes with no exceptions. The rendered history arrays have maximum reported weighted-mean discrepancy 1.3322676295501878e-15 and a largest rendered theta of 1.000000000000002; these are explicitly floating copies of the exact feasible histories, not exact constraint violations. Both scientific reviewers completed the selected full-certificate intake, including every literal long-number fragment and its complete reference structure.

## Final artifacts and visual acceptance

The delivered script is the reviewed source package, 323,121 bytes, SHA256 259d449f96def51a949edcca2f1ce685c84427e1e59cc1b8b607e078c99b9478. Static extraction verifies all eleven embedded source bodies exactly; final rendering and artifact checks run through its declared entry points. Its opaque white borderless entropy legend masks only the decorative guide underneath the legend and preserves every ordinate, coordinate, font and curve.

The actual artifact checker passes. All 57,217 CSV rows—four entropy inputs, both panels' reference and envelope ordinates, both panels' required maxima/guides, the formal endpoint, and all inset copies—match the accepted saved data. The PDF is one 504 by 172.8 point page (7.0 by 2.4 inches), with embedded fonts and no image objects; the minimum base lettering is 8 points. Its SHA256 is 2404d02d8127a4a0a15aec3bda072cf1b872f1a33d0927067e543a3b6307f476. The CSV SHA256 is 2d9d2a2c6e40c3a6c04b06f697d54f95696a87a0c5de13ccae0eb6d118b11557.

The author inspected the final rendered image. Both full y-labels remain on the canvas, the entropy peak annotation clears the legend, the legends/inset are legible, and the required filled maxima, open formal endpoint, complementary shading and late uniform-reference annotations are present. The black crossing dot and dotted Page guide keep their computed coordinates. Their separation is visibly resolved in the prescribed inset; the main-panel marker footprint is wider than that small separation, so no distinct main-panel gap is claimed. No coordinate offset was introduced. The inset displays the prescribed limits, reference curves, admissible bands, half line, Page guide and crossing dot.

The saved grid contains the original 201 evenly spaced centres, all specified insertions and the preserved old normalized union, refined to 3,575 common normalized centres at both levels. Direct interpolation is used, with no fitted smoothing. The independently reconstructed maximum path change is 0.02205434930255083 pt, below 0.1 pt for both the main/inset height bounds. Nonnegativity, endpoint values, all four maxima/capacity ordering, uniform-history feasibility, crossing and terminal optimizing histories, ordered response bounds with complementary shares, and the whole-post-Page equality are checked separately. These facts discharge the figure's numerical/rendering J checks. Subsection compilation with its float, occupied caption/float area, manuscript scaling and the 29-page limit remain DEFERRED to the lead; the figure file does not claim those checks.

SOURCES: 192.168.1.93:<internal>/HISTORY_PACK_owner215.md (identical inherited whole-read provenance); 192.168.1.93:<internal>/WRITE_PAGE_SIGNED_v1.md, governing §3 A–J; 192.168.1.93:<internal>/OWNER215_brief.txt; 192.168.1.93:<internal>/PAGEFIG_AUTHOR_original.tex; 192.168.1.93:<internal>/PAGEFIG_REDERIVATION_original.tex; 192.168.1.93:<internal>/PAGEFIG_AUTHOR_COMPARISON.md; 192.168.1.93:<internal>/PAGEFIG_SOLE_COMPARISON.md; 192.168.1.93:<internal>/page_curve_amps_R1.py (eleven hash-verified embedded bodies); 192.168.1.93:<internal>/raw (immutable pair/anchor/spectral records and approved saved-only projections); 192.168.1.93:<internal>/validation.json; 192.168.1.93:<internal>/final_evidence; 192.168.1.93:<internal>/spectral_evidence_primary1920; 192.168.1.93:<internal>/final_plot_artifact_check.json; 192.168.1.93:<internal>/figure_geometry.json. No route-2 figure delivery or box output was read.
