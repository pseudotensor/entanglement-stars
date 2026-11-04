# reviewer A (lead subagent) clock-growth estimate (answer to reviewer B clockbar); full derivation: lead scratchpad fable_clock_derivation.md

Full derivation (with the symbolic checks and the kernel numerics) is in `<internal>/fable_clock_derivation.md`; check scripts `chk_clock.py`, `chk_family.py`, `chk_family_avg.py` sit beside it.

## Conclusion

reviewer B's estimate (C) can be carried out exactly with the paper's lattice rates, and it splits cleanly into a part that time dilation makes harmless and a part that nothing in the paper controls. With the frozen (held-projector) reference, the exact result is

  D⁺Λ ≤ Ṡ [A₁ + ½A₂],  Λ := 4πG∫₀ᴿ|J|/k ds ≥ L,

with Ṡ an explicit lattice constant and A₁, A₂ two explicit clock-weighted slice functionals that do not themselves bound the clock. This is a genuine reduction (floor ⇐ bounded A₁, A₂, both measurable on the 573 records), but it is not D⁺L ≤ a + bL with floor-free constants: the general a priori bounds on A₁, A₂ carry the factor e^{(ℓ+4a)g_max}, g_max = sup 4πG|J|/k the pointwise clock-normalized gradient, and that factor is tamed only by the hydrodynamic condition g_max ≲ 1/ℓ, which is itself a floor (L ≤ R/ℓ). With the paper's declared polarization response the estimate degenerates further to Riccati type (quadratic in L on reviewer B's ε-family). So: concede the all-time Grönwall; the deciding calculation is named below.

## (1) The estimate with the actual rates

Identities used (all verified symbolically, residual 0): r k' = 4πGρ/k − 4πGrJ − 3k/2; (ln N)' = 4πGJ/k; K_r = v'/N = k/2 − 4πGρ/k; and the windowed charge identity (NṀ)' = 4πr²(Nρ̇ + vJ̇), i.e. N(s)Ṁ(s) = 4π∫₀ˢσ²(Nρ̇ + vJ̇)dσ, which with the paper's sources is reviewer B's −iTr(C[h̄_s,h̄]) − ⟨ū_s, W''u̇⟩.

Matter balance is clock-proportional, exactly. With h̄ = Σ_y(N̄_y e_y + v̄_y·p_y), 0 ≤ C ≤ I, rank e_x ≤ 8, ‖e_x‖ ≤ ε_x = √(m²+3/2a²), ‖p_x‖ ≤ 1/2a, hopping locality (25/63/129 interacting sites) and |v̄_y| ≤ KR N̄_y (nonnegative kernel, reviewer B's barrier |w| ≤ Kr):

  a³|ρ̇_m,x| ≤ [400ε_x² + 1512ε_xKR/a] N̄ˡᵒᶜ_x,  a³|J̇_m,x| ≤ [504ε_x/a + 1548KR/a²] N̄ˡᵒᶜ_x.

Call the bracket Ṡ. No floor, no jets, nothing about C beyond ‖C‖ ≤ 1. This is the time-dilation fact: matter stops where the clock that drives it stops.

Inserting into (C) (Dini derivative, two-sided functional):

  Term I = 4πG∫|J̇|/k ≤ Ṡ·A₁,  A₁ := 4πG∫₀ᴿ N̄ˡᵒᶜ/k ds;
  Term II = 4πG∫|J||Ṁ|/(2kM) ≤ (Ṡ/2)·A₂,  A₂ := 4πG∫₀ᴿ(|J|/k)Φ(s)ds,
  Φ(s) := 4π∫₀ˢσ²N̄ˡᵒᶜ(σ)N(σ)(1+Kσ)dσ / (N(s)M(s)).

What A₁, A₂ do and do not control. On reviewer B's ε-family (k=ε, J=jr; G=¼, j=0.02, R=10) I computed: A₁ → 5.0 (point sampling) or 5.0 + O(ℓ/R)·L with a box-averaged clock; A₂/L → 1.75, independent of ℓ ∈ {0, 0.5, 1}. So on the family D⁺Λ ≤ Ṡ[5 + (0.9 + O(ℓ/R))Λ]: linear, floor-free constants, even as L → ∞. The mechanism is concentration: the clock is so steep that the clock-weighted inner integral in Φ lives within ε/(4πGjs) of s, and the enclosed-mass rate is dilated away.

In general that concentration is not available. The a priori bounds are N̄ˡᵒᶜ(σ) ≤ N(σ)e^{(ℓ+4a)g_max} and Φ(s) = O(e^{(ℓ+4a)g_max}·Θ(s)(1+KR)/k(s)²), Θ the inner/outer clock ratio; the kernel of Term II is 2G/(k³s³), two inverse strains beyond Λ's integrand. The sup-bound A₂ ≤ Φ_max Λ is quadratic in L on the family (Φ_max ≈ 1.7L) while A₂ itself is linear; the gap is exactly the concentration. A linear closure with floor-free constants therefore needs (i) g_max resolved on the averaging scale, (ii) Θ bounded (clock roughly monotone), (iii) a mass margin on current-carrying spheres or the concentration mechanism. Condition (i) alone gives L ≤ Rg_max ≤ R/(ℓ+4a), a floor. So no bound of the form a + bL is provable from (S) alone; what is provable is the reduction above.

The blocking term, named. It is the enclosed-mass response −JṀ/(2kM) in (C), and it blocks through two concrete mismatches. First, the clock that drives the matter is N̄ = W_ℓ†N, averaged over ℓ, while the clock the rows assign at the sphere is N(s); when the clock is not resolved on ℓ the time-dilation protection is lost by e^{ℓg}. Second, the polarization response: W is degree one, so W'' is degree −1 and W''[ū]ū = 0 (Euler); hence ⟨ū_s, W''u̇⟩ = ⟨[W'',χ_s]ū, u̇⟩ is a collar term at s carrying one clock factor (from u̇ = N·relative rate), where the matter flux carries two. So Ṁ_ref/M ≈ 8πGγP(jets)(relative rates)/(k²s) with no clock factor left, and from the displayed W₂ = −γ∫(v')²/N, ṙ_v = 2γ∂_rK̇_r contains γ(K_r + 4πGrJ)·(4πGJ/k)·(Ṁ/M): the pointwise gradient g undivided by any clock. Term II with the reference becomes ∝ γ∫|J|/(k³s), ~1/ε² ~ L² on the family. The reference is an instantaneous functional of the fields, not time-dilated; reviewer B's 1/N polarization coefficient reappears here as the missing clock factor. The constraint-inverse hypothesis does not rescue this: the prescribed-source inverse kernel contains 4πG/k(s₀) for a current spike at s₀, so a uniform ‖F_u⁻¹‖ in the natural norm is a pointwise mass margin k ≥ k_* ≈ 4πG/‖F_u⁻¹‖, after which Λ ≤ 4πGB_JR/k_* follows on the slice with no Grönwall at all. The reported ‖F_u⁻¹‖ ≤ 2.9 is a floor or not depending on the unspecified norm.

## (2) Propagating |J| ≤ cρ

Not from the dynamics, but the condition can be weakened before it is given up. Lemma W (new; uses only the signed identity 4πG∫ρ/k = Rk(R) + ½∫k + 4πG∫rJ, no sign of ρ): if pointwise |J| ≤ cρ + c'k², then 4πG∫|J|/k ≤ c[3RK/2 + 2πGB_JR²] + 4πGc'RK, hence a two-sided clock bound. This permits ρ < 0 wherever mass is enclosed and a current excess proportional to the mean enclosed density 2GM/r³, because ∫k ≤ RK. Any additive constant instead of c'k² is fatal (the ε-family has |J| ≤ jR).

Splitting C − P_★ into band-diagonal and interband parts, s = s_kin + s_coh − t: the kinetic bound |J_kin| ≤ ν₃ρ_kin holds at all times for free, since 0 ≤ C ≤ I is preserved and |q_r| ≤ ν₃ε is a property of the dispersion (app:charge), with the caveat that pointwise on the lattice it holds only after averaging over the interference scale (a two-plane-wave positive-band packet has sites with negative Re ψ*(h₀ψ)). The coherence and polarization parts are signed with no current-energy relation: J_pol = 2γ∂_rK_r ≠ 0 where ρ_pol = −γK_r² = 0. So the weakest condition that still yields the floor is

  |J_coh − t_J| − ν₃(ρ_coh − t_N) ≤ (1−ν₃)ρ + c'·2GM/r³,

the non-kinetic current excess dominated by the kinetic slack plus the mean enclosed density. It is a slice condition, measurable on the records, and at its boundary the full rates ρ̇_m − ṙ_N, J̇_m − ṙ_v have no sign (reviewer B's objection stands). What the dynamics does preserve: 0 ≤ C ≤ I and spec C, the boundary charge E (hence only the outer margin k(R)² = 2GE/R³), and the clock-proportionality (S) of the matter rates.

## (3) Strongest honest theorem, one sentence, the calculation

Theorem. Let (C,N,v) be an admitted inward-branch spherical Hartree history on [0,R], N(R)=1, regular centre, held (field-independent) reference, with |ρ_ℓ| ≤ B_ρ, |J_ℓ| ≤ B_J (hence k ≤ K). Then D⁺Λ ≤ Ṡ[A₁ + ½A₂] with Ṡ, A₁, A₂ as above; if A₁ ≤ a₀ + a₁Λ and A₂ ≤ 2b₀ + 2b₁Λ on [0,T], then Λ(t) ≤ e^{Ṡ(a₁+b₁)t}[Λ(0) + Ṡ(a₀+b₀)t] and min N ≥ e^{−Λ}. With the polarization response an additional term γC_P[A₃ + g_maxA₄], A₃ = ∫|J|/(k³s), enters, giving only a finite lower bound on any collapse time, T_* ≥ ∫_{Λ(0)}^∞ dΛ/bound(Λ). Separately (Lemma W), on any slice satisfying |J| ≤ cρ + c'k² the clock is bounded by the explicit Λ₀ above, with no mass margin.

Sentence for the paper: "With the held reference, the clock integral Λ = 4πG∫|J_ℓ|/k obeys D⁺Λ ≤ Ṡ[A₁ + ½A₂] with an explicit lattice constant and two clock-weighted slice functionals that are finite on every regular slice and do not themselves bound the clock, so the clock cannot vanish in finite time while A₁ and A₂ stay bounded; on 573 stored records min N ≥ 0.072, and whether the exact dynamics, including the polarization response, keeps A₁ and A₂ bounded is not established."

The single remaining calculation: bound sup_t A₁ and sup_t A₂ along the exact history, which is the statement that the clock-normalized gradient 4πGJ_ℓ/k stays resolved on the averaging scale ℓ (so the matter is driven by the clock the rows assign) and that the polarization collar flux at each sphere is bounded by a multiple of the enclosed mass there, a relative mass-rate bound |Ṁ|/M ≤ μ; given either, Grönwall closes with floor-free constants. Concession: I cannot derive either from the paper's bounds, and the second fails on the ε-family's scaling.

Caveats. Numerics taken as supplied. The constant Ṡ is an operator-norm bound, orders of magnitude above hydrodynamic rates, so T_* from Theorem B is structural, not useful numerically. The polarization analysis uses the displayed longitudinal W₂ term, not the audited cubic tensor. The stored histories' held projector is level-following, so their reference response is neither zero nor W''u̇.