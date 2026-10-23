# Locate the 100-200 r_s excess of the sampled collar-flux integral over the exact energy fall inside 16 r_s.
# Inputs: the B2 hold run's per-record L_at_t (summary json, held records, one per 1 r_s) and the
# reconstructed E_in(16 r_s) per 10 r_s from m8_backprop_n1026_v1.npz. No propagation.
import json, numpy as np
S = json.load(open("<internal>/v6s_B2hold_k1_summary.json"))
recs = [r for r in S["records"] if r.get("held_step")]
rs = 64.0
t = np.array([r["t"] for r in recs]) / rs          # 100 .. 499, 1 r_s apart
L = np.array([r["L_at_t"] for r in recs])
B = np.load("<internal>/m8_backprop_n1026_v1.npz", allow_pickle=True)
tr = np.array(B["t_rs"], float)
E = np.array(json.loads(str(B["E_in"]))["16"], float)
o = np.argsort(tr); tr, E = tr[o], E[o]
# calibrate the left-sum factor against the note's 100-200 value 0.97946e-3
raw = L[(t >= 100) & (t < 200)].sum()
fac = 0.97946e-3 / raw
print("raw left sum 100-200", raw, "factor to note", fac, "(dt = 1 r_s = 64 time units)")
fac = rs   # dt = 1 r_s = 64 time units; L_at_t is already per chain (reproduces the note to 1.4e-6 relative)
for a, b in [(100, 200), (200, 300), (300, 400), (400, 500)]:
    m = (t >= a) & (t < b)
    dE = E[tr == a][0] - E[tr == b][0]
    print(f"per 100: {a}-{b}: flux {fac*L[m].sum():.5e}  fall {dE:.5e}")
print("\nper 10 r_s, 100-200 (left sums of per-record L, 10 samples each):")
print(" interval  fall_inside_16  flux_sum  flux-fall  cumulative(flux-fall)  min/max L*fac in interval")
cum = 0.0
for a in range(100, 200, 10):
    b = a + 10
    m = (t >= a) & (t < b)
    dE = E[tr == a][0] - E[tr == b][0]
    F = fac * L[m].sum()
    cum += F - dE
    print(f" {a}-{b}  {dE:.4e}  {F:.4e}  {F-dE:+.3e}  {cum:+.3e}  {fac*L[m].min():+.2e} {fac*L[m].max():+.2e}")
print("\nper 10 r_s, 200-500 differences (flux - fall):")
row = []
for a in range(200, 500, 10):
    b = a + 10; m = (t >= a) & (t < b)
    dE = E[tr == a][0] - E[tr == b][0]; F = fac * L[m].sum()
    row.append(F - dE)
print(" ".join(f"{x:+.1e}" for x in row))
print(" sum 200-500", sum(row), " max |per-10|", max(abs(x) for x in row))
print("\nper-record L*fac at t = 100..140:")
for a in range(100, 141):
    print(f"  {a}: {fac*L[t==a][0]:+.3e}", end="\n" if (a-99) % 6 == 0 else "")
print()
