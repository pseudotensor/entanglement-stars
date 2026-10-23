# D13 v1.1 checks (solve_cold8 section (3)): (1) the collar split's window size and FFT grid in the stored d13 npz/json;
# (2) L_H(kappa_T) (kappa 1 + 2) at the windows starting at 20 r_s, from the D13_class_v1 json.
import json, numpy as np
P = "<internal>/prod"
z = np.load(P + "v6o_B2_k1_d13.npz", allow_pickle=True)
ks = [k for k in z.files if "collar" in k.lower() or "split" in k.lower()]
print("npz keys with collar/split:", ks[:20], "of", len(z.files))
for k in ks[:10]:
    a = z[k]; print(" ", k, a.shape, a.dtype)
d = json.load(open(P + "v6o_B2_k1_d13.json"))
Ls = set(); sv = set()
for t, c in d["clocks"].items():
    cs = c["per_channel"]["1"].get("collar_split")
    if cs: Ls.add(cs["L"]); sv.add(cs["sites_valid"])
print("stored collar_split L values:", Ls, "sites_valid:", sv, "-> FFT length M = 4L =", [4 * L for L in Ls],
      "grid 2pi/M =", [2 * np.pi / (4 * L) for L in Ls], "; 2pi/L =", [2 * np.pi / L for L in Ls])
C = json.load(open("<internal>/d13_class_v1.json"))
def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in k.lower() for s in ("allow", "l_h", "lh")):
                print(p + "/" + k, str(v)[:400])
            walk(v, p + "/" + k)
    elif isinstance(o, list) and o and isinstance(o[0], (dict, list)):
        walk(o[0], p + "[0]")
walk(C)
