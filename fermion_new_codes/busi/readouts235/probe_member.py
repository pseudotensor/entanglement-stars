# Probe: list keys/shapes of a collapse member's fields npz and json top-level structure
import sys, json, numpy as np
base = sys.argv[1]
z = np.load(base + "_fields.npz", allow_pickle=True)
for k in z.files: print("npz", k, z[k].shape, z[k].dtype, (str(z[k])[:400] if z[k].ndim==0 else ""))
j = json.load(open(base + ".json"))
def show(o, pre="", depth=0):
    if depth > 2: return
    if isinstance(o, dict):
        for k, v in list(o.items())[:60]:
            t = type(v).__name__; ln = len(v) if hasattr(v, "__len__") and not isinstance(v, str) else ""
            print(pre + k, t, ln, (repr(v)[:100] if not isinstance(v, (dict, list)) else ""))
            if isinstance(v, dict): show(v, pre + "  ", depth + 1)
            if isinstance(v, list) and v and isinstance(v[0], dict) and depth < 2: print(pre + "  [0] keys:", list(v[0].keys())[:80])
show(j)
