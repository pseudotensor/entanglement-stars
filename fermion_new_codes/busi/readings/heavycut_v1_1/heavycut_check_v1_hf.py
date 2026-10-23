#!/usr/bin/env python3
"""heavycut_check_v1_hf.py: read-only listing of the private HF dataset (pseudotensor/qi2-fermion-bulk) for the
B2 orbitals slices, to check the delivery's sentence "The HF dataset holds the same 37 slices" (summary item 6).
The token is read from <internal>/gr_exports.sh (HUGGING_FACE_HUB_TOKEN_PRIVATE) and never printed.
Nothing is downloaded. Output: OUT/heavycut_check_v1_hf.out and OUT/heavycut_check_v1_hf.json."""
import json, os, re, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "<internal>/heavycut_v1_1"
REPO = "pseudotensor/qi2-fermion-bulk"
tok = None
for line in open("<internal>/gr_exports.sh"):
    m = re.match(r"\s*export\s+HUGGING_FACE_HUB_TOKEN_PRIVATE\s*=\s*['\"]?([^'\"\s]+)", line)
    if m:
        tok = m.group(1)
if tok is None:
    print("no HUGGING_FACE_HUB_TOKEN_PRIVATE in gr_exports.sh"); sys.exit(1)
from huggingface_hub import HfApi
api = HfApi(token=tok)
files = api.list_repo_files(REPO, repo_type="dataset")
pat = re.compile(r"orbitals_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_T(\d+\.\d+)\.npz$")
hits = sorted((float(pat.search(f).group(1)), f) for f in files if pat.search(f))
expect = [12, 16, 20, 24, 28, 36] + list(range(40, 101, 2))
out = {"repo": REPO, "n_files_total": len(files), "B2n4096_orbitals": [f for _, f in hits], "clocks": [t for t, _ in hits],
       "n": len(hits), "equals_37_page_clocks": [t for t, _ in hits] == [float(x) for x in expect],
       "dirs": sorted(set(os.path.dirname(f) for _, f in hits))}
with open(os.path.join(OUT, "heavycut_check_v1_hf.out"), "w") as f:
    for k, v in out.items():
        if k != "B2n4096_orbitals":
            f.write("%s: %s\n" % (k, v)); print(k, ":", v)
    f.write("files:\n" + "\n".join(out["B2n4096_orbitals"]) + "\n")
json.dump(out, open(os.path.join(OUT, "heavycut_check_v1_hf.json"), "w"), indent=1)
