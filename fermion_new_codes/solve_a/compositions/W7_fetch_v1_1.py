#!/usr/bin/env python
"""W7_fetch_v1_1.py (the solve lane; COMP2 v1.1, reviewer A's W7 ruling 1). Re-fetches the ext100 orbital slices at T/r_s 48, 50, 52, 54, 56
from the private HF dataset pseudotensor/qi2-fermion-bulk into --out-dir and checks each file's md5 against the box manifest that recorded
the move (<internal>/MANIFEST_B_ext100_slices.md5, paths ./solve/B_ext100_slices/<name>). The token is read from the environment
(HF_TOKEN, set by the caller from gr_exports.sh) and never printed. Deletes nothing; refuses to overwrite a file whose md5 already matches.
Usage (box): HF_TOKEN=... python W7_fetch_v1_1.py --out-dir <internal>/slices
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

REPO = "pseudotensor/qi2-fermion-bulk"
MANIFEST = "<internal>/MANIFEST_B_ext100_slices.md5"
NAMES = ["orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T%07.2f.npz" % t for t in (48.0, 50.0, 52.0, 54.0, 56.0)]


def md5(p, block=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        while True:
            c = fh.read(block)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN not set")
    from huggingface_hub import hf_hub_download, HfApi
    man = {}
    for line in open(MANIFEST):
        p = line.split()
        if len(p) == 2 and "/solve/B_ext100_slices/" in p[1] and p[1].endswith(".npz"):
            man[os.path.basename(p[1])] = (p[0], p[1][2:] if p[1].startswith("./") else p[1])
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, ".hf_cache")
    files = set(HfApi(token=tok).list_repo_files(REPO, repo_type="dataset"))
    out = {"repo": REPO, "manifest": MANIFEST, "files": {}}
    for nm in NAMES:
        want, rpath = man[nm]
        dst = os.path.join(a.out_dir, nm)
        rec = {"repo_path": rpath, "in_repo": rpath in files, "manifest_md5": want}
        if os.path.exists(dst) and md5(dst) == want:
            rec.update({"md5": want, "match": True, "note": "already present"})
        else:
            p = hf_hub_download(REPO, rpath, repo_type="dataset", token=tok, cache_dir=cache)
            shutil.copyfile(p, dst + ".part")
            os.replace(dst + ".part", dst)
            m = md5(dst)
            rec.update({"md5": m, "match": m == want, "bytes": os.path.getsize(dst)})
        out["files"][nm] = rec
        print("%s: repo path %s, md5 %s, manifest %s, match %s" % (nm, rpath, rec["md5"], want, rec["match"]), flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    with open(os.path.join(a.out_dir, "fetch_v1_1.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
