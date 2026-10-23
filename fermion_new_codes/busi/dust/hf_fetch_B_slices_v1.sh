#!/bin/bash
# Fetch B's continuation slices T/r_s 48 to 56 (the dust tail's second route, WEB40G task busi 1) from the owner's dataset
# pseudotensor/qi2-fermion-bulk, solve/B_ext100_slices/ (uploaded by the solve lane, RUN_RECORD 08:46 to 08:48 UTC), into this lane's own space
# on the box's /data disk. Each file's sha256 is checked against the dataset's LFS sha256. The token only inside the python process, from
# ${QI2_ROOT}/gr_exports.sh; never printed or logged. Usage (box): setsid nohup bash hf_fetch_B_slices_v1.sh > LOG 2>&1 &
OUT=${QI2_ROOT}/B_ext100_slices; mkdir -p $OUT
( source ${QI2_ROOT}/gr_exports.sh > /dev/null 2>&1
  ${QI2_ROOT}/python - "$OUT" <<'PY'
import os, sys, hashlib, time
from huggingface_hub import HfApi, hf_hub_download
out = sys.argv[1]; repo = "pseudotensor/qi2-fermion-bulk"
api = HfApi(token=os.environ.get("HUGGING_FACE_HUB_TOKEN_PRIVATE"))
files = [x for x in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="solve/B_ext100_slices", recursive=True) if getattr(x, "size", None)]
want = [x for x in files if any(("_T%04d.00" % t) in x.path for t in (48, 50, 52, 54, 56))]
print("listed %d files, fetching %d: %s" % (len(files), len(want), [os.path.basename(x.path) for x in want]), flush=True)
for x in want:
    t0 = time.time()
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=x.path, local_dir=out, token=os.environ.get("HUGGING_FACE_HUB_TOKEN_PRIVATE"))
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    rsha = getattr(getattr(x, "lfs", None), "sha256", None)
    print("%s: %d bytes (remote %d), sha256 %s, remote %s, %s (%.0f s)" % (os.path.basename(p), os.path.getsize(p), x.size, h.hexdigest(), rsha,
          "MATCH" if (rsha == h.hexdigest() and os.path.getsize(p) == x.size) else "MISMATCH", time.time() - t0), flush=True)
PY
)
echo "fetch exit $?"
