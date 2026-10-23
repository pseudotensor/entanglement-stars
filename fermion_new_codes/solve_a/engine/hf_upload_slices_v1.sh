#!/bin/bash
# hf_upload_slices_v1.sh (the lane's analyst builder; COMMS rule 9, busi's red team of RULING_box_runs_v1 defect 1: free disk on the box): uploads
# the named files to the owner's private dataset pseudotensor/qi2-fermion-bulk under solve/<subdir>/, verifies each remote file's size against the
# local one, and deletes the local file only when it matches (and --delete is given). The token is read only inside the uploading process from
# ${QI2_ROOT}/gr_exports.sh (HUGGING_FACE_HUB_TOKEN_PRIVATE); never printed or written. Usage (box): bash hf_upload_slices_v1.sh <subdir> <--delete|--keep> file...
set -u
SUB=${1:?subdir}; MODE=${2:?--delete or --keep}; shift 2
source ${QI2_ROOT}/gr_exports.sh > /dev/null 2>&1
${QI2_ROOT}/python - "$SUB" "$MODE" "$@" <<'PY'
import os, sys, time
from huggingface_hub import HfApi
sub, mode, files = sys.argv[1], sys.argv[2], sys.argv[3:]
api = HfApi(token=os.environ.get("HUGGING_FACE_HUB_TOKEN_PRIVATE"))
repo = "pseudotensor/qi2-fermion-bulk"
for f in files:
    name = "solve/%s/%s" % (sub, os.path.basename(f))
    size = os.path.getsize(f)
    t0 = time.time()
    api.upload_file(path_or_fileobj=f, path_in_repo=name, repo_id=repo, repo_type="dataset", commit_message="solve lane: %s" % name)
    info = list(api.get_paths_info(repo_id=repo, paths=[name], repo_type="dataset"))
    rsize = info[0].size if info else None
    ok = (rsize == size)
    print("%s UTC: %s -> %s: local %d bytes, remote %s bytes, %s (%.0f s)" % (time.strftime("%H:%M:%S", time.gmtime()), f, name, size, rsize, "MATCH" if ok else "MISMATCH", time.time() - t0), flush=True)
    if ok and mode == "--delete":
        os.remove(f)
        print("  deleted local %s" % f, flush=True)
PY
