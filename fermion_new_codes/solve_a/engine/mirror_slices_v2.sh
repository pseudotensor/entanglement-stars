#!/bin/bash
# mirror_slices_v2.sh (solve builder 8, analyst; engine spec 49a.2): the slice mirrors' version 2. Replaces mirror_v36bm_v1.sh and
# pair_mirror_ell4a_v1.sh (and is the mirror of the v32 member's v36b continuation on pod 2). Each pass (every 3 min): every orbitals slice under
# the pod's states dir (all tags) is copied to /data as soon as it has stabilized (size unchanged since the previous pass and its mtime older than
# 120 s: the engine's np.savez is not atomic), md5-compared box against pod; the pod keeps its newest 2 slices (over all tags), every older one
# is deleted on the pod only after an md5-equal /data copy exists. Optionally (HF prefix given) each /data copy is uploaded to the owner's
# dataset pseudotensor/qi2-fermion-bulk under that prefix (remote size and LFS sha256 checked), the token only inside the uploading python
# process from ${QI2_ROOT}/gr_exports.sh, never printed or logged. /data floor 200 GB.
# Usage (box): setsid nohup bash mirror_slices_v2.sh NAME HOST PORT PODSTATES DST [HFPREFIX] > /dev/null 2>&1 < /dev/null &
set -u
NAME=${1:?}; HOST=${2:?}; PORT=${3:?}; PD=${4:?}; DST=${5:?}; HFP=${6:-}
PY=${QI2_ROOT}/python
K="-i ${QI2_ROOT}/id_rsa -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30"
SSH="ssh -n $K -p $PORT $HOST"
LOG=$DST/mirror_v2_$NAME.log; SEEN=$DST/.mirror_v2_seen
mkdir -p $DST; touch $DST/hf_done.txt $DST/md5_done.txt $SEEN
log() { echo "$(date -u "+%F %T UTC") $*" >> $LOG; }
hf_upload() {
  ( source ${QI2_ROOT}/gr_exports.sh > /dev/null 2>&1
    $PY - "$1" "$HFP" <<'PYEOF'
import os, sys, time, hashlib
from huggingface_hub import HfApi
f, pre = sys.argv[1], sys.argv[2]; name = pre.rstrip("/") + "/" + os.path.basename(f)
api = HfApi(token=os.environ.get("HUGGING_FACE_HUB_TOKEN_PRIVATE")); repo = "pseudotensor/qi2-fermion-bulk"
size = os.path.getsize(f); h = hashlib.sha256()
with open(f, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 24), b""):
        h.update(blk)
sha = h.hexdigest(); t0 = time.time()
try:
    api.upload_file(path_or_fileobj=f, path_in_repo=name, repo_id=repo, repo_type="dataset", commit_message="solve lane: %s" % name)
    info = list(api.get_paths_info(repo_id=repo, paths=[name], repo_type="dataset"))
except Exception as e:
    print("upload error %s: %s" % (name, type(e).__name__)); sys.exit(2)
rsize = info[0].size if info else None
lfs = getattr(info[0], "lfs", None) if info else None
rsha = getattr(lfs, "sha256", None) if lfs is not None else None
ok = (rsize == size) and (rsha == sha)
print("%s: %d bytes, %s (%.0f s)" % (name, size, "MATCH" if ok else "MISMATCH", time.time() - t0)); sys.exit(0 if ok else 1)
PYEOF
  )
}
log "start pid $$ name $NAME pod $HOST:$PORT $PD -> $DST hf '${HFP}'"
while true; do
  # rel path, size, mtime age (s), sorted by clock then tag
  LIST=$($SSH "cd $PD 2>/dev/null && now=\$(date +%s) && for f in */orbitals_*.npz; do [ -e \"\$f\" ] && echo \"\$f \$(stat -c %s \"\$f\") \$((now - \$(stat -c %Y \"\$f\")))\"; done" 2>/dev/null | awk '{k=$1; sub(/.*_T/, "", k); print k" "$0}' | sort -k1,1 | cut -d' ' -f2-)
  N=$(echo "$LIST" | grep -c .)
  i=0
  while read -r rel sz age; do
    [ -n "${rel:-}" ] || continue; i=$((i + 1)); n=$(basename $rel)
    prev=$(grep "^$rel " $SEEN | tail -1 | cut -d' ' -f2)
    echo "$rel $sz" >> $SEEN
    if ! grep -q "^$n " $DST/md5_done.txt; then
      [ "$prev" = "$sz" ] && [ "$age" -ge 120 ] || continue
      free=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
      [ "$free" -ge 200 ] || { log "/data free $free GB < 200: no copy"; continue; }
      scp -q $K -P $PORT $HOST:$PD/$rel $DST/$n.part && mv $DST/$n.part $DST/$n || { log "copy fail $rel"; continue; }
      a=$(md5sum $DST/$n | cut -c1-32); b=$($SSH "md5sum $PD/$rel" | cut -c1-32)
      if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "$n $a" >> $DST/md5_done.txt; log "copied $rel md5 $a"
      else log "MD5 MISMATCH $rel box $a pod $b (retry next pass)"; rm -f $DST/$n; continue; fi
    fi
    if [ $i -le $((N - 2)) ] && [ -e $DST/$n ]; then $SSH "rm -f $PD/$rel" && log "pod copy removed (older than the newest 2): $rel"; fi
  done <<< "$LIST"
  if [ -n "$HFP" ]; then
    for f in $DST/orbitals_*.npz; do
      [ -e "$f" ] || continue; grep -q "^$(basename $f) " $DST/hf_done.txt && continue
      grep -q "^$(basename $f) " $DST/md5_done.txt || continue
      if out=$(hf_upload "$f"); then log "HF $out"; echo "$(basename $f) uploaded $(date -u +%T)" >> $DST/hf_done.txt; else log "HF retry later: $out"; fi
    done
  fi
  tail -n 2000 $SEEN > $SEEN.t && mv $SEEN.t $SEEN
  sleep 180
done
