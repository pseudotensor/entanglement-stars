#!/bin/bash
# c_mirror_v3.sh (the busi lane): version 2 parameterised: c_mirror_v3.sh POD_HOST POD_PORT BASE MEMBER (s4|s2) RUNTAG. The pod directory BASE
# (for example ${QI2_ROOT}/busi_compC_r2), the pid file BASE/C32_<member>.pid, the logs BASE/logs/C32_<member>.log and guard_<member>.log, the dataset
# folder busi/compC_C32_<member>_<RUNTAG>_slices/, the box record folder results/compC/C32_<member>_<RUNTAG>, the transit
# ${QI2_ROOT}/busi_compC_mirror<member>_<RUNTAG>. Version 2's header follows.
# c_mirror_v2.sh (the busi lane; COMMS rules 9 and 10; inbox #71): version 1 with the transit on the box's /data disk
# (${QI2_ROOT}/busi_compC_mirror, 3.5 TB free) in place of /home, whose 82 GB free sat below version 1's floor and near the 80 GB the
# solve lane keeps free there for B's guard; the floor is checked on /data. The logs and lists stay in results/compC/C32mirror.
# Version 1's header follows.
# c_mirror_v1.sh (the busi lane; COMMS rules 9 and 10; inbox #71): the slice mirror and record sync of Computation C's sigma/a 4 member on pod 5
# (${QI2_ROOT}/busi_compC on the container disk), after the solve lane's pair_mirror_v1.sh. Slices orbitals_<tag>_T*.npz (both fields, complex128)
# and light_<tag>_T*.npz (the light channels): the latest 2 orbitals slices stay on the pod while the member runs (its restart point); every
# other slice, once its size is stable and its zip readable, is copied to the box md5-checked, the pod copy deleted on a match, uploaded to
# pseudotensor/qi2-fermion-bulk under busi/compC_C32_s4_slices/ with the remote size and LFS sha256 checked, the box copy deleted on a match.
# One slice in transit at a time; the box keeps FLOOR_GB free. Every pass the record files (json, fields, spectra, handoff, s11 npz, the logs)
# are copied to the box. The token only inside the uploading python process from ${QI2_ROOT}/gr_exports.sh; never printed or logged.
# Never touches any process: the member's state is read from its pid file and /proc only.
# Usage (box): setsid nohup bash c_mirror_v3.sh HOST PORT BASE MEMBER RUNTAG > ${QI2_ROOT}/mirror.log 2>&1 < /dev/null &
set -u
PH=${1:?pod host}; PP=${2:?pod port}; PB=${3:?pod base}; M=${4:?member}; RT=${5:?run tag}
SSH="ssh -n -i ${QI2_ROOT}/id_rsa -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PP $PH"
PODDIR=$PB/states
LOC=${QI2_ROOT}/C32mirror_${M}_${RT}; TR=${QI2_ROOT}/${M}_${RT}; REC=${QI2_ROOT}/C32_${M}_${RT}
PY=${QI2_ROOT}/python
FLOOR_GB=200
mkdir -p $LOC $TR $REC
ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
log() { echo "$(ts) $*"; }
box_free_gb() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }

hf_upload() {  # $1 local file; prints one line; returns 0 only on size and sha256 match
  ( source ${QI2_ROOT}/gr_exports.sh > /dev/null 2>&1
    $PY - "$1" "busi/compC_C32_${M}_${RT}_slices/" <<'PY'
import os, sys, time, hashlib
from huggingface_hub import HfApi
f = sys.argv[1]; name = sys.argv[2] + os.path.basename(f)
api = HfApi(token=os.environ.get("HUGGING_FACE_HUB_TOKEN_PRIVATE"))
repo = "pseudotensor/qi2-fermion-bulk"
size = os.path.getsize(f)
h = hashlib.sha256()
with open(f, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 24), b""):
        h.update(blk)
sha = h.hexdigest(); t0 = time.time()
try:
    api.upload_file(path_or_fileobj=f, path_in_repo=name, repo_id=repo, repo_type="dataset", commit_message="busi lane: %s" % name)
    info = list(api.get_paths_info(repo_id=repo, paths=[name], repo_type="dataset"))
except Exception as e:
    print("upload error %s: %s" % (name, type(e).__name__)); sys.exit(2)
rsize = info[0].size if info else None
lfs = getattr(info[0], "lfs", None) if info else None
rsha = getattr(lfs, "sha256", None) if lfs is not None else None
ok = (rsize == size) and (rsha == sha)
print("%s: local %d bytes sha256 %s, remote %s bytes sha256 %s, %s (%.0f s)" % (name, size, sha, rsize, rsha, "MATCH" if ok else "MISMATCH", time.time() - t0))
sys.exit(0 if ok else 1)
PY
  )
}

log "c_mirror_v3 start pid $$ (pod $PH:$PP $PB member $M run $RT) (transit $TR; /data floor ${FLOOR_GB} GB; /home free $(df -BG --output=avail /home | tail -1 | tr -dc 0-9) GB untouched)"
while true; do
  # 1. finish any box copy whose upload has not yet matched
  for f in $TR/*.npz; do
    [ -e "$f" ] || continue
    if out=$(hf_upload "$f"); then log "HF $out"; rm -f "$f"; log "deleted box copy $(basename $f)"; echo "$(basename $f) uploaded $(ts)" >> $LOC/done.txt
    else log "HF retry later: $out"; fi
  done
  # 2. the records to the box (small files; overwritten each pass)
  for f in $($SSH "ls $PB/results/C32/ 2>/dev/null" 2>/dev/null); do
    scp -q -o BatchMode=yes -o ConnectTimeout=30 -P $PP $PH:$PB/results/C32/$f $REC/$f.part 2>/dev/null && mv $REC/$f.part $REC/$f
  done
  scp -q -o BatchMode=yes -o ConnectTimeout=30 -P $PP $PH:"$PB/status.txt $PB/logs/C32_${M}.log $PB/logs/guard_${M}.log" $REC/ 2>/dev/null
  # 3. the pod's slices and the member's state
  PID=$($SSH "cat $PB/C32_${M}.pid 2>/dev/null" 2>/dev/null)
  RUNNING=$($SSH "test -n '$PID' && test -d /proc/$PID && echo 1 || echo 0" 2>/dev/null)
  ORB=$($SSH "find $PODDIR -name 'orbitals_*.npz' -printf '%P\\n' 2>/dev/null | sort" 2>/dev/null)
  LIG=$($SSH "find $PODDIR -name 'light_*.npz' -printf '%P\\n' 2>/dev/null | sort" 2>/dev/null)
  USE=$($SSH "du -sBG $PB 2>/dev/null | cut -f1; df -BG --output=avail / | tail -1" 2>/dev/null | tr '\n' ' ')
  if [ "${RUNNING:-1}" = "0" ]; then LIST="$ORB $LIG"; else LIST="$(echo "$ORB" | head -n -2) $LIG"; fi
  log "pass: member running ${RUNNING:-?}; to mirror [$(echo $LIST)]; pod busi_compC use and container free $USE; box free $(box_free_gb) GB"
  for n in $LIST; do
    [ "$(ls $TR/*.npz 2>/dev/null | wc -l)" -ge 1 ] && break   # one slice in transit at a time
    s1=$($SSH "stat -c %s $PODDIR/$n" 2>/dev/null); sleep 120; s2=$($SSH "stat -c %s $PODDIR/$n" 2>/dev/null)
    [ -n "$s1" ] && [ "$s1" = "$s2" ] || { log "$n still growing ($s1 -> $s2)"; continue; }
    $SSH "python -c \"import zipfile,sys; zipfile.ZipFile(sys.argv[1]).namelist()\" $PODDIR/$n" 2>/dev/null || { log "$n zip not readable yet"; continue; }
    need=$(( s2 / 1073741824 + 1 + FLOOR_GB ))
    [ "$(box_free_gb)" -ge "$need" ] || { log "box free $(box_free_gb) GB < $need GB: pull of $n waits"; break; }
    PMD5=$($SSH "md5sum $PODDIR/$n | cut -d' ' -f1" 2>/dev/null)
    t0=$(date +%s)
    scp -q -o BatchMode=yes -o ConnectTimeout=30 -P $PP $PH:$PODDIR/$n $TR/$n.part && mv $TR/$n.part $TR/$n
    BMD5=$(md5sum $TR/$n 2>/dev/null | cut -d' ' -f1)
    if [ -n "$PMD5" ] && [ "$PMD5" = "$BMD5" ]; then
      $SSH "rm -f $PODDIR/$n" && log "$n copied ($s2 bytes, $(( $(date +%s) - t0 )) s), md5 $BMD5 MATCH, pod copy deleted"
      echo "$n $s2 $BMD5" >> $LOC/md5_pod_box.txt
      if out=$(hf_upload "$TR/$n"); then log "HF $out"; rm -f "$TR/$n"; log "deleted box copy $n"; echo "$n uploaded $(ts)" >> $LOC/done.txt
      else log "HF retry later: $out"; fi
    else
      log "$n md5 MISMATCH (pod $PMD5 box $BMD5): box copy removed, pod copy kept"; rm -f $TR/$n
    fi
  done
  if [ "${RUNNING:-1}" = "0" ] && [ -z "$(echo $ORB $LIG)" ] && [ "$(ls $TR/*.npz 2>/dev/null | wc -l)" -eq 0 ]; then
    log "member ended, every slice mirrored: c_mirror_v3 ends"; break
  fi
  sleep 600
done
