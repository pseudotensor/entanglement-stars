#!/bin/bash
# slice_keep_last2_v1.sh (busi, analyst; ReviewA_s4fail_v1 sec. 3: "a slice every 0.5 r_s, keeping the last two"): beside a running member,
# every 5 minutes (and once after the engine exits) keeps the two newest COMPLETE slices in M/states and deletes the older ones. A slice is
# complete when its size is unchanged over 60 s and a newer one, or the engine's exit, follows it; the newest is never touched while the
# engine runs. Only regular files named orbitals_<TAG>_T*.npz in M/states are candidates: never a symlink, never anything outside states/
# (the resume slice lives in RES). Every deletion is logged (name, size, mtime) to M/logs/slice_keep.tsv.
#   bash slice_keep_last2_v1.sh M TAG ENGINE_PID
set -u
M=$1; TAG=$2; EP=$3; LOGF=$M/logs/slice_keep.tsv
prune() {
  mapfile -t S < <(ls $M/states/orbitals_${TAG}_T*.npz 2>/dev/null | sort)
  n=${#S[@]}; running=$1
  # the complete set: all but the newest while the engine runs; every file once it has exited (sizes stable over 60 s)
  [ $running = 1 ] && n=$((n - 1))
  [ $n -le 2 ] && return
  for ((k = 0; k < n - 2; k++)); do
    f=${S[$k]}; [ -L "$f" ] && continue; [ -f "$f" ] || continue
    s1=$(stat -c %s "$f"); sleep 60; s2=$(stat -c %s "$f"); [ "$s1" = "$s2" ] || continue
    echo -e "$(date -u +%FT%TZ)\tdelete\t$(basename $f)\t$s2\t$(stat -c %y "$f")" >> $LOGF
    rm -f -- "$f"
  done
}
while kill -0 $EP 2>/dev/null; do sleep 300; prune 1; done
prune 0
echo "$(date -u +%FT%TZ) engine $EP gone; final prune done" >> $LOGF
