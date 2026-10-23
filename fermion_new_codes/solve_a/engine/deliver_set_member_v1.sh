#!/bin/bash
# Delivers one r_s 32 set member's COLLAPSE_records_v1 (engine spec 20.5 (1), "then the set") from the box into the deliveries folder: the
# records md, the data npz, the masks and eigenvalues npz, the replay's exactness table and the rows' table (its title naming engine_rows_v2),
# one LARGE_FILES line per npz and one DELIVERIES.md line. Run from the worktree's solve/. Arg: the member's tag. Builder analyst.
set -e
t=$1
D=${QI2_ROOT}/solve_a
B=compute-host:${QI2_ROOT}/engine
scp -q $B/COLL/COLLAPSE_records_v1_$t.md $D/
scp -q $B/COLL/COLLAPSE_records_v1_${t}_data.npz $D/collapse_records/
scp -q $B/H100b/REPLAY2/replay2_${t}_spectra.npz $B/H100b/REPLAY2/replay2_${t}.md $D/collapse_records/
scp -q $B/ROWS/rows_${t}_v1.md results/engine/ROWS/
sed "s/(engine_rows_v1.py; the stresses from/(engine_rows_v2.py, whose records mode is version 1's unchanged; the stresses from/" results/engine/ROWS/rows_${t}_v1.md > $D/collapse_records/rows_${t}_v1.md
cp $D/COLLAPSE_records_v1_$t.md results/engine/COLL/
for f in $D/collapse_records/COLLAPSE_records_v1_${t}_data.npz $D/collapse_records/replay2_${t}_spectra.npz; do
  s=$(stat -c %s "$f"); h=$(sha256sum "$f" | cut -d' ' -f1); b=$(basename $f)
  printf '%s\t%s\t%s\t%s\t%s\n' "deliveries/solve_a/collapse_records/$b" "$s" "$h" "this host: $f" "COLLAPSE_records_v1 of the r_s 32 member $t (the data or the masks and eigenvalues); never in git" >> results/LARGE_FILES.tsv
done
n=$(grep -c "^| [0-9]" $D/collapse_records/replay2_${t}.md || true)
ex=$(grep -o "the largest source differences from the stored shell totals: eps [0-9.e+-]*, J [0-9.e+-]*" $D/collapse_records/replay2_${t}.md || true)
echo "- $(date -u '+%Y-%m-%d %H:%M UTC') solve_a: to .lane and .hreduce, COLLAPSE_records_v1 for the r_s 32 member $t (COLLAPSE_records_v1_$t.md with collapse_records/: the data npz, the masks and eigenvalues, the replay's exactness table ($n records; $ex), the rows' table; engine spec 20.5 (1); the labels of 22.3 and 22.4)." >> ${QI2_ROOT}/DELIVERIES.md
echo "delivered $t"
