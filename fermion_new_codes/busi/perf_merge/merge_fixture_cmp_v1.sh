#!/bin/bash
# merge_fixture_cmp_v1.sh (busi, MERGE-1): the fixture comparisons. Each pair "SET:A SET:B" by cmp_rd13m_v1.py (records and head but the
# walls, the fields, spectra and handoff npz and the 0.125 slice, every array np.array_equal) -> fixture/cmp/A_vs_B.{txt,json}, the verdict
# line appended to fixture/cmp/summary.txt.   merge_fixture_cmp_v1.sh "F1:M0 F1:R0" "F1:Mb F1:Rb" ...
M=${QI2_ROOT}/MERGE_build; PY=${QI2_ROOT}/python
mkdir -p $M/fixture/cmp
for P in "$@"; do
  set -- $P; sa=${1%%:*}; a=${1##*:}; sb=${2%%:*}; b=${2##*:}
  A=$M/fixture/$sa/$a; B=$M/fixture/$sb/$b
  nice -n 10 env OMP_NUM_THREADS=1 $PY $M/build/cmp_rd13m_v1.py $A fx_$a $B fx_$b --slices-a $A/states --slices-b $B/states --out $M/fixture/cmp/${a}_vs_${b}.json > $M/fixture/cmp/${a}_vs_${b}.txt 2>&1
  echo "$a vs $b rc $?: $(head -1 $M/fixture/cmp/${a}_vs_${b}.txt)" >> $M/fixture/cmp/summary.txt
done
