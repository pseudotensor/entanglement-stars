#!/bin/bash
# prints the number of this lane's busy compute processes: python processes of jon whose working directory is under ${QI2_ROOT}/solve_a
# and whose lifetime CPU share exceeds 30 percent (idle parents and pool managers excluded).  Builder analyst.
n=0
while read -r pid pcpu comm; do
  [ "$comm" = "python" ] || continue
  awk -v c="$pcpu" "BEGIN{exit !(c > 30)}" || continue
  d=$(readlink /proc/$pid/cwd 2>/dev/null)
  case "$d" in ${QI2_ROOT}/solve_a*) n=$((n+1));; esac
done < <(ps -u jon -o pid=,pcpu=,comm=)
echo $n
