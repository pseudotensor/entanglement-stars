#!/bin/bash
# gate_v6r_chain.sh (busi PERF PG-A1..A3): after arm A (v6q) ends, the other arms one at a time on the same GPU, then the comparisons.
S=${QI2_ROOT}/v6r; W=${QI2_ROOT}/gate_v6r_box; PY=${QI2_ROOT}/python; GPU=${1:-1}
until grep -q "^A_v6q exit" $W/status.txt; do sleep 20; done
bash $S/gate_v6r_box.sh B_v6r page_history_v6r.py $GPU 4 2
bash $S/gate_v6r_box.sh C_v6o page_history_v6o.py $GPU 4 2
bash $S/gate_v6r_box.sh D_v6r_off page_history_v6r.py $GPU 4 2 --v6r-share-traces off --v6r-full-cache off
bash $S/gate_v6r_box.sh E_v6r_pc256 page_history_v6r.py $GPU 4 2 --prop-chunk 256
cd $W
for p in "A_v6q B_v6r" "A_v6q D_v6r_off" "A_v6q E_v6r_pc256" "C_v6o B_v6r --code-names-differ" "C_v6o A_v6q --code-names-differ"; do
  set -- $p; $PY $S/cmp_v6r_gate.py runs/$1 runs/$2 $3 > logs/cmp_$1_vs_$2.log 2>&1; echo "cmp $1 vs $2 rc $? $(tail -1 logs/cmp_$1_vs_$2.log)" >> status.txt
done
echo "CHAIN DONE $(date -u +%FT%TZ)" >> status.txt
