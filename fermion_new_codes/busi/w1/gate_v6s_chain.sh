#!/bin/bash
# gate_v6s_chain.sh (busi lane, inbox #309): the v6s gates in series on one GPU (one compute process at a time): A v6q and B v6s at the defaults
# (kappas 1-2, T 0 to 2 r_s) and their bitwise comparison; C the interior readout with the dense and exterior-side checks (kappa 1, T 0 to 10 r_s);
# D the kappa = 0 smoke with the checks (T 0 to 4 r_s); E the lean interior mode on the same chain (kappa 1, T 0 to 2 r_s, readouts every 2 r_s).
GPU=${1:-2}
W=${QI2_ROOT}/page_verify; C=${QI2_ROOT}/code_pod7; S=$C/code_d13; PY=${QI2_ROOT}/python
cd $W
bash $S/gate_v6s_box.sh A_v6q page_history_v6q.py $GPU 1-2 2
bash $S/gate_v6s_box.sh B_v6s page_history_v6s.py $GPU 1-2 2
$PY $S/cmp_v6s_gate.py runs/A_v6q runs/B_v6s > logs/cmp_A_B.txt 2>&1; echo "cmp A B exit $? $(date -u +%FT%TZ)" >> status.txt
bash $S/gate_v6s_box.sh C_int page_history_v6s.py $GPU 1-1 10 --interior-readout check
bash $S/gate_v6s_box.sh D_k0 page_history_v6s.py $GPU 1-1 4 --kappa-zero --interior-readout check
bash $S/gate_v6s_box.sh E_lean page_history_v6s.py $GPU 1-1 2 --lean-interior --interior-every-rs 2
echo "chain done $(date -u +%FT%TZ)" >> status.txt
