#!/bin/bash
# solve 8a7b5274, PERF H3 A/B on h100-f: mirror the outputs h100-f ${QI2_ROOT}/H3AB_*, scripts, logs, reads, stage md5s} ->
# box ${QI2_ROOT}/h100f through this machine, then md5 both ends (inputs are box files already: not copied).
H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ProxyCommand='ssh -o HostKeyAlias=workvm -J compute-host -p 2222 compute-host -W %h:%p' -i ${QI2_ROOT}/id_rsa compute-host"
B=compute-host; M=${QI2_ROOT}/H3_AB_h100f_v1
SP=${QI2_ROOT}/h3ab
set -e
LIST='gate/H3AB_* h3ab_* code/h3ab_ptracer_v1.py threads.env run_all.out H3AB_read_v1.json STAGE_MD5_*.txt'
bash -c "$H 'cd ${QI2_ROOT}/busi_compC && tar -cf - $LIST 2>/dev/null'" | ssh $B "mkdir -p $M/h100f && tar -C $M/h100f -xf -"
bash -c "$H 'cd ${QI2_ROOT}/busi_compC && find $LIST -type f | sort | xargs md5sum'" > $SP/mirror_h100f.md5
ssh $B "cd $M/h100f && find $LIST -type f | sort | xargs md5sum" > $SP/mirror_box.md5
cmp $SP/mirror_h100f.md5 $SP/mirror_box.md5 && echo "MIRROR MD5 EQUAL ($(wc -l < $SP/mirror_box.md5) files)"
cat $SP/mirror_h100f.md5 | ssh $B "cat > $M/MD5SUMS_h100f.txt"
