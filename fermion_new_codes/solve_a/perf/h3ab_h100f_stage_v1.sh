#!/bin/bash
# solve 8a7b5274, PERF H3 A/B on h100-f (lead inbox #253; busi ready_jobs_v3 sec. 35 / 20.3): stage BASE=${QI2_ROOT}/busi_compC on h100-f by
# streaming from the box (box -> this machine -> h100-f), then md5 every file on both ends. Overwrites nothing (refuses if BASE exists).
#   code/   = box compC/v9c4 stage (engine_twofield_v9c3c c6f15b2e, engine_twofield_v9c4 152ec1c6, assign_primitive_v1f 9c9abc1b, cmp_v9c3_runs 35045e35,
#             engine_twofield_v8_selftest for the compare), no __pycache__
#   engine/ = box compC_engine_v36 *.py (325 files, fingerprint 1973800e under en_US sort / c381cd18 under LC_ALL=C)
#   pod_launch_C_v12t.sh f28d75c2 (the spec's launcher; reference only, its segment mode is reproduced by h3ab_arm_v1.sh)
#   states/ = c64s8L T0013.00 slice d17763e7;  results/C64v4L/ = the c64s8L_T13 run files (json 0323356a)
H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ProxyCommand='ssh -o HostKeyAlias=workvm -J compute-host -p 2222 compute-host -W %h:%p' -i ${QI2_ROOT}/id_rsa compute-host"
B=compute-host; BASE=${QI2_ROOT}/busi_compC
ST=${QI2_ROOT}/v9c4; EN=${QI2_ROOT}/compC_engine_v36
LA=${QI2_ROOT}/pod_launch_C_v12t.sh; C=${QI2_ROOT}/c64s8L_T13
SL=orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz
SP=${QI2_ROOT}/h3ab
mkdir -p $SP; set -e
bash -c "$H 'test ! -e $BASE && mkdir -p $BASE/code $BASE/engine $BASE/states $BASE/results/C64v4L $BASE/gate'" || { echo "refusing: $BASE exists on h100-f"; exit 2; }
echo "code $(date -u)"
ssh $B "tar -C $ST --exclude=__pycache__ -cf - ." | bash -c "$H 'tar -C $BASE/code -xf -'"
ssh $B "tar -C $EN -cf - --exclude=__pycache__ \$(cd $EN && ls *.py)" | bash -c "$H 'tar -C $BASE/engine -xf -'"
ssh $B "cat $LA" | bash -c "$H 'cat > $BASE/pod_launch_C_v12t.sh'"
echo "run files $(date -u)"
ssh $B "tar -C $C/results/C64v4L -cf - ." | bash -c "$H 'tar -C $BASE/results/C64v4L -xf -'"
echo "slice $(date -u)"
ssh $B "cat $C/states/$SL" | bash -c "$H 'tee $BASE/states/$SL | md5sum > $BASE/states/slice.stream.md5'"
echo "md5 $(date -u)"
L='md5sum pod_launch_C_v12t.sh code/*.py states/*.npz results/C64v4L/*; (cd engine && ls | wc -l && md5sum *.py | sort -k2 | md5sum && LC_ALL=C md5sum *.py | LC_ALL=C sort -k2 | md5sum)'
ssh $B "set -e; T=\$(mktemp -d); cd \$T; cp -p $LA .; mkdir code engine states results; ln -s $ST/*.py code/; ln -s $EN/*.py engine/; ln -s $C/states/$SL states/; ln -s $C/results/C64v4L results/C64v4L; $L; cd /; rm -rf \$T" > $SP/stage_box.md5
bash -c "$H 'cd $BASE && $L'" > $SP/stage_h100f.md5
cat $SP/stage_h100f.md5 | bash -c "$H 'cat > $BASE/STAGE_MD5_h100f.txt'"; cat $SP/stage_box.md5 | bash -c "$H 'cat > $BASE/STAGE_MD5_box.txt'"
bash -c "$H 'cat $BASE/states/slice.stream.md5'"
if cmp $SP/stage_box.md5 $SP/stage_h100f.md5; then echo "STAGE MD5 EQUAL ($(wc -l < $SP/stage_box.md5) lines)"; else diff $SP/stage_box.md5 $SP/stage_h100f.md5; echo "STAGE MD5 DIFFER"; exit 3; fi
echo "STAGE DONE $(date -u)"
