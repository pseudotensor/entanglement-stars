#!/bin/bash
# solve 8a7b5274: mirror h100-f ${QI2_ROOT}/out (+ scripts, logs) -> box ${QI2_ROOT}/C32_h100f_checks_v1 through
# this machine, then md5 both ends. Idempotent (re-streams everything; slices are 3.5 GB each).
H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ProxyCommand='ssh -o HostKeyAlias=workvm -J compute-host -p 2222 compute-host -W %h:%p' -i ${QI2_ROOT}/id_rsa compute-host"
B=compute-host; M=${QI2_ROOT}/C32_h100f_checks_v1
set -e
bash -c "$H 'cd ${QI2_ROOT}/comp1 && tar -cf - out *.sh *.py *.log *.out *.txt *.md5 2>/dev/null'" | ssh $B "mkdir -p $M/h100f && tar -C $M/h100f -xf -"
bash -c "$H 'cd ${QI2_ROOT}/comp1 && find out -type f | sort | xargs md5sum'" > ${QI2_ROOT}/mirror_h100f.md5
ssh $B "cd $M/h100f && find out -type f | sort | xargs md5sum" > ${QI2_ROOT}/mirror_box.md5
cmp ${QI2_ROOT}/mirror_h100f.md5 ${QI2_ROOT}/mirror_box.md5 && echo "MIRROR MD5 EQUAL ($(wc -l < ${QI2_ROOT}/mirror_box.md5) files)"
cat ${QI2_ROOT}/mirror_h100f.md5 | ssh $B "cat > $M/MD5SUMS_h100f_out.txt"
