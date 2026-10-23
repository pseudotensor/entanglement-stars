#!/bin/bash
# solve 8a7b5274: stream COMP1 h100-f check inputs box -> this machine -> h100-f (tar streams), then md5 both ends.
H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ProxyCommand='ssh -o HostKeyAlias=workvm -J compute-host -p 2222 compute-host -W %h:%p' -i ${QI2_ROOT}/id_rsa compute-host"
B=compute-host
R2D=${QI2_ROOT}/T0080.00
TAG=C32_s4_ell4_tau2_twofield_v3_off
S60=${QI2_ROOT}/slices
S50=${QI2_ROOT}/s4offc50_pod1off50
set -e
bash -c "$H 'mkdir -p ${QI2_ROOT}/compC ${QI2_ROOT}/compC_engine_v36 ${QI2_ROOT}/R2run ${QI2_ROOT}/slices'"
echo "code $(date -u)"
ssh $B "tar -C ${QI2_ROOT}/compC -cf - engine_twofield_v3.py engine_twofield_v8.py" | bash -c "$H 'tar -C ${QI2_ROOT}/compC -xf -'"
ssh $B "tar -C ${QI2_ROOT}/compC_engine_v36 --exclude=__pycache__ -cf - ." | bash -c "$H 'tar -C ${QI2_ROOT}/compC_engine_v36 -xf -'"
echo "run files $(date -u)"
ssh $B "tar -C $R2D -cf - collapse_$TAG.json collapse_${TAG}_fields.npz collapse_${TAG}_handoff.npz collapse_${TAG}_s11.npz collapse_${TAG}_spectra.npz MD5SUMS" | bash -c "$H 'tar -C ${QI2_ROOT}/R2run -xf -'"
echo "slice 60 $(date -u)"
ssh $B "tar -C $S60 -cf - orbitals_${TAG}_T0060.00.npz" | bash -c "$H 'tar -C ${QI2_ROOT}/slices -xf -'"
echo "slice 50 $(date -u)"
ssh $B "tar -C $S50 -cf - orbitals_${TAG}_T0050.00.npz" | bash -c "$H 'tar -C ${QI2_ROOT}/slices -xf -'"
echo "md5 $(date -u)"
ssh $B "cd ${QI2_ROOT}/code && md5sum compC/engine_twofield_v3.py compC/engine_twofield_v8.py && cd compC_engine_v36 && find . -type f ! -path './__pycache__/*' | sort | xargs md5sum | md5sum; cd $R2D && md5sum collapse_$TAG.json collapse_${TAG}_fields.npz collapse_${TAG}_handoff.npz collapse_${TAG}_s11.npz collapse_${TAG}_spectra.npz; md5sum $S60/orbitals_${TAG}_T0060.00.npz $S50/orbitals_${TAG}_T0050.00.npz" > ${QI2_ROOT}/md5_box.txt
bash -c "$H 'cd ${QI2_ROOT}/code && md5sum compC/engine_twofield_v3.py compC/engine_twofield_v8.py && cd compC_engine_v36 && find . -type f | sort | xargs md5sum | md5sum; cd ${QI2_ROOT}/R2run && md5sum collapse_$TAG.json collapse_${TAG}_fields.npz collapse_${TAG}_handoff.npz collapse_${TAG}_s11.npz collapse_${TAG}_spectra.npz; md5sum ${QI2_ROOT}/orbitals_${TAG}_T0060.00.npz ${QI2_ROOT}/orbitals_${TAG}_T0050.00.npz'" > ${QI2_ROOT}/md5_h100f.txt
echo "STAGE DONE $(date -u)"
