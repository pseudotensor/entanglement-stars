#!/bin/bash
# solve 8a7b5274, COMP1 h100-f checks: run ONLY after the coordinator's "GO h100-f".
# 1) copy the two orbitals slices box -> this machine -> h100-f (md5 computed on the received stream); 2) verify every input md5 on h100-f
# against the box values below; 3) build the run scripts and launch run_all.sh (check1_v3, check1_v8, check2_v3, one after the other, GPU 0).
H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ProxyCommand='ssh -o HostKeyAlias=workvm -J compute-host -p 2222 compute-host -W %h:%p' -i ${QI2_ROOT}/id_rsa compute-host"
B=compute-host
TAG=C32_s4_ell4_tau2_twofield_v3_off
S60=${QI2_ROOT}/orbitals_${TAG}_T0060.00.npz
S50=${QI2_ROOT}/orbitals_${TAG}_T0050.00.npz
DS=${QI2_ROOT}/slices
set -e
echo "copy T60 $(date -u)"
ssh $B "cat $S60" | bash -c "$H 'tee $DS/orbitals_${TAG}_T0060.00.npz | md5sum > $DS/T60.stream.md5'"
echo "copy T50 $(date -u)"
ssh $B "cat $S50" | bash -c "$H 'tee $DS/orbitals_${TAG}_T0050.00.npz | md5sum > $DS/T50.stream.md5'"
echo "verify $(date -u)"
bash -c "$H 'cat > ${QI2_ROOT}/inputs_expected.md5'" <<M
6c276323cf9cba10aefd68db77d61aa5  ${QI2_ROOT}/orbitals_${TAG}_T0060.00.npz
04385a846dfe358ff6575039b4efa065  ${QI2_ROOT}/orbitals_${TAG}_T0050.00.npz
b9e675aaa2dbb94bc6d12ba6cac7fa9a  ${QI2_ROOT}/collapse_${TAG}.json
95eac8352a27dd845964239a21ebd4f1  ${QI2_ROOT}/collapse_${TAG}_fields.npz
d0c700c2909a9344db393d7299ec76ed  ${QI2_ROOT}/collapse_${TAG}_handoff.npz
760a7fd80126e2355ef4f1ee61c0f365  ${QI2_ROOT}/collapse_${TAG}_spectra.npz
3f360fa38e31e9c50dd859eb9c634954  ${QI2_ROOT}/collapse_${TAG}_s11.npz
2e3e74fe597799bfaaf9f74dafb2e9d2  ${QI2_ROOT}/engine_twofield_v3.py
7305db4c1303ca4445377c752ef60f6a  ${QI2_ROOT}/engine_twofield_v8.py
69c38fbb6aa0f0acef69579df16e2091  ${QI2_ROOT}/engine_collapse_v36.py
fc6faf26a353f6d904a5242874e25586  ${QI2_ROOT}/engine_collapse_v5.py
d142154a1e454fed344f1b0ddacf71e3  ${QI2_ROOT}/engine_mixing_v1.py
460664cd52b8babd42f9a9c227499662  ${QI2_ROOT}/engine_ops_v3.py
3db7b1dbb174a7a1b79c6b865d535ade  ${QI2_ROOT}/engine_ops_v34_cert.py
c5759e7852fa5446ba22b6f72ab4d869  ${QI2_ROOT}/engine_stress_v2.py
5225e959b9e0bc730062041d60d800c8  ${QI2_ROOT}/engine_vacflow_v1.py
b436094601ba130bf54d920ce6892264  ${QI2_ROOT}/page_ops_v5.py
M
bash -c "$H 'cd ${QI2_ROOT}/comp1 && cat in/slices/T60.stream.md5 in/slices/T50.stream.md5 && md5sum -c inputs_expected.md5 > inputs_md5_check.txt 2>&1; cat inputs_md5_check.txt; (cd code/compC_engine_v36 && find . -type f | sort | xargs md5sum | md5sum) | tee -a inputs_md5_check.txt; grep -q FAILED inputs_md5_check.txt && { echo MD5 FAILED; exit 3; }; grep -q 8c9a09e6f8c1632ea1eaf02e34f3c999 inputs_md5_check.txt || { echo ENGINE DIR TREE MD5 MISMATCH; exit 4; }; echo ALL INPUT MD5 OK'"
echo "scripts $(date -u)"
SD=${QI2_ROOT}/compositions
for f in h100f_launch_checks_v1.py h100f_compare_check1_v1.py h100f_compare_check2_v1.py; do cat $SD/$f | bash -c "$H 'cat > ${QI2_ROOT}/$f'"; done
bash -c "$H 'cd ${QI2_ROOT}/comp1 && md5sum h100f_*.py'"; (cd $SD && md5sum h100f_*.py)
echo "launch $(date -u)"
bash -c "$H 'cd ${QI2_ROOT}/comp1 && ${QI2_ROOT}/python -I h100f_launch_checks_v1.py > launch.out 2>&1 && cat launch.out && (nohup bash run_all.sh > run_all.out 2>&1 < /dev/null &) ; sleep 2; ps -u qi2 -o pid,etime,args | grep [e]ngine_twofield'"
echo "GO DONE $(date -u)"
