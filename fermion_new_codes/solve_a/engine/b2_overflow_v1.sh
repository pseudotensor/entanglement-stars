#!/bin/bash
# b2_overflow_v1.sh (pod 4; solve builder 5): B2 stopped at 17:18 UTC when pod 4's /workspace quota (94 GB) filled (the box mirror waited on
# the box's free-space floor since 12:43 UTC). Frees /workspace for B2's restart: the crash run's 4 output files copied to ${QI2_ROOT}/B2crash (the
# restart's --resume-from); slices T24 and T28 and the crash run's T52 (record 52.0 never reached the json) moved to the container disk
# ${QI2_ROOT}/B2_overflow by copy, cmp, then delete (T52 renamed _crashrun). Log ${QI2_ROOT}/b2_overflow_v1.log.
set -u
S=${QI2_ROOT}/c19_rs64_A_R2_tau8_ell4_avg_B2n4096; O=${QI2_ROOT}/B2_overflow; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
mkdir -p $O ${QI2_ROOT}/B2crash
ts() { date -u '+%H:%M:%S UTC'; }
for f in collapse_${T}.json collapse_${T}_fields.npz collapse_${T}_handoff.npz collapse_${T}_spectra.npz ${T}.log; do
  cp -p ${QI2_ROOT}/$f ${QI2_ROOT}/$f && cmp -s ${QI2_ROOT}/$f ${QI2_ROOT}/$f && echo "$(ts) crash copy $f ok" || echo "$(ts) crash copy $f FAILED"
done
for p in "T0024.00 T0024.00" "T0028.00 T0028.00" "T0052.00 T0052.00_crashrun"; do
  set -- $p; src=$S/orbitals_${T}_$1.npz; dst=$O/orbitals_${T}_$2.npz
  cp $src $dst.part && cmp -s $src $dst.part && mv $dst.part $dst && rm -f $src && echo "$(ts) moved $1 -> $dst (cmp equal)" || echo "$(ts) move $1 FAILED (source kept)"
done
echo "$(ts) done; workspace $(df -h /workspace | tail -1 | awk '{print $3" used "$4" free"}')"
