#!/bin/bash
# make_kit_s16A_v3.sh (busi; = v2 with the DTP-parameterized arm script; the B' s = 16 stage-A kit with checkpoint/resume: run_bprime_v5.py and run_s16A_arm_v2.sh; box only).
set -eu
CODE=${QI2_ROOT}/code
PREP=${QI2_ROOT}/prep_v5
DECL=${QI2_ROOT}/stageA_decl_v2
KIT=${1:-${QI2_ROOT}/kit_s16A_v3}
rm -rf "$KIT"; mkdir -p "$KIT/v3/engine_copy" "$KIT/code_v6" "$KIT/prep"
for f in run_bprime_v5.py pt3_core.py pt3b_cube.py pt3b_kernels.py pt3b_v2lib.py ladder_read_v1.py; do cp "$CODE/v3/$f" "$KIT/v3/"; done
cp "$CODE/v3/engine_copy/"*.py "$KIT/v3/engine_copy/"
cp "$CODE/code_v6/page_ops_v6.py" "$KIT/code_v6/"
cp "$CODE/v3/run_s16A_arm_v2.sh" "$CODE/v3/run_ladder_v1.sh" "$KIT/"
for t in F1_s16_d0 F1_s16_d0_eta2_bb F1_s16_d0_onset225_bb; do for w in "" _sup _exit; do cp "$PREP/${t}${w}_prep.npz" "$PREP/${t}${w}_prep.json" "$KIT/prep/"; done; done
cp "$DECL"/stageA_manifest_v1.json "$DECL"/hole_packet_v1.npz "$DECL"/cin_*_v1.npz "$KIT/prep/"
cd "$KIT" && find . -type f ! -name MD5SUMS | sort | xargs md5sum > MD5SUMS
cd "$(dirname "$KIT")" && tar czf "$(basename "$KIT").tgz" "$(basename "$KIT")" && md5sum "$(basename "$KIT").tgz"
