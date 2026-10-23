#!/bin/bash
# make_kit_s16A_v1.sh (busi; assembles the B' s = 16 stage-A kit on the box: the benchmark kit's code unchanged, the three F1 preps, the
# stage-A declarations (manifest, hole packet, covariances) and run_s16A_v1.sh).
set -eu
CODE=${QI2_ROOT}/code
PREP=${QI2_ROOT}/prep_v4
DECL=${QI2_ROOT}/stageA_decl_v1
KIT=${1:-${QI2_ROOT}/kit_s16A_v1}
rm -rf "$KIT"; mkdir -p "$KIT/v3/engine_copy" "$KIT/code_v6" "$KIT/prep"
for f in run_bprime_v4.py pt3_core.py pt3b_cube.py pt3b_kernels.py pt3b_v2lib.py; do cp "$CODE/v3/$f" "$KIT/v3/"; done
cp "$CODE/v3/engine_copy/"*.py "$KIT/v3/engine_copy/"
cp "$CODE/code_v6/page_ops_v6.py" "$KIT/code_v6/"
cp "$CODE/v3/run_s16A_v1.sh" "$KIT/"
for t in F1_s16_d0 F1_s16_d0_eta2_bb F1_s16_d0_onset225_bb; do cp "$PREP/${t}_prep.npz" "$PREP/${t}_prep.json" "$KIT/prep/"; done
cp "$DECL"/stageA_manifest_v1.json "$DECL"/hole_packet_v1.npz "$DECL"/cin_*_v1.npz "$KIT/prep/"
cd "$KIT" && find . -type f ! -name MD5SUMS | sort | xargs md5sum > MD5SUMS
cd "$(dirname "$KIT")" && tar czf "$(basename "$KIT").tgz" "$(basename "$KIT")" && md5sum "$(basename "$KIT").tgz"
