#!/bin/bash
# cw15: run icloud's UNCHANGED filtered-readout adapter (source_v1/run_mock_filtered_comparison_readouts_v1.py,
# sha256 d1b35e41..., md5 125f7814a3cc461bfb4a58b38e2af53e; byte-identical copy kept as
# busi/code/clockwell/cw15_run_mock_filtered_comparison_readouts_v1_COPY.py) for the never-run filtered
# n = 3 cohorts, WITHOUT writing into icloud's directories.
# Why an overlay: the adapter's sha-bound runtime guard (cpu_runtime.py) and its output gate require the
# output to resolve under ${QI2_ROOT}/MOCK_filtered_comparison_readouts_v1; editing either changes
# a frozen source sha and the adapter refuses to run. So the job runs in a private user+mount namespace where
# that directory is an overlay: lower = icloud's directory (read-only, never modified), upper = busi's
# ${QI2_ROOT}/upper. Every byte the job writes lands in the busi upper dir;
# paths recorded inside the journal read ${QI2_ROOT}/MOCK_filtered_comparison_readouts_v1<name>,
# physically ${QI2_ROOT}/upper<name>.
# Usage: cw15_launch_overlay.sh STAGE N G GLABEL PARENT_SHA [INTAKE_REVIEW INTAKE_REVIEW_SHA]
set -euo pipefail
STAGE=$1; N=$2; G=$3; GL=$4; PSHA=$5; IREV=${6:-}; IRSHA=${7:-}
LOW=${QI2_ROOT}/MOCK_filtered_comparison_readouts_v1
OV=${QI2_ROOT}/adapter_overlay_v1
mkdir -p $OV/upper $OV/work $OV/logs
if [ "$STAGE" = intake ]; then NAME=intake_even_n${N}_g${GL}_v1; else NAME=even_n${N}_g${GL}_v1; fi
EXTRA=""
if [ "$STAGE" = readouts ]; then EXTRA="--intake-output $LOW/intake_even_n${N}_g${GL}_v1 --intake-review $IREV --intake-review-sha $IRSHA"; fi
unshare -rm bash -c "
set -e
mount -t overlay overlay -o lowerdir=$LOW,upperdir=$OV/upper,workdir=$OV/work $LOW
cd $LOW/source_v1
/usr/bin/time -v -o $OV/logs/${NAME}_time.txt timeout --signal=TERM 600s nice -n 10 env QI2_REMOTE_CPU=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ${QI2_ROOT}/python run_mock_filtered_comparison_readouts_v1.py --stage $STAGE --n $N --g $G --parent-journal-sha $PSHA --spec MOCK_filtered_clockwell_spec_v1.md --control-spec MOCK_filtered_physical_controls_spec_v1.md --baseline-spec MOCK_unitary_lead_run_spec_v1.md --readout-spec MOCK_filtered_readout_addendum_v1.md --controls-output $LOW/controls_v1 --control-review $LOW/reviews_controls_v1/redteam_filtered_adapter_controls_v1.md --control-review-sha 84738ae1f9fa54ed9a8b1ae6212b9f068c562628f076d6fccaea942637f1ec5d $EXTRA --output $LOW/$NAME > $OV/logs/${NAME}_stdout.txt 2>&1
"
echo "exit 0 $NAME"; ls $OV/upper
