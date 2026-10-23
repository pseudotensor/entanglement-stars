#!/bin/bash
# busi, PERF RD-1c: the box GPU parts in sequence on one box GPU (1 or 2, never 0), taken when free: no compute process above 1000 MiB (the
# resident litellm processes on GPU 1 hold 260 MiB each) and utilization below 10 percent at 3 checks 60 s apart. 8 threads, nice 10.
#   (b) rd1c_percall_v1.py on the T0013.00 slice (kappa 1..10) and RD-4's card arm B5 records 13.34375, 13.625, 14.0 (kappa 1, 5, 10)
#   (c, box smoke) H12 = v9c7 SYM + tabulated + --heavy-eigh banded, then (a) G7 = v9c7 defaults, G6 = v9c6 defaults, then T12 = v9c6 SYM +
#   tabulated (H12's reference), all by run_seg_v9c7.sh on the same GPU; then the compares.
# Usage. rd1c_box_chain_v1.sh   (log ${QI2_ROOT}/box_chain.log)
ST=${QI2_ROOT}/v9c7; W=${QI2_ROOT}/RD1c_build; PY=${QI2_ROOT}/python
free_gpu() {  # prints the index of a free GPU among 1 2, or nothing
  for g in 1 2; do
    u=$(nvidia-smi --query-gpu=uuid,utilization.gpu --format=csv,noheader,nounits -i $g); uu=${u%%,*}; ut=${u##*, }
    big=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v U=$uu '$1==U && $2>1000' | wc -l)
    [ "$big" = 0 ] && [ "$ut" -lt 10 ] && { echo $g; return; }
  done
}
echo "chain start $(date -u +%FT%TZ)"
G=""; ok=0; last=""
while [ $ok -lt 3 ]; do
  g=$(free_gpu); if [ -n "$g" ] && [ "$g" = "$last" ]; then ok=$((ok+1)); elif [ -n "$g" ]; then ok=1; else ok=0; fi; last=$g; [ $ok -lt 3 ] && sleep 60
done
G=$last; echo "GPU $G free $(date -u +%FT%TZ): $(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader -i $G)"
export CUDA_VISIBLE_DEVICES=$G BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
echo "== (b) per call $(date -u +%FT%TZ)"
nice -n 10 $PY $ST/rd1c_percall_v1.py $W/percall --slice ${QI2_ROOT}/orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz \
  --run-dir ${QI2_ROOT}/RD4AB_B5_20261021T081537Z --records 13.34375,13.625,14.0 --record-channels 1,5,10 --gpu 0 > $W/logs/percall.log 2>&1
echo "rc $? $(grep 'rd1c_percall_v1 (b)' $W/logs/percall.log)"
for arm in H12 G7 G6 T12; do bash $ST/run_seg_v9c7.sh $arm $G; done
echo "== compares $(date -u +%FT%TZ)"
TAG=C64_s8_ell8_tau2_twofield_v4L; S=$W/seg; export CUDA_VISIBLE_DEVICES=""
echo "-- (a) G7 against G6 (bitwise)"; $PY $ST/cmp_v9c3_runs.py $S/G6/$TAG $S/G7/$TAG --args-equal
echo "-- (a) G7 against RD-4's stored G6 (box GPU 1, 21 Oct) and G6 against it"; $PY $ST/cmp_v9c3_runs.py ${QI2_ROOT}/$TAG $S/G7/$TAG --args-equal; $PY $ST/cmp_v9c3_runs.py ${QI2_ROOT}/$TAG $S/G6/$TAG --args-equal
echo "-- (a) no v9c7 perf line in G7's log after the command line: $(tail -n +2 $S/G7/engine_run.log | grep -c 'v9c7 perf')"
echo "-- box smoke H12 against T12 (cmp_rd4_v1, the sec. 7 budget lines)"; $PY $ST/cmp_rd4_v1.py $S/T12/$TAG $S/H12/$TAG --strip label,engine_v34 --top 8
echo "-- H12 json args against T12's: $($PY $ST/cmp_v9c3_runs.py $S/T12/$TAG $S/H12/$TAG --args-equal | grep -o 'args differ in [^;]*')"
echo "-- H12's v9c7 perf lines"; grep -h "v9c7 perf" $S/H12/engine_run.log | cut -c1-400
echo "chain done $(date -u +%FT%TZ)"
