#!/bin/bash
# run.sh, kit c_twofield_v9c2 (solve bench_gputypes v1): the twofield engine engine_twofield_v9c2.py (6f4a0e28) at its defaults (= v9c1 84938595
# numbers) with --anchor-passage lz, c64s8L's segment from the regenerated T0013.00 slice (d17763e7) to T/rs END (default 13.5 = 16 records of
# tau/rs 2/64): pod_launch_C_v12t.sh (f28d75c2)'s segment mode (busi ready_jobs_v3 sec. 17 B0 / 20.6) reproduced without its 76,600 MiB card
# guard: the snapshot copy of the run files, its args.cut_transport rewritten record -> sector (cut_transport_before kept) and args.anchor_passage
# set to "lz" (anchor_passage_before kept), then the engine call with v12t's COMMON for c64s8L, lagged, sector, no slices.
# Engine modules: engine/ (325 files, LC_ALL=C list fingerprint c381cd18). Usage: bash run.sh GPU [NT] [END]
KIT=$(cd $(dirname $0) && pwd); CLASS=c_twofield_v9c2; ENVNAME=np1263; STACK="1.26.3 1.14.1 2.4.1+cu124"
. $KIT/bench_lib.sh
END=${3:-13.5}
TAG=C64_s8_ell8_tau2_twofield_v4L; SLICE=$KIT/input/orbitals_${TAG}_T0013.00.npz; SNAP=$D/snap
mkdir -p $SNAP $D/states; cp -p $KIT/input/run/collapse_${TAG}* $SNAP/
$PY -c "import json,sys; p=sys.argv[1]; J=json.load(open(p)); c=J['args'].get('cut_transport','record'); (c != 'sector') and (J['args'].__setitem__('cut_transport_before', c), J['args'].__setitem__('cut_transport', 'sector'), json.dump(J, open(p,'w'), default=float))" $SNAP/collapse_$TAG.json || exit 4
$PY -c "import json,sys; p=sys.argv[1]; k,v=sys.argv[2].split('=',1); J=json.load(open(p)); J['args'][k+'_before']=J['args'].get(k); J['args'][k]=json.loads(v); json.dump(J, open(p,'w'), default=float)" $SNAP/collapse_$TAG.json 'anchor_passage="lz"' || exit 4
export BUSI_ENGINE_DIR=$KIT/engine BUSI_PAGE_DIR=$KIT/engine PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$CLASS END $END NT $NT" > $D/kit.txt; md5sum $KIT/code/engine_twofield_v9c2.py >> $D/kit.txt
COMMON="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 56 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve lagged --cut-transport sector --light-max-pieces 8192"
cd $D && nice -n 10 $PY $KIT/bench_exec.py --gpu $GPU --out $D --log $D/log.txt -- $PY $KIT/code/engine_twofield_v9c2.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell 8 --slice-dir $D/states --gpu 0 --out $D --tag $TAG --resume-slice $SLICE --resume-run $SNAP/$TAG --segment-end-T-over-rs $END --anchor-passage lz
rc=$?; $PY $KIT/bench_cmp.py summary c $D; echo "run dir $D rc $rc"; exit $rc
