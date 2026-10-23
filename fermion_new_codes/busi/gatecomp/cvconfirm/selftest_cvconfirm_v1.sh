#!/bin/bash
# selftest_cvconfirm_v1.sh (busi, analyst; CVCONFIRM, ReviewA_cvcause_v1 sec. 3): the field injection's self-test ON THE BOX CPU
# (CUDA_VISIBLE_DEVICES empty, no --gpu, nice 10, at most 8 threads in all), a small fixture of CV2048's shape: CV2048.args' block word
# for word with r_s/a 16, ell 2, prep-ell 2, tau 1 (tau/r_s 1/16 as CV2048), K 10, --dust-set isotropic, --fields collocation, and the
# two CPU substitutions --eigh dense (banded needs CUDA) and --pair-workers 0 (the thread budget). X-fixture = 256 sites (16 r_s), the
# source P-fixture = 512 sites (32 r_s): X's chain is the inner half of P's, as CV2048's is job (b)'s.
#   Xf   parent 248f25f3, fresh, T 0 -> 0.3125, slice 0.0625                       (X)
#   Sf   parent, 512 sites, fresh, T 0 -> 0.125, slice 0.125
#   Su   parent, resumed from Sf's 0.125 slice, --nk-maxiter 0, one record (0.1875), slice 0.1875: an unconverged kept iterate, as job (b)'s 3.3125
#   Su2  parent, resumed from Su's 0.1875 slice with the line's own Newton block, to 0.3125     (Su2's run files = the P-fixture's records)
#   A1P  parent, resumed from Xf's 0.0625 slice to 0.3125
#   A1   rd13mfi, option ABSENT, the same                    test (a): A1 = A1P bitwise (cmp_prefix_v1: json records and npz rows)
#   A2   rd13mfi, --inject-fields <Su2's 0.1875 extract> --inject-T-over-rs 0.1875, the same
#                                                            test (b): A2's 0.1875 applied fields = the source's at the 256 mapped sites, bitwise;
#                                                                      A2's 0.125 record (before the injection) = A1's bitwise; the comparator on (Xf, Su2, A1, A2)
#   A2s  rd13mfi, injecting A1P's own 0.1875 row (self-injection)    test (c): every npz row = A1's; json records = A1's but the field_inject entry
#   R*   refusals at start (test (d)): T off the clock; T at the slice's clock (from a full fields npz); a source whose radii are shifted; the option half given
set -u
C=${QI2_ROOT}/cvconfirm; CODE=$C/code16d_cvc; K=$C/code; PY=${QI2_ROOT}/python; O=$C/selftest
[ -e $O ] && { echo "refusing: $O exists"; exit 1; }
mkdir -p $O
P0=engine_collapse_v36nq_rd13mf.py; P1=engine_collapse_v36nq_rd13mfi.py
B="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 10 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes --T-over-rs 0.3125 \
--rs 16 --ell 2 --prep-ell 2 --tau 1 --dust-set isotropic --mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact \
--nk-line-search halving --nk-halvings 6 --nk-restarts 2 --nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean \
--end-above 0.001 --pair-workers 0 --fields collocation --eigh dense --jvp eigenbasis --nk-cache-host off"
BU=$(echo "$B" | sed 's/--nk-maxiter 30/--nk-maxiter 0/')   # no Newton step: the kept iterate is the first solve's output (residual about 2.5e-5 here), unconverged
cd $CODE; md5sum $P0 $P1 engine_fields_v16.py > $O/md5_inputs.txt
run() { local n=$1 thr=$2 eng=$3; shift 3; mkdir -p $O/$n/states; echo "$eng $*" > $O/$n/cmd.txt
  ( cd $CODE && env CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr NUMBA_NUM_THREADS=$thr PYTHONDONTWRITEBYTECODE=1 \
      nice -n 10 $PY -u $CODE/$eng "$@" --out $O/$n --slice-dir $O/$n/states --tag fx > $O/$n/log.txt 2>&1 )
  echo "$(date -u +%FT%TZ) $n rc $?" >> $O/DONE.txt; }
sl() { ls $O/$1/states/orbitals_fx_T*.npz | head -1; }   # each run writes one slice
: > $O/DONE.txt
echo "$(date -u +%FT%TZ) start" >> $O/DONE.txt
run Xf 4 $P0 $B --nsites 256 --slices-c128 0.0625 &
( run Sf 4 $P0 $B --nsites 512 --segment-end-T-over-rs 0.125 --slices-c128 0.125
  run Su 4 $P0 $BU --nsites 512 --resume-slice $(sl Sf 0000.12) --resume-run $O/Sf/fx --segment-end-T-over-rs 0.1875 --slices-c128 0.1875
  run Su2 4 $P0 $B --nsites 512 --resume-slice $(sl Su 0000.19) --resume-run $O/Su/fx --segment-end-T-over-rs 0.3125 --slices-c128 40 ) &
wait
XS=$(sl Xf 0000.06); RV="$B --nsites 256 --resume-slice $XS --resume-run $O/Xf/fx --segment-end-T-over-rs 0.3125 --slices-c128 40"
nice -n 10 $PY $K/inject_extract_v1.py $O/Su2 fx 0.1875 $O/inject_Su2_T0.1875.npz > $O/extract.log 2>&1
run A1P 4 $P0 $RV &
run A1 4 $P1 $RV &
wait
nice -n 10 $PY $K/inject_extract_v1.py $O/A1P fx 0.1875 $O/inject_A1P_T0.1875.npz >> $O/extract.log 2>&1
run A2 4 $P1 $RV --inject-fields $O/inject_Su2_T0.1875.npz --inject-T-over-rs 0.1875 &
run A2s 4 $P1 $RV --inject-fields $O/inject_A1P_T0.1875.npz --inject-T-over-rs 0.1875 &
wait
# refusals (each exits at the option's checks; nothing evolves)
$PY - $O/inject_Su2_T0.1875.npz $O/inject_shifted.npz <<'PYX'
import sys, numpy as np
Z = np.load(sys.argv[1], allow_pickle=True); d = {k: Z[k] for k in Z.files}; d["R"] = d["R"] + 0.5
np.savez(sys.argv[2], **d)
PYX
run R_offclock 2 $P1 $RV --inject-fields $O/inject_Su2_T0.1875.npz --inject-T-over-rs 0.19
run R_atslice 2 $P1 $RV --inject-fields $O/A1P/collapse_fx_fields.npz --inject-T-over-rs 0.0625   # the full fields npz (holds 0.0625)
run R_shifted 2 $P1 $RV --inject-fields $O/inject_shifted.npz --inject-T-over-rs 0.1875
run R_half 2 $P1 $RV --inject-fields $O/inject_Su2_T0.1875.npz
echo "$(date -u +%FT%TZ) ALL DONE" >> $O/DONE.txt
