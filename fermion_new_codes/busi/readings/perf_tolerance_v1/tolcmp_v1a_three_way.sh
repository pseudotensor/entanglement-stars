#!/bin/bash
# tolcmp_v1a_three_way.sh (busi, analyst, 21 Oct): a replay read with tolcmp_v1a under the ruling's sec. 4 as amended by erratum v1a:
# (A) X against D (gate A, Newton fields line 2.5e-6), each arm against the production (gate B), and X against the production with the
# same-class test. Each DIR holds collapse_TAG{,_fields,_spectra,_handoff}.{json,npz} (symlinks are fine).
#   tolcmp_v1a_three_way.sh XDIR DDIR PRODDIR OUTDIR TAG AFTER RS SLICE_X SLICE_D [X name] [D name] [prod name]
set -u
XD=$1; DD=$2; PD=$3; O=$4; TAG=$5; AFT=$6; RS=$7; SX=$8; SD=$9; NX=${10:-X}; ND=${11:-D}; NP=${12:-production}
PY=${PY:-${QI2_ROOT}/python}; TCP=${TCP:-${QI2_ROOT}/tolcmp_v1a.py}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
TC="nice -n 10 $PY -I $TCP"; mkdir -p $O
$TC $XD $DD --tag $TAG --after $AFT --rs $RS --nameA "$NX" --nameB "$ND" --slice-a $SX --slice-b $SD --gate A --engine newton --out $O/tol_X_vs_D.json > $O/tol_X_vs_D.txt 2>&1; echo "X-D rc $?"
$TC $DD $PD --tag $TAG --after $AFT --rs $RS --nameA "$ND" --nameB "$NP" --gate B --engine newton --out $O/tol_D_vs_prod.json > $O/tol_D_vs_prod.txt 2>&1; echo "D-prod rc $?"
$TC $XD $PD --tag $TAG --after $AFT --rs $RS --nameA "$NX" --nameB "$NP" --gate B --engine newton --out $O/tol_X_vs_prod.json > $O/tol_X_vs_prod.txt 2>&1; echo "X-prod rc $?"
$TC $XD $PD --tag $TAG --after $AFT --rs $RS --nameA "$NX" --nameB "$NP" --gate B --engine newton --same-class $O/tol_X_vs_prod.json --default-vs-prod $O/tol_D_vs_prod.json --out $O/tol_X_vs_prod_sc.json > $O/tol_X_vs_prod_sc.txt 2>&1; echo "X-prod same-class rc $?"
cd $O && md5sum tol_*.txt tol_*.json > MD5SUMS_tolcmp_v1a.txt
