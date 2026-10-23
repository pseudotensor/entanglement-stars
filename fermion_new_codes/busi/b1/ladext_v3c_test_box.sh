#!/bin/bash
# busi B1 ladext v3c (reviewer A REVIEWA_B1_split_unruled_v1 Ruling 2): branch tests on the box CPU (nice 10, 8 threads); nothing launched
set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
B=${QI2_ROOT}/b1; D=$B/ladext_v3c_test; DRYD=$B/ladext_dry; S=$D/r2_snapshot
C3=$D/code/engine_b1_tests_v3c.py; C2=$B/kit_v3b/code/engine_b1_tests_v3b.py
mkdir -p $D/out $D/syn
cd $D
COM="--m 0.4 --g 0.25"
dec() { grep '^LADEXT decision' ; }
# synthetic member lists: the dry check's 6 plus held on f 9.4815, held on 4.7407 only, and the real 3 rows + a 4th complete without hold
nice -n 10 $PY -I - $DRYD $D/syn $S <<'PYEOF'
import json, sys, os, glob, shutil
dry, out, snap = sys.argv[1:4]
for p in glob.glob(os.path.join(dry, "syn_*.json")):
    shutil.copy(p, out)
def row(f, rs, root, T1, Rt, Tl=32.0):
    return {"f": f, "rs": rs, "records": int(Tl / 0.125) + 1, "root_formed": root, "first_root_T_over_rs": T1, "R_t_last": Rt, "T_last_over_rs": Tl}
base = [row(2.4, 8.1, False, None, None), row(3.0, 10.125, False, None, None)]
json.dump({"rows": base + [row(4.7407, 15.9998625, True, 9.625, None), row(9.4815, 32.0000625, True, 8.0, 30.0)]},
          open(os.path.join(out, "syn_E_touch_f4.7407_held_f9.4815.json"), "w"), indent=1)
json.dump({"rows": base + [row(4.7407, 15.9998625, True, 9.625, 14.0), row(9.4815, 32.0000625, True, 8.0, 30.0)]},
          open(os.path.join(out, "syn_F_held_f4.7407.json"), "w"), indent=1)
json.dump({"rows": base + [row(4.7407, 15.9998625, True, 9.625, None), row(9.4815, 32.0000625, True, 8.0, None)]},
          open(os.path.join(out, "syn_G_touch_f4.7407_none_held_9.4815.json"), "w"), indent=1)
json.dump({"rows": base + [row(4.7407, 15.9998625, False, None, None), row(9.4815, 32.0000625, True, 12.0, None)]},
          open(os.path.join(out, "syn_H_first_root_on_f9.4815_none_held.json"), "w"), indent=1)
json.dump({"rows": base + [row(4.7407, 15.9998625, True, 9.625, None), row(9.4815, 32.0000625, True, 8.0, None, Tl=13.0)]},
          open(os.path.join(out, "syn_I_touch_f4.7407_f9.4815_partial.json"), "w"), indent=1)
print("synthetic lists:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "syn_*.json"))))
PYEOF
echo "== 1. existing 21.2 ladder json (20 r_s): v3b | v3c"
for c in $C2 $C3; do nice -n 10 $PY $c ladext --out $D/out/existing --rows-json $DRYD/ladder_R2.json --T-over-rs 20 $COM --md $D/out/t.md --json-out - | dec; done
echo "== 2. synthetic member lists (32 r_s): v3b then v3c; the json kept for v3c"
for s in $D/syn/syn_*.json; do
  n=$(basename $s .json); echo "-- $n"
  nice -n 10 $PY $C2 ladext --out $D/out/$n --rows-json $s --T-over-rs 32 $COM --md $D/out/t.md --json-out - | dec | sed 's/^/v3b /'
  nice -n 10 $PY $C3 ladext --out $D/out/$n --rows-json $s --T-over-rs 32 $COM --md $D/out/t.md --json-out $D/out/$n/v3c.json | dec | sed 's/^/v3c /'
done
echo "== 3. R2's extension as it stands (read-only snapshot; f 9.4815 running): records path, v3b | v3c"
for c in $C2 $C3; do nice -n 10 $PY $c ladext --out $S --tag b1ladx --members 2.4,3.0,4.7407,9.4815 --T-over-rs 32 $COM --md $D/out/t.md --json-out - | grep '^LADEXT'; done
echo "-- R2's own v3b json (copied):"; $PY -I -c "import json;d=json.load(open('$S/collapse_b1ladx_ladder_ext.json'));print(d['branch'],'|',d['decision'])"
echo "== 4. records path, the real 3 members + a synthetic complete f 9.4815 member (the f 4.7407 records rescaled to r_s 32.0000625, roots, none at 32 r_s)"
mkdir -p $D/out/recfull; for f in 2.4 3.0 4.7407; do ln -sf $S/collapse_b1ladx_f${f}_records.jsonl $D/out/recfull/; done
nice -n 10 $PY -I - $S $D/out/recfull <<'PYEOF'
import json, sys, os
s, d = sys.argv[1:3]
src = [json.loads(x) for x in open(os.path.join(s, "collapse_b1ladx_f4.7407_records.jsonl")) if x.strip()]
rs = 32.0000625
with open(os.path.join(d, "collapse_b1ladx_f9.4815_records.jsonl"), "w") as fh:
    for k, r in enumerate(src):
        r = dict(r); r["T"] = k * 0.125 * rs; r["T_over_rs"] = k * 0.125
        fh.write(json.dumps(r) + "\n")
print("synthetic f9.4815:", len(src), "records; R_t_last", src[-1].get("R_t"))
PYEOF
for c in $C2 $C3; do nice -n 10 $PY $c ladext --out $D/out/recfull --tag b1ladx --members 2.4,3.0,4.7407,9.4815 --T-over-rs 32 $COM --md $D/out/t.md --json-out $D/out/recfull/$(basename $c .py).json | dec; done
echo "== 5. ksea's reader (b1_card_v3b.sh line 123 verbatim) on each v3c json"
for j in $D/out/*/v3c.json $D/out/recfull/engine_b1_tests_v3c.json; do
  L=$j; RS=$($PY -c "import json,sys;d=json.load(open('$L'));sys.exit(1) if not str(d.get('decision','')).startswith('B1 point') else print(d['rs_B1'])") && r="takes rs_B1 $RS" || r="refuses"
  echo "$(basename $(dirname $j)): branch $($PY -c "import json;d=json.load(open('$L'));print(d['branch'], 'E_hold', d['E_hold'])") -> ksea $r"
done
echo "== 6. stop rule (b1_card_v3b.sh case verbatim) on the v3c branch with members done n: kills only on (i)/(ii) with n < 4"
for br in "(i)" "(ii)" "incomplete" "below the extension" "(iii)"; do case "$br" in "(i)"|"(ii)"|"held(engine line)") echo "$br: stop (n<4 kill)";; *) echo "$br: go on";; esac; done
echo "v3c (ii) with E_hold null needs all members complete, so n = 4 and no kill; partial lists give incomplete (syn_I, sec. 3)"
md5sum $C3 $S/* | sed 's#'$D'/##'
echo TEST_DONE
