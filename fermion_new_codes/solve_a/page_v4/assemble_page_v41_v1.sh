#!/bin/bash
# assemble_page_v41_v1.sh (the solve lane; PAGE_solved_history_v4.1): reruns the v1b box pipeline, fetches the outputs, fills the placeholders of the prose
# parts and concatenates the delivery PAGE_solved_history_v4.1.md with the figure, its data file and the per-pair export beside it. Local work: copies and text.
set -e
L=${QI2_ROOT}/page_v4.1; C=${QI2_ROOT}/page_v4; DEL=${QI2_ROOT}/solve_a
scp -q $C/page_v4_data_v1b.py $C/page_v4_tables_v1b.py $C/page_v4_fig_v1b.py $C/run_page_v4_box_v1b.sh compute-host:${QI2_ROOT}/code
ssh compute-host 'cd ${QI2_ROOT}/page_v4 && bash code/run_page_v4_box_v1b.sh' | tail -n 8
scp -q compute-host:${QI2_ROOT}/{page_v4_tables_v1b.md,page_v4_readout_export_v1b.tsv,PAGE_fig_v4.1.png,PAGE_fig_v4.1_data.npz,MANIFEST.md5,OUTPUTS.md5,page_v4_data_v1.json} $L/
python3 - <<'PY'
import hashlib
L="${QI2_ROOT}/page_v4.1"; C="${QI2_ROOT}/page_v4"
def md5(p):
    return hashlib.md5(open(p,"rb").read()).hexdigest()
scripts="\n".join("- %s  %s" % (md5(C+"/"+f), C+"/"+f) for f in ("page_v4_data_v1b.py","page_v4_tables_v1b.py","page_v4_fig_v1b.py","run_page_v4_box_v1b.sh","assemble_page_v41_v1.sh"))
outs="\n".join("- " + l.strip() for l in open(L+"/OUTPUTS.md5"))
man="\n".join("- " + l.strip() for l in open(L+"/MANIFEST.md5"))
texts=[("${QI2_ROOT}/ReviewA_reading_PAGE_v4_v1.md","the correctness reading of version 4 (both parts; the corrections applied here)"),
       ("${QI2_ROOT}/PAGE_solved_history_v4.md","version 4 as issued"),
       ("${QI2_ROOT}/W1_B2_composed_v3.md","the busi lane's B2 K = 18 composition (its sections 1, 4 to 6: run (a), the tally, the sentences S1 to S5 with numbers)"),
       ("${QI2_ROOT}/W1_compose_ruling_v2.md","the W1 compose ruling v2 (the v1f rule, S1 to S3, runs (a) to (d), the geometric criterion of section 2(b))"),
       ("${QI2_ROOT}/w1_postpass_v1f.py","the post-pass (the lane copy; the box copy has the same md5)"),
       ("${QI2_ROOT}/ReviewA_reading_B2_k24_k32_k48_probes_J7_v1.md","the reading of the kappa 24, 32, 48 probes (its section 3; 76/24 ruled formed, tail open)"),
       ("${QI2_ROOT}/REVIEWA_RULING_O1_D13_protocol_v1_final.md","the D13 protocol ruling (the sayable set, section 1)"),
       ("${QI2_ROOT}/PAGE_solved_history_v3.md","version 3 (the marks' definitions, section 6)"),
       ("${QI2_ROOT}/PAGE_solved_history_v3b.md","version 3b (the holds' readings; the withheld list)"),
       ("${QI2_ROOT}/ReviewA_cold_reading_PAGE_solved_history_v3.md","the cold reading of version 3"),
       ("${QI2_ROOT}/ReviewA_cold_reading_PAGE_solved_history_v3b.md","the cold reading of version 3b"),
       ("${QI2_ROOT}/PAGE_solved_history_v3_summary.json","version 3's summary json (copied to the box as in_v3_summary.json: the figure block's marks, the s4 run's flux and the A-to-100 rows)"),
       ("${QI2_ROOT}/requirements_new.txt","the requirements (the Page-curve paragraphs of sections 2 and 5)"),
       ("${QI2_ROOT}/07_page_curve.tex","the paper's section (the marked sentences at lines 46, 207, 227)"),
       ("${QI2_ROOT}/06_scaling.tex","the paper's T_d table (line 338)")]
tx="\n".join("- %s  %s  (%s)" % (md5(p), p, d) for p,d in texts)
p1=open(L+"/PAGE_v41_part1.md").read()
p3=open(L+"/PAGE_v41_part3.md").read().replace("<<SCRIPTS_MD5>>", scripts).replace("<<OUTPUTS_MD5>>", outs).replace("<<MANIFEST>>", man).replace("<<TEXT_MD5>>", tx)
tables=open(L+"/page_v4_tables_v1b.md").read()
out=p1+tables+p3
assert "<<" not in out
open(L+"/PAGE_solved_history_v4.1.md","w").write(out)
print("assembled", len(out), "chars")
PY
cp $L/PAGE_solved_history_v4.1.md $DEL/PAGE_solved_history_v4.1.md; cp $L/PAGE_fig_v4.1.png $DEL/PAGE_fig_v4.1.png; cp $L/PAGE_fig_v4.1_data.npz $DEL/PAGE_fig_v4.1_data.npz; cp $L/page_v4_readout_export_v1b.tsv $DEL/PAGE_v4.1_readout_export.tsv
md5sum $DEL/PAGE_solved_history_v4.1.md $DEL/PAGE_fig_v4.1.png $DEL/PAGE_fig_v4.1_data.npz $DEL/PAGE_v4.1_readout_export.tsv
for w in "mutual information" "frozen" "prediction" "remnant" "static " "<<"; do echo "$w: $(grep -c -i "$w" $DEL/PAGE_solved_history_v4.1.md)"; done
