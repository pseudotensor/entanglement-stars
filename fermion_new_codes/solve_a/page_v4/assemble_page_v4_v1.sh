#!/bin/bash
# assemble_page_v4_v1.sh (the solve lane; PAGE_solved_history_v4): reruns the box pipeline, fetches the outputs, fills the placeholders of the prose
# parts and concatenates the delivery PAGE_solved_history_v4.md with the figure beside it. Local work: file copies and text only.
set -e
L=${QI2_ROOT}/page_v4; C=${QI2_ROOT}/page_v4; DEL=${QI2_ROOT}/solve_a
ssh compute-host 'cd ${QI2_ROOT}/page_v4 && bash code/run_page_v4_box_v1.sh' | tail -n 8
scp -q compute-host:${QI2_ROOT}/{page_v4_tables_v1.md,PAGE_fig_v4.png,PAGE_fig_v4_data.npz,MANIFEST.md5,OUTPUTS.md5,page_v4_data_v1.json} $L/
cd $L
python3 - <<'PY'
import json, hashlib, subprocess, os
L="${QI2_ROOT}/page_v4"; C="${QI2_ROOT}/page_v4"
D=json.load(open(L+"/page_v4_data_v1.json")); SB=D["sub16_B2"]; cl=D["B2"]["clocks"]; CUTS=D["cuts"]
RB={(x["t"],x["cut"]):x for x in D["B2"]["readout"]}
rows=[]
for i,t in enumerate(cl):
    for j,lab in enumerate(CUTS):
        d=SB["D16_w"][i][j]; S=D["B2"]["SUM"][i][j]; v=(RB.get((t,lab)) or {}).get("verdict")
        if d is not None and t>=40-1e-9: rows.append((t,lab,d,S,v,18*abs(d)/abs(S)))
formed=[x for x in rows if x[4] in ("formed, tail open","converged")]; nf=[x for x in rows if x[4] not in ("formed, tail open","converged")]
big=sorted([x for x in formed if x[5]>0.005], key=lambda x:-x[5])
last=max(x[0] for x in rows)
if SB["final"]:
    status="complete to 100 r_s (run (c) of the W1 ruling v2: --interp-sub 16, kappa 18 only, --lean; wall %.2f h)" % (SB["wall_s"]/3600.0)
    openline="The 16-sub-step control exists for one channel per history (kappa 18 on B2, kappa 14 on s8); the other channels' sub-step error is inferred from it, not measured."
else:
    status="NOT complete at the time of this composition (pending; the run is in progress on the box GPU, its rows reaching T/r_s %g; the table prints what exists)" % last
    openline="The B2 sub-step control (kappa 18 at 16 sub-steps) is pending: its rows reach %g r_s at this composition; the s8 control is complete." % last
rng_txt="over the %d readout pairs to %g r_s |D16| is %.1e to %.1e nats; as the share 18 |D16| it is below 0.5 percent of the sum at %d of the %d formed pairs and %s at the rest (%s); at the not-formed and within-the-floor pairs, where the sums are small, it is %.2f to %.1f percent; the kappa 18 formed energy 288 E_18 under 16 sub-steps is %.1f to %.1f percent above its 8-sub-step value over 40 to %g r_s" % (
    len(rows), last, min(abs(x[2]) for x in rows), max(abs(x[2]) for x in rows), len(formed)-len(big), len(formed), ("%.1f to %.1f percent" % (min(x[5] for x in big)*100, max(x[5] for x in big)*100)) if big else "-", ", ".join("%g/%s %.2f" % (x[0],x[1],x[5]*100) for x in big), min(x[5] for x in nf)*100, max(x[5] for x in nf)*100,
    min(100*(SB["E_kappa_w_sub16"][i]/SB["E_kappa_w_sub8"][i]-1) for i,t in enumerate(cl) if t>=40-1e-9 and SB["E_kappa_w_sub16"][i] is not None), max(100*(SB["E_kappa_w_sub16"][i]/SB["E_kappa_w_sub8"][i]-1) for i,t in enumerate(cl) if t>=40-1e-9 and SB["E_kappa_w_sub16"][i] is not None), last)
p1=open(L+"/PAGE_v4_part1.md").read()
p1=p1.replace("<<SUB16_STATUS>>", status).replace("<<SUB16_OPEN>>", openline)
p1=p1.replace("where its rows exist, D16 = 288 [x_18(16) - x_18(8)] is at most 1.4e-7 to 1.0e-4 nats per pair, below 0.05 percent of the sum as 18 |D16| at every formed pair (0.1 to 2.5 percent at not-formed pairs where the sum is small) <<SUB16_RANGE>>", "D16 = 288 [x_18(16) - x_18(8)]: " + rng_txt)
def md5(p):
    return hashlib.md5(open(p,"rb").read()).hexdigest()
scripts="\n".join("- %s  %s" % (md5(C+"/"+f), C+"/"+f) for f in ("page_v4_data_v1.py","page_v4_tables_v1.py","page_v4_fig_v1.py","run_page_v4_box_v1.sh","assemble_page_v4_v1.sh"))
outs="\n".join("- " + l.strip() for l in open(L+"/OUTPUTS.md5"))
man="\n".join("- " + l.strip() for l in open(L+"/MANIFEST.md5"))
texts=[("${QI2_ROOT}/W1_B2_composed_v3.md","the busi lane's B2 K = 18 composition (its sections 4 to 6: the tally, the sentences S1 to S5 with numbers, run (a))"),
       ("${QI2_ROOT}/W1_compose_ruling_v2.md","the W1 compose ruling v2 (the v1f rule, S1 to S3, runs (a) to (d))"),
       ("${QI2_ROOT}/w1_postpass_v1f.py","the post-pass (the lane copy; the box copy has the same md5)"),
       ("${QI2_ROOT}/ReviewA_reading_B2_k24_k32_k48_probes_J7_v1.md","the reading of the kappa 24, 32, 48 probes (its section 3 the tail sentence; 76/24 ruled formed, tail open)"),
       ("${QI2_ROOT}/REVIEWA_RULING_O1_D13_protocol_v1_final.md","the D13 protocol ruling (the sayable set, section 1)"),
       ("${QI2_ROOT}/PAGE_solved_history_v3.md","version 3 (the marks' definitions, section 6)"),
       ("${QI2_ROOT}/PAGE_solved_history_v3b.md","version 3b (the holds' readings; the withheld list)"),
       ("${QI2_ROOT}/ReviewA_cold_reading_PAGE_solved_history_v3.md","the cold reading of version 3"),
       ("${QI2_ROOT}/ReviewA_cold_reading_PAGE_solved_history_v3b.md","the cold reading of version 3b"),
       ("${QI2_ROOT}/PAGE_solved_history_v3_summary.json","version 3's summary json (copied to the box as in_v3_summary.json: the figure block's marks and the s4 run's flux)"),
       ("${QI2_ROOT}/requirements_new.txt","the requirements (the Page-curve paragraphs of sections 2 and 5)"),
       ("${QI2_ROOT}/07_page_curve.tex","the paper's section (the three marked sentences at lines 46, 207, 227)"),
       ("${QI2_ROOT}/06_scaling.tex","the paper's T_d table (line 338)")]
tx="\n".join("- %s  %s  (%s)" % (md5(p), p, d) for p,d in texts)
p3=open(L+"/PAGE_v4_part3.md").read().replace("<<SCRIPTS_MD5>>", scripts).replace("<<OUTPUTS_MD5>>", outs).replace("<<MANIFEST>>", man).replace("<<TEXT_MD5>>", tx)
tables=open(L+"/page_v4_tables_v1.md").read()
out=p1+tables+p3
assert "<<" not in out
open(L+"/PAGE_solved_history_v4.md","w").write(out)
print("assembled", len(out), "chars; sub16 final", SB["final"], "last", last)
PY
cp $L/PAGE_solved_history_v4.md $DEL/PAGE_solved_history_v4.md; cp $L/PAGE_fig_v4.png $DEL/PAGE_fig_v4.png; cp $L/PAGE_fig_v4_data.npz $DEL/PAGE_fig_v4_data.npz
cp $L/MANIFEST.md5 $L/page_v4_MANIFEST.md5 2>/dev/null || true
md5sum $DEL/PAGE_solved_history_v4.md $DEL/PAGE_fig_v4.png $DEL/PAGE_fig_v4_data.npz
grep -c -i -E "mutual information|frozen|prediction" $DEL/PAGE_solved_history_v4.md || true
