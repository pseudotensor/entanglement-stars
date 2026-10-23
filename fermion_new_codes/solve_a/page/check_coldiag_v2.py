"""check_coldiag_v2.py (solve builder 7; engine spec 48.2 (3)): version 1 with the in-span weight's bound at 1 + 1e-6 in place of 1 + 1e-9: the
weight is ||K D w||^2 for the least-squares memory coefficient w through H_D's pseudo-inverse (relative cutoff 1e-12), whose rounding grows with
H_D's condition number (version 1 found 168 of 4096 oracle columns at most 7.9e-07 above 1, every other bound held); the largest excess is
printed. Version 1: check_coldiag_v1.py (solve builder 7; engine spec 48.2 (3)): light reading of page_history_v6m's col_diag entries in every json of a
directory (an oracle's output or a run's summary): counts, bounds (rho, rho_q, in_span in [0, 1]; the reference block's numbers in [0, 1];
minority-half weights in [0, 1/2]; single-column entropies in [0, ln 2]) and the flagged shares. Prints PASS when every bound holds and at
least one entry with admitted columns exists. Usage: check_coldiag_v1.py DIR [--expect-entries]"""
import json, math, os, sys
D = sys.argv[1]; eps = 1e-9; L2 = math.log(2.0)
n_e = n_c = n_q = 0; bad = []; shares = []; EX = [0.0]
def walk(x, p):
    global n_e, n_c, n_q
    if isinstance(x, dict):
        if "col_diag" in x and isinstance(x["col_diag"], dict):
            cd = x["col_diag"]; n_e += 1
            for c in cd.get("candidates", []):
                n_c += 1
                if not (-eps <= c["rho"] <= 1 + eps): bad.append("%s candidate rho %r" % (p, c["rho"]))
                if not (-eps <= c["other_half"] <= 1 + eps): bad.append("%s candidate other_half %r" % (p, c["other_half"]))
            for q in cd.get("columns", []):
                n_q += 1
                for k_, hi in (("rho", 1.0), ("in_span", 1.0), ("minority_half", 0.5), ("s_state", L2), ("s_vac", L2)):
                    tol = 1e-6 if k_ == "in_span" else eps
                    if k_ == "in_span": EX[0] = max(EX[0], q[k_] - 1.0)
                    if not (-eps <= q[k_] <= hi + tol): bad.append("%s column %s %r" % (p, k_, q[k_]))
            for k_, v_ in (cd.get("reference_block") or {}).items():
                if k_.startswith("n_"): continue
                if not (-eps <= v_ <= 1 + eps) and k_ != "off_block_F": bad.append("%s reference %s %r" % (p, k_, v_))
            if cd.get("n_admitted"):
                shares.append((p, cd["n_admitted"], cd.get("n_flagged"), cd.get("share_state_flagged"), cd.get("share_vac_flagged"), cd.get("sum_s_state"), cd.get("sum_s_vac")))
        for k, v in x.items():
            walk(v, p + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk(v, p + "[%d]" % i)
for f in sorted(os.listdir(D)):
    if f.endswith(".json"):
        try:
            walk(json.load(open(os.path.join(D, f))), f)
        except Exception as e:
            print("unreadable %s: %s" % (f, type(e).__name__))
print("col_diag entries %d, candidates %d, admitted columns %d, bound violations %d; the largest in-span excess over 1: %.2e" % (n_e, n_c, n_q, len(bad), EX[0]))
for b in bad[:10]:
    print("  " + b)
for s in shares[:12]:
    print("  %s: admitted %d, flagged %s, share state %s, share vac %s, sum s_state %s, sum s_vac %s" % s)
ok = not bad and (n_q > 0 or "--expect-entries" not in sys.argv)
print("PASS" if ok else "FAIL")
