"""read_48_1_3_v1.py (solve builder 7; engine spec 48.1 (3); light parsing): both runs' root columns through T/r_s 12.5 (the v32 member and the
ended tightened v36 run) and the v32 member's residuals, evaluation counts and continuation stages from 12.5 to its latest record.
Usage: read_48_1_3_v1.py V32_JSON TOL10_JSON OUT_MD"""
import json, sys
def load(f):
    d = json.load(open(f)); return [x for x in d["records"] if x.get("E") is not None]
a, b = load(sys.argv[1]), load(sys.argv[2]); out = []
P = out.append
P("# The sigma/a 2 member: both runs' roots through T/r_s 12.5 and the v32 member's solve from 12.5 (engine spec 48.1 (3))\n")
P("Composed by code/engine/read_48_1_3_v1.py (light parsing) from the v32 member's json (pod 2, mirrored by sync_v32_v3 to <internal>/v32_member; %d records to T/r_s %.4f) and the ended tightened run's json (pod 1 <internal>/v36member_r2, copied md5-equal; %d records to %.4f). Every number is the jsons'.\n" % (len(a), a[-1]["T"] / 64, len(b), b[-1]["T"] / 64))
def g(x, k, f="%.6f"):
    v = x.get(k); return (f % v) if isinstance(v, (int, float)) else "-"
bt = {round(x["T"] / 64, 6): x for x in b}
P("## Roots (trapping zeros of N - |v|) from 11.5 to 12.5\n")
P("| T/r_s | v32: zeros | v32: R_t | v32: 2gE | tol10: zeros | tol10: R_t | R_t difference | min(N-|v|) v32 | tol10 |")
P("|---|---|---|---|---|---|---|---|---|")
for x in a:
    t = round(x["T"] / 64, 6)
    if 11.5 - 1e-9 <= t <= 12.5 + 1e-9:
        y = bt.get(t, {})
        z1 = ", ".join("%.4f" % z for z in (x.get("trapping_zeros") or [])) or "none"; z2 = ", ".join("%.4f" % z for z in (y.get("trapping_zeros") or [])) or "none"
        d = (x["R_t"] - y["R_t"]) if isinstance(x.get("R_t"), float) and isinstance(y.get("R_t"), float) else None
        tg = 2 * 0.25 * x["E"]
        P("| %.4f | %s | %s | %.4f | %s | %s | %s | %s | %s |" % (t, z1, g(x, "R_t", "%.4f"), tg, z2, g(y, "R_t", "%.4f"), ("%+.2e" % d) if d is not None else "-", g(x, "min_N_minus_absv"), g(y, "min_N_minus_absv")))
fa = next((x for x in a if x.get("trapping_zeros")), None); fb = next((x for x in b if x.get("trapping_zeros")), None)
P("\nFirst record with a trapping zero: v32 member T/r_s %s, outer root %s; tightened run T/r_s %s, outer root %s.\n" % (
    "%.4f" % (fa["T"] / 64) if fa else "none", g(fa or {}, "R_t", "%.4f"), "%.4f" % (fb["T"] / 64) if fb else "none", g(fb or {}, "R_t", "%.4f")))
P("## The v32 member's solve from 12.5 to its latest record\n")
P("| T/r_s | converged | kept residual | confirming residual | confirming dE/E | meets stop rule | evaluations | continuation | seconds | outer root |")
P("|---|---|---|---|---|---|---|---|---|---|")
for x in a:
    if x["T"] / 64 >= 12.5 - 1e-9:
        s = x.get("solve_v32") or {}; c = s.get("continuation")
        cs = c if isinstance(c, str) else (json.dumps(c)[:80] if c else "-")
        P("| %.4f | %s | %s | %s | %s | %s | %s | %s | %.0f | %s |" % (x["T"] / 64, x.get("converged"), g(s, "kept_residual", "%.2e"), g(s, "confirming_residual", "%.2e"), g(s, "confirming_dE", "%.2e"),
          s.get("meets_stop_rule"), s.get("evaluations"), cs, x.get("seconds") or float("nan"), g(x, "R_t", "%.4f")))
open(sys.argv[3], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[3])
