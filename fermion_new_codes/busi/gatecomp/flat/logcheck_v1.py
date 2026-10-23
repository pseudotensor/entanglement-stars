#!/usr/bin/env python
"""logcheck_v1.py (busi, GATECOMP; ReviewA_cvvoid_v1 sec. 2, inbox #426): the LOG-ONLY centre check beside a flat member. Per record:
N(0) (the stored node value N_nodes[0], a point value of the engine's dense lapse), the six-term of record at r = sigma, 2 sigma, 4 sigma
(each radius's value, its dominant term and the d_T mode), and the Landau-Zener sub-step counts nsub per channel with nsub_max (from the
json record's continuation entries, as the engine logs them).
The six-term is the stopper's and the comparator's (gate_common_v1.SixTerm: the dense fields re-solved from the record's stored eps_tot,
J_tot by the field module's solve_constraints_ivp, collar flag, support 1e-14; derivatives Richardson at h = sigma/32 by gate_profiles_v1's
kfields; d_T central between the adjacent records). A record is written once its successor is in the fields npz (central d_T, as the gate
reads it); with --final the last record is written too, with the backward d_T, labelled.
READ ONLY: it opens the run json and fields npz (the npz copied to a private temp first, as the stopper does), never writes into the run
dir, never signals any process, never touches nsub, the stopper or the engine. Its only output is OUT (default MEMBER/LOGCHECK.tsv, beside
the stopper's log; not appended into the stopper's own .out, whose writer holds a truncating fd and would overwrite appended bytes).
  logcheck_v1.py MEMBER_DIR TAG CODE_DIR --sigma S [--run run] [--period 120] [--once] [--final] [--out FILE]
  (CODE_DIR = the field module's dir, as the stopper's third argument; gate_common_v1.py and gate_profiles_v1.py found as the stopper finds them)
Post hoc on stored records: --once --final with MEMBER_DIR/--run pointing at the stored copy and --out elsewhere.
"""
import argparse, datetime as dt, json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(os.path.dirname(HERE), "code"), os.path.join(os.path.dirname(HERE), "kits", "code"), "<internal>/code", "<internal>/code"):
    if os.path.exists(os.path.join(p, "gate_common_v1.py")):
        sys.path.insert(0, p); break
import gate_common_v1 as GC  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("member"); ap.add_argument("tag"); ap.add_argument("code")
ap.add_argument("--sigma", type=float, required=True); ap.add_argument("--run", default="run")
ap.add_argument("--period", type=float, default=120.0); ap.add_argument("--once", action="store_true"); ap.add_argument("--final", action="store_true")
ap.add_argument("--out", default=None); ap.add_argument("--tmp", default=None)
a = ap.parse_args()
RUN = os.path.join(a.member, a.run)
OUT = a.out or os.path.join(a.member, "LOGCHECK.tsv")
TMP = a.tmp or os.path.join(os.path.dirname(os.path.abspath(OUT)), ".logcheck_fields.npz")
JP = os.path.join(RUN, "collapse_%s.json" % a.tag); FP = os.path.join(RUN, "collapse_%s_fields.npz" % a.tag)
six = GC.load_sixterm(a.code)
GP = six.GP
sig = a.sigma
RADII = (sig, 2.0 * sig, 4.0 * sig)
TERMS = ("abs(v'/N)", "abs(v/(N r))", "abs(N'/N)", "sqrt(abs(D k_r))", "sqrt(abs(D k_t))", "sqrt(abs(N''/N))")
HEAD = ("# logcheck_v1 (ReviewA_cvvoid_v1 sec. 2): member %s tag %s sigma %g; six-term of record (sigma Lambda_geom) at r = %s a\n"
        "T_over_rs\tN0\tsix_r1\tdom_r1\tsix_r2\tdom_r2\tsix_r4\tdom_r4\tdT_mode\tnsub_max\tnsub_per_channel\tconverged\twritten_utc\n") % (
    a.member, a.tag, sig, ", ".join("%g" % r for r in RADII))


def done_clocks():
    if not os.path.exists(OUT):
        return set()
    s = set()
    for ln in open(OUT):
        if ln.startswith("#") or ln.startswith("T_over_rs"):
            continue
        try:
            s.add(round(float(ln.split("\t")[0]), 6))
        except ValueError:
            pass
    return s


def six_at(F, k, rs_unused=None):
    """the six-term at RADII for fields row k: [(value, dominant)], mode."""
    Td = np.asarray(F["T"], float); n = len(Td)
    S0 = six.solve(F, k)
    if S0 is None:
        return None, "solve failed"
    Sp = six.solve(F, k - 1) if k - 1 >= 0 else None; Sn = six.solve(F, k + 1) if k + 1 < n else None
    r = np.asarray(RADII, float); h = sig / 32.0
    k0 = GP.kfields(S0[0], S0[1], r, h)
    kp = GP.kfields(Sp[0], Sp[1], r, h) if Sp else None; kn = GP.kfields(Sn[0], Sn[1], r, h) if Sn else None
    if kp is not None and kn is not None:
        dTr = (kn["kr"] - kp["kr"]) / (Td[k + 1] - Td[k - 1]); dTt = (kn["kt"] - kp["kt"]) / (Td[k + 1] - Td[k - 1]); mode = "central"
    elif kp is not None:
        dTr = (k0["kr"] - kp["kr"]) / (Td[k] - Td[k - 1]); dTt = (k0["kt"] - kp["kt"]) / (Td[k] - Td[k - 1]); mode = "backward"
    else:
        dTr = (kn["kr"] - k0["kr"]) / (Td[k + 1] - Td[k]); dTt = (kn["kt"] - k0["kt"]) / (Td[k + 1] - Td[k]); mode = "forward"
    Dkr = (dTr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dTt + k0["v"] * k0["kt1"]) / k0["N"]
    T6 = {"abs(v'/N)": np.abs(k0["kr"]), "abs(v/(N r))": np.abs(k0["kt"]), "abs(N'/N)": np.abs(k0["N1"] / k0["N"]),
          "sqrt(abs(D k_r))": np.sqrt(np.abs(Dkr)), "sqrt(abs(D k_t))": np.sqrt(np.abs(Dkt)), "sqrt(abs(N''/N))": np.sqrt(np.abs(k0["N2"] / k0["N"]))}
    out = []
    for i in range(len(r)):
        dom = max(TERMS, key=lambda t: T6[t][i])
        out.append((float(sig * T6[dom][i]), dom))
    return out, mode


def poll(final):
    try:
        J = json.load(open(JP))
        shutil.copy2(FP, TMP)
        F = np.load(TMP, allow_pickle=True)
    except Exception as e:  # the engine may be rewriting a file: next poll
        print("%s read deferred: %s" % (dt.datetime.utcnow().strftime("%H:%M:%S"), e), flush=True)
        return 0
    rs = float(J["args"]["rs"])
    recs = GC.recs_by_T(J, rs); rows = GC.rows_by_T(F, rs)
    have = done_clocks(); n = len(F["T"]); wrote = 0
    new = not os.path.exists(OUT)
    with open(OUT, "a") as fo:
        if new:
            fo.write(HEAD)
        for t in sorted(rows):
            if t in have or t not in recs:
                continue
            k = rows[t]
            if k + 1 >= n and not final:
                continue
            res, mode = six_at(F, k)
            rec = recs[t]
            cont = rec.get("continuation") or {}
            ns = ",".join("%s:%s" % (c, (cont[c] or {}).get("nsub")) for c in sorted(cont, key=lambda s: int(s)))
            N0 = float(np.asarray(F["N_nodes"][k], float)[0])
            if res is None:
                cols = ["nan", "-"] * 3
            else:
                cols = []
                for v, d in res:
                    cols += ["%.6f" % v, d]
            fo.write("%.6f\t%.9f\t%s\t%s%s\t%s\t%s\t%s\t%s\n" % (t, N0, "\t".join(cols), mode, " (last record, provisional)" if k + 1 >= n else "",
                                                               rec.get("nsub_max"), ns, rec.get("converged"), dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
            wrote += 1
        fo.flush()
    six.cache.clear()
    return wrote


if a.once:
    print("logcheck_v1: %d records written to %s" % (poll(a.final), OUT), flush=True)
    sys.exit(0)
while True:
    w = poll(False)
    if w:
        print("%s logcheck_v1: %d records written" % (dt.datetime.utcnow().strftime("%H:%M:%S"), w), flush=True)
    time.sleep(a.period)
