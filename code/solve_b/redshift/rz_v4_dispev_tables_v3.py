"""rz_v4_dispev_tables_v3.py (REDSHIFT_received_v4.1, inbox #59): TABLES script (v3 = v2 calling its groups ARRIVAL CLUSTERS, not distinct emission events, and its statistics cluster-based; numbers unchanged; v2 = v1 grouping the A rows by class and clock when their arrivals form one consecutive run on the 1 a grid, with each group's emission-time spread printed, and every B row a distinct launch; v1 grouped by emission times within 1e-3 a, which split one clock's arrival samples where G_D x 1 a exceeds 1e-3 a) for rz_v3_dispev_v1's outputs on the exact-release passes
(rz_rays_v4n, rz_rays_v4nc), the dispersive rays launched at the headline events; v4 = rz_v3_dispev_tables_v1 with the changes the v3 checks
ask for (ReviewA_rzchk4 sec. 6 items 1, 2; ReviewB_rzchk4 secs. 4.1-4.4):
  - the rows are grouped into DISTINCT EMISSION EVENTS: rows of one class and one clock (A: clock index; B: offset delta) whose emission times
    agree within 1e-3 a are one event (the A rows are arrival samples 1 a apart of one emission event); every count is printed per row and per
    distinct event, and the medians are over distinct events (each event read on its largest-L_T row);
  - 'earlier' and 'less redshifted' are counted separately, per member and ap_e, with every exception listed;
  - L_F/L_T is a ratio of logarithms, not a frequency ratio; L_gK = L_K + ln A_D (the static receiver on reviewer B's phase-rate convention) is
    printed beside L_gF = L_F + ln A_D, the two kept distinct;
  - scope: the four members P64_s4, P64_s2, P64_s4_tau4, S4_128_s8 are named as such wherever a statement covers them only.
usage: rz_v4_dispev_tables_v1.py OUTDIR DISPEVDIR members...
"""
import json, os, sys
import numpy as np

OUT, DD = sys.argv[1], sys.argv[2]
MEM = sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
FOUR = ("P64_s4", "P64_s2", "P64_s4_tau4", "S4_128_s8")


def fmt(x, p=3):
    if x is None:
        return "-"
    try:
        if not np.isfinite(x):
            return "-"
    except TypeError:
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return "%.*e" % (max(p - 1, 1), x)
    return "%.*f" % (p, x)


SPREAD = {}


def groups(ev, rs=None, m=None):
    """distinct emission events: A rows of one class and one clock whose arrivals form one consecutive run on the 1 a arrival grid are one
    event (their emission-time spread is recorded in SPREAD[m]); every B row is a distinct launch (record, delta). Returns a group id per row."""
    gid = [None] * len(ev); ng = 0
    keys = {}
    for k, d in enumerate(ev):
        if d["cls"].startswith("A"):
            keys.setdefault((d["cls"], d["clock"]), []).append(k)
        else:
            gid[k] = ng; ng += 1
    for (c, ck), ks in keys.items():
        td = sorted(ev[k]["tD"] for k in ks)
        if len(ks) > 1 and abs((td[-1] - td[0]) - (len(ks) - 1)) > 1e-6:      # not one consecutive run: each row its own event
            for k in ks:
                gid[k] = ng; ng += 1
            if m is not None:
                SPREAD.setdefault(m, {})["%s.%s" % (c, ck)] = None
            continue
        for k in ks:
            gid[k] = ng
        ng += 1
        if m is not None:
            te = [ev[k]["te"] for k in ks]
            SPREAD.setdefault(m, {})["%s.%s" % (c, ck)] = float(max(te) - min(te))
    order = {}
    for g in gid:                                                          # number the events in row order
        order.setdefault(g, len(order))
    return [order[g] for g in gid]


D1 = ["**Table 9-H1v4. Dispersive rays at the headline events themselves** (rz_v3_dispev_v1.py on the exact-release passes rz_rays_v4n / "
      "rz_rays_v4nc; this table: rz_v4_dispev_tables_v3.py; massless radial channel; the class's top 5 rows by L_T at 4 r_s inside the "
      "solver-certified record interval; A1 r_e ≥ 1 a and A2 at the exact emission events of the exact-release clock implementation from the "
      "backward map, B at its launch event; C not used). Per row: the ARRIVAL CLUSTER it belongs to (K#: the A rows of one class and one "
      "clock whose arrivals form one consecutive run on the 1 a grid; their emission times differ, by the spread printed after Table 9-H2v4, "
      "so a cluster is not one emission event; every B row is its own launch, a cluster of one); the stored L_T and its arrival "
      "T_D/r_s; the continuum front re-integrated forward from the event (L_T, a reproduction check; '-' where the stored arrival is the last "
      "record); per emitted lattice momentum ap_e: the dispersive ray's arrival T_D/r_s and L_F = ln(ε(p_e)/Ω_D) (reviewer A's convention F; coordinate "
      "frequency at the receiver), L_K = ln(ω_K/Ω_D) (reviewer B's phase rate along the normal-comoving emitter), and the static receiver's "
      "L_gF = L_F + ln A_D and L_gK = L_K + ln A_D (two conventions, kept distinct); nr = not received by the last record; ‡ = the ray's ap passed π/2.\n",
      "| member | cluster | class | stored L_T (T_D/r_s) | continuum from the event: L_T | ap_e 0.05: T_D/r_s, L_F / L_K / L_gF / L_gK | ap_e 0.1 | ap_e 0.2 | ap_e 0.3 |",
      "|---|---|---|---|---|---|---|---|---|"]
D2 = ["\n**Table 9-H2v4. Per member and ap_e, over the headline events of Table 9-H1v4, counted per row and per ARRIVAL CLUSTER** "
      "(rows / clusters; cluster-based counts, not counts of distinct emission events): received by the last record; the largest L_F, L_gF and L_gK; those whose L_F exceeds reviewer B's crossover estimate "
      "ln(2/ap_e²) (6.68, 5.30, 3.91, 3.10 at ap_e 0.05, 0.1, 0.2, 0.3; a breakdown scale of the continuum approximation, not a cap); arriving "
      "earlier than the stored arrival; arriving less redshifted (L_F < L_T); then over the clusters (each read on its largest-L_T row; cluster-weighted): the "
      "median of L_F/L_T (a RATIO OF LOGARITHMS, not a frequency-reduction factor) and of the arrival shift T_D(dispersive) − T_D(stored) in r_s; "
      "the largest L_F − L_T (positive = the dispersive ray more redshifted); and the largest |L_K − L_F|.\n",
      "| member | ap_e | rows / clusters | received | largest L_F / L_gF / L_gK | L_F > ln(2/ap_e²) | earlier | less redshifted (L_F < L_T) | median L_F/L_T (clusters) | median ΔT_D (r_s, clusters) | largest L_F − L_T | largest abs(L_K − L_F) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
EXC = ["\n**Table 9-H3v4. The exceptions** (rz_v4_dispev_tables_v3.py): every arrival cluster and ap_e whose dispersive ray is received "
       "but NOT less redshifted than the stored continuum value (L_F ≥ L_T), and every one NOT earlier (T_D(dispersive) ≥ T_D(stored)); then per "
       "member the largest dispersive L_F at ap_e ≤ 0.1 against the largest stored L_T over the member's tested events (A1 r_e ≥ 1 a, A2, B).\n",
       "| member | cluster | class | stored L_T (T_D/r_s) | ap_e | L_F | L_F − L_T | dispersive T_D/r_s | which |",
       "|---|---|---|---|---|---|---|---|---|"]
MX = ["\n| member | largest stored L_T over the tested events (class) | largest dispersive L_F at ap_e 0.05 / 0.1 (class) | dispersive above continuum? |",
      "|---|---|---|---|"]
res = {}
allpairs = dict(rows=0, ev=set())
for m in MEM:
    p = os.path.join(DD, "dispev_%s.json" % m)
    if not os.path.exists(p):
        continue
    J = json.load(open(p)); rs = J["rs"]; aps = J["aps"]
    ev = J["events"]
    gid = groups(ev, rs, m)
    first = {}
    for k, g in enumerate(gid):
        if g not in first or ev[k]["LT"] > ev[first[g]]["LT"]:
            first[g] = k
    for k, d in enumerate(ev):
        cells = []
        for apv in aps:
            q = d["disp"].get("%g" % apv, {})
            if q.get("status") == 1:
                cells.append("%s, %s / %s / %s / %s%s" % (fmt(q["tD"] / rs, 2), fmt(q["LF"]), fmt(q["LK"]), fmt(q["LgF"]), fmt(q.get("LgK")), " ‡" if q.get("branch") else ""))
            elif q.get("status") == -1:
                cells.append("capped")
            else:
                cells.append("nr%s" % (" ‡" if q.get("branch") else ""))
        D1.append("| %s | K%d | %s | %s (%s) | %s | %s |" % (m, gid[k] + 1, d["cls"], fmt(d["LT"]), fmt(d["tD"] / rs, 3), fmt(d.get("cont_LT")), " | ".join(cells)))
    r = {}
    ng = len(first)
    for apv in aps:
        a_ = "%g" % apv
        cr = float(np.log(2 / apv ** 2))
        rcv = [k for k, d in enumerate(ev) if d["disp"].get(a_, {}).get("status") == 1]
        rcv_g = sorted(set(gid[k] for k in rcv))
        q_ = lambda k: ev[k]["disp"][a_]
        LF = [q_(k)["LF"] for k in rcv]; LgF = [q_(k)["LgF"] for k in rcv]
        LgK = [q_(k).get("LgK") for k in rcv if q_(k).get("LgK") is not None and np.isfinite(q_(k).get("LgK"))]
        over = [k for k in rcv if q_(k)["LF"] > cr]; over_g = sorted(set(gid[k] for k in over))
        earl = [k for k in rcv if q_(k)["tD"] < ev[k]["tD"]]; earl_g = sorted(set(gid[k] for k in earl))
        less = [k for k in rcv if q_(k)["LF"] < ev[k]["LT"]]; less_g = sorted(set(gid[k] for k in less))
        reps = [first[g] for g in rcv_g]
        rat = [q_(k)["LF"] / ev[k]["LT"] for k in reps if ev[k]["LT"] > 0]
        dt = [(q_(k)["tD"] - ev[k]["tD"]) / rs for k in reps]
        dlt = [q_(k)["LF"] - ev[k]["LT"] for k in rcv]
        kf = [abs(q_(k)["LK"] - q_(k)["LF"]) for k in rcv if np.isfinite(q_(k)["LK"])]
        r[a_] = dict(n=len(ev), ng=ng, rcv=len(rcv), rcv_g=len(rcv_g), maxLF=max(LF) if LF else None, maxLgF=max(LgF) if LgF else None,
                     maxLgK=max(LgK) if LgK else None, over=len(over), over_g=len(over_g), over_ids=["%s|%d" % (m, g) for g in over_g],
                     earl=len(earl), earl_g=len(earl_g), less=len(less), less_g=len(less_g), crossover=cr,
                     med_ratio=float(np.median(rat)) if rat else None, med_dt=float(np.median(dt)) if dt else None,
                     max_dLT=max(dlt) if dlt else None, maxKF=max(kf) if kf else None)
        q = r[a_]
        D2.append("| %s | %g | %d / %d | %d / %d | %s / %s / %s | %d / %d | %d / %d | %d / %d | %s | %s | %s | %s |" % (
            m, apv, q["n"], q["ng"], q["rcv"], q["rcv_g"], fmt(q["maxLF"]), fmt(q["maxLgF"]), fmt(q["maxLgK"]), q["over"], q["over_g"], q["earl"], q["earl_g"],
            q["less"], q["less_g"], fmt(q["med_ratio"]), fmt(q["med_dt"], 2), fmt(q["max_dLT"]), fmt(q["maxKF"])))
        for g in rcv_g:
            k = first[g]; qq = q_(k)
            whys = []
            if qq["LF"] >= ev[k]["LT"]:
                whys.append("not less redshifted")
            if qq["tD"] >= ev[k]["tD"]:
                whys.append("not earlier")
            if whys:
                EXC.append("| %s | K%d | %s | %s (%s) | %g | %s | %s | %s | %s |" % (m, g + 1, ev[k]["cls"], fmt(ev[k]["LT"]), fmt(ev[k]["tD"] / rs, 3), apv, fmt(qq["LF"]),
                                                                                  fmt(qq["LF"] - ev[k]["LT"]), fmt(qq["tD"] / rs, 3), "; ".join(whys)))
                r.setdefault("_exc", []).append(dict(g=g, cls=ev[k]["cls"], LT=ev[k]["LT"], ap=apv, LF=qq["LF"], d=qq["LF"] - ev[k]["LT"], why=whys))
    kmax = int(np.argmax([d["LT"] for d in ev]))
    dm = {}
    for a_ in ("0.05", "0.1"):
        rc = [k for k, d in enumerate(ev) if d["disp"].get(a_, {}).get("status") == 1]
        if rc:
            kk = max(rc, key=lambda k: ev[k]["disp"][a_]["LF"]); dm[a_] = (ev[kk]["disp"][a_]["LF"], ev[kk]["cls"])
    above = any(v[0] > ev[kmax]["LT"] for v in dm.values())
    MX.append("| %s | %s (%s) | %s | %s |" % (m, fmt(ev[kmax]["LT"]), ev[kmax]["cls"],
                                             " / ".join("%s (%s)" % (fmt(dm[a_][0]), dm[a_][1]) if a_ in dm else "-" for a_ in ("0.05", "0.1")), "yes" if above else "no"))
    res[m] = dict(summary=r, n_events=len(ev), n_distinct=ng, max_stored=(ev[kmax]["LT"], ev[kmax]["cls"]), disp_max={k: list(v) for k, v in dm.items()}, disp_above=above,
                  cont_check=max((abs(d["cont_LT"] - d["LT"]) for d in ev if d.get("cont_LT") is not None and np.isfinite(d["cont_LT"])), default=None),
                  n_cont=sum(1 for d in ev if d.get("cont_LT") is not None and np.isfinite(d["cont_LT"])))
# ---- the scope statements, computed
def span(mems, key):
    v = [res[m]["summary"][a]["%s" % key] for m in mems if m in res for a in res[m]["summary"] if not a.startswith("_") and res[m]["summary"][a][key] is not None]
    return (min(v), max(v)) if v else (None, None)


four = [m for m in FOUR if m in res]; rest = [m for m in res if m not in FOUR]
apsall = sorted(set(a for m in res for a in res[m]["summary"] if not a.startswith("_")), key=float)
S = dict(four=four, rest=rest, ratio_four=span(four, "med_ratio"), ratio_rest=span(rest, "med_ratio"), ratio_all=span(list(res), "med_ratio"),
         over_rows_four=sum(res[m]["summary"][a]["over"] for m in four for a in apsall if a in res[m]["summary"]),
         over_g_four=len(set(i for m in four for a in apsall if a in res[m]["summary"] for i in res[m]["summary"][a]["over_ids"])),
         over_pairs_g_four=sum(res[m]["summary"][a]["over_g"] for m in four for a in apsall if a in res[m]["summary"]),
         over_rows_all=sum(res[m]["summary"][a]["over"] for m in res for a in apsall if a in res[m]["summary"]),
         over_g_all=len(set(i for m in res for a in apsall if a in res[m]["summary"] for i in res[m]["summary"][a]["over_ids"])),
         rows_four=sum(res[m]["summary"][a]["n"] for m in four for a in apsall if a in res[m]["summary"]),
         maxLF_four={a: max((res[m]["summary"][a]["maxLF"] for m in four if a in res[m]["summary"] and res[m]["summary"][a]["maxLF"] is not None), default=None) for a in apsall},
         maxLgF_four={a: max((res[m]["summary"][a]["maxLgF"] for m in four if a in res[m]["summary"] and res[m]["summary"][a]["maxLgF"] is not None), default=None) for a in apsall},
         maxLgK_four={a: max((res[m]["summary"][a]["maxLgK"] for m in four if a in res[m]["summary"] and res[m]["summary"][a]["maxLgK"] is not None), default=None) for a in apsall},
         earlier_all=all(res[m]["summary"][a]["earl"] == res[m]["summary"][a]["rcv"] for m in res for a in apsall if a in res[m]["summary"]),
         less_all_four=all(res[m]["summary"][a]["less"] == res[m]["summary"][a]["rcv"] for m in four for a in apsall if a in res[m]["summary"]),
         exc_members=sorted(set(m for m in res if res[m]["summary"].get("_exc"))),
         exc_max={m: max(e["d"] for e in res[m]["summary"]["_exc"]) for m in res if res[m]["summary"].get("_exc")},
         disp_above=[m for m in res if res[m]["disp_above"]])
res["_scope"] = S
D2.append("\nScope, computed: median L_F/L_T over arrival clusters, per member and ap_e: %s to %s on %s; %s to %s on %s. Crossover exceedances on "
          "those four members: %d (row, ap_e) pairs, %d (arrival cluster, ap_e) pairs, %d arrival clusters (over all members: %d pairs of rows, "
          "%d clusters). Every received dispersive ray arrives earlier than the stored arrival: %s. On the four members every received "
          "dispersive ray is less redshifted: %s. Members with a received ray not less redshifted or not earlier (Table 9-H3v4): %s." % (
              fmt(S["ratio_four"][0]), fmt(S["ratio_four"][1]), ", ".join(four), fmt(S["ratio_rest"][0]), fmt(S["ratio_rest"][1]), ", ".join(rest),
              S["over_rows_four"], S["over_pairs_g_four"], S["over_g_four"], S["over_rows_all"], S["over_g_all"], "yes" if S["earlier_all"] else "NO",
              "yes" if S["less_all_four"] else "NO", ", ".join("%s (largest L_F − L_T %s)" % (m, fmt(S["exc_max"][m])) for m in S["exc_members"]) or "none"))
D2.append("\nEmission-time spread of each A arrival cluster's rows (a; the rows' emission times differ by this much): %s." % "; ".join("%s %s" % (m, ", ".join("%s %s" % (k, ("%.2e" % v) if v is not None else "not one run")
                                                                                                for k, v in SPREAD.get(m, {}).items())) for m in res if not m.startswith("_")))
res["_spread"] = SPREAD
D2.append("\nContinuum reproduction from the emission side (forward DOP853 from the event against the stored L_T), max abs difference (rows "
          "checked): %s." % "; ".join("%s %s (%d)" % (m, fmt(res[m]["cont_check"]), res[m]["n_cont"]) for m in res if not m.startswith("_")))
open(os.path.join(OUT, "dispev4_tables.md"), "w").write("\n".join(D1 + D2 + EXC + MX) + "\n")
json.dump(res, open(os.path.join(OUT, "dispev4_tables.json"), "w"), indent=1, default=float)
print("\n".join(D2 + EXC + MX))
