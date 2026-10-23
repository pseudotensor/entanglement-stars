"""b4_light_clocks_v1.py (busi f52b1152, 20 Oct; REVIEWA_B4_rs16_v1.md sec. 3): the light run's extra checkpoint clocks on a B4 history, at the
matched-clock values of the r_s/a 64 and 32 readings. Matched clocks run from the formation record (the first record with R_t non-null, the
ruling's clock origin) in units of r_s ln(r_s/a): t_B4 = t0_B4 + (ln S_B4 / ln S)(t_S - t0_S), every clock in units of each object's own r_s.
  64 (B2, 417a4018; the reading v6o_B2_k1 collects every 2 r_s, plus 15 and 30): first root 12.125 (the first episode, to 17.25), and from
     18.375 the trapped history to 100; both origins are listed, since the rule "first R_t record" gives 12.125.
  32 (C3set19 collapse_c19_rs32_A_R2_tau8_ell4_avg_r145, 93d0df1c, first root 16.5): no light reading exists yet; it is assumed to use the
     production's 2 r_s collection (2, 4, ..., 100). Override with --clocks32.
Each matched clock is rounded to the history's record grid (tau / r_s) and kept when it is inside [t0_B4, last record]. Prints the comma list
for --extra-ck-rs on stdout and the table to --table.
Usage: python -I b4_light_clocks_v1.py <B4 record json> [--table file] [--clocks64 a,b,..] [--clocks32 a,b,..] [--t0-64 12.125,18.375] [--t0-32 16.5]"""
import argparse, json, math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record"); ap.add_argument("--table", default=None)
    ap.add_argument("--clocks64", default=",".join("%g" % x for x in sorted(set([15.0, 30.0] + [2.0 * k for k in range(1, 51)]))))
    ap.add_argument("--clocks32", default=",".join("%g" % (2.0 * k) for k in range(1, 51)))
    ap.add_argument("--t0-64", dest="t0_64", default="12.125,18.375"); ap.add_argument("--t0-32", dest="t0_32", default="16.5")
    a = ap.parse_args()
    D = json.load(open(a.record))
    rs = float(D["args"]["rs"]); tau = float(D["args"]["tau"]); dt = tau / rs
    recs = D["records"]
    roots = [r for r in recs if r.get("R_t") is not None]
    if not roots:
        raise SystemExit("no record with R_t non-null: no clock origin (the gate should have stopped this history)")
    t0 = float(roots[0]["T_over_rs"]); tlast = float(recs[-1]["T_over_rs"])
    rows = []; ks = set()
    for S, t0s, cl in ((64.0, a.t0_64, a.clocks64), (32.0, a.t0_32, a.clocks32)):
        f = math.log(rs) / math.log(S)
        for t0S in [float(x) for x in t0s.split(",") if x.strip()]:
            for tS in [t0S] + [float(x) for x in cl.split(",") if x.strip()]:
                if tS < t0S - 1e-9:
                    continue
                t = t0 + f * (tS - t0S)
                k = int(round(t / dt)); tg = k * dt
                if tg < t0 - 1e-9 or tg > tlast + 1e-9:
                    continue
                rows.append((S, t0S, tS, t, tg)); ks.add(k)
    out = ",".join("%g" % (k * dt) for k in sorted(ks))
    if a.table:
        with open(a.table, "w") as fh:
            fh.write("# B4 light run's matched clocks on %s (r_s/a %g, tau %g, grid %g r_s; origin t0 %.4f r_s, last record %.4f)\n\n" % (a.record, rs, tau, dt, t0, tlast))
            fh.write("| reading r_s/a | its origin | its clock | matched clock here | on the record grid |\n|---|---|---|---|---|\n")
            for r in rows:
                fh.write("| %g | %g | %g | %.4f | %g |\n" % r)
            fh.write("\n%d distinct clocks\n" % len(ks))
    print(out)


if __name__ == "__main__":
    main()
