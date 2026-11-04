"""rz_v3_law_v2.py (REDSHIFT_received_v4.1, inbox #59): TABLES script (v2 = v1 with the ray-mask label on every row of a member whose ray mask is unavailable, in L1v3 and L2v3; no computation changes). v1 = rz_v2_law_v1's fits of the received redshift against the hover and
receiver-clock integrals, with the changes the v2 checks ask for (ReviewB_rzchk3 secs. 4.1 and 6; ReviewA_rzchk3 sec. 6 items 3 and 11):
  - the fitted quantity is named and both are fitted: L_T = ln(1/G_D) (v2's) and L_D = ln(1/g_D) = L_T + ln A_D(u) (the static receiver);
  - the population is masked as rz_v3_head_v1 defines 'resolved' (emission r_e >= 1 a, the clock not inside the first site by the emission,
    the ray's smallest |r| >= 1 a; rz_rays_v4m), beside v2's unmasked population;
  - T_x, the zone exit of the ray arriving at u, is the stored hover_t0 of the backward pass: the step midpoint (h = 0.5 a grid) at which the
    ray, integrated backward from its arrival, is first found inside the root's linear zone; arrivals are 1 a apart, so dT_x/du from central
    differences takes multiples of 0.25 only. It is printed with that label and not used as evidence.
Per clock with at least 20 trapped arrivals inside the solver-certified record interval: (K) L = s K_H + c on the ray's own hover integral;
(Q) L = s Q + c, Q(u) = int kappa_R(T_x(u')) du'; (E) one exponential in u, kappa_fit/kappa_H; out of fit (first half fitted, second half
scored): constant / exponential / K_H + c / Q + c (unit slope), the smallest rms wins. usage: rz_v3_law_v1.py OUTDIR SPEC [SPEC ...]
"""
import json, os, sys
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

OUT = sys.argv[1]
SPECS = [s.split("=", 1) for s in sys.argv[2:]]
os.makedirs(OUT, exist_ok=True)


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


def rng(v, p=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    a, b = fmt(min(v), p), fmt(max(v), p)
    return a if a == b else "%s to %s" % (a, b)


def lsq(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, float(np.sqrt(np.mean((y - A @ c) ** 2)))


def fits(uu, y, x, kx, tx, kH):
    Q = np.concatenate([[0.0], np.cumsum(0.5 * (kx[1:] + kx[:-1]) * np.diff(uu))])
    cK, rK = lsq(x, y); cQ, rQ = lsq(Q, y)
    pe = np.polyfit(uu, y, 1)
    n = len(y); h1 = np.arange(n) < n // 2; h2 = ~h1
    r_c = float(np.sqrt(np.mean((y[h2] - np.mean(y[h1])) ** 2)))
    p1 = np.polyfit(uu[h1], y[h1], 1); r_e = float(np.sqrt(np.mean((y[h2] - np.polyval(p1, uu[h2])) ** 2)))
    c1 = np.mean(y[h1] - x[h1]); r_k = float(np.sqrt(np.mean((y[h2] - (x[h2] + c1)) ** 2)))
    c2 = np.mean(y[h1] - Q[h1]); r_q = float(np.sqrt(np.mean((y[h2] - (Q[h2] + c2)) ** 2)))
    scores = dict(constant=r_c, exponential=r_e, K_H=r_k, Q=r_q)
    return dict(sK=float(cK[0]), cK=float(cK[1]), rK=rK, sQ=float(cQ[0]), rQ=rQ, kfit=float(pe[0] / kH), scores=scores, win=min(scores, key=scores.get))


L1 = ["**Table L1v3. The fits per comoving clock, on the resolved population** (rz_v3_law_v2.py): backward events at 4 r_s with a hover in "
      "the outer root's linear zone, inside the solver-certified record interval, masked (r_e ≥ 1 a, clock outside the first site at emission, "
      "ray's smallest |r| ≥ 1 a). For L_T = ln(1/G_D): (K) slope / intercept / rms on the ray's own hover integral K_H; (Q) slope / rms on "
      "Q(u) = ∫κ_R(T_x(u')) du'; (E) κ_fit/κ_H; out-of-fit rms constant / exponential / K_H + c / Q + c and the winner. For L_D = ln(1/g_D): "
      "(K) slope and the out-of-fit winner. dT_x/du median: T_x is read on the h = 0.5 a step grid and the arrivals are 1 a apart, so this "
      "median takes multiples of 0.25 only (a label, not a measurement).\n",
      "| member | clock | arrivals (of v2's) | u/r_s range | L_T: (K) slope / intercept / rms | L_T: (Q) slope / rms | L_T: (E) κ_fit/κ_H | L_T out-of-fit rms: const / exp / K_H / Q | L_T winner | L_D: (K) slope | L_D winner | dT_x/du median (grid-quantized) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
res = {}
for name, prefix in SPECS:
    if not os.path.exists(prefix + ".npz"):
        continue
    Z = dict(np.load(prefix + ".npz")); S = json.load(open(prefix + ".json"))
    rs = S["rs"]; g = S["g"]; T = Z["T"]
    Tc_nom = (S["T_cert_rs"] if S["T_cert_rs"] is not None else -1.0) * rs
    k = int(np.argmin(np.abs(T - Tc_nom))); Tc = float(T[k]) if abs(T[k] - Tc_nom) < 1e-3 * rs else Tc_nom
    rec = Z["receivers"]; i4 = int(np.argmin(np.abs(rec - 4 * rs)))
    key = [kk for kk in Z if kk.startswith("B") and kk.endswith("_trec") and abs(float(kk[1:-5].rstrip("_")) - rec[i4]) < 1e-6]
    if not key:
        continue
    key = key[0][:-4]
    u = Z[key + "trec"]; lnG = np.log(Z[key + "fD"])[:, None] - Z[key + "psie"]
    with np.errstate(invalid="ignore"):
        lnA = 0.5 * np.log(Z[key + "ND_"] ** 2 - Z[key + "vD_"] ** 2)
    Ke = Z[key + "K_e"]; hv = Z[key + "hover_e"]; Tx = Z[key + "hover_t0"]
    te = Z[key + "te"]; re = Z[key + "re"]
    rmask = (key + "rmin_e") in Z
    mR = (Z[key + "rmin_e"] >= 1.0) if rmask else np.ones(lnG.shape, bool)
    tg = Z["em_tg"]; RR = Z["em_R"]; ne = RR.shape[1]
    tsub = np.full(ne, np.inf)
    for e in range(ne):
        okc = (tg >= Z["em_t0"][e] - 1e-9) & np.isfinite(RR[:, e]) & (np.abs(RR[:, e]) < 1.0)
        if okc.any():
            tsub[e] = float(tg[np.argmax(okc)])
    resm = (re >= 1.0) & (te < tsub[None, :]) & mR
    kapR = np.nan_to_num(Z["RT_kap"], nan=0.0)
    nA1 = int(np.sum(Z["em_t0"] == 0))
    rows = []
    for e in range(lnG.shape[1]):
        base = np.isfinite(lnG[:, e]) & np.isfinite(Ke[:, e]) & (hv[:, e] > 0) & (u <= Tc + 1e-9) & np.isfinite(Tx)
        ok = base & resm[:, e]
        if ok.sum() < 20:
            if base.sum() >= 20:
                rows.append(dict(e=e, n=int(ok.sum()), n_v2=int(base.sum()), masked_out=True))
                L1.append("| %s | %s%d | %d (%d) | - | fewer than 20 resolved arrivals | - | - | - | - | - | - | - |" % (name + ("" if rmask else " (e, c only: ray mask unavailable)"), "A1." if e < nA1 else "A2.", e, int(ok.sum()), int(base.sum())))
            continue
        uu = u[ok]; yT = -lnG[ok, e]; yD = yT + lnA[ok]; x = Ke[ok, e]; tx = Tx[ok]
        kx = np.interp(tx, T, kapR)
        E_u = np.interp(uu, T, Z["E"]); kH = float(np.mean(1.0 / (4 * g * E_u)))
        fT = fits(uu, yT, x, kx, tx, kH); fD = fits(uu, yD, x, kx, tx, kH)
        du0 = np.median(np.diff(u)); cont = (np.diff(uu) < 1.01 * du0)
        dT = np.full(len(uu), np.nan)
        for j in range(1, len(uu) - 1):
            if cont[j - 1] and cont[j]:
                dT[j] = (tx[j + 1] - tx[j - 1]) / (uu[j + 1] - uu[j - 1])
        row = dict(e=e, n=int(ok.sum()), n_v2=int(base.sum()), u0=float(uu[0] / rs), u1=float(uu[-1] / rs), T=fT, D=fD,
                   dTmed=float(np.nanmedian(dT)) if np.isfinite(dT).any() else None, masked_out=False)
        rows.append(row)
        s = fT["scores"]
        L1.append("| %s | %s%d | %d (%d) | %s..%s | %s / %s / %s | %s / %s | %s | %s / %s / %s / %s | %s | %s | %s | %s |" % (
            name + ("" if rmask else " (e, c only: ray mask unavailable)"), "A1." if e < nA1 else "A2.", e, row["n"], row["n_v2"], fmt(row["u0"], 2), fmt(row["u1"], 2), fmt(fT["sK"]), fmt(fT["cK"]), fmt(fT["rK"]),
            fmt(fT["sQ"]), fmt(fT["rQ"]), fmt(fT["kfit"]), fmt(s["constant"]), fmt(s["exponential"]), fmt(s["K_H"]), fmt(s["Q"]), fT["win"], fmt(fD["sK"]), fD["win"],
            fmt(row["dTmed"])))
    res[name] = dict(rows=rows, rmask=bool(rmask))
L2 = ["\n**Table L2v3. Per member** (from Table L1v3, resolved population): clocks fitted; ranges over the clocks of the (K) slope, the (Q) slope and "
      "the (E) κ_fit/κ_H for L_T; the out-of-fit winners counted (constant / exponential / K_H / Q) for L_T and for L_D; and the members' "
      "statement read off the L_T winners.\n",
      "| member | clocks fitted (dropped by the masks) | L_T (K) slope range | L_T (Q) slope range | L_T (E) κ_fit/κ_H range | L_T winners | L_D winners | L_D (K) slope range |",
      "|---|---|---|---|---|---|---|---|"]
for name, d in res.items():
    rows = [r for r in d["rows"] if not r["masked_out"]]
    nd = sum(1 for r in d["rows"] if r["masked_out"])
    if not rows:
        L2.append("| %s | 0 (%d) | - | - | - | - | - | - |" % (name, nd))
        continue
    wT = [r["T"]["win"] for r in rows]; wD = [r["D"]["win"] for r in rows]
    cnt = lambda w: "%d / %d / %d / %d" % (w.count("constant"), w.count("exponential"), w.count("K_H"), w.count("Q"))
    d["summary"] = dict(n=len(rows), dropped=nd, wT=dict(constant=wT.count("constant"), exponential=wT.count("exponential"), K_H=wT.count("K_H"), Q=wT.count("Q")),
                        wD=dict(constant=wD.count("constant"), exponential=wD.count("exponential"), K_H=wD.count("K_H"), Q=wD.count("Q")),
                        sK=[min(r["T"]["sK"] for r in rows), max(r["T"]["sK"] for r in rows)], sQ=[min(r["T"]["sQ"] for r in rows), max(r["T"]["sQ"] for r in rows)],
                        kfit=[min(r["T"]["kfit"] for r in rows), max(r["T"]["kfit"] for r in rows)], sKD=[min(r["D"]["sK"] for r in rows), max(r["D"]["sK"] for r in rows)])
    L2.append("| %s%s | %d (%d) | %s | %s | %s | %s | %s | %s |" % (name, "" if d["rmask"] else " (e, c only: ray mask unavailable)", len(rows), nd, rng([r["T"]["sK"] for r in rows]),
                                                             rng([r["T"]["sQ"] for r in rows]), rng([r["T"]["kfit"] for r in rows]), cnt(wT), cnt(wD), rng([r["D"]["sK"] for r in rows])))
open(os.path.join(OUT, "law_tables.md"), "w").write("\n".join(L1 + L2) + "\n")
json.dump(res, open(os.path.join(OUT, "law_tables.json"), "w"), indent=1, default=float)
print("\n".join(L2))
