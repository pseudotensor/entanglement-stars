"""rz_v4_variants_v4.py (REDSHIFT_received_v4.1, inbox #59): TABLES script; v4 = v3 naming the fixed-emission statuses as sampled-map statuses (non-receipt not verified); numbers unchanged; v3 = v2 with the captions naming this script; v2 = v1 with Table V1b carrying the B class too (by launch label); v1 = rz_v3_variants_v1 with the changes the v3 checks ask for
(ReviewB_rzchk4 secs. 2.3, 2.4; ReviewA_rzchk4 sec. 6 items 7, 9):
  - one sign convention everywhere: dL_T = L_T(variant or completion) - L_T(main or reference), dT_D = T_D(variant) - T_D(main) (v3 printed
    the change of ln G_D, the opposite sign, at fixed arrival, and main minus variant at fixed emission);
  - the fixed-emission comparison prints each main event's status in the variant: received (bracketed by consecutive arrivals 1 a apart),
    not received by the variant's certified end (the label lies past the variant's last received emission of that clock), before the variant's
    first received emission of that clock, or inside its received range with no adjacent bracket (a gap); none of these is merged into another;
  - Table V1b: the blueshift maxima's own change under each variant (the main run's most blueshifted received event per class, at fixed arrival);
  - Table V1 compares the stored passes (rz_rays_v4: main and variants both release the later clocks one step early, so the A2 columns compare
    like implementations); Table V2 compares the exact-release passes (rz_rays_v4n reference, rz_rays_v4nc completions); the completions have
    no propagation-variant passes and are untested under them.
Nothing is re-integrated here. usage: rz_v4_variants_v4.py OUTDIR RAYSDIR RAYSN_DIR COMPN_DIR
"""
import glob, json, os, sys
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

OUT, RAYS, RAYSN, COMPN = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
VARS = (("app_hold_cubic", "", "the production hold"), ("dense_lin_quintic", "", "the dense fields"), ("bar_lin_cubic", "h025", "step h/2"),
        ("bar_pchipt_cubic", "", "PCHIP in time"), ("bar_lin_pchip", "", "PCHIP in r"), ("bar_lin_linear", "", "linear in r"))
STATUS = ("bracketed", "label past the variant's last sampled received emission of the clock (non-receipt not verified)",
          "label before the variant's first sampled received emission of the clock", "no adjacent bracket on the sampled map")


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


def load(p):
    if not os.path.exists(p + ".npz"):
        return None
    Z = dict(np.load(p + ".npz")); Z["_S"] = json.load(open(p + ".json"))
    return Z


def bwd(Z):
    S = Z["_S"]; rs = S["rs"]; T = Z["T"]
    Tc_nom = (S["T_cert_rs"] if S["T_cert_rs"] is not None else -1.0) * rs
    k = int(np.argmin(np.abs(T - Tc_nom))); Tc = float(T[k]) if abs(T[k] - Tc_nom) < 1e-3 * rs else Tc_nom
    rec = Z["receivers"]; i4 = int(np.argmin(np.abs(rec - 4 * rs)))
    key = [kk for kk in Z if kk.startswith("B") and kk.endswith("_trec") and abs(float(kk[1:-5].rstrip("_")) - rec[i4]) < 1e-6][0][:-4]
    LT = -(np.log(Z[key + "fD"])[:, None] - Z[key + "psie"])
    LTf = -(np.log(Z["F_fD"][:, i4]) + Z["F_psiD"][:, i4])
    return dict(trec=Z[key + "trec"], LT=LT, re=Z[key + "re"], taue=Z[key + "taue"], Tc=Tc, rs=rs, nA1=int(np.sum(Z["em_t0"] == 0)),
                Bl=dict(kind=Z["L_kind"], krec=Z["L_k_rec"].astype(int), delta=Z["L_delta"], LT=LTf, tD=Z["F_tD"][:, i4]))


def fixem(B0, B1, i, e, Tlim1):
    """the variant's L_T and arrival at the main event's tau_e on the same clock: (status, LT, T_D)."""
    ts = B0["taue"][i, e]
    tr = B1["trec"]; ta = B1["taue"][:, e]; lg = B1["LT"][:, e]
    ok = np.isfinite(ta) & np.isfinite(lg) & (tr <= Tlim1 + 1e-9)
    if not ok.any():
        return (1, None, None)
    if ts > np.max(ta[ok]):
        return (1, None, None)
    if ts < np.min(ta[ok]):
        return (2, None, None)
    idx = np.where(ok[:-1] & ok[1:] & (np.diff(tr) < 1.01 * np.median(np.diff(tr))))[0]
    lo = np.minimum(ta[idx], ta[idx + 1]); hi = np.maximum(ta[idx], ta[idx + 1])
    br = idx[(lo <= ts) & (ts <= hi) & (ta[idx + 1] != ta[idx])]
    if not len(br):
        return (3, None, None)
    j = br[np.argmin(np.abs(tr[br] - B0["trec"][i]))]
    w = (ts - ta[j]) / (ta[j + 1] - ta[j])
    return (0, float(lg[j] + w * (lg[j + 1] - lg[j])), float(tr[j] + w * (tr[j + 1] - tr[j])))


def head_events(B):
    cert = (B["trec"] <= B["Tc"] + 1e-9)[:, None] & np.isfinite(B["LT"])
    ne = B["LT"].shape[1]; a1 = np.arange(ne) < B["nA1"]
    out = {}
    for cls, m in (("A1 r_e>=1a", cert & a1[None, :] & (B["re"] >= 1.0)), ("A1 sub-site", cert & a1[None, :] & (B["re"] < 1.0)), ("A2", cert & ~a1[None, :])):
        if m.any():
            i, e = np.unravel_index(np.argmax(np.where(m, B["LT"], -np.inf)), B["LT"].shape)
            out[cls] = (int(i), int(e), float(B["LT"][i, e]))
        else:
            out[cls] = None
    return out


def blue_events(B):
    """the most blueshifted received event per class (smallest L_T < 0) inside the interval."""
    cert = (B["trec"] <= B["Tc"] + 1e-9)[:, None] & np.isfinite(B["LT"])
    ne = B["LT"].shape[1]; a1 = np.arange(ne) < B["nA1"]
    out = {}
    for cls, m in (("A1", cert & a1[None, :]), ("A2", cert & ~a1[None, :])):
        mm = m & (B["LT"] < 0)
        if mm.any():
            i, e = np.unravel_index(np.argmin(np.where(mm, B["LT"], np.inf)), B["LT"].shape)
            out[cls] = (int(i), int(e), float(B["LT"][i, e]))
    return out


def compare(B0, B1, Tlim1):
    d1 = {round(float(t), 4): i for i, t in enumerate(B1["trec"])}
    ne = min(B0["LT"].shape[1], B1["LT"].shape[1])
    rows = []
    for i, t in enumerate(B0["trec"]):
        j = d1.get(round(float(t), 4))
        if j is None or t > B0["Tc"] + 1e-9:
            continue
        for e in range(ne):
            rows.append((i, j, e, B0["LT"][i, e], B1["LT"][j, e] if B1["trec"][j] <= Tlim1 + 1e-9 else np.nan))
    a = np.array([r_[3] for r_ in rows]); b = np.array([r_[4] for r_ in rows])
    both = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[both]; deep = both & (a >= 5)
    fe = {s: 0 for s in range(4)}; fe_d, fe_t, fe_dd = [], [], []
    for (i, j, e, x, y) in rows:
        if not np.isfinite(x):
            continue
        st, lt, td = fixem(B0, B1, i, e, Tlim1)
        fe[st] += 1
        if st == 0:
            fe_d.append(lt - x); fe_t.append(td - B0["trec"][i])
            if x >= 5:
                fe_dd.append(lt - x)
    idx = {(r_[0], r_[2]): k for k, r_ in enumerate(rows)}
    H0 = head_events(B0)
    dh, dhf = {}, {}
    for cls in ("A1 r_e>=1a", "A1 sub-site", "A2"):
        ev = H0.get(cls)
        if ev is None:
            dh[cls] = dhf[cls] = None
            continue
        k = idx.get((ev[0], ev[1]))
        dh[cls] = float(b[k] - a[k]) if k is not None and np.isfinite(a[k]) and np.isfinite(b[k]) else None
        st, lt, td = fixem(B0, B1, ev[0], ev[1], Tlim1)
        dhf[cls] = (st, float(lt - ev[2]) if st == 0 else None, float(td - B0["trec"][ev[0]]) if st == 0 else None)
    bl = {}
    for cls, ev in blue_events(B0).items():
        k = idx.get((ev[0], ev[1]))
        bl[cls] = (ev[2], float(b[k] - a[k]) if (k is not None and np.isfinite(b[k])) else None, float(B0["trec"][ev[0]] / B0["rs"]))
    L0, L1 = B0["Bl"], B1["Bl"]
    lab1 = {(int(k), round(float(dl), 9)): i for i, (k, dl, kd) in enumerate(zip(L1["krec"], L1["delta"], L1["kind"])) if kd == 1}
    bd = []; b_lost = 0
    for i, (k, dl, kd) in enumerate(zip(L0["krec"], L0["delta"], L0["kind"])):
        if kd != 1 or not (np.isfinite(L0["LT"][i]) and L0["tD"][i] <= B0["Tc"] + 1e-9):
            continue
        j = lab1.get((int(k), round(float(dl), 9)))
        if j is None or not (np.isfinite(L1["LT"][j]) and L1["tD"][j] <= Tlim1 + 1e-9):
            b_lost += 1
            continue
        bd.append(L1["LT"][j] - L0["LT"][i])
    mb = (L0["kind"] == 1) & np.isfinite(L0["LT"]) & (L0["tD"] <= B0["Tc"] + 1e-9) & (L0["LT"] < 0)
    if mb.any():
        i = int(np.where(mb)[0][np.argmin(L0["LT"][mb])])
        j = lab1.get((int(L0["krec"][i]), round(float(L0["delta"][i]), 9)))
        ok1 = j is not None and np.isfinite(L1["LT"][j]) and L1["tD"][j] <= Tlim1 + 1e-9
        bl["B"] = (float(L0["LT"][i]), float(L1["LT"][j] - L0["LT"][i]) if ok1 else None, float(L0["tD"][i] / B0["rs"]))
    sgn = lambda v: (float(v[np.argmax(np.abs(v))]) if len(v) else None)
    return dict(both=int(both.sum()), only0=int((np.isfinite(a) & ~np.isfinite(b)).sum()), only1=int((~np.isfinite(a) & np.isfinite(b)).sum()),
                dmax=sgn(d), dmax5=sgn((b - a)[deep]), dh=dh, dhf=dhf, fe=fe, fe_max=sgn(np.array(fe_d)), fe_max5=sgn(np.array(fe_dd)),
                fe_tmax=sgn(np.array(fe_t)), B_max=sgn(np.array(bd)), B_n=len(bd), B_lost=int(b_lost), blue=bl)


V1 = ["**Table V1v4. Propagation variants on the headline population, at fixed arrival and at fixed emission label** (rz_v4_variants_v4.py; the "
      "stored passes, rz_rays_v4, main and variants alike releasing the later clocks one step early, so the A2 columns compare like "
      "implementations). Signed changes are VARIANT MINUS MAIN: dL_T = L_T(variant) − L_T(main), dT_D = T_D(variant) − T_D(main); each 'largest' "
      "is the signed change of largest magnitude. Fixed arrival (clock, T_D): received in both / main only / variant only; the largest dL_T over all "
      "matched and over L_T ≥ 5 in the main run; dL_T at the main run's headline events of A1 r_e ≥ 1 a / A1 sub-site / A2 (the TESTED A "
      "headline events; B is compared by label in the last column, not at fixed arrival). Fixed emission label (clock, τ_e): the main's received "
      "events by their status on the variant's SAMPLED map, its consecutive arrivals 1 a apart inside its certified interval (bracketed / label "
      "past the variant's last sampled received emission of that clock / before its first / no adjacent bracket; sampled-map statuses: "
      "non-receipt is not verified, since no event is located directly), the largest dL_T among those received in the variant, over L_T ≥ 5, the largest dT_D, "
      "and dL_T (dT_D) at the A1 r_e ≥ 1 a and A2 headline events or their status. B by label (record, δ from each pass's own root): the largest dL_T "
      "(matched; not received in the variant).\n",
      "| member | variant | fixed arrival: both / main only / variant only | fixed arrival: largest dL_T all / L_T ≥ 5 | fixed arrival dL_T at the headline events: A1 r_e≥1a / A1 sub-site / A2 | fixed emission, sampled-map status: bracketed / past the last sampled / before the first sampled / no adjacent bracket | fixed emission: largest dL_T all / L_T ≥ 5; largest dT_D (a) | fixed emission at the headline events: A1 r_e≥1a / A2 | B by label: largest dL_T (matched; not received) |",
      "|---|---|---|---|---|---|---|---|---|"]
V1b = ["\n**Table V1b. The blueshift maxima under the variants** (ReviewA_rzchk4 item 9): per member, variant and class, the main run's most "
       "blueshifted received event inside its certified interval (its L_T < 0 and arrival T_D/r_s) and the variant's dL_T at the same (clock, T_D) "
       "('not received' where the variant has no arrival there); B: the main run's most blueshifted received launch and the variant's change at the "
       "same launch label (record, δ).\n",
       "| member | variant | A1: main L_T (T_D/r_s); dL_T | A2: main L_T (T_D/r_s); dL_T | B by label: main L_T (T_D/r_s); dL_T |",
       "|---|---|---|---|---|"]
res = {}
members = sorted(set(os.path.basename(p)[len("rays_"):].split("_bar_lin_cubic")[0] for p in glob.glob(os.path.join(RAYS, "rays_*_bar_lin_cubic.npz"))))
for m in members:
    Z0 = load(os.path.join(RAYS, "rays_%s_bar_lin_cubic" % m))
    if Z0 is None:
        continue
    B0 = bwd(Z0)
    for var, tag, lab in VARS:
        if var == "bar_lin_cubic" and not tag:
            continue
        Z1 = load(os.path.join(RAYS, "rays_%s_%s%s" % (m, var, ("_" + tag) if tag else "")))
        if Z1 is None:
            continue
        B1 = bwd(Z1)
        r = compare(B0, B1, B1["Tc"])
        r.update(var=var, tag=tag)
        res.setdefault(m, []).append(r)
        fh = lambda q: ("%s (%s)" % (fmt(q[1]), fmt(q[2], 2))) if (q and q[0] == 0) else (STATUS[q[0]] if q else "-")
        V1.append("| %s | %s%s (%s) | %d / %d / %d | %s / %s | %s / %s / %s | %d / %d / %d / %d | %s / %s; %s | %s / %s | %s (%d; %d) |" % (
            m, var, (" " + tag) if tag else "", lab, r["both"], r["only0"], r["only1"], fmt(r["dmax"]), fmt(r["dmax5"]),
            fmt(r["dh"].get("A1 r_e>=1a")), fmt(r["dh"].get("A1 sub-site")), fmt(r["dh"].get("A2")), r["fe"][0], r["fe"][1], r["fe"][2], r["fe"][3],
            fmt(r["fe_max"]), fmt(r["fe_max5"]), fmt(r["fe_tmax"], 2), fh(r["dhf"].get("A1 r_e>=1a")), fh(r["dhf"].get("A2")), fmt(r["B_max"]), r["B_n"], r["B_lost"]))
        if r["blue"]:
            fb = lambda q: ("%s (%s); %s" % (fmt(q[0]), fmt(q[2], 3), fmt(q[1]) if q[1] is not None else "not received")) if q else "-"
            V1b.append("| %s | %s%s | %s | %s | %s |" % (m, var, (" " + tag) if tag else "", fb(r["blue"].get("A1")), fb(r["blue"].get("A2")), fb(r["blue"].get("B"))))
# ---- completions on the exact-release passes
V2 = ["\n**Table V2v4. The interior completions of the reference collapse** (rz_v4_variants_v4.py on the exact-release passes: rz_rays_v4n for "
      "the reference, rz_rays_v4nc for the held-source and widened completions; the completions have no propagation-variant passes and are untested "
      "under them). Split at the completions' branch T/r_s 11.125 by the ARRIVAL time; signed changes COMPLETION MINUS REFERENCE. Per completion: "
      "flights ending before the branch (matched, largest dL_T); after it (matched, largest dL_T, median dL_T); at fixed emission label after the "
      "branch, the reference's received events by their status on the completion's sampled map (bracketed / label past its last sampled "
      "received emission of the clock / before its first / no adjacent bracket; non-receipt not verified) and the largest dL_T and dT_D among the received; and the largest L_T inside the solver-certified record interval for A1 with "
      "r_e ≥ 1 a, for the A1 and A2 clocks, and for A1, A2 and B (reference in brackets).\n",
      "| completion | before 11.125: matched, largest dL_T | after: matched, largest dL_T, median dL_T | after, fixed emission, sampled-map status: bracketed / past the last sampled / before the first sampled / no adjacent bracket; largest dL_T; largest dT_D (a) | max L_T, A1 r_e ≥ 1 a (reference) | max L_T, A1 and A2 clocks (reference) | max L_T, A1, A2 and B (reference) |",
      "|---|---|---|---|---|---|---|"]


def popmax(B):
    cert = (B["trec"] <= B["Tc"] + 1e-9)[:, None] & np.isfinite(B["LT"])
    ne = B["LT"].shape[1]; a1 = np.arange(ne) < B["nA1"]
    m1 = cert & a1[None, :] & (B["re"] >= 1.0)
    v1 = float(np.max(B["LT"][m1])) if m1.any() else np.nan
    v12 = float(np.max(B["LT"][cert])) if cert.any() else np.nan
    Bl = B["Bl"]; mb = (Bl["kind"] == 1) & np.isfinite(Bl["LT"]) & (Bl["tD"] <= B["Tc"] + 1e-9)
    vb = float(np.max(Bl["LT"][mb])) if mb.any() else np.nan
    return v1, v12, max(v12, vb) if np.isfinite(vb) else v12


Z0 = load(os.path.join(RAYSN, "rays_P64_s4_bar_lin_cubic"))
if Z0 is not None:
    B0 = bwd(Z0); P0 = popmax(B0)
    for cname in ("P64_s4_heldsrc", "P64_s4_widened"):
        Z1 = load(os.path.join(COMPN, "rays_%s_bar_lin_cubic" % cname))
        if Z1 is None:
            continue
        B1 = bwd(Z1); P1 = popmax(B1)
        d1 = {round(float(t), 4): i for i, t in enumerate(B1["trec"])}
        ne = min(B0["LT"].shape[1], B1["LT"].shape[1])
        rows = [(i, d1[round(float(t), 4)], e, B0["LT"][i, e], B1["LT"][d1[round(float(t), 4)], e]) for i, t in enumerate(B0["trec"])
                if round(float(t), 4) in d1 and t <= min(B0["Tc"], B1["Tc"]) + 1e-9 for e in range(ne)]
        a = np.array([r_[3] for r_ in rows]); b = np.array([r_[4] for r_ in rows]); tr = np.array([B0["trec"][r_[0]] for r_ in rows])
        both = np.isfinite(a) & np.isfinite(b)
        pre = both & (tr < 11.125 * B0["rs"]); post = both & (tr >= 11.125 * B0["rs"])
        fe = {s: 0 for s in range(4)}; fe_d, fe_t = [], []
        for (i, j, e, x, y) in rows:
            if not np.isfinite(x) or B0["trec"][i] < 11.125 * B0["rs"]:
                continue
            st, lt, td = fixem(B0, B1, i, e, B1["Tc"])
            fe[st] += 1
            if st == 0:
                fe_d.append(lt - x); fe_t.append(td - B0["trec"][i])
        sgn = lambda v: (float(v[np.argmax(np.abs(v))]) if len(v) else None)
        r = dict(n_pre=int(pre.sum()), pre_max=sgn((b - a)[pre]), n_post=int(post.sum()), post_max=sgn((b - a)[post]),
                 post_med=float(np.median((b - a)[post])) if post.any() else None, fe=fe, fe_max=sgn(np.array(fe_d)), fe_tmax=sgn(np.array(fe_t)),
                 popmax=list(P1), popmax_ref=list(P0))
        res[cname] = r
        V2.append("| %s | %d, %s | %d, %s, %s | %d / %d / %d / %d; %s; %s | %s (%s) | %s (%s) | %s (%s) |" % (
            cname, r["n_pre"], fmt(r["pre_max"]), r["n_post"], fmt(r["post_max"]), fmt(r["post_med"]), fe[0], fe[1], fe[2], fe[3], fmt(r["fe_max"]),
            fmt(r["fe_tmax"], 2), fmt(P1[0]), fmt(P0[0]), fmt(P1[1]), fmt(P0[1]), fmt(P1[2]), fmt(P0[2])))
open(os.path.join(OUT, "variants_tables.md"), "w").write("\n".join(V1 + V1b + V2) + "\n")
json.dump(res, open(os.path.join(OUT, "variants_tables.json"), "w"), indent=1, default=float)
print("\n".join(V1 + V1b + V2))
