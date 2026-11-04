"""rz_v4_head_v3.py (REDSHIFT_received_v4.1, inbox #59): TABLES script. v3 = v2 with the ray-mask label on Table I1c's and I1e's rows and on I1's cross-member resolved rows, Table I1e on every member, and H3's caption stating its domain ends exactly; no computation changes. v2 = v1 with captions naming this script and the exact-release passes (v1 named rz_v3_head_v1 and called H1's unmasked columns v2's, untrue for A2 there) and P64_s4_ext200's row note naming its early-release A2 clocks and its unreproducible whole window; no computation changes. v1 = rz_v3_head_v1 read from the rz_rays_v4n passes (the later-released
clocks released exactly at T_rel; the ray's smallest |r| with its emission endpoint and each step's cubic minimum), with H3's diagnostics and H4's
counts also on the resolved domain, the members whose ray mask is unavailable labelled in every row, H6's whole-window values marked when they
arrive past the admitted interval, I1's tail rule stated and its per-clock endpoints exported, the received proper-time shares on both domains,
and I1c's last column named for what it counts (ReviewB_rzchk4 secs. 1.4, 3.1-3.5; ReviewA_rzchk4 sec. 6 items 5, 6, 10). v3 =
rz_v2_head_v1's headline and event-level tables, read from the
rz_rays_v4m ray passes (v4 plus the running minimum of |r| along every front; the launches and values of v4 are reproduced, so with the
masks off every v2 number is reproduced), with the changes the v2 checks ask for (ReviewB_rzchk3 secs. 2.2-2.3, 4.1-4.5, 5; ReviewA_rzchk3
sec. 6 items 7, 9, 10, 12, 14, 15). Nothing is re-integrated here. Box CPU, one thread.
usage: rz_v3_head_v1.py OUTDIR SPEC [SPEC ...], SPEC = member=RAYS_PREFIX (prefix without .npz/.json)

Observables, as v2: at the fixed radius R_D = 4 r_s, L_T = ln(1/G_D), G_D = d tau_e/dT_D = c_D e^{-I}/D_e (the coordinate-clock ratio);
L_D = ln(1/g_D) = L_T + ln A_D, g_D = G_D/A_D, A_D = sqrt(N_D^2 - v_D^2) (the static receiver's frequency ratio). Each L_T maximum is printed
with the L_D of the SAME event; the independent maximum of L_D over the same population is printed in its own column (reviewer B 2.3).
Masks (reviewer B 4.1), per event: (e) emission radius r_e >= 1 a; (c) clock continuation valid: the clock has not entered the first site
(|r| < 1 a on its stored worldline, grid h) at or before the emission (comoving clocks only); (r) the ray's smallest |r| between emission
and arrival >= 1 a (rz_rays_v4m). 'resolved' = all that apply (A: e, c, r; B and C: e, r). v2's populations are the unmasked ones (A1 split
by (e) alone).
Certification: 'inside the solver-certified record interval' = arrival at or before the member's certified record (v2's 'certified').
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


def episodes(flag):
    out, s = [], None
    for k, f in enumerate(flag):
        if f and s is None:
            s = k
        if (not f) and s is not None:
            out.append((s, k - 1)); s = None
    if s is not None:
        out.append((s, len(flag) - 1))
    return out


def load(prefix):
    Z = dict(np.load(prefix + ".npz", allow_pickle=False)); S = json.load(open(prefix + ".json"))
    return Z, S


def member(name, prefix):
    Z, S = load(prefix)
    rs = S["rs"]; T = Z["T"]
    Tc_nom = (S["T_cert_rs"] if S["T_cert_rs"] is not None else -1.0) * rs
    k = int(np.argmin(np.abs(T - Tc_nom)))
    Tc = float(T[k]) if abs(T[k] - Tc_nom) < 1e-3 * rs else Tc_nom
    Tg = S["T_gate_rs"] * rs if (S.get("T_gate_rs") is not None and np.isfinite(S["T_gate_rs"])) else np.inf
    Tb = S["T_birth_rs"] * rs if S.get("T_birth_rs") is not None else np.inf
    rec = Z["receivers"]; i4 = int(np.argmin(np.abs(rec - 4 * rs)))
    key = None
    for kk in Z:
        if kk.startswith("B") and kk.endswith("_trec") and abs(float(kk[1:-len("_trec")].rstrip("_")) - rec[i4]) < 1e-6:
            key = kk[:-len("trec")]
    M = dict(name=name, S=S, Z=Z, rs=rs, T=T, Tc=Tc, Tc_nom=Tc_nom, Tg=Tg, Tb=Tb, i4=i4, key=key)
    # clock continuation: the first stored grid time each clock is inside the first site
    tg = Z["em_tg"]; RR = Z["em_R"]; ne = RR.shape[1]
    tsub = np.full(ne, np.inf)
    for e in range(ne):
        ok = (tg >= Z["em_t0"][e] - 1e-9) & np.isfinite(RR[:, e]) & (np.abs(RR[:, e]) < 1.0)
        if ok.any():
            tsub[e] = float(tg[np.argmax(ok)])
    M["tsub"] = tsub
    B = {}
    if key is not None:
        trec = Z[key + "trec"]
        lnG = np.log(Z[key + "fD"])[:, None] - Z[key + "psie"]
        ND, vD = Z[key + "ND_"], Z[key + "vD_"]
        with np.errstate(invalid="ignore"):
            lnA = 0.5 * np.log(ND ** 2 - vD ** 2)
        te = Z[key + "te"]; re = Z[key + "re"]
        mE = re >= 1.0; mC = te < tsub[None, :]
        rminB = Z.get(key + "rmin_e")
        if rminB is None:                                   # a v4 pass (no ray radius stored): the (r) mask is not applied, said so
            rminB = np.full(re.shape, np.nan); mR = np.ones(re.shape, bool)
        else:
            mR = rminB >= 1.0
        B = dict(trec=trec, lnG=lnG, lnA=lnA, te=te, taue=Z[key + "taue"], re=re, I=Z[key + "psie"], cD=Z[key + "fD"],
                 hover=Z[key + "hover_e"], K=Z[key + "K_e"], ht0=Z[key + "hover_t0"], ht1=Z[key + "hover_t1"], rmin=rminB,
                 mE=mE, mC=mC, mR=mR, res=mE & mC & mR)
    M["B"] = B
    with np.errstate(invalid="ignore", divide="ignore"):
        lzf = np.log(Z["F_fD"][:, i4]) + Z["F_psiD"][:, i4]
        mat = Z["L_kind"] >= 2
        lzf = np.where(mat, lzf - np.log(Z["L_De"]), lzf)
        lnAf = 0.5 * np.log(Z["F_ND"][:, i4] ** 2 - Z["F_vD"][:, i4] ** 2)
    mEf = Z["L_re"] >= 1.0
    M["rmask"] = "F_rmin_D" in Z
    rminF = Z["F_rmin_D"][:, i4] if M["rmask"] else np.full(len(mEf), np.nan)
    mRf = (rminF >= 1.0) if M["rmask"] else np.ones(len(mEf), bool)
    M["F"] = dict(lnG=lzf, lnA=lnAf, tD=Z["F_tD"][:, i4], te=Z["L_te"], re=Z["L_re"], kind=Z["L_kind"], delta=Z["L_delta"], krec=Z["L_k_rec"],
                  I=-Z["F_psiD"][:, i4], cD=Z["F_fD"][:, i4], De=np.where(mat, Z["L_De"], 1.0), taue=Z["L_tau_e"], psimax=Z["F_psimax"],
                  status=Z["F_status"], r_end=Z["F_r_end"], rmin=rminF, mE=mEf, mR=mRf, res=mEf & mRf)
    M["eps"] = episodes(np.isfinite(Z["RT_Rout"]))
    M["rtag"] = "" if M["rmask"] else " (e, c only: ray mask unavailable)"
    return M


def pg_share(te, tD, Tg):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((tD - np.maximum(te, Tg)) / (tD - te), 0.0, 1.0)


res = {}
H1 = ["**Table H1. The headline, renamed and split** (rz_v4_head_v3.py on the exact-release passes rz_rays_v4n and rz_rays_v4nc: v3's columns, with the later-released clocks (A2) released at T_rel exactly, Table A2Xv2 printing the change; P64_s4_ext200 is the exception named in its row). At R_D = 4 r_s, over "
      "arrivals inside the solver-certified record interval. In each 'a / b' cell, a is the largest L_T = ln(1/G_D) of the class (G_D = dτ_e/dT_D, "
      "the coordinate-clock ratio) and b is L_D = ln(1/g_D) = L_T + ln A_D of THE SAME EVENT (g_D the static receiver's frequency ratio, "
      "A_D = √(N_D² − v_D²) at the arrival); the 'max L_D' columns are L_D maximised independently over the same population. 'resolved' applies "
      "the masks (e) r_e ≥ 1 a, (c) the clock has not entered the first site by the emission, (r) the ray's smallest |r| ≥ 1 a.\n",
      "| member | inside the solver-certified record interval to T/r_s | A1, r_e ≥ 1 a: L_T / L_D same event; max L_D | A1 resolved: L_T / L_D; max L_D | A1 sub-site: L_T / L_D | A2: L_T / L_D; max L_D | A2 resolved: L_T / L_D; max L_D | B: L_T / L_D; max L_D | B resolved: L_T / L_D; max L_D | C: L_T / L_D; max L_D |",
      "|---|---|---|---|---|---|---|---|---|---|"]
H2 = ["\n**Table H2. The headline events** (as v2, with the masks): per member and class, the event that sets the class's largest L_T in Table H1 "
      "(unmasked classes of v2 and the resolved ones): clock (A: index; B: offset δ in a; C: quantile), emission T_e, proper time τ_e (A only), "
      "radius r_e, arrival T_D, I = ∫∂_r c dT along the ray, c_D, ln A_D, D_e, the ray's smallest |r| (a), and the masks: post-gate share of the "
      "flight (one-term gate), pre-birth emission, sub-site emission (e), clock inside the first site before the emission (c).\n",
      "| member | class | clock | T_e/r_s | τ_e/r_s | r_e (a) | T_D/r_s | I | c_D | ln A_D | D_e | ray's smallest abs r (a) | post-gate share (one-term gate) | pre-birth | sub-site emission | clock inside the first site before it |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
H3 = ["\n**Table H3. Clock domains** (as v2; reviewer B rzchk3 secs. 4.5 and 5): per comoving clock, from its own worldline and the backward map: release; "
      "first time inside the first site (r < 1 a, † past the certified end); first time inside the matter root (†); the last emission received inside "
      "the solver-certified record interval on the backward map (T_e, τ_e, r_e there); the last forward launch received; the received share of the "
      "clock's proper time; two energy-transfer DIAGNOSTICS on the backward map: ∫G_D dτ_e/Δτ (weighted by the coordinate-clock ratio; v2's) "
      "and ∫g_D dτ_e/Δτ (weighted by the static receiver's ratio). Neither is a transmission probability (no transmission treatment; a "
      "time-dependent background can supply energy). Two domains, each with Δτ = τ(t_end) − τ(T_rel) read on the clock's stored worldline "
      "samples (step 0.5 a): UNMASKED (v2's and v3's: every received event, sub-site emissions included; t_end the last stored sample at or "
      "before min(the clock's last finite sample, the certified end)) and RESOLVED (v4: only events passing the masks (e), (c), (r), summed "
      "over consecutive arrivals both resolved and 1 a apart; t_end the last stored sample strictly before the sample at which the clock is "
      "first inside the first site, one sample (0.5 a) before the printed first-entry time, or at or before the certified end if that is "
      "earlier; printed as the domain end).\n",
      "| member | clock | release T/r_s | r_0/r_s | inside the first site from T/r_s | inside the matter root from T/r_s | last received emission (backward): T_e/r_s, τ_e/r_s, r_e (a) | last received forward launch T_e/r_s | unmasked: received proper-time share / ∫G_D dτ_e/Δτ / ∫g_D dτ_e/Δτ | resolved: domain end T/r_s; share / ∫G_D dτ_e/Δτ / ∫g_D dτ_e/Δτ |",
      "|---|---|---|---|---|---|---|---|---|---|"]
H4 = ["\n**Table H4. Blueshifted arrivals and tick-map flags** (reviewer A rzchk 2(d), 7.4; reviewer B rzchk3 sec. 5): per member, over arrivals inside the "
      "solver-certified record interval at 4 r_s, per class: arrivals blueshifted in the coordinate-clock sense (L_T < 0, G_D > 1) and for the static "
      "receiver (L_D < 0, g_D > 1), with the largest −L_T. Tick-map flags on the backward map (A1, A2): monotonicity reversals (consecutive "
      "arrivals 1 a apart whose τ_e decreases); bunched pairs (consecutive one-sided slopes dτ_e/dT_D differing by more than a factor 2, the larger "
      "above 1). These are flags; no cause (caustic or otherwise) is assigned. Clocks with either H3 diagnostic above 1, on the unmasked and on "
      "the resolved domain.\n",
      "| member | A1: blueshifted L_T<0 / L_D<0, largest −L_T | A2: same | B: same | C: same | backward: monotonicity reversals / bunched pairs | clocks with ∫G_D dτ_e/Δτ > 1 / ∫g_D dτ_e/Δτ > 1: unmasked; resolved |",
      "|---|---|---|---|---|---|---|"]
H5 = ["\n**Table H5. Post-gate shares on the flights inside the solver-certified record interval** (as v2): the share of each flight after this "
      "pass's ONE-TERM gate T_gate (σ max|∂_r v̄| > 0.1 on the back-averaged site fields); the six-term and ‖K‖_op gates are not applied here.\n",
      "| member | T_b/r_s | T_gate/r_s (one-term) | one-term gate before the first root? | A1 median / max | A2 median / max | B median / max | C median / max |",
      "|---|---|---|---|---|---|---|---|"]
H6 = ["\n**Table H6. Largest received L_T over all classes, with the class named** (as v2; reviewer B rzchk3 sec. 2.1): inside the solver-certified "
      "record interval, unmasked and resolved, with the L_D of the same event and the independent L_D maximum; and over the whole stored window.\n",
      "| member | inside: max L_T (class) / L_D same event | inside, resolved: max L_T (class) / L_D same event | inside: max L_D (class) | inside, resolved: max L_D (class) | whole window: max L_T (class) |",
      "|---|---|---|---|---|---|"]
H7 = ["\n**Table H7. The dust emitters' validity** (as v2).\n",
      "| member | quantile | launches | timelike | rejected | median abs Δ ln D_e | max abs Δ ln D_e |",
      "|---|---|---|---|---|---|---|"]
H8 = ["\n**Table H8. ln A_D and the transfer beyond 4 r_s** (as v2, with the breakdown behind reviewer A rzchk3 item 9): ln A_D median over arrivals "
      "inside the solver-certified record interval, backward (A) / forward (B, C); then, per receiver beyond 4 r_s and per launch class, on "
      "forward launches received at both 4 r_s and that receiver inside the interval: count and largest abs(ln G_D(r) − ln G_D(4 r_s)). v2's "
      "column is the outermost receiver over B and C; v1's exterior check took every forward front (A, B, C) reaching every receiver ≥ 4 r_s.\n",
      "| member | ln A_D median: backward A / forward B, C | receiver r/r_s | A (forward clock launches): n, max abs | B: n, max abs | C: n, max abs | B and C (v2's column at the outermost): n, max abs |",
      "|---|---|---|---|---|---|---|"]
I1 = ["\n**Table I1. Received-sample minima over declared selections** (reviewer B rzchk p. 12; rzchk3 secs. 4.2-4.4; reviewer A rzchk3 item 10). For each "
      "selection, the SMALLEST L_T and L_D over the received events in it inside the solver-certified record interval, and counts. These are "
      "sample minima over received events, not bounds over all signals, and no signal band is specified. (a) a clock's whole received worldline; "
      "(b) a clock's last Δτ of proper time before its last eligible received emission, the endpoint taken FOR EACH POPULATION SEPARATELY (an "
      "outcome-relative selection with no censored count; the masks move the endpoint, so the masked and unmasked (b) rows are different "
      "selections, not one selection with fewer events: P64_s2's resolved A1 row ends earlier on its clocks; the endpoints per clock are in Table I1e); (c) root-neighbourhood launches B within δ ≤ 1, 4, 16 a outside a standing outer root, emitted "
      "inside the interval, with the not-received launches split: received after the interval's end (inside the stored window) / not received by "
      "the last stored record. A1 and A2 apart, unmasked and resolved; for (a) the received share of the clocks' proper time over the population's own "
      "domain (unmasked: Table H3's unmasked share; resolved: its resolved share).\n",
      "| member | population | (a) whole worldline: min L_T / min L_D (n; received proper-time share over the population's domain) | (b) last 0.01 r_s | (b) last 0.1 r_s | (b) last 1 r_s |",
      "|---|---|---|---|---|---|"]
I1c = ["\n**Table I1c. B launches near a standing root** (selection (c)): min L_T / min L_D over the received; counts received inside the interval / "
       "received after its end / not received by the last stored record; and the launches emitted at the certified end's record itself (present only where "
       "the stored records run past it, since no B launch is made at a history's last record).\n",
       "| member | population | δ ≤ 1 a: min L_T / min L_D (received inside, received after the end, not received by the last stored record) | δ ≤ 4 a | δ ≤ 16 a | launches at the certified end's record (δ ≤ 1, 4, 16 a) |",
       "|---|---|---|---|---|---|"]
for name, prefix in SPECS:
    if not os.path.exists(prefix + ".npz"):
        continue
    M = member(name, prefix)
    Z, S, rs, Tc, Tg, Tb = M["Z"], M["S"], M["rs"], M["Tc"], M["Tg"], M["Tb"]
    B, Fw = M["B"], M["F"]
    r = dict(Tc=Tc / rs, Tg=Tg / rs if np.isfinite(Tg) else None, Tb=Tb / rs if np.isfinite(Tb) else None, rmask=bool(M["rmask"]), prefix=prefix)
    nA1 = int(np.sum(Z["em_t0"] == 0)); ne = len(Z["em_t0"])
    a1 = np.arange(ne) < nA1
    cells, evs = {}, []
    if B:
        cert = (B["trec"] <= Tc + 1e-9)[:, None] & np.isfinite(B["lnG"])
        LG = -B["lnG"]; Lg = LG + B["lnA"][:, None]
        sub = ~B["mE"]
        for cls, cm, subsel in (("A1 r_e≥1a", a1, ~sub), ("A1 resolved", a1, B["res"]), ("A1 sub-site", a1, sub), ("A1 all", a1, np.ones_like(sub)),
                                ("A2", ~a1, np.ones_like(sub)), ("A2 resolved", ~a1, B["res"])):
            m = cert & cm[None, :] & subsel
            if not m.any():
                cells[cls] = None
                continue
            i, e = np.unravel_index(np.argmax(np.where(m, LG, -np.inf)), LG.shape)
            jg = np.unravel_index(np.argmax(np.where(m, Lg, -np.inf)), Lg.shape)
            ev = dict(cls=cls, clock=int(e), Te=float(B["te"][i, e] / rs), taue=float(B["taue"][i, e] / rs), re=float(B["re"][i, e]), TD=float(B["trec"][i] / rs),
                      I=float(B["I"][i, e]), cD=float(B["cD"][i]), lnA=float(B["lnA"][i]), De=1.0, LG=float(LG[i, e]), Lg=float(Lg[i, e]), Lg_max=float(Lg[jg]),
                      Lg_max_clock=int(jg[1]), Lg_max_TD=float(B["trec"][jg[0]] / rs), rmin=float(B["rmin"][i, e]),
                      pg=float(pg_share(B["te"][i, e], B["trec"][i], Tg)), prebirth=bool(B["te"][i, e] < Tb), subsite=bool(B["re"][i, e] < 1.0),
                      clocksub=bool(not B["mC"][i, e]))
            cells[cls] = ev
            if cls != "A1 all":
                evs.append(ev)
    certf = np.isfinite(Fw["lnG"]) & (Fw["tD"] <= Tc + 1e-9)
    for cls, km in (("B", Fw["kind"] == 1), ("B resolved", (Fw["kind"] == 1) & Fw["res"]), ("C", Fw["kind"] >= 2), ("C resolved", (Fw["kind"] >= 2) & Fw["res"])):
        m = certf & km
        if not m.any():
            cells[cls] = None
            continue
        LG = -Fw["lnG"]; Lg = LG + Fw["lnA"]
        i = int(np.argmax(np.where(m, LG, -np.inf))); jg = int(np.argmax(np.where(m, Lg, -np.inf)))
        ev = dict(cls=cls, clock=("δ %s" % fmt(Fw["delta"][i], 3)) if cls.startswith("B") else ("R_%d" % (50 if Fw["kind"][i] == 2 else 90)), Te=float(Fw["te"][i] / rs),
                  taue=None, re=float(Fw["re"][i]), TD=float(Fw["tD"][i] / rs), I=float(Fw["I"][i]), cD=float(Fw["cD"][i]), lnA=float(Fw["lnA"][i]),
                  De=float(Fw["De"][i]), LG=float(LG[i]), Lg=float(Lg[i]), Lg_max=float(Lg[jg]), Lg_max_TD=float(Fw["tD"][jg] / rs), Lg_max_Te=float(Fw["te"][jg] / rs),
                  rmin=float(Fw["rmin"][i]), pg=float(pg_share(Fw["te"][i], Fw["tD"][i], Tg)), prebirth=bool(Fw["te"][i] < Tb), subsite=bool(Fw["re"][i] < 1.0),
                  clocksub=False)
        cells[cls] = ev
        evs.append(ev)
    r["cells"] = cells
    c = lambda k: ("%s / %s; %s" % (fmt(cells[k]["LG"]), fmt(cells[k]["Lg"]), fmt(cells[k]["Lg_max"]))) if cells.get(k) else "-"
    c2 = lambda k: ("%s / %s" % (fmt(cells[k]["LG"]), fmt(cells[k]["Lg"]))) if cells.get(k) else "-"
    H1.append("| %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (name, "" if M["rmask"] else " (records not locatable: its v1 pass, whose A2 clocks are released one step early; ray mask (r) unavailable, so its 'resolved' columns apply the emission and clock filters only; its records equal P64_s4's through T/r_s 40; its whole-window value is unreproducible)", fmt(Tc / rs, 4), c("A1 r_e≥1a"), c("A1 resolved"), c2("A1 sub-site"), c("A2"), c("A2 resolved"),
                                                               c("B"), c("B resolved"), c("C")))
    for ev in evs:
        H2.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, ev["cls"] + (M["rtag"] if "resolved" in ev["cls"] else ""), ev["clock"], fmt(ev["Te"], 4), fmt(ev["taue"], 4), fmt(ev["re"], 3), fmt(ev["TD"], 4), fmt(ev["I"]), fmt(ev["cD"], 4), fmt(ev["lnA"], 4),
            fmt(ev["De"], 4), fmt(ev["rmin"], 3), fmt(ev["pg"], 3), "yes" if ev["prebirth"] else "no", "yes" if ev["subsite"] else "no", "yes" if ev["clocksub"] else "no"))
    # ---- H3 clock domains
    tg = Z["em_tg"]; RR = Z["em_R"]; TT = Z["em_tau"]; T = Z["T"]
    jg_ = np.clip(np.searchsorted(T, tg, side="right") - 1, 0, len(T) - 2); wg = (tg - T[jg_]) / (T[jg_ + 1] - T[jg_])
    Rt_g = (1 - wg) * Z["RT_Rout"][jg_] + wg * Z["RT_Rout"][jg_ + 1]
    clk = []
    for e in range(ne):
        t0 = Z["em_t0"][e]
        valid = (tg >= t0 - 1e-9) & np.isfinite(RR[:, e])
        ss = np.where(valid & (np.abs(RR[:, e]) < 1.0))[0]
        ins = np.where(valid & np.isfinite(Rt_g) & (RR[:, e] < Rt_g))[0]
        last_b = (np.nan, np.nan, np.nan); share = ratio = ratio_g = np.nan
        if B:
            col = B["lnG"][:, e]
            okc = np.isfinite(col) & (B["trec"] <= Tc + 1e-9)
            if okc.any():
                ii = np.where(okc)[0]
                j = ii[np.argmax(B["taue"][ii, e])]
                last_b = (B["te"][j, e] / rs, B["taue"][j, e] / rs, B["re"][j, e])
                vend = np.where(valid & (tg <= Tc + 1e-9))[0]
                tau_tot = TT[vend[-1], e] - TT[vend[0], e] if len(vend) > 1 else np.nan
                tr = B["trec"][ii]; ta = B["taue"][ii, e]; G = np.exp(col[ii]); g = np.exp(col[ii] - B["lnA"][ii])
                o = np.argsort(tr); tr, ta, G, g = tr[o], ta[o], G[o], g[o]
                d_ta = np.diff(ta)
                if np.isfinite(tau_tot) and tau_tot > 0:
                    share = float(np.sum(np.abs(d_ta)) / tau_tot)
                    ratio = float(np.sum(0.5 * (G[1:] + G[:-1]) * np.abs(d_ta)) / tau_tot)
                    ratio_g = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.abs(d_ta)) / tau_tot)
        km = (Fw["kind"] == 0) & (Z["L_eid"] == e) & certf
        lastf = float(np.max(Fw["te"][km]) / rs) if km.any() else np.nan
        tsb = float(tg[ss[0]]) if len(ss) else None; tin = float(tg[ins[0]]) if len(ins) else None
        # v4: the resolved domain
        share_r = ratio_r = ratio_gr = np.nan; tend_r = np.nan
        if B:
            tlim = min(tsb if tsb is not None else np.inf, Tc)
            vr = np.where(valid & (tg <= tlim + 1e-9) & (tg < (tsb if tsb is not None else np.inf)))[0]
            if len(vr) > 1:
                tend_r = float(tg[vr[-1]])
                tau_r = TT[vr[-1], e] - TT[vr[0], e]
                okr = np.isfinite(B["lnG"][:, e]) & (B["trec"] <= Tc + 1e-9) & B["res"][:, e]
                ir = np.where(okr)[0]
                if len(ir) > 1 and tau_r > 0:
                    o_ = np.argsort(B["trec"][ir]); ir = ir[o_]
                    adj = np.diff(B["trec"][ir]) < 1.01 * np.median(np.diff(B["trec"]))
                    dta = np.abs(np.diff(B["taue"][ir, e]))[adj]
                    Gr = np.exp(B["lnG"][ir, e]); gr = np.exp(B["lnG"][ir, e] - B["lnA"][ir])
                    share_r = float(np.sum(dta) / tau_r)
                    ratio_r = float(np.sum((0.5 * (Gr[1:] + Gr[:-1]))[adj] * dta) / tau_r)
                    ratio_gr = float(np.sum((0.5 * (gr[1:] + gr[:-1]))[adj] * dta) / tau_r)
        clk.append(dict(e=e, t0=t0 / rs, sub=tsb / rs if tsb is not None else None, ins=tin / rs if tin is not None else None,
                        sub_cert=bool(tsb is not None and tsb <= Tc + 1e-9), ins_cert=bool(tin is not None and tin <= Tc + 1e-9),
                        last_b=[float(x) for x in last_b], lastf=lastf, share=share, ratio=ratio, ratio_g=ratio_g, share_r=share_r, ratio_r=ratio_r,
                        ratio_gr=ratio_gr, tend_r=tend_r / rs if np.isfinite(tend_r) else None))
        dag = lambda t, cflag: "" if (t is None or cflag) else "†"
        H3.append("| %s%s | %s%d | %s | %s | %s%s | %s%s | %s, %s, %s | %s | %s / %s / %s | %s; %s / %s / %s |" % (
            name, M["rtag"], "A1." if t0 == 0 else "A2.", e, fmt(t0 / rs, 2), fmt(Z["r0s"][e] / rs, 2), fmt(clk[-1]["sub"], 4), dag(tsb, clk[-1]["sub_cert"]),
            fmt(clk[-1]["ins"], 4), dag(tin, clk[-1]["ins_cert"]), fmt(last_b[0], 4), fmt(last_b[1], 4), fmt(last_b[2], 3), fmt(lastf, 4), fmt(share, 3),
            fmt(ratio, 3), fmt(ratio_g, 3), fmt(clk[-1]["tend_r"], 4), fmt(share_r, 3), fmt(ratio_r, 3), fmt(ratio_gr, 3)))
    r["clocks"] = clk
    # ---- H4 blueshifts and flags
    def blue(cls_mask_b=None, fwd_mask=None):
        if cls_mask_b is not None:
            m = (B["trec"] <= Tc + 1e-9)[:, None] & np.isfinite(B["lnG"]) & cls_mask_b[None, :]
            v = B["lnG"][m]; vg = (B["lnG"] - B["lnA"][:, None])[m]
        else:
            m = certf & fwd_mask
            v = Fw["lnG"][m]; vg = (Fw["lnG"] - Fw["lnA"])[m]
        return (int(np.sum(v > 0)), int(np.sum(vg > 0)), float(np.max(v)) if len(v) else np.nan)
    bl = {}
    if B:
        bl["A1"] = blue(a1); bl["A2"] = blue(~a1)
        dec = bun = 0
        for e in range(ne):
            ok = np.isfinite(B["lnG"][:, e]) & (B["trec"] <= Tc + 1e-9)
            ii = np.where(ok)[0]
            if len(ii) < 3:
                continue
            tr = B["trec"][ii]; ta = B["taue"][ii, e]
            cont = np.diff(tr) < 1.01 * np.median(np.diff(B["trec"]))
            dta = np.diff(ta) / np.diff(tr)
            dec += int(np.sum((dta < 0) & cont))
            sl = dta[:-1]; sr = dta[1:]; cc = cont[:-1] & cont[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                big = np.maximum(np.abs(sl), np.abs(sr)); small = np.minimum(np.abs(sl), np.abs(sr))
                bun += int(np.sum(cc & (big > 1.0) & (big > 2.0 * small)))
        bl["fold"] = (dec, bun)
    bl["B"] = blue(fwd_mask=Fw["kind"] == 1); bl["C"] = blue(fwd_mask=Fw["kind"] >= 2)
    nr = sum(1 for c_ in clk if np.isfinite(c_["ratio"]) and c_["ratio"] > 1); nrg = sum(1 for c_ in clk if np.isfinite(c_["ratio_g"]) and c_["ratio_g"] > 1)
    nrr = sum(1 for c_ in clk if np.isfinite(c_["ratio_r"]) and c_["ratio_r"] > 1); nrgr = sum(1 for c_ in clk if np.isfinite(c_["ratio_gr"]) and c_["ratio_gr"] > 1)
    r["blue"] = bl; r["n_ratio_gt1"] = nr; r["n_ratio_g_gt1"] = nrg; r["n_ratio_gt1_res"] = nrr; r["n_ratio_g_gt1_res"] = nrgr
    H4.append("| %s%s | %s | %s | %s | %s | %s | %d / %d; %d / %d |" % (name, M["rtag"], *[("%d / %d, %s" % (bl[k][0], bl[k][1], fmt(bl[k][2]))) if k in bl else "-" for k in ("A1", "A2", "B", "C")],
                                                         ("%d / %d" % bl["fold"]) if "fold" in bl else "-", nr, nrg, nrr, nrgr))
    # ---- H5
    sh = {}
    if B:
        for cls, cm in (("A1", a1), ("A2", ~a1)):
            m = (B["trec"] <= Tc + 1e-9)[:, None] & np.isfinite(B["lnG"]) & cm[None, :]
            ii, ee = np.where(m)
            v = pg_share(B["te"][ii, ee], B["trec"][ii], Tg)
            sh[cls] = (float(np.median(v)), float(np.max(v))) if len(v) else (np.nan, np.nan)
    for cls, km in (("B", Fw["kind"] == 1), ("C", Fw["kind"] >= 2)):
        m = certf & km
        v = pg_share(Fw["te"][m], Fw["tD"][m], Tg)
        sh[cls] = (float(np.median(v)), float(np.max(v))) if m.any() else (np.nan, np.nan)
    r["pg"] = sh
    H5.append("| %s | %s | %s | %s | %s |" % (name, fmt(Tb / rs if np.isfinite(Tb) else None, 4), fmt(Tg / rs if np.isfinite(Tg) else None, 4),
                                               "-" if not (np.isfinite(Tb) and np.isfinite(Tg)) else ("yes" if Tg < Tb else "no"),
                                               " | ".join("%s / %s" % (fmt(sh[k][0], 3), fmt(sh[k][1], 3)) if k in sh else "-" for k in ("A1", "A2", "B", "C"))))
    # ---- H6
    def allmax(lim, resolved, use_g):
        best = (-np.inf, None, None, None)
        if B:
            m = (B["trec"] <= lim + 1e-9)[:, None] & np.isfinite(B["lnG"])
            if resolved:
                m = m & B["res"]
            LG = -B["lnG"]; Lg = LG + B["lnA"][:, None]
            X = Lg if use_g else LG
            for cls, cm in (("A1", a1), ("A2", ~a1)):
                mm = m & cm[None, :]
                if mm.any():
                    q = np.unravel_index(np.argmax(np.where(mm, X, -np.inf)), X.shape)
                    if X[q] > best[0]:
                        best = (float(X[q]), cls, float(Lg[q]), float(B["trec"][q[0]]))
        for cls, km in (("B", Fw["kind"] == 1), ("C", Fw["kind"] >= 2)):
            mm = np.isfinite(Fw["lnG"]) & (Fw["tD"] <= lim + 1e-9) & km
            if resolved:
                mm = mm & Fw["res"]
            LG = -Fw["lnG"]; Lg = LG + Fw["lnA"]
            X = Lg if use_g else LG
            if mm.any():
                q = int(np.argmax(np.where(mm, X, -np.inf)))
                if X[q] > best[0]:
                    best = (float(X[q]), cls, float(Lg[q]), float(Fw["tD"][q]))
        return best
    a_c = allmax(Tc, False, False); a_cr = allmax(Tc, True, False); g_c = allmax(Tc, False, True); g_cr = allmax(Tc, True, True); a_w = allmax(np.inf, False, False)
    r["allmax"] = dict(cert=a_c, cert_res=a_cr, cert_g=g_c, cert_g_res=g_cr, whole=a_w)
    past = a_w[3] is not None and a_w[3] > Tc + 1e-9
    H6.append("| %s%s | %s (%s) / %s | %s (%s) / %s | %s (%s) | %s (%s) | %s (%s)%s |" % (name, M["rtag"], fmt(a_c[0]), a_c[1], fmt(a_c[2]), fmt(a_cr[0]), a_cr[1], fmt(a_cr[2]),
                                                                       fmt(g_c[0]), g_c[1], fmt(g_cr[0]), g_cr[1], fmt(a_w[0]), a_w[1],
                                                                       (" †, arriving at T/r_s %s past the admitted interval%s" % (fmt(a_w[3] / rs, 3), "; unreproducible: records not locatable" if not M["rmask"] else "")) if past else ""))
    # ---- H7 dust validity (as v2)
    dr_ = Z["dust_R"]
    for qi, qn in ((0, "R_50"), (1, "R_90")):
        km = Fw["kind"] == 2 + qi
        n = int(km.sum())
        if n == 0:
            continue
        Vc = np.gradient(dr_[:, qi], T)
        Vf = np.concatenate([np.diff(dr_[:, qi]) / np.diff(T), [np.nan]])
        kk = Fw["krec"][km].astype(int)
        Ne = Z["L_Ne"][km]; ve = Z["L_ve"][km]
        with np.errstate(invalid="ignore", divide="ignore"):
            bc = (Vc[kk] - ve) / Ne; bf = (Vf[kk] - ve) / Ne
            okc = np.abs(bc) < 1; okf = np.abs(bf) < 1
            lc = 0.5 * np.log((1 - bc) / (1 + bc)); lf = 0.5 * np.log((1 - bf) / (1 + bf))
        both = okc & okf & np.isfinite(lc) & np.isfinite(lf)
        d = np.abs(lc - lf)[both]
        H7.append("| %s | %s | %d | %d | %d | %s | %s |" % (name, qn, n, int(okc.sum()), int(n - okc.sum()), fmt(np.median(d)) if len(d) else "-", fmt(np.max(d)) if len(d) else "-"))
    # ---- H8 with the breakdown
    lnA_b = float(np.nanmedian(B["lnA"][B["trec"] <= Tc + 1e-9])) if B else np.nan
    lnA_f = float(np.nanmedian(Fw["lnA"][certf & (Fw["kind"] >= 1)])) if (certf & (Fw["kind"] >= 1)).any() else np.nan
    rec = Z["receivers"]
    tr8 = {}
    for io in range(len(rec)):
        if rec[io] <= 4 * rs + 1e-6:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            lzo = np.log(Z["F_fD"][:, io]) + Z["F_psiD"][:, io]
            lzo = np.where(Fw["kind"] >= 2, lzo - np.log(Z["L_De"]), lzo)
        both = certf & np.isfinite(lzo) & (Z["F_tD"][:, io] <= Tc + 1e-9)
        row = {}
        for cls, km in (("A", Fw["kind"] == 0), ("B", Fw["kind"] == 1), ("C", Fw["kind"] >= 2), ("BC", Fw["kind"] >= 1)):
            dd = (lzo - Fw["lnG"])[both & km]
            row[cls] = (int(len(dd)), float(np.max(np.abs(dd))) if len(dd) else None, float(np.median(dd)) if len(dd) else None)
        tr8["%.4f" % (rec[io] / rs)] = row
        H8.append("| %s | %s / %s | %s%s | %d, %s | %d, %s | %d, %s | %d, %s |" % (name, fmt(lnA_b, 4), fmt(lnA_f, 4), fmt(rec[io] / rs, 3), " (r_c)" if io == S.get("i_rc") else "",
                                                                            row["A"][0], fmt(row["A"][1]), row["B"][0], fmt(row["B"][1]), row["C"][0], fmt(row["C"][1]), row["BC"][0], fmt(row["BC"][1])))
    io = int(np.argmax(rec))
    r["transfer"] = dict(outer=float(rec[io] / rs), n=tr8.get("%.4f" % (rec[io] / rs), {}).get("BC", (0,))[0],
                         med=tr8.get("%.4f" % (rec[io] / rs), {}).get("BC", (0, None, None))[2], mx=tr8.get("%.4f" % (rec[io] / rs), {}).get("BC", (0, None))[1],
                         lnA_b=lnA_b, lnA_f=lnA_f, by_receiver=tr8)
    # ---- I1: (a), (b) per A1, A2 apart, unmasked and resolved
    iv = {}
    if B:
        cert = (B["trec"] <= Tc + 1e-9)[:, None] & np.isfinite(B["lnG"])
        LG = -B["lnG"]; Lg = LG + B["lnA"][:, None]
        for pop, pm in (("A1", a1[None, :] & np.ones_like(cert)), ("A1 resolved", a1[None, :] & B["res"]), ("A2", (~a1)[None, :] & np.ones_like(cert)),
                        ("A2 resolved", (~a1)[None, :] & B["res"]), ("A1+A2 (v2)", np.ones_like(cert))):
            cm = cert & pm
            key_ = "share_r" if "resolved" in pop else "share"
            sh_ = [c_[key_] for c_ in clk if np.isfinite(c_[key_]) and ((c_["t0"] == 0) == pop.startswith("A1") or pop.startswith("A1+A2"))]
            endp = {}
            d = dict(a=((float(np.min(LG[cm])), float(np.min(Lg[cm])), int(cm.sum())) if cm.any() else None),
                     share=((float(np.min(sh_)), float(np.max(sh_))) if sh_ else None))
            for dt_ in (0.01, 0.1, 1.0):
                mm = np.zeros_like(cert)
                for e in range(ne):
                    ce = cm[:, e]
                    if not ce.any():
                        continue
                    tl = np.max(B["taue"][ce, e])
                    mm[:, e] = ce & (B["taue"][:, e] >= tl - dt_ * rs)
                    if dt_ == 0.01:
                        jl = np.where(ce)[0][np.argmax(B["taue"][ce, e])]
                        endp["%s%d" % ("A1." if e < nA1 else "A2.", e)] = [float(tl / rs), float(B["te"][jl, e] / rs), float(B["re"][jl, e])]
                d["b%g" % dt_] = (float(np.min(LG[mm])), float(np.min(Lg[mm])), int(mm.sum())) if mm.any() else None
            d["endpoints"] = endp
            iv[pop] = d
            fi = lambda t: ("%s / %s (%d)" % (fmt(t[0]), fmt(t[1]), t[2])) if t else "-"
            I1.append("| %s%s | %s | %s | %s | %s | %s |" % (name, M["rtag"] if "resolved" in pop else "", pop, (fi(d["a"])[:-1] + "; %s to %s)" % (fmt(d["share"][0], 3), fmt(d["share"][1], 3))) if (d["a"] and d["share"]) else fi(d["a"]),
                                                       fi(d["b0.01"]), fi(d["b0.1"]), fi(d["b1"])))
    # ---- I1c: B near a standing root, with the censoring split
    Rout = Z["RT_Rout"]
    kr = np.clip(Fw["krec"].astype(int), 0, len(Rout) - 1)
    LGf = -Fw["lnG"]; Lgf = LGf + Fw["lnA"]
    after = np.isfinite(Fw["lnG"]) & (Fw["tD"] > Tc + 1e-9)
    never = ~np.isfinite(Fw["tD"])
    for pop, pm in (("B", np.ones(len(Fw["kind"]), bool)), ("B resolved", Fw["res"])):
        cc = {}
        for dl in (1.0, 4.0, 16.0):
            sel = (Fw["kind"] == 1) & (Fw["delta"] > 0) & (Fw["delta"] <= dl) & np.isfinite(Rout[kr]) & (Fw["te"] <= Tc + 1e-9)
            rcv = sel & certf & pm
            selp = sel & (pm | ~certf)            # the resolved mask is a property of a received ray; unreceived launches are counted in full
            cc["c%g" % dl] = ((float(np.min(LGf[rcv])), float(np.min(Lgf[rcv]))) if rcv.any() else (np.nan, np.nan)) + \
                (int(rcv.sum()), int((selp & after).sum()), int((selp & never).sum()), int((sel & (np.abs(Fw["te"] - Tc) < 1e-6)).sum()))
        iv["c " + pop] = cc
        I1c.append("| %s | %s | %s | %s | %s | %s |" % (name + (M["rtag"] if "resolved" in pop else ""), pop, *["%s / %s (%d, %d, %d)" % (fmt(cc["c%g" % d][0]), fmt(cc["c%g" % d][1]), cc["c%g" % d][2], cc["c%g" % d][3], cc["c%g" % d][4])
                                                                   for d in (1.0, 4.0, 16.0)], ", ".join(str(cc["c%g" % d][5]) for d in (1.0, 4.0, 16.0))))
    r["intervals"] = iv
    res[name] = r
# ---- across members
rt_ = lambda m, pop: " (e, c only: ray mask unavailable)" if ("resolved" in pop and not res[m]["rmask"]) else ""
I1.append("\n**Across members**, per selection and population: the smallest of the members' minima (member), over the members printed above.\n")
I1.append("| selection | population | smallest L_T (member) | smallest L_D (member) |")
I1.append("|---|---|---|---|")
for ik, lab in (("a", "(a) whole received worldline"), ("b0.01", "(b) last 0.01 r_s"), ("b0.1", "(b) last 0.1 r_s"), ("b1", "(b) last 1 r_s")):
    for pop in ("A1", "A1 resolved", "A2", "A2 resolved", "A1+A2 (v2)"):
        vals = [(res[m]["intervals"][pop][ik][0], res[m]["intervals"][pop][ik][1], m) for m in res if res[m]["intervals"].get(pop) and res[m]["intervals"][pop].get(ik)]
        if not vals:
            I1.append("| %s | %s | - | - |" % (lab, pop))
            continue
        a_ = min(vals, key=lambda x: x[0]); b_ = min(vals, key=lambda x: x[1])
        I1.append("| %s | %s | %s (%s) | %s (%s) |" % (lab, pop, fmt(a_[0]), a_[2] + rt_(a_[2], pop), fmt(b_[1]), b_[2] + rt_(b_[2], pop)))
for dl in (1.0, 4.0, 16.0):
    for pop in ("B", "B resolved"):
        vals = [(res[m]["intervals"]["c " + pop]["c%g" % dl][0], res[m]["intervals"]["c " + pop]["c%g" % dl][1], m) for m in res
                if res[m]["intervals"].get("c " + pop) and np.isfinite(res[m]["intervals"]["c " + pop]["c%g" % dl][0])]
        if not vals:
            I1.append("| (c) δ ≤ %g a | %s | - | - |" % (dl, pop))
            continue
        a_ = min(vals, key=lambda x: x[0]); b_ = min(vals, key=lambda x: x[1])
        I1.append("| (c) δ ≤ %g a | %s | %s (%s) | %s (%s) |" % (dl, pop, fmt(a_[0]), a_[2] + rt_(a_[2], pop), fmt(b_[1]), b_[2] + rt_(b_[2], pop)))
I1e = ["\n**Table I1e. The (b) selections' endpoints per clock** (ReviewB_rzchk4 sec. 3.4, ReviewB_rzchk5): on every member with comoving clocks, each clock's last "
       "eligible received emission inside the solver-certified record interval, per population (τ_e/r_s, T_e/r_s, r_e in a); the masked and "
       "unmasked selections end at different emissions wherever these differ.\n",
       "| member | clock | A1 or A2: τ_e/r_s, T_e/r_s, r_e (a) | resolved: τ_e/r_s, T_e/r_s, r_e (a) |",
       "|---|---|---|---|"]
for m in res:
    ivm = res[m]["intervals"]
    for base in ("A1", "A2"):
        e0 = ivm.get(base, {}).get("endpoints", {}); e1 = ivm.get(base + " resolved", {}).get("endpoints", {})
        for ck in sorted(set(e0) | set(e1), key=lambda x: int(x.split(".")[1])):
            f3 = lambda v: ("%s, %s, %s" % (fmt(v[0], 4), fmt(v[1], 4), fmt(v[2], 3))) if v else "-"
            I1e.append("| %s | %s | %s | %s |" % (m + ("" if res[m]["rmask"] else " (e, c only: ray mask unavailable)"), ck, f3(e0.get(ck)), f3(e1.get(ck))))
open(os.path.join(OUT, "head_tables.md"), "w").write("\n".join(H1 + H2 + H3 + H4 + H5 + H6 + H7 + H8 + I1 + I1c + I1e) + "\n")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(res), open(os.path.join(OUT, "head_tables.json"), "w"), indent=1, default=float)
print("\n".join(H1))
