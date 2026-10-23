#!/usr/bin/env python
"""f256s4_fieldfail_repro_v1.py (busi, analyst; box CPU, no card): F256_s4 ended at record T/rs 0.50 on a failed field solve
("the downward stretch at r = 11.2239 failed: Required step size is less than spacing between numbers"). The exact failing sources
(the first Newton evaluation at 0.50) are not stored: they need the propagated state at 0.50 and its eigendecompositions (GPU). This script
re-solves the stored record sources (fields npz, records 0.3125 to 0.4375) with the deployed field module (code16d_flat/engine_fields_v16.py,
md5 7a69faad), then sources extrapolated towards 0.50 along the records' own drift, on both routes, and for every failing solve reads
(i) the collocation solution with the switch rule off (U's true profile, its positive minimum), (ii) the stretch's approach to the fold
(U' -> 0 at s > 0), (iii) three candidate field-route variants (the switch level lowered; rtol 1e-10; a fold exit back to r).
Read-only on the run files; writes one json.
    python f256s4_fieldfail_repro_v1.py --code DIR --fields NPZ --out JSON
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--lam", default="0,0.25,0.5,0.75,1.0,1.1,1.25,1.5,1.75,2.0")
    ap.add_argument("--records", default="0.3125,0.375,0.4375")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    os.environ["QI2_FIELDS_ROUTE"] = "ivp"
    import engine_fields_v16 as FLD
    from scipy.integrate import solve_ivp
    z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(z["R"], float); ell = int(z["ell"]); g = float(z["g"]) if "g" in z.files else a.g
    Ts = np.asarray(z["T"], float) / 256.0
    E = np.asarray(z["eps_tot"], float); J = np.asarray(z["J_tot"], float)
    out = {"code": a.code, "fields": a.fields, "T_over_rs": Ts.tolist(), "g": g, "ell": ell, "cases": []}
    orig_UL = FLD.integrate_UL

    def ivp(eps, Jt, **kw):
        t0 = time.time()
        Nn, vn, Wn, info = FLD.solve_constraints_ivp(eps, Jt, g, R, ell, rtol=kw.get("rtol", 1e-12), atol=kw.get("atol", 1e-16),
                                                     collar_policy="flag", support_rel=1e-14)
        res = {"failed": info.get("failed"), "s": time.time() - t0}
        if not info.get("failed"):
            gg = info["gauge"]
            res.update({"gauge": gg["label"][:160], "zeros": [[round(q["r"], 4), q["kind"]] for q in gg["zeros"]], "stretches": gg["stretches"],
                        "near_tangencies": gg["near_tangencies"], "U_scale": gg["U_scale"], "U_switch": gg["U_switch"],
                        "N_min": float(np.min(Nn)), "N_centre": info["ivp"]["N_centre"], "regime_max": info["regime"]["max_ratio"],
                        "centre_regular": info["centre_series_check"]["regular_start"]})
        return res, (Nn, vn, info)

    def colloc_profile(eps, Jt):
        """the collocation solution with the switch rule off (eps_switch 0): U's sampled profile on (r_0, r_collar] and its minima."""
        FLD._RD2["eps_switch"] = 0.0
        try:
            st, o = FLD._rd2_collocation(eps, Jt, g, R, ell, 1e-12, 1e-16, "flag", 1e-14, 1.0 / 64.0)
        finally:
            FLD._RD2["eps_switch"] = 1e-6
        if st != "ok":
            return {"ok": False, "why": o}
        info = o[3]
        src = info["src"]
        r = np.arange(ell / 64.0, info["r_collar"], ell / 64.0)
        rg = np.linspace(0.0, info["r_collar"], 4001)
        U_scale = float(np.sum(0.5 * (np.abs(src.rho(rg[1:])) * rg[1:] ** 2 + np.abs(src.rho(rg[:-1])) * rg[:-1] ** 2) * np.diff(rg))) * 8.0 * np.pi * g
        # U from the dense W of the collocation is not exported; integrate U' = 8 pi g r^2 (rho + q J) on the fine grid as a check instead
        Ucol = info.get("_rd2_U")
        return {"ok": True, "U_scale": U_scale, "gauge": info["gauge"]["label"][:160] if "gauge" in info else None,
                "fields_route": {k: v for k, v in info.get("fields_route", {}).items() if not isinstance(v, np.ndarray)},
                "N_min": float(np.min(o[0]))}

    def u_profile_ivp_noswitch(eps, Jt):
        """U(r) and U'(r) by the r-integration alone with the switch level at 1e-14 U_scale (the stretch never entered while U > 1e-14 U_scale)."""
        FLD.integrate_UL = functools.partial(orig_UL, eps_switch=1e-14)
        try:
            res, (Nn, vn, info) = ivp(eps, Jt)
        finally:
            FLD.integrate_UL = orig_UL
        if res["failed"]:
            return res, None
        src = info["src"]
        rr = np.arange(ell / 64.0, min(info["r_collar"], 40.0), ell / 256.0)
        segs, why, counts, zeros = orig_UL(src, g, ell / 64.0, info["r_collar"], FLD.centre_coefficients(src, g, ell), 1e-12, 1e-16, ell, eps_switch=1e-14)
        pw = FLD.Piecewise(FLD.centre_coefficients(src, g, ell), ell / 64.0, segs)
        U, L = pw(rr)
        Us = counts["U_scale"]
        k_ = [i for i in range(1, len(rr) - 1) if U[i] < U[i - 1] and U[i] <= U[i + 1]]
        mins = [[float(rr[i]), float(U[i] / Us)] for i in k_][:12]
        res["U_minima_r_U_over_Uscale"] = mins
        res["rho_J_at_minima"] = [[float(src.rho(m[0])), float(src.J(m[0]))] for m in mins]
        res["U_over_Uscale_at"] = {str(x): float(np.interp(x, rr, U) / Us) for x in (2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 11.2239, 12.0, 14.0, 16.0, 20.0)}
        return res, (rr, U / Us, L)

    def fold_probe(eps, Jt):
        """the downward stretch the production enters, rerun alone with the event's data: s from s_sw to 0, |U'| and dr/ds along it."""
        src = FLD.SmoothedSources5(eps, Jt, R, ell, FLD.coarse_grid(R, ell)[-1], 1e-14)
        rho_, J_ = src.rho, src.J
        fpi = 4.0 * np.pi * g
        r_nodes = FLD.coarse_grid(R, ell)
        i_need = int(np.ceil((src.R_last + 8.0 * ell) / ell)); r_collar = float(r_nodes[min(i_need, len(r_nodes) - 1)])
        rg = np.linspace(0.0, r_collar, 4001)
        U_scale = float(np.sum(0.5 * (np.abs(rho_(rg[1:])) * rg[1:] ** 2 + np.abs(rho_(rg[:-1])) * rg[:-1] ** 2) * np.diff(rg))) * 2.0 * fpi
        U_sw = 1e-6 * U_scale
        segs, why, counts, zeros = orig_UL(src, g, ell / 64.0, r_collar, FLD.centre_coefficients(src, g, ell), 1e-12, 1e-16, ell)
        rec = {"why": why, "zeros": zeros, "counts": {k: v for k, v in counts.items()}}
        # find the downward switch event: rerun the last positive r-segment's end
        last = segs[-1] if segs else None
        if last is None:
            return rec
        r_e = float(last[1])
        Ue, Le = last[2](np.array([r_e]))
        Ue = float(np.atleast_1d(Ue)[0]); Le = float(np.atleast_1d(Le)[0])
        rec["switch_event"] = {"r": r_e, "U_over_Uscale": Ue / U_scale}

        def Up(r, s):
            return 2.0 * fpi * r * r * (float(rho_(r)) - (s / np.sqrt(r)) * float(J_(r)))

        def rhs_s(s, y):
            up = Up(y[0], s)
            return [2.0 * s / up, 2.0 * fpi * y[0] ** 1.5 * float(J_(y[0])) / up]
        s0 = float(np.sqrt(max(Ue, 0.0)))
        st = solve_ivp(rhs_s, (s0, 0.0), [r_e, Le], method="DOP853", rtol=1e-12, atol=1e-16, dense_output=True)
        ss = st.t; rr = st.y[0]
        ups = np.array([Up(r_, s_) for r_, s_ in zip(rr, ss)])
        rec["stretch"] = {"status": int(st.status), "message": st.message, "n_steps": int(len(ss)), "s_start": s0, "s_end": float(ss[-1]),
                          "U_end_over_Uscale": float(ss[-1] ** 2 / U_scale), "r_end": float(rr[-1]),
                          "Uprime_start": float(ups[0]), "Uprime_end": float(ups[-1]), "Uprime_ratio_end_start": float(ups[-1] / ups[0]),
                          "drds_end": float(2.0 * ss[-1] / ups[-1]) if ups[-1] != 0 else None,
                          "last_steps_ds": [float(x) for x in np.diff(ss)[-5:]]}
        return rec

    def make_fold_exit_UL():
        """candidate (c), the fold exit: integrate_UL's source with one insertion before the downward stretch: at the switch event the
        r-integration is probed below the switch with two terminal events, U' = 0 crossed upward (a positive minimum, the fold) and U = 0
        crossed downward; when the minimum comes first with U > 0 the r-integration is kept to it (counted as a near-tangency) and the loop
        goes on in r; otherwise (a true zero, or the probe failing) the original stretch runs unchanged. Production code untouched."""
        import inspect
        src_txt = inspect.getsource(orig_UL)
        anchor = "                st = s_stretch(r_e, float(ye[1]), float(np.sqrt(max(float(ye[0]), 0.0))), 0.0)\n"
        assert src_txt.count(anchor) == 1
        ins = (
            "                def _ev_min(r_, y_):\n"
            "                    return 2.0 * fpi * r_ * r_ * (float(rho_(r_)) - np.sqrt(max(y_[0], 0.0) / r_) * float(J_(r_)))\n"
            "                _ev_min.terminal = True; _ev_min.direction = 1\n"
            "                def _ev_zero(r_, y_):\n"
            "                    return y_[0]\n"
            "                _ev_zero.terminal = True; _ev_zero.direction = -1\n"
            "                try:\n"
            "                    _pr = solve_ivp(rhs_pos, (r_e, r_collar), np.array(ye, float), method='DOP853', dense_output=True, rtol=rtol, atol=atol, events=[_ev_min, _ev_zero])\n"
            "                except Exception:\n"
            "                    _pr = None\n"
            "                if _pr is not None and _pr.success and _pr.status == 1 and len(_pr.t_events[0]) and not len(_pr.t_events[1]):\n"
            "                    _rm = float(_pr.t_events[0][0])\n"
            "                    segs.append((r_e, _rm, _pr.sol, 1, int(_pr.nfev))); counts['near_tangencies'] += 1\n"
            "                    counts.setdefault('fold_exits', []).append([r_e, _rm, float(_pr.y_events[0][0][0]) / U_scale])\n"
            "                    r = _rm; y = np.array(_pr.y_events[0][0], float)\n"
            "                    continue\n")
        src_txt = src_txt.replace(anchor, ins + anchor).replace("def integrate_UL(", "def integrate_UL_foldexit(")
        ns = dict(vars(FLD))
        exec(compile(src_txt, "<integrate_UL_foldexit>", "exec"), ns)
        return ns["integrate_UL_foldexit"]
    UL_fold = make_fold_exit_UL()

    def fold_exit_variant(eps, Jt):
        FLD.integrate_UL = UL_fold
        try:
            res, (Nn, vn, info) = ivp(eps, Jt)
            if not res["failed"]:
                res["fold_exits"] = FLD.integrate_UL(info["src"], g, ell / 64.0, info["r_collar"], FLD.centre_coefficients(info["src"], g, ell),
                                                     1e-12, 1e-16, ell)[2].get("fold_exits")
        finally:
            FLD.integrate_UL = orig_UL
        return res, (Nn, vn)

    cases = []
    idx = {round(t, 4): i for i, t in enumerate(Ts)}
    for t in [float(x) for x in a.records.split(",") if x]:
        i = idx[t]
        cases.append(("record %.4f" % t, E[i], J[i]))
    i1, i0, im = idx[0.4375], idx[0.375], idx[0.3125]
    dE1 = E[i1] - E[i0]; dJ1 = J[i1] - J[i0]
    for lam in [float(x) for x in a.lam.split(",")]:
        cases.append(("lin 0.4375 + %.2f x (0.4375 - 0.375)" % lam, E[i1] + lam * dE1, J[i1] + lam * dJ1))
    cases.append(("quad to 0.5", 3 * E[i1] - 3 * E[i0] + E[im], 3 * J[i1] - 3 * J[i0] + J[im]))
    for name, eps, Jt in cases:
        c = {"case": name}
        c["ivp"], _ = ivp(eps, Jt)
        c["r_integration_switch_1e-14"], prof = u_profile_ivp_noswitch(eps, Jt)
        c["collocation_switch_off"] = colloc_profile(eps, Jt)
        if c["ivp"]["failed"]:
            c["fold_probe"] = fold_probe(eps, Jt)
            FLD.integrate_UL = functools.partial(orig_UL, eps_switch=1e-8)
            try:
                c["variant_switch_1e-8"], _ = ivp(eps, Jt)
            finally:
                FLD.integrate_UL = orig_UL
            c["variant_rtol_1e-10"], _ = ivp(eps, Jt, rtol=1e-10, atol=1e-14)
            c["variant_fold_exit"], fe = fold_exit_variant(eps, Jt)
            FLD.integrate_UL = functools.partial(orig_UL, eps_switch=1e-14)
            try:
                _r, (N14, v14, _i) = ivp(eps, Jt)
            finally:
                FLD.integrate_UL = orig_UL
            if fe[0] is not None and N14 is not None:
                c["fold_exit_vs_switch_1e-14"] = {"max_dN": float(np.max(np.abs(fe[0] - N14))), "max_dv": float(np.max(np.abs(fe[1] - v14)))}
        print(json.dumps({"case": name, "ivp_failed": c["ivp"]["failed"], "zeros": c["ivp"].get("zeros"),
                          "minima": (c["r_integration_switch_1e-14"].get("U_minima_r_U_over_Uscale") or [])[:4]}), flush=True)
        cases_out = c
        out["cases"].append(cases_out)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
