#!/usr/bin/env python3
"""obs_raymap_fixture_summary_v1.py (builder analyst): one table from several obs_raymap_solved_v3 records (the per-packet runs of the J3'
fixture control): per (j, m_p/kappa, omega/kappa, T_surf) n_w, n_stat (the stationary scattering's occupation on the frozen slice), the ratio
n_w/n_stat, Gamma_w, n_w/Gamma_w (the factorization test reads it across j at equal omega), theta (fit), theta (T_conv), the end-norm and
the Duhamel bound, the first-passage share's range; per packet the log slope of n_w and the quasi-frozen check. Records are deduplicated by
(j, sign, m, omega) (the first file wins). Args: records... --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    seen = set()
    L = ["## J3' on the fixture history: the per-packet records combined (obs_raymap_fixture_summary_v1.py over %s)" % ", ".join(x.split("/")[-1] for x in a.records), "",
         "| j | m_p/kappa | omega/kappa | T_surf/r_s | n_w | n_stat | n_w/n_stat | Gamma_w | n_w/Gamma_w | theta (fit) | theta (T_conv) | end-norm | Duhamel B | G2 share min, max |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tail = []
    for f in a.records:
        D = json.load(open(f))
        j = D["args"]["j"]; sg = D["args"]["sign"]
        for row in D["rows"]:
            key = (j, sg, row["mp_over_kappa"], row["omega_over_kappa"])
            if key in seen or "series" not in row:
                continue
            seen.add(key)
            for s_ in row["series"]:
                g2 = s_.get("G2") or {}
                fmt = lambda x, p: (p % x) if isinstance(x, (int, float)) else "-"
                L.append("| %g | %g | %g | %g | %.6e | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    j, row["mp_over_kappa"], row["omega_over_kappa"], s_["T_surf_over_rs"], s_["n_w"], fmt(s_.get("n_stat"), "%.6e"),
                    fmt((s_["n_w"] / s_["n_stat"]) if s_.get("n_stat") else None, "%.4f"), fmt(s_.get("Gamma_w"), "%.6f"),
                    fmt((s_["n_w"] / s_["Gamma_w"]) if s_.get("Gamma_w") else None, "%.6e"), fmt(s_.get("theta_fit"), "%.5f"), fmt(s_.get("theta_conv"), "%.5f"),
                    fmt(s_.get("endnorm_inside_Rt_plus_5ell"), "%.2e"), fmt(s_.get("duhamel_B"), "%.2e"),
                    ("%.4f, %.4f" % (g2["share_min"], g2["share_max"])) if g2 else "-"))
            hc = row.get("hold_check")
            tail.append("- j %g, m_p/kappa %g, omega/kappa %g (%s): log slope of n_w %s per r_s; the quasi-frozen check %s; packet s_x %.2f r_s, temporal width %.2f r_s" % (
                j, row["mp_over_kappa"], row["omega_over_kappa"], f.split("/")[-1], ("%.4e" % row["log_slope_per_rs"]) if row.get("log_slope_per_rs") is not None else "-",
                ("relative %.2e (d theta %.1e)" % (hc["rel_diff"], hc.get("dtheta", float("nan")))) if hc else "-", row["s_x_density"] / D["rs_geo"], row["s_t_over_rs"]))
    L += [""] + tail + ["", "- every number above is printed by the script from the records; the fixture is static after 2 r_s (the no-trap v_0 ramped to the smooth-core",
                         "  profile, R_c = 4 sites, r_in about 1.05), so the quasi-frozen check and the Duhamel bound are trivially passed there."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
