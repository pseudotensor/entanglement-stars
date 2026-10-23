"""b7_table_v1.py: prints the B7 markdown tables from b7_birth_v2.json (no computation beyond differences)."""
import json, sys
R = json.load(open(sys.argv[1]))
print("| member | class | m a | r_s/a | sigma/a | tau/a | R0/r_s | T_f | step | T_birth linear | T_birth cubic (records) | error |cubic - linear| | shift T_birth - T_f (linear; cubic) | shift / step (cubic) | r at birth, cubic (a) | R_t at T_f (a) | json-series zero: linear; cubic |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for o in R:
    a = o["args"]
    cr = ", ".join("%g" % t for t in o["cubic_records"])
    print("| %s | %s | %g | %g | %g | %g | %g | %.6f | %.6f | %.6f | %.6f (%s) | %.1e | %+.6f; %+.6f | %.3f | %.2f | %.3f | %.6f; %.6f |" % (
        o["member"], o["class"], a["m"], a["rs"], a["ell"], a["tau"], a["R0_over_rs"], o["T_f"], o["step"], o["T_birth_lin"],
        o["T_birth_cubic"], cr, o["interp_error"], o["shift_lin"], o["shift_cubic"], o["shift_cubic"] / o["step"], o["r_birth_cubic"],
        o["R_t_f"], o["T_jsonseries_lin"], o["T_jsonseries_cubic"]))
print()
print("| member | json md5 | fields npz | fields md5 | records used (fields row) | converged flags (json) | max |rebuilt - json| of min(N-|v|) | max |rebuilt - json| zero | bracket m (prev; first) | script |")
print("|---|---|---|---|---|---|---|---|---|---|")
for o in R:
    ch = o["rebuild_checks"].values()
    dm = max(abs(c["d_min"]) for c in ch)
    dz = max((c["max_d_zero"] if isinstance(c["max_d_zero"], float) else 9e9) for c in ch)
    print("| %s | %s | %s | %s | %s | %s | %.1e | %.1e | %.4f; %.4f | %s |" % (o["member"], o["json_md5"], o["fields"], o["fields_md5"],
        ", ".join("%s (%s)" % (k, v) for k, v in o["fields_rows_used"].items()), ", ".join("%s" % v for v in o["converged_window"].values()),
        dm, dz, o["bracket_m"][0], o["bracket_m"][1], o["script"]))
