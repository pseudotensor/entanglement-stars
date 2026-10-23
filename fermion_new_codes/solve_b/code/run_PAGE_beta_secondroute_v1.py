"""Matched-clock raw-spectrum/collar diagnostic, PAGE specification section 5."""

import argparse
import json
import math
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output)
    import numpy as np
    from scipy.integrate import solve_ivp

    spectral = json.loads(args.spectra.read_text())["rows"]
    lookup = {(r["kappa"],r["clock_rs"],r["cut_rs"]):r for r in spectral}
    metadata = {}
    record_flows = {}
    parent_checks = []
    for channel in range(1,19):
        path = args.inputs/f"v6o_B2_k{channel}_d13.json"
        raw = json.loads(path.read_text())
        metadata[channel] = {float(t):r["per_channel"][str(channel)]
                             for t,r in raw["clocks"].items()}
        summary = json.loads(path.with_name(path.name.replace("_d13.json","_summary.json")).read_text())
        settings = summary["settings"]
        if ((settings["g"] is not None and float(settings["g"]) != 0.25) or
                (settings["rs_nominal"] is not None and float(settings["rs_nominal"]) != 64) or
                int(settings["nsites"]) != 8192):
            raise ValueError("Incompatible radiation parent or scale")
        parent_checks.append({"kappa":channel,"nsites":settings["nsites"],
                              "g_argument":settings["g"],"rs_argument":settings["rs_nominal"],
                              "source_record":settings["record"],"source_fields":settings["fields"],
                              "missing_argument_source":"B2 applied-field data and named nominal-scale contract"})
        record_flows[channel] = {int(r["k"]):r["per_channel"][str(channel)]["collar_flow"]
                                 for r in summary["records"]}
    with np.load(args.fields,allow_pickle=False) as fields:
        rs = 64.0
        times = fields["T"]/rs
        radii = fields["R"]/rs
        clocks = fields["N_applied"]
        flows = fields["v_applied"]
        g = float(fields["g"])

    def speed(t,r):
        if not times[0] <= t <= times[-1] or not radii[0] <= r <= radii[-1]:
            raise ValueError("Ray outside saved applied fields")
        it = min(max(int(np.searchsorted(times,t)-1),0),len(times)-2)
        ir = min(max(int(np.searchsorted(radii,r)-1),0),len(radii)-2)
        ft,fr = (t-times[it])/(times[it+1]-times[it]), (r-radii[ir])/(radii[ir+1]-radii[ir])
        def linear(array):
            return ((1-ft)*((1-fr)*array[it,ir]+fr*array[it,ir+1])
                    +ft*((1-fr)*array[it+1,ir]+fr*array[it+1,ir+1]))
        return linear(clocks)-abs(linear(flows))

    def emission(arrival,step,tolerance):
        def event(t,y): return y[0]-2
        event.terminal = True
        result = solve_ivp(lambda t,y:[speed(t,y[0])], (arrival,0), [16.0],
                           events=event, method="DOP853", max_step=step,
                           rtol=tolerance, atol=tolerance/10)
        if not result.success or len(result.t_events[0]) != 1:
            raise RuntimeError(f"Ray failed to reach 2 r_s for arrival {arrival}")
        return float(result.t_events[0][0])

    grid = list(range(40,101,2))
    ray_rows = []
    for arrival in grid:
        coarse = emission(arrival,0.0625,1e-9)
        fine = emission(arrival,0.03125,1e-11)
        ray_rows.append({"arrival_rs": arrival,"coarse_emission_rs": coarse,
                         "emission_rs": fine,"refinement_difference_rs": fine-coarse})
        (args.output/"ray_map.json").write_text(json.dumps(ray_rows,indent=2)+"\n")
    emit = {r["arrival_rs"]:r["emission_rs"] for r in ray_rows}
    coarse_emit = {r["arrival_rs"]:r["coarse_emission_rs"] for r in ray_rows}
    all_times = sorted(metadata[1])

    def entropy_at(channels,cut,t,nearest=False):
        if t < all_times[0] or t > all_times[-1]:
            raise ValueError("Entropy extrapolation is prohibited")
        if nearest:
            chosen = min(all_times,key=lambda x:abs(x-t))
            brackets = [(chosen,1.0)]
        elif t in all_times:
            brackets = [(t,1.0)]
        else:
            index = int(np.searchsorted(all_times,t))
            lo,hi = all_times[index-1:index+1]
            frac = (t-lo)/(hi-lo)
            brackets = [(lo,1-frac),(hi,frac)]
        total, valid, checked = [], True, True
        for clock, fraction in brackets:
            terms = []
            for channel in channels:
                row = lookup.get((channel,clock,cut))
                if row is None or row["complement_excess"] is None:
                    return None,False,False
                valid &= (row["checkpoint_state_spectra_valid"] and
                          row["checkpoint_vacuum_spectra_valid"])
                checked &= "direct_state" in row
                terms.append(16*channel*row["complement_excess"])
            total.append(fraction*math.fsum(terms))
        return math.fsum(total),valid,checked

    temperature_checks = []
    flow_checks = []
    for channel, channel_rows in metadata.items():
        for clock,row in channel_rows.items():
            expected = 1/(8*math.pi*g*row["E"])
            temperature_checks.append({"kappa":channel,"clock_rs":clock,
                                       "absolute_residual":row["T_H_now"]-expected,
                                       "relative_residual":row["T_H_now"]/expected-1})
            if clock not in metadata[1] or not math.isclose(
                    row["E"], metadata[1][clock]["E"],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("Channels do not share charge and clock")
            if (not math.isclose(row["two_gE"],2*g*row["E"],rel_tol=1e-12,abs_tol=1e-12)
                    or not math.isclose(times[int(row["k"])],clock,rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError("Raw charge or nominal clock disagrees with applied-field parent")
        channel_times = sorted(channel_rows)
        for a,b in zip(channel_times[:-1],channel_times[1:]):
            lo,hi = int(channel_rows[a]["k"]),int(channel_rows[b]["k"])
            integrated = math.fsum(record_flows[channel][k] for k in range(lo,hi))
            difference = channel_rows[b]["collar_flow_acc"]-channel_rows[a]["collar_flow_acc"]
            flow_checks.append({"kappa":channel,"start_rs":a,"end_rs":b,
                                "record_sum":integrated,"accumulator_difference":difference,
                                "unweighted_convention_residual":difference-integrated})
    windows = []
    cases = [((1,),(1,),"k1"),((1,2),(1,2),"k1_k2"),
             (tuple(range(1,19)),tuple(range(1,19)),"K18_L18"),
             (tuple(range(1,19)),(1,2),"K18_L12_unmatched")]
    with (args.output/"beta_rows.jsonl").open("w") as progress:
        for channels,energy_channels,label in cases:
            for a in grid:
                for b in grid:
                    if b-a < 4: continue
                    de = math.fsum(16*k*(metadata[k][b]["collar_flow_acc"]-
                                              metadata[k][a]["collar_flow_acc"]) for k in energy_channels)
                    ta = 1/(8*math.pi*g*metadata[1][a]["E"])
                    tb = 1/(8*math.pi*g*metadata[1][b]["E"])
                    for cut in (2,16):
                        ea,eb = (emit[a],emit[b]) if cut == 2 else (a,b)
                        sa,va,ca = entropy_at(channels,cut,ea)
                        sb,vb,cb = entropy_at(channels,cut,eb)
                        row = {"channels":list(channels),"energy_channels":list(energy_channels),
                               "label":label,"cut_rs":cut,
                               "start_rs":a,"end_rs":b,"emission_start_rs":ea,
                               "emission_end_rs":eb,"delta_energy":de,
                               "temperature_start":ta,"temperature_end":tb,
                               "delta_entropy":None if sa is None or sb is None else sb-sa,
                               "beta":None if de == 0 or sa is None or sb is None else
                                   (ta+tb)/2*(sb-sa)/de,
                               "energy_sign": "positive" if de>0 else "negative" if de<0 else "zero",
                               "known_spectra_valid":va and vb,
                               "all_endpoints_direct_checked":ca and cb,
                               "route":"doubled-plus complement exported spectra"}
                        if cut == 2 and de != 0:
                            sna,vna,cna = entropy_at(channels,cut,ea,nearest=True)
                            snb,vnb,cnb = entropy_at(channels,cut,eb,nearest=True)
                            sca,vca,cca = entropy_at(channels,cut,coarse_emit[a])
                            scb,vcb,ccb = entropy_at(channels,cut,coarse_emit[b])
                            row["nearest_beta"] = None if sna is None or snb is None else (ta+tb)/2*(snb-sna)/de
                            row["coarse_ray_beta"] = None if sca is None or scb is None else (ta+tb)/2*(scb-sca)/de
                            row["nearest_known_spectra_valid"] = vna and vnb
                            row["nearest_all_direct_checked"] = cna and cnb
                            row["coarse_ray_known_spectra_valid"] = vca and vcb
                            row["coarse_ray_all_direct_checked"] = cca and ccb
                        windows.append(row)
                        progress.write(json.dumps(row,allow_nan=False)+"\n")
                        progress.flush()
    extrema = []
    for _,_,label in cases:
        for cut in (2,16):
            for clean_only in (False,True):
                subset = [row for row in windows if row["label"]==label and
                          row["cut_rs"]==cut and row["end_rs"]-row["start_rs"]>=6 and
                          row["beta"] is not None and (not clean_only or row["known_spectra_valid"])]
                extrema.append({"label":label,"cut_rs":cut,"width_min_rs":6,
                                "selection":"no known invalid input" if clean_only else "all arithmetic rows",
                                "count":len(subset),"minimum":min(subset,key=lambda r:r["beta"]) if subset else None,
                                "maximum":max(subset,key=lambda r:r["beta"]) if subset else None})
    result = {"scope":"Raw complement-spectrum diagnostic with its pointwise physicality "
                      "and sector-source limitations; no D0 classification.",
              "runtime":runtime,"fields":str(args.fields.resolve()),
              "rs_nominal":rs,"g":g,"spatial_interpolation":"linear",
              "temporal_interpolation":"linear in stored applied fields",
              "parent_checks":parent_checks,"ray_rows":ray_rows,
              "temperature_checks":temperature_checks,"flow_checks":flow_checks,
              "windows":windows,
              "extrema":extrema}
    (args.output/"beta.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"saved","windows":len(windows),"output":str(args.output/"beta.json")}))


if __name__ == "__main__":
    main()
