"""Tables from this lane's saved results only; no solver or first-route comparison."""

import argparse
import json
import math
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require_remote_cpu(args.output)
    base = args.output
    def load(name): return json.loads((base/name).read_text())
    spectrum = load("spectral_entropy.json")
    beta = load("beta.json")
    ledger = load("ledger.json")
    thermal = load("thermal_R_endpoint/thermal_R.json")
    def index(data): return {(r["kappa"],r["clock_rs"],r["cut_rs"]):r for r in data["rows"]}
    production = index(spectrum)
    charge_path = Path('<internal>/v6o_B2_k1_d13.json')
    charge_raw = json.loads(charge_path.read_text())
    charge = next(row for key,row in charge_raw['clocks'].items() if float(key)==100)['per_channel']['1']
    energy, g = float(charge['E']), float(beta['g'])
    radius = 2*g*energy
    area = 4*math.pi*radius**2
    area_normalization = {'clock_rs':100,'E_a':energy,'g':g,'two_gE':charge.get('two_gE'),
        'r_E_over_a':radius,'A_over_a2':area,'S_BH':area/(4*g),'S_eta':0.9192*area,
        'charge_source':str(charge_path),'fields_source':beta['fields'],'eta_input':0.9192}
    def aggregate(data,channels,clock,cut,field):
        rows = [data.get((k,clock,cut)) for k in channels]
        available = all(r is not None and r.get(field) is not None for r in rows)
        return {"value":math.fsum(16*r["kappa"]*r[field] for r in rows) if available else None,
                "available":available,
                "no_known_state_failure":all(r is not None and r["checkpoint_state_spectra_valid"] for r in rows),
                "no_known_vacuum_failure":all(r is not None and r["checkpoint_vacuum_spectra_valid"] for r in rows),
                "saved_spectra_complete":all(r is not None and r.get("checkpoint_saved_spectra_complete",True) for r in rows)}
    targets = []
    for clock,cut in ((50,2),(50,30),(100,2),(100,30),(60,30),(70,30)):
        for channels in ((1,),tuple(range(1,19))):
            for route in ("direct_excess","complement_excess"):
                targets.append({"clock_rs":clock,"cut_rs":cut,"channels":list(channels),
                                "route":route,**aggregate(production,channels,clock,cut,route)})
    area_normalization['ratios'] = [{**r,'over_S_BH':r['value']/area_normalization['S_BH'] if r['value'] is not None else None,
        'over_S_eta':r['value']/area_normalization['S_eta'] if r['value'] is not None else None}
        for r in targets if r['clock_rs']==100 and r['cut_rs']==2 and len(r['channels'])==18]
    spectral_gaps = []
    for clock,cut in ((50,2),(50,30),(100,2),(100,30),(60,30),(70,30)):
        for channels in ((1,),tuple(range(1,19))):
            for field in ('state_complement_gap','state_sector_gap','vacuum_sector_gap'):
                spectral_gaps.append({'clock_rs':clock,'cut_rs':cut,'channels':list(channels),
                    'field':field,**aggregate(production,channels,clock,cut,field)})
            direct = aggregate(production,channels,clock,cut,'direct_excess')
            comp = aggregate(production,channels,clock,cut,'complement_excess')
            spectral_gaps.append({'clock_rs':clock,'cut_rs':cut,'channels':list(channels),
                'field':'direct_minus_complement_excess',
                'value':direct['value']-comp['value'] if direct['available'] and comp['available'] else None,
                'no_known_state_failure':direct['no_known_state_failure'] and comp['no_known_state_failure']})
    residuals = {
        "ray_refinement_max_rs":max(abs(r["refinement_difference_rs"]) for r in beta["ray_rows"]),
        "temperature_absolute_max":max(abs(r["absolute_residual"]) for r in beta["temperature_checks"]),
        "temperature_relative_max":max(abs(r["relative_residual"]) for r in beta["temperature_checks"]),
        "unweighted_flow_residual_max":max(abs(r["unweighted_convention_residual"]) for r in beta["flow_checks"]),
        "unweighted_per_channel_ledger_telescoping_max":max(abs(r["telescoping_residual"]) for r in ledger["rows"]),
        "unweighted_per_channel_ledger_balance_max":max(abs(r["balance_residual"]) for r in ledger["rows"]),
        "weighted_aggregate_ledger_telescoping_max":max(abs(r["telescoping_residual"]) for r in ledger["aggregates"] if r['status']=='formed'),
        "weighted_aggregate_ledger_balance_max":max(abs(r["balance_residual"]) for r in ledger["aggregates"] if r['status']=='formed'),
        "ledger_saved_second_term_difference_max":max(abs(r["saved_second_term_difference"]) for r in ledger["rows"])}
    scattering = {}
    for setting in ("base","refined"):
        folder = base/("thermal_R" if setting=="base" else "thermal_R_endpoint")
        samples = []
        for k in range(1,9):
            samples.extend(json.loads(line) for line in (folder/f"scattering_k{k}_{setting}.jsonl").read_text().splitlines())
        scattering[setting] = {"samples":len(samples),
            "unitarity_absolute_max":max(abs(r["unitarity_residual"]) for r in samples),
            "Jost_relative_residual_max":max(r["Jost_relative_residual"] for r in samples),
            "small_transmission_samples":sum(r["small_transmission_conditioning"] for r in samples)}
    spreads = []
    cases = [("piecewise",tuple(range(1,9)),(50,100),(16,24,30,48),"direct_excess"),
             ("sub16",(18,),(50,100),(2,16,24,30,48),"complement_excess"),
             ("completion_fs",(1,2),(20,30,40),(2,16,24,30,48),"direct_excess"),
             ("completion_pa",(1,2),(20,30,40),(2,16,24,30,48),"direct_excess"),
             ("scale_s8",tuple(range(1,19)),(20,30,40),(2,16,24,30,48),"direct_excess")]
    for name,channels,clocks,cuts,route in cases:
        control = index(load(name+"/spectral_entropy.json"))
        for clock in clocks:
            for cut in cuts:
                a = aggregate(control,channels,clock,cut,route)
                b = aggregate(production,channels,clock,cut,route)
                spreads.append({"control":name,"channels":list(channels),"clock_rs":clock,
                                "cut_rs":cut,"route":route,"control_value":a,"production_value":b,
                                "difference":a["value"]-b["value"] if a["available"] and b["available"] else None,
                                "conditional_times18_proxy":18*abs(a["value"]-b["value"]) if name=="sub16" and a["available"] and b["available"] else None})
    completion_ranges = []
    for clock in (20,30,40):
        for cut in (2,16,24,30,48):
            available = [r for r in spreads if r["control"] in ("completion_fs","completion_pa")
                         and r["clock_rs"]==clock and r["cut_rs"]==cut]
            values = [r["control_value"]["value"] for r in available]
            prod = aggregate(production,(1,2),clock,cut,"direct_excess")
            values.append(prod["value"])
            completion_ranges.append({"clock_rs":clock,"cut_rs":cut,"channels":[1,2],
                "range":max(values)-min(values) if all(v is not None for v in values) else None,
                "all_no_known_state_failure":prod["no_known_state_failure"] and all(r["control_value"]["no_known_state_failure"] for r in available)})
    chosen_windows = [(46,100),(80,100),(40,46)]+[(a,a+6) for a in range(46,95,6)]
    beta_rows = [r for r in beta["windows"] if (r["start_rs"],r["end_rs"]) in chosen_windows]
    primary_ledger = [r for r in ledger["aggregates"] if
        (r["clock_start_rs"],r["clock_end_rs"]) in [(a,a+14) for a in (26,36,46,56,66,76,86)]]
    result = {"targets":targets,"area_normalization":area_normalization,"spectral_gaps":spectral_gaps,
              "residuals":residuals,"scattering_controls":scattering,
              "thermal_rates":thermal["rates"],"thermal_channels":thermal["channels"],
              "thermal_high_tail_bounds":thermal["high_frequency_tail_bounds"],
              "thermal_high_tail_relative":thermal["high_frequency_relative_bounds"],
              "control_spreads":spreads,"completion_ranges":completion_ranges,
              "beta_primary_windows":beta_rows,"beta_extrema":beta["extrema"],
              "primary_ledger":primary_ledger,"invalid_spectra":load("invalid_spectra.json")["invalid_spectra"]}
    (base/"tables.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    def fmt(value):
        if value is None: return "unavailable"
        if isinstance(value,bool): return "yes" if value else "no"
        if isinstance(value,float): return f"{value:.10g}"
        return str(value)
    lines = ["# Saved numerical tables","", "Arithmetic from this lane's outputs. No cross-route judgment.",""]
    def table(title,headers,rows):
        lines.extend([title,"","| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"])
        for row in rows: lines.append("| "+" | ".join(fmt(value) for value in row)+" |")
        lines.append("")
    table("Residual controls",["quantity","value"],list(residuals.items()))
    table("Charge-area normalization at T/r_s=100",['quantity','value'],
          [(key,area_normalization[key]) for key in ('E_a','g','two_gE','r_E_over_a','A_over_a2','S_BH','S_eta')])
    table("Cut2 18-channel area ratios",['route','nats','over S_BH','over S_eta'],
          [(r['route'],r['value'],r['over_S_BH'],r['over_S_eta']) for r in area_normalization['ratios']])
    table("Weighted spectrum-route gaps",['T/r_s','cut/r_s','channels','gap','nats','no known state failure'],
          [(r['clock_rs'],r['cut_rs'],len(r['channels']),r['field'],r['value'],r['no_known_state_failure']) for r in spectral_gaps])
    table("Thermal reference on the finite domain",["setting","I_E","I_N","I_S","R"],
          [(s,r["energy"],r["number"],r["entropy"],r["R"]) for s,r in thermal["rates"].items()])
    table("Scattering controls",["setting","samples","max unitarity residual","max relative Jost residual"],
          [(s,r["samples"],r["unitarity_absolute_max"],r["Jost_relative_residual_max"]) for s,r in scattering.items()])
    table("Cut excess; channel one includes its weight 16",["T/r_s","cut/r_s","channels","route","nats","no known state failure"],
          [(r["clock_rs"],r["cut_rs"],len(r["channels"]),r["route"],r["value"],r["no_known_state_failure"]) for r in targets])
    table("Broad beta windows",["set","cut/r_s","start","end","delta S","delta E","beta","no known invalid spectra"],
          [(r["label"],r["cut_rs"],r["start_rs"],r["end_rs"],r["delta_entropy"],r["delta_energy"],r["beta"],r["known_spectra_valid"])
           for r in beta_rows if (r["start_rs"],r["end_rs"]) in chosen_windows[:3]])
    table("Six-r_s beta windows",['set','cut/r_s','start','end','beta','no known invalid spectra'],
          [(r['label'],r['cut_rs'],r['start_rs'],r['end_rs'],r['beta'],r['known_spectra_valid'])
           for r in beta_rows if r['end_rs']-r['start_rs']==6])
    table("Beta extrema over windows of width at least six",['set','cut/r_s','selection','count','minimum','min window','maximum','max window'],
          [(r['label'],r['cut_rs'],r['selection'],r['count'],
            r['minimum']['beta'] if r['minimum'] else None,
            str((r['minimum']['start_rs'],r['minimum']['end_rs'])) if r['minimum'] else None,
            r['maximum']['beta'] if r['maximum'] else None,
            str((r['maximum']['start_rs'],r['maximum']['end_rs'])) if r['maximum'] else None)
           for r in beta['extrema']])
    table("Primary weighted ledger",['channels','cut/r_s','start','end','second term','current','jumps minus vacuum','balance residual'],
          [(str(r['channels']),r['cut_rs'],r['clock_start_rs'],r['clock_end_rs'],
            r.get('second_term_D'),r.get('current'),r.get('jumps_minus_vacuum'),r.get('balance_residual'))
           for r in primary_ledger])
    def pair(row,key): return fmt(row["production_value"][key])+" / "+fmt(row["control_value"][key])
    table("Control sensitivities; flags list production / control; no histories averaged",
          ["control","T/r_s","cut/r_s","channels","difference/nats","available","spectra complete","no known state failure"],
          [(r["control"],r["clock_rs"],r["cut_rs"],len(r["channels"]),r["difference"],
            pair(r,"available"),pair(r,"saved_spectra_complete"),pair(r,"no_known_state_failure")) for r in spreads])
    (base/"tables.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"saved","table":str(base/"tables.md"),"data":str(base/"tables.json")}))


if __name__ == "__main__": main()
