"""Second reader: independent audit of the three busi-run filtered intake + readout jobs (n=3, g=0,1/2,2)
against icloud's accepted n=3, g=1 intake/readout. Read-only. Output JSON to argv[1]."""
import json, hashlib, sys, os
U = "<internal>/upper"
IC = "<internal>/MOCK_filtered_comparison_readouts_v1"
PHYS = "<internal>/MOCK_filtered_physical_v1"

def sha(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def recs(p):
    out = []
    with open(p) as f:
        for l in f:
            l = l.strip()
            if l: out.append(json.loads(l))
    return out
def base(d): return {os.path.basename(k): v for k, v in d.items()}

ref_i = recs(os.path.join(IC, "intake_even_n3_g1_v1", "written_report.md"))
ref_r = recs(os.path.join(IC, "even_n3_g1_v1", "written_report.md"))
def kd(rs): return {r["kind"]: r["data"] for r in rs}
RI = kd(ref_i); RR = kd(ref_r)
res = {"reference": {"intake_md5": sha(os.path.join(IC, "intake_even_n3_g1_v1", "written_report.md"), "md5"),
                     "readout_md5": sha(os.path.join(IC, "even_n3_g1_v1", "written_report.md"), "md5")}}
for g, gl in (("0", 0.0), ("0p5", 0.5), ("2", 2.0)):
    ip = os.path.join(U, "intake_even_n3_g%s_v1" % g, "written_report.md")
    jp = os.path.join(U, "intake_even_n3_g%s_v1" % g, "records.ndjson")
    rp = os.path.join(U, "even_n3_g%s_v1" % g, "written_report.md")
    I = recs(ip); K = kd(I); R = recs(rp); KR = kd(R)
    t = K["termination"]; c = K["filtered_saved_intake_complete"]
    parent = c["parent_journal"]
    roster = c["actual_action_roster"]
    r = {
        "intake_md5": sha(ip, "md5"), "intake_journal_md5": sha(jp, "md5"),
        "intake_sha256": sha(ip), "journal_sha256": sha(jp),
        "term_ok": t["status"] == "complete" and t["error"] is None and t["all_pass"] is True and t["sources_unchanged"] is True and t.get("runtime_admitted") is True and t.get("threads") == 1,
        "terminal_last": I[-1]["kind"] == "termination",
        "kind_roster_equal_ref": [x["kind"] for x in I] == [x["kind"] for x in ref_i],
        "intake_sources_eq_term_sources": K["intake"]["sources"] == t["sources"],
        "sources_eq_ref_basename_sha": base(t["sources"]) == base(RI["termination"]["sources"]),
        "n_sources": len(t["sources"]),
        "adapter_controls_eq_ref": K["accepted_filtered_adapter_controls"] == RI["accepted_filtered_adapter_controls"],
        "n_g": [c["n"], c["g"]] == [3, gl],
        "parent_journal": parent,
        "parent_sha_recorded_eq_now": c["parent_journal_sha256"] == sha(parent),
        "roster_34": len(roster) == 34 and sorted(roster) == sorted([[k, j / 4] for k in ("pure", "registered") for j in range(17)]),
        "excluded_256": len(c["excluded_wall_actions"]) == 256,
        "locators_eq_ref_count": len(c["locators"]) == len(RI["filtered_saved_intake_complete"]["locators"]),
        "physical_sources_eq_ref": base(c["actual_sources"]) == base(RI["filtered_saved_intake_complete"]["actual_sources"]),
        "n_physical_sources": len(c["actual_sources"]),
        "readout_md5": sha(rp, "md5"),
        "readout_term": {k: KR["termination"].get(k) for k in ("status", "error", "all_pass", "sources_unchanged")},
        "readout_kind_roster_eq_ref": [x["kind"] for x in R] == [x["kind"] for x in ref_r],
        "readout_sources_eq_ref": base(KR["termination"]["sources"]) == base(RR["termination"]["sources"]),
        "readout_saved_intake_binds_this_intake": KR["accepted_filtered_saved_intake"].get("journal_sha256") == sha(jp),
        "adapter_source_md5": sha(os.path.join(IC, "source_v1", "run_mock_filtered_comparison_readouts_v1.py"), "md5"),
    }
    res["n3_g" + g] = r
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
