#!/usr/bin/env python3
"""Cold reading of W2_readout_v1, the second inspection: the anchor epoch against v36l key by key, the exact crossings, the late start bins, the pre-12.5 moves (box, read-only)."""
import json
B="<internal>/lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p"
A=json.load(open(B+"_anc/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_anc.json"))
L=json.load(open("<internal>/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f.json"))
Pj=json.load(open(B+"_v36v_anc/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc.json"))
E0=A["head"]["E0"]
MA={round(r["T_over_rs"],6):r for r in A["records"]}; ML={round(r["T_over_rs"],6):r for r in L["records"]}; MP={round(r["T_over_rs"],6):r for r in Pj["records"]}
def canon(o): return json.dumps(o,sort_keys=True,separators=(",",":"))
print("## anc vs v36l epoch: which dg_v36f sub-keys differ, and are the bin quantities equal?")
import collections
cnt=collections.Counter(); 
for T in sorted(k for k in MA if 12.625-1e-9<=k<=14.5+1e-9):
    da=MA[T]["dg_v36f"]; dl=ML[T]["dg_v36f"]
    ks=sorted(set(da)|set(dl)); diff=[k for k in ks if canon(da.get(k))!=canon(dl.get(k))]
    for k in diff: cnt[k]+=1
    same=[k for k in ("E_rec","E_start","E_end","E_pred","r_chain_over_E_rec","switch_sum_over_E_rec","n_switch","rest_over_E_rec","level","nsub","attempts","iterations","converged","tracked_set_changes","node_E") if canon(da.get(k))==canon(dl.get(k))]
    notsame=[k for k in ("E_rec","E_start","E_end","E_pred","r_chain_over_E_rec","switch_sum_over_E_rec","n_switch","rest_over_E_rec","level","nsub","attempts","iterations","converged","tracked_set_changes","node_E") if canon(da.get(k))!=canon(dl.get(k))]
    print(" %.3f differing dg keys: %s | bin keys NOT equal: %s" % (T, diff, notsame))
print(" counts:", dict(cnt))
# show an example of a differing non-timing key
T=12.625; da=MA[T]["dg_v36f"]; dl=ML[T]["dg_v36f"]
for k in ("history","substeps","attempt_list","continuation_nodes_final"):
    if canon(da.get(k))!=canon(dl.get(k)):
        if isinstance(da.get(k),list) and da.get(k) and isinstance(da[k][0],dict):
            d0=da[k][0]; l0=dl[k][0]; kk=[x for x in sorted(set(d0)|set(l0)) if canon(d0.get(x))!=canon(l0.get(x))]
            print("  12.625 %s[0] differing sub-keys: %s" % (k, kk))
            for x in kk[:12]:
                print("     %s: anc %s | v36l %s" % (x, repr(d0.get(x))[:120], repr(l0.get(x))[:120]))
        else:
            print("  12.625 %s differs: anc %s | v36l %s" % (k, repr(da.get(k))[:200], repr(dl.get(k))[:200]))
for k in ("M1_balance","stress","v36","engine_v34","label","slice_saved"):
    a_=MA[T].get(k); l_=ML[T].get(k)
    if isinstance(a_,dict) and isinstance(l_,dict):
        kk=[x for x in sorted(set(a_)|set(l_)) if canon(a_.get(x))!=canon(l_.get(x))]
        print("  top-level %s differing sub-keys: %s" % (k, kk[:15]))
        for x in kk[:4]: print("     %s: anc %s | v36l %s" % (x, repr(a_.get(x))[:140], repr(l_.get(x))[:140]))
    else:
        print("  top-level %s: anc %s | v36l %s" % (k, repr(a_)[:160], repr(l_)[:160]))
# the fields: compare the stored field-like arrays if any (E_dust_shells, H_in, N_min, R_t, min_N_minus_absv, n_pocket, trapping_zeros)
for T in (12.625, 13.625, 14.5):
    print(" %.3f equal: E %s E_dust %s H_in %s R_t %s minN %s n_pocket %s zeros %s iterations %s converged %s" % (T, MA[T]["E"]==ML[T]["E"], MA[T]["E_dust_shells"]==ML[T]["E_dust_shells"], MA[T]["H_in"]==ML[T]["H_in"], MA[T]["R_t"]==ML[T]["R_t"], MA[T]["min_N_minus_absv"]==ML[T]["min_N_minus_absv"], MA[T]["n_pocket"]==ML[T]["n_pocket"], MA[T]["trapping_zeros"]==ML[T]["trapping_zeros"], MA[T]["iterations"]==ML[T]["iterations"], MA[T]["converged"]==ML[T]["converged"]))
print("\n## the exact (gap 0) crossings: flags and the sub-step's prim flags")
for T,si in ((23.25,0),(23.375,1)):
    s=MP[T]["dg_v36f"]["substeps"][si]; c=s["prim"]["verify"]["crossings"]
    for x in c:
        print(" %.3f sub %d: ch %s %s->%s dec %s gap_min %s gap_min2 %s jump_fit %s jump_grid %s jump_prev %s falls %s flags %s fit_ok %s fit_why %s tie %s knife %s mirrored %s s_star %s" % (T, s["sub"], x["channel"], x["i"], x["n"], x["decision"], x.get("gap_min"), x.get("gap_min2"), x.get("jump_fit"), x.get("jump_grid"), x.get("jump_prev"), x.get("falls"), x.get("flags"), x.get("fit_ok"), x.get("fit_why"), x.get("tie"), x.get("knife"), x.get("mirrored"), x.get("s_star")))
    p=s["prim"]; print("   sub prim: stable %s held %s jumps_fall %s printable %s level_stable %s n_crossings %s n_diabatic %s n_adiabatic %s n_unresolved %s revert %s" % (p["stable"],p["held"],p["jumps_fall"],p["printable"],p["level_stable"],p["n_crossings"],p["n_diabatic"],p["n_adiabatic"],p["n_unresolved"],p.get("revert")))
print("\n## late records with start bins above 1e-9: the bins")
for T in (33.75, 38.25, 38.375, 38.5, 38.625, 38.75, 38.875):
    r=MP[T]; d=r["dg_v36f"]; prev=MP[round(T-0.125,6)]["E"]
    print(" %.3f dE %+.3e start %+.3e chord %+.3e rec %+.3e lvl %s att %s it %s conv %s" % (T,(r["E"]-prev)/E0,(d["E_start"]-d["E_rec"])/E0,(d["E_end"]-d["E_start"])/E0,(r["E"]-d["E_end"])/E0,d["level"],d["attempts"],r["iterations"],r["converged"]))
print("\n## the pre-12.5 v36f records: largest |dE| before 12.375, and pockets at 12.25/12.375/12.5")
prevE=None; big=[]
for T in sorted(k for k in MP if k<=12.5+1e-9):
    r=MP[T]
    if prevE is not None: big.append((abs(r["E"]-prevE)/E0,T))
    prevE=r["E"]
big.sort(reverse=True); print(" top |dE| :", [("%.3f"%t,"%.2e"%v) for v,t in big[:5]])
print(" pockets 12.25/12.375/12.5:", [sum(v for v in (MP[t]["n_pocket"] or {}).values() if v is not None) for t in (12.25,12.375,12.5)])
