#!/usr/bin/env python
"""s16A_gate_corr_v1.py (busi; inbox #503 (A)): emits PAGE_Bprime_s16A_gate_v1_correction_20261023.md whole, the dated correction section
to PAGE_Bprime_s16A_gate_v1.md (issued records are not edited). Recomputes the residual numbers it quotes from the md5-checked clock and 2b arms."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s16A_end_lib_v2 as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--gate", required=True, help="the issued PAGE_Bprime_s16A_gate_v1.md (md5 recorded)")
    ap.add_argument("--prov", required=True, help="PAGE_Bprime_s16A_provenance_v1.md (md5 recorded)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K1 = E.load_run(os.path.join(a.bp, "s16A_qh1_v1"), "s16A_1_F1_d0_dt0.5_T100_both")
    B2 = E.load_run(os.path.join(a.bp, "s16A_qh1_v1"), "s16A_2b_F1_onset225_dt0.5_T100_both")
    p1 = E.eigp(E.G_at(K1, E.T3)); p2 = E.eigp(E.G_at(B2, E.T3))
    r1 = float((1 - p1).sum()); r2 = float((1 - p2).sum()); d = r2 - r1
    out = []; P = out.append
    P("# PAGE B' s = 16, stage A gate: dated correction section, 23 Oct 2026 (busi lane)")
    P("")
    P("Emitted whole by busi/code/pagetests/v3/s16A_gate_corr_v1.py; numbers recomputed from the md5-checked arms, none retyped. This is the dated correction section to PAGE_Bprime_s16A_gate_v1.md (md5 %s), which stays as issued and remains the technical acceptance record (inbox #503 (A); ReviewA_gatex_v1, ReviewB_gatex_v1). NOTHING PRINTS. Table A and Note A are NOT added here: they wait for second_route's second route of gate items 1, 2 and 4." % E.md5(a.gate)[:8])
    P("")
    P("## Correction 1 (to sec. 6, 'Correction (f)', and sec. 9): boundary dependence, not boundary-set")
    P("")
    P("Issued text: 'the late residual is set by the sink boundary'. Corrected text: **the late residual shows boundary DEPENDENCE.** At 1e2 L, at equal dt (0.5 a), %.3f of the onset-2 L joint residual Sum(1 - p) = %.3f is onset-induced (the onset-2.25 L residual is %.3f): boundary dependence of the sink2L model, not the whole residual. The 2 to 2.25 L shell itself is unmeasured. No retention statement rests on the sink2L residual." % (d, r1, r2))
    P("")
    P("## Correction 2 (to sec. 1, prep wording)")
    P("")
    P("Issued text: 'prep npz md5 recorded in both: 876dd5c6 and 876dd5c6'. Corrected: 876dd5c6 is the md5 of the SOURCE RECORD (P64_ref_c19r11) that the prep code writes into every prep manifest's npz_md5 field, not the prep's md5. The prep every stage A arm loaded is kit_s16A_v2/prep/F1_s16_d0_prep.npz, md5 b97c1f87 (2b: its onset prep f7e9cf93), as shown in PAGE_Bprime_s16A_provenance_v1.md (md5 %s). No propagated number changes." % E.md5(a.prov)[:8])
    P("")
    P("Inputs: %s/%s.npz (md5 %s), %s/%s.npz (md5 %s); this script (md5 %s)." % (K1["dir"], K1["tag"], K1["md5_npz"][:8], B2["dir"], B2["tag"], B2["md5_npz"][:8], E.md5(os.path.abspath(__file__))[:8]))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
