#!/usr/bin/env python
"""reset_terms_readout_v1.py (solve builder 8, analyst; engine spec 49.2 (1)): the energy identity's reset terms per record on A's stored
records, a readout with no producer change and no rerun of A.

On A's chain (page_history_v6l's History and ChannelOps: the record's applied fields on the engine's sites, the vacuum exterior at E_k beyond,
the massless channel's clocked operator h_k = h(N_k, v_k), channels as given), per record k and channel: the dense eigendecomposition of h_k,
P_k = 1(h_k < 0), E_ref,k = sum of the negative eigenvalues = Tr[h_k P_k], and with the previous record's occupied columns
X_{k-1}: Tr[h_k P_{k-1}]. Then
    the reset term       Q_k  = Tr[h_k (P_{k-1} - P_k)] = Tr[h_k P_{k-1}] - E_ref,k        (>= 0: P_k minimizes Tr[h_k P] at its rank),
    the field-change term D_k = Tr[(h_k - h_{k-1}) P_{k-1}] = Tr[h_k P_{k-1}] - E_ref,k-1,
and E_ref,k - E_ref,k-1 = D_k - Q_k exactly; its one numerical input, Tr[h_{k-1} P_{k-1}] = E_ref,k-1, is checked per record by evaluating the
trace on the stored columns against the eigenvalue sum (the identity residual printed). The occupied-set
change is flagged two ways: the rank change n_neg,k - n_neg,k-1, and the stored engine masks (--spectra: the run's spectra npz, mask_kappa rows;
a record whose mask differs from the previous record's is flagged), where Q_k is first order (a reset proper); between records with one occupied
set it is second order in the record step (Hellmann-Feynman). Per checkpoint interval (every --checkpoint-every-rs): Sum Q_k, Sum D_k, the
change of E_ref, and beside them A's own row of that checkpoint read from --a-summary (W_acc_interval, the trapezoid, the identity residuals;
printed as A stored them, every key that exists).

Every per-record number is saved as it is computed (json rows, and the negative-eigenvalue spectra in an npz at the end) so later tables need no
rerun (COMMS rule 10); --resume continues from the json, recomputing record k-1's eigendecomposition once.
Usage: python reset_terms_readout_v1.py --record J --fields F [--spectra S] --nsites 8192 --channels 1,2 --T-end-rs 40 --out DIR --tag TAG
       [--a-summary A.json] [--device cuda:0] [--resume]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import page_ops_v6 as OPS  # noqa: E402
from page_history_v6l import History, ChannelOps, tr_blocked  # noqa: E402


def log(msg):
    print("%s %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--spectra", default=None)
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--channels", default="1,2")
    ap.add_argument("--T-end-rs", dest="T_end_rs", type=float, default=40.0)
    ap.add_argument("--checkpoint-every-rs", dest="ck_rs", type=float, default=2.0)
    ap.add_argument("--a-summary", dest="a_summary", default=None)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    H = History(a.record, a.fields, a.nsites, fields_key="applied")
    rs = H.rs_nom
    ks = [k for k in range(len(H)) if H.T[k] <= a.T_end_rs * rs + 1e-9]
    kap = [int(x) for x in a.channels.split(",")]
    ops = {k_: ChannelOps(k_, H.R, dev) for k_ in kap}
    S = np.load(a.spectra) if a.spectra else None
    TS = np.asarray(S["T"], float) if S is not None else None

    def stored_mask(T, k_):
        if S is None or ("mask_%d" % k_) not in S.files:
            return None
        i = int(np.argmin(np.abs(TS - T)))
        return None if abs(TS[i] - T) > 1e-9 else np.asarray(S["mask_%d" % k_][i])

    jpath = os.path.join(a.out, "reset_terms_%s.json" % a.tag)
    out = {"script": "reset_terms_readout_v1.py", "args": vars(a), "label": "the reset terms Q_k = Tr[h_k (P_{k-1} - P_k)] per record on A's stored "
           "records (the records' applied fields, A's chain and channels); a readout, no producer change", "rows": [], "checkpoints": []}
    done = set()
    if a.resume and os.path.exists(jpath):
        out = json.load(open(jpath)); done = {r["k"] for r in out["rows"]}
        log("resume: %d records done" % len(done))
    prev = {}
    spectra = {k_: [] for k_ in kap}
    t0 = time.time()
    for idx, k in enumerate(ks):
        if k in done and (idx + 1 < len(ks) and ks[idx + 1] in done):
            continue
        N, v = H.fields(k)
        row = {"k": k, "T_over_rs": float(H.T[k]) / rs, "E": float(H.E[k]), "per_channel": {}}
        for k_ in kap:
            c = ops[k_]
            h = c.h_csr(N, v); Bh = c.banded(h)
            e, V = OPS.eigh_csr(h, dev)
            neg = e < 0; nneg = int(neg.sum())
            E_ref = float(e[neg].sum())
            pc = {"n_neg": nneg, "E_ref": E_ref}
            m = stored_mask(float(H.T[k]), k_)
            if k_ in prev:
                Xp, E_ref_p, nneg_p, m_p, Bh_p = prev[k_]
                tr_hk_Pp = tr_blocked(Bh, Xp)
                tr_hp_Pp = tr_blocked(Bh_p, Xp)
                pc.update({"tr_hk_Pprev": tr_hk_Pp, "tr_hprev_Pprev": tr_hp_Pp, "Q": tr_hk_Pp - E_ref, "D": tr_hk_Pp - E_ref_p,
                           "identity_residual": tr_hp_Pp - E_ref_p,
                           "rank_change": nneg - nneg_p,
                           "stored_mask_changed": (None if (m is None or m_p is None) else bool(not np.array_equal(m, m_p)))})
                # the identity's residual: Tr[h_{k-1} P_{k-1}] evaluated as a trace on the stored columns against the eigenvalue sum E_ref,k-1
            Xn = V[:, neg].contiguous()
            del V
            prev[k_] = (Xn, E_ref, nneg, m, Bh)
            spectra[k_].append(e[neg].cpu().numpy())
            row["per_channel"][str(k_)] = pc
            del h, e
            torch.cuda.empty_cache() if dev.type == "cuda" else None
        w = {k_: 16.0 * k_ for k_ in kap}
        if all("Q" in row["per_channel"][str(k_)] for k_ in kap):
            row["Q_weighted"] = sum(w[k_] * row["per_channel"][str(k_)]["Q"] for k_ in kap)
            row["D_weighted"] = sum(w[k_] * row["per_channel"][str(k_)]["D"] for k_ in kap)
            row["flag"] = any((row["per_channel"][str(k_)].get("rank_change") or 0) != 0 or row["per_channel"][str(k_)].get("stored_mask_changed")
                              for k_ in kap)
        if k not in done:
            out["rows"].append(row)
        if idx % 4 == 0 or idx == len(ks) - 1:
            log("T/r_s %.3f: %s (%.0f s)" % (row["T_over_rs"], "; ".join("kappa %d Q %s D %s n_neg %d" % (
                k_, ("%.3e" % row["per_channel"][str(k_)]["Q"]) if "Q" in row["per_channel"][str(k_)] else "-",
                ("%.3e" % row["per_channel"][str(k_)]["D"]) if "D" in row["per_channel"][str(k_)] else "-",
                row["per_channel"][str(k_)]["n_neg"]) for k_ in kap), time.time() - t0))
            json.dump(out, open(jpath + ".tmp", "w"), default=float); os.replace(jpath + ".tmp", jpath)
    # checkpoint intervals
    rows = sorted(out["rows"], key=lambda r: r["k"])
    A = None
    if a.a_summary and os.path.exists(a.a_summary):
        A = json.load(open(a.a_summary))
        A = A.get("rows") or A.get("checkpoints") or []
    step = a.ck_rs
    cks = []
    t_edges = np.arange(0.0, a.T_end_rs + 1e-9, step)
    for t_lo, t_hi in zip(t_edges[:-1], t_edges[1:]):
        rr = [r for r in rows if t_lo + 1e-9 < r["T_over_rs"] <= t_hi + 1e-9 and "Q_weighted" in r]
        if not rr:
            continue
        ck = {"T_lo_over_rs": float(t_lo), "T_hi_over_rs": float(t_hi), "records": len(rr), "flagged_records": [r["T_over_rs"] for r in rr if r.get("flag")],
              "sum_Q_weighted": sum(r["Q_weighted"] for r in rr), "sum_D_weighted": sum(r["D_weighted"] for r in rr),
              "sum_Q_flagged_weighted": sum(r["Q_weighted"] for r in rr if r.get("flag")),
              "max_identity_residual": max(abs(r["per_channel"][c]["identity_residual"]) for r in rr for c in r["per_channel"]),
              "per_channel": {c: {"sum_Q": sum(r["per_channel"][c]["Q"] for r in rr), "sum_D": sum(r["per_channel"][c]["D"] for r in rr)} for c in rr[0]["per_channel"]}}
        if A:
            m = [x for x in A if abs(float(x.get("t_over_rs", -1)) - t_hi) < 1e-6]
            if m:
                ck["A_row"] = {kk: m[0][kk] for kk in m[0] if any(s in kk for s in ("W_acc", "trapezoid", "W_hdot", "identity_residual", "jump", "E_ref", "W_creation"))}
        cks.append(ck)
    out["checkpoints"] = cks
    json.dump(out, open(jpath + ".tmp", "w"), default=float); os.replace(jpath + ".tmp", jpath)
    np.savez(os.path.join(a.out, "reset_terms_%s_negspectra.npz" % a.tag), **{"neg_e_%d_%03d" % (k_, i): s for k_ in kap for i, s in enumerate(spectra[k_])})
    with open(os.path.join(a.out, "reset_terms_%s.md" % a.tag), "w") as f:
        f.write("# The energy identity's reset terms per record on A's stored records (engine spec 49.2 (1); reset_terms_readout_v1.py)\n\n")
        f.write("Weighted sums (16 kappa) per checkpoint interval; flagged = a rank change or a stored-mask change at the record. Every number is the script's.\n\n")
        f.write("| interval T/r_s | records | flagged records | Sum Q | Sum Q on flagged | Sum D | max identity residual | A's row (as stored) |\n|---|---|---|---|---|---|---|---|\n")
        for ck in cks:
            f.write("| %.1f to %.1f | %d | %s | %.6e | %.6e | %.6e | %.2e | %s |\n" % (ck["T_lo_over_rs"], ck["T_hi_over_rs"], ck["records"],
                    ", ".join("%.3f" % x for x in ck["flagged_records"]) or "none", ck["sum_Q_weighted"], ck["sum_Q_flagged_weighted"], ck["sum_D_weighted"],
                    ck["max_identity_residual"], json.dumps(ck.get("A_row"), default=float) if ck.get("A_row") else "-"))
    log("done: %d records, %d intervals" % (len(rows), len(cks)))


if __name__ == "__main__":
    main()
