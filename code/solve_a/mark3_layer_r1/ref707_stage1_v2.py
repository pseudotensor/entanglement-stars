# ref707_stage1_v2 (= v1 7a92560c with ONLY the V0 roundoff tolerance of ADDENDUM_707_ref_v1a; #707 REF screening, stage 1; frozen ADDENDUM_707_ref_v1.md md5 87ef1164; 2026-10-31): READ-ONLY, box CPU.
#   python ref707_stage1_v1.py <out prefix> <member label> <fields.npz> <tau4 fields.npz|none> <slice.npz> [<slice.npz> ...]
# Held sources (engine_collapse_v19 sources(), l.1196-1204) vs paper eq:dm-reference sources on stored complex128 slices:
#   eps_held = sum_k w [dens(h0,sea) - dens(h0,R2) + dens(h0,dust,f)];  eps_dm = sum_k w [dens(h0,sea) - dens(h0,P*) + dens(h0,dust,f) - (dW2_N(u) - dW2_N(u*))]
#   (J likewise with Q and dW2_v); P* = negative band of the clocked chain at (N_cinit, v_cinit) (engine_collapse_v19 l.1002-1007).
# Gates V0 (chain = W2 module chain), V1 (held = stored eps_tot/J_tot), V2 (re-solve held = stored fields). Rows under dm by re-solve.
import os, sys, json, time, hashlib
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
ENG = '<internal>/code'
ENG_MD5 = {'engine_collapse_v19.py': '9e4e6abf4f3a23d97ac8ba79dda85785', 'engine_collapse_v5.py': 'fc6faf26a353f6d904a5242874e25586',
           'engine_fields_v14.py': 'dc864dfeafbf233f5669b961a51cb7fc', 'engine_vacflow_v1.py': '5225e959b9e0bc730062041d60d800c8'}
W2DIR = '<internal>/code'; W2MD5 = '6f9ab6853241a38f73314e132f318ffd'
T0 = time.time()


def md5f(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


for fn, m_ in ENG_MD5.items(): assert md5f(os.path.join(ENG, fn)) == m_, 'engine md5 %s' % fn
assert md5f(os.path.join(W2DIR, 'vaclaw_w2_v3.py')) == W2MD5
sys.path.insert(0, ENG); sys.path.insert(1, W2DIR)
import engine_collapse_v5 as C5
C5.backend(None)
import engine_collapse_v19 as V19
import engine_fields_v14 as FLD
import vaclaw_w2_v3 as W2
import m3l_lib_v2 as L


def log(s): print('%6.0fs %s' % (time.time() - T0, s), flush=True)


def main():
    prefix, member, fpath, t4path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]; slices = sys.argv[5:]
    Z = np.load(fpath); R = np.asarray(Z['R'], float); ns = len(R); Tz = np.asarray(Z['T'], float)
    g = float(Z['g']); ell = int(Z['ell']); sol = json.loads(str(Z['solver']))
    V19.OPT['collar_policy'] = sol.get('collar_policy', 'flag'); V19.OPT['support_rel'] = sol.get('support_rel', 1e-14); V19.OPT['site_fields'] = sol.get('site_fields', 'averaged')
    Ncs, vcs = np.asarray(Z['N_cinit'], float), np.asarray(Z['v_cinit'], float)
    m = 0.2
    s0 = np.load(slices[0]); kappas = [int(k) for k in s0['kappas']]; wts = [float(w) for w in s0['weights']]; rs = float(s0['rs'])
    chs = {k: C5.Chan(k, m, ns) for k in kappas}
    out = dict(member=member, fields=fpath, fields_md5=md5f(fpath), R_is_n_plus_1=bool(np.allclose(R, np.arange(1, ns + 1))), kappas=kappas, weights=wts, rs=rs,
               solver=sol, V0={}, slices={})
    # V0
    for k in kappas:
        hs = chs[k].hs.toarray(); hb = L.band_to_dense(L.heavy_band(ns, k, m, np.ones(ns), np.zeros(ns)))
        Qs = chs[k].Qs.toarray(); Qb = L.band_to_dense(L.heavy_band(ns, 0, 0.0, np.zeros(ns), np.ones(ns)))
        out['V0'][str(k)] = dict(h_max_diff=float(np.abs(hs - hb).max()), Q_max_diff=float(np.abs(Qs - Qb).max()))
    w2ok = out['R_is_n_plus_1'] and all(v['h_max_diff'] <= 1e-14 and v['Q_max_diff'] <= 1e-14 for v in out['V0'].values())
    out['V0_pass'] = bool(w2ok); log('V0 %s' % json.dumps(out['V0']))
    # P* per channel
    Pst = {}
    for k in kappas:
        e, U = C5.eigT(chs[k].clocked_dense(Ncs, vcs)); Pst[k] = U[:, e < 0].contiguous(); del U
    log('P* built; n_neg %s' % [int(Pst[k].shape[1]) for k in kappas])
    dWs = {k: W2.dW2(Ncs, vcs, k=float(k), m=m) for k in kappas}
    # spread file
    T4 = np.load(t4path) if t4path != 'none' else None
    for sp_ in slices:
        T_ = time.time(); S = np.load(sp_); Tsl = float(S['T']); i = int(np.argmin(np.abs(Tz - Tsl))); lab = '%.2f' % (Tsl / rs)
        Ni, vi = np.asarray(Z['N'][i], float), np.asarray(Z['v'][i], float); Na, va = np.asarray(Z['N_applied'][i], float), np.asarray(Z['v_applied'][i], float)
        eh = np.zeros(ns); Jh = np.zeros(ns); dP_e = np.zeros(ns); dP_J = np.zeros(ns); dW_e = np.zeros(ns); dW_J = np.zeros(ns); dWa_e = np.zeros(ns); dWa_J = np.zeros(ns)
        for k, w in zip([int(x) for x in S['kappas']], [float(x) for x in S['weights']]):
            ch = chs[k]; T_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.complex128)
            sea = T_t(S['sea_%d' % k]); R2 = T_t(S['R2_%d' % k]); du = T_t(S['dust_%d' % k]); f = np.asarray(S['f_%d' % k], float)
            hs, hR, hd = C5.dens(ch.h, sea), C5.dens(ch.h, R2), C5.dens(ch.h, du, f)
            Qs_, QR, Qd = C5.dens(ch.Q, sea), C5.dens(ch.Q, R2), C5.dens(ch.Q, du, f)
            hP, QP = C5.dens(ch.h, Pst[k]), C5.dens(ch.Q, Pst[k])
            eh += w * (hs - hR + hd); Jh += w * (Qs_ - QR + Qd)
            dP_e += w * (hR - hP); dP_J += w * (QR - QP)
            gN, gv = W2.dW2(Ni, vi, k=float(k), m=m); gNa, gva = W2.dW2(Na, va, k=float(k), m=m); sN, sv = dWs[k]
            dW_e -= w * (gN - sN); dW_J -= w * (gv - sv); dWa_e -= w * (gNa - sN); dWa_J -= w * (gva - sv)
            del sea, R2, du
        es, Js = np.asarray(Z['eps_tot'][i], float), np.asarray(Z['J_tot'][i], float); se, sJ = float(np.abs(es).max()), float(np.abs(Js).max())
        V1 = dict(eps=float(np.abs(eh - es).max() / se), J=float(np.abs(Jh - Js).max() / sJ))
        edm, Jdm = eh + dP_e + (dW_e if w2ok else 0.0), Jh + dP_J + (dW_J if w2ok else 0.0)
        Nh, vh, _, ih = V19.solve(eh, Jh, g, R, ell, Ni); Nd, vd, _, idm = V19.solve(edm, Jdm, g, R, ell, Ni)
        V2 = dict(N=float(np.abs(Nh - Ni).max()), v=float(np.abs(vh - vi).max()))
        def nrm(a): return dict(max=float(np.abs(a).max()), L2=float(np.sqrt(np.sum(a * a))))
        rec = dict(T=Tsl, T_over_rs=Tsl / rs, record=i, T_record=float(Tz[i]), V1=V1, V1_pass=bool(V1['eps'] <= 1e-10 and V1['J'] <= 1e-10), V2=V2, V2_pass=bool(max(V2.values()) <= 1e-9),
                   scale=dict(eps=se, J=sJ), dS_eps=nrm(edm - eh), dS_J=nrm(Jdm - Jh), dS_eps_Pstar=nrm(dP_e), dS_J_Pstar=nrm(dP_J), dS_eps_W2=nrm(dW_e), dS_J_W2=nrm(dW_J),
                   dS_eps_W2_applied_variant=nrm(dWa_e), dS_J_W2_applied_variant=nrm(dWa_J),
                   rows_dm=dict(dN_max=float(np.abs(Nd - Ni).max()), dv_max=float(np.abs(vd - vi).max()), dN_vs_heldresolve=float(np.abs(Nd - Nh).max()), dv_vs_heldresolve=float(np.abs(vd - vh).max()),
                                E_held=float(FLD.charge(ih, g)), E_dm=float(FLD.charge(idm, g)), residual_W=idm.get('residual_W'), residual_invN=idm.get('residual_invN'), failed=idm.get('failed')))
        if T4 is not None:
            T4z = np.asarray(T4['T'], float); j = int(np.argmin(np.abs(T4z - Tsl)))
            rec['spread'] = dict(T4=float(T4z[j]), dT=float(T4z[j] - Tsl), S_eps=float(np.abs(np.asarray(T4['eps_tot'][j]) - es).max()), S_J=float(np.abs(np.asarray(T4['J_tot'][j]) - Js).max()),
                                 S_N=float(np.abs(np.asarray(T4['N'][j]) - Ni).max()), S_v=float(np.abs(np.asarray(T4['v'][j]) - vi).max()))
        out['slices'][lab] = rec
        json.dump(out, open('%s_%s.json' % (prefix, member), 'w'), indent=1, default=float)
        np.savez_compressed('%s_%s_T%s.npz' % (prefix, member, lab), eps_held=eh, J_held=Jh, eps_dm=edm, J_dm=Jdm, dP_e=dP_e, dP_J=dP_J, dW_e=dW_e, dW_J=dW_J, N_dm=Nd, v_dm=vd)
        log('%s T/rs %s rec %d V1 %s V2 %s dS_eps %s dS_J %s rows %s spread %s (%.0fs)' % (member, lab, i, V1, V2, rec['dS_eps'], rec['dS_J'], rec['rows_dm'], rec.get('spread'), time.time() - T_))
    log('DONE')


if __name__ == '__main__':
    main()
