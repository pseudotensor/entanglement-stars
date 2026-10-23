"""s16A_end_lib_v2.py (busi): additions to s16A_end_lib_v1 (unchanged, re-exported) for the stage A gate package (inbox #488):
singular-value multiplicity, the antiunitary mirror check from the prep bases, subset location by energy, quantiles. Never edit after first use."""
import numpy as np

from s16A_end_lib_v1 import *  # noqa: F401,F403
import s16A_end_lib_v1 as E1


def top_sv_mult(M, rel=1e-6):
    """Top singular value, its multiplicity (values within rel of it) and the next distinct value."""
    sv = np.linalg.svd(M, compute_uv=False)
    top = sv[0]
    m = int((np.abs(sv - top) <= rel * top).sum())
    nxt = float(sv[m]) if m < len(sv) else float("nan")
    return float(top), m, nxt, sv


def mirror_C(prep_npz):
    """C = (J F_s)^dag F_e with J psi = (-psi3*, psi2*, psi1*, -psi0*) on the component-major U_sub of the prep (second_route's antiunitary)."""
    z = np.load(prep_npz)
    U = z["U_sub"]; ns = int(z["n_supply"]); n = U.shape[0] // 4
    P = U.reshape(4, n, -1)
    J = np.stack([-P[3].conj(), P[2].conj(), P[1].conj(), -P[0].conj()]).reshape(4 * n, -1)
    C = J[:, :ns].conj().T @ U[:, ns:]
    return C, float(np.max(np.abs(C.conj().T @ C - np.eye(C.shape[1])))), z


def mirror_residuals(G, C, ns):
    """Block identity G_ee = C^dag conj(G_ss) C and the cross-block identity conj(G_se) = C G_se^dag conj(C) (J^2 = +1)."""
    Gss, Gee, Gse = G[:ns, :ns], G[ns:, ns:], G[:ns, ns:]
    r_blk = float(np.max(np.abs(Gee - C.conj().T @ Gss.conj() @ C)))
    r_x = float(np.max(np.abs(Gse.conj() - C @ Gse.conj().T @ C.conj())))
    return r_blk, r_x


def locate(e_all, e_sub):
    idx = []; used = set()
    for x in e_sub:
        cand = [i for i in np.argsort(np.abs(e_all - x)) if i not in used]
        idx.append(cand[0]); used.add(cand[0])
    idx = np.array(idx)
    return idx, float(np.max(np.abs(e_all[idx] - e_sub)))


def q(x, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    return [float(np.quantile(np.asarray(x), t)) for t in qs]
