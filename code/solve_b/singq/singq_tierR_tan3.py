# singq_tierR_tan3.py -- BATCHED versions of singq_tierR_tan2 (a8528e92): tan_pass_b / maj_pass_b process K columns at once (arrays shaped (n, K));
# arithmetic per column identical to the single-column passes. Used by singq_vinv_v2.
import numpy as np
import numba as nb


@nb.njit(cache=True)
def tan_pass_b(drl, dJl, y, hd, M, w, sp_, lam, Tp, tail, MI, G, dtail):
    n, K = drl.shape; I = n - 1
    dM = np.zeros((n, K))
    for i in range(I):
        a4i = 4.0 * np.pi * y[i] * y[i]; a4n = 4.0 * np.pi * y[i + 1] * y[i + 1]
        c0 = 1.0 + 0.5 * hd * sp_[i]; den = 1.0 - 0.5 * hd * sp_[i + 1]
        for k in range(K):
            num = c0 * dM[i, k] + 0.5 * hd * a4i * (drl[i, k] + w[i] * dJl[i, k]) + 0.5 * hd * a4n * (drl[i + 1, k] + w[i + 1] * dJl[i + 1, k])
            dM[i + 1, k] = num / den
    dw = np.zeros((n, K)); dsp = np.zeros((n, K))
    for j in range(1, n):
        if M[j] > 0:
            a4 = 4.0 * np.pi * y[j] * y[j]
            for k in range(K):
                dw[j, k] = w[j] * dM[j, k] / (2.0 * M[j])
                dsp[j, k] = a4 * w[j] * dJl[j, k] / (2.0 * M[j]) - sp_[j] * dM[j, k] / (2.0 * M[j])
    dlam = np.zeros((n + 1, K))
    denI = 1.0 - 0.5 * hd * sp_[I]
    for k in range(K):
        dTp = 0.0
        if MI > 0:
            dTp = -np.sqrt(2.0 * G) / 2.0 * (dtail[k] / np.sqrt(MI) - tail * dM[I, k] / (2.0 * MI ** 1.5))
        dlam[I, k] = dTp / denI + (1.0 + Tp) * 0.5 * hd * dsp[I, k] / (denI * denI)
    for j in range(I - 1, 0, -1):
        den = 1.0 - 0.5 * hd * sp_[j]; rho = (1.0 + 0.5 * hd * sp_[j]) / den; c1 = lam[j + 1] * hd / (den * den)
        for k in range(K):
            dlam[j, k] = dlam[j + 1, k] * rho + c1 * dsp[j, k]
    dN = np.zeros((n, K))
    for i in range(1, I):
        for k in range(K):
            dN[i, k] = 0.5 * (dlam[i, k] + dlam[i + 1, k])
    for k in range(K):
        dN[I, k] = dlam[I, k]
        dN[0, k] = (4.0 * dN[1, k] - dN[2, k]) / 3.0
    return dM, dN, dw


@nb.njit(cache=True)
def maj_pass_b(drl, dJl, y, hd, M, w, sp_, lam, Tp, tail, MI, G, dtail):
    n, K = drl.shape; I = n - 1
    dM = np.zeros((n, K))
    for i in range(I):
        a4i = 4.0 * np.pi * y[i] * y[i]; a4n = 4.0 * np.pi * y[i + 1] * y[i + 1]
        c0 = abs(1.0 + 0.5 * hd * sp_[i]); den = 1.0 - 0.5 * hd * sp_[i + 1]
        for k in range(K):
            num = c0 * dM[i, k] + 0.5 * hd * a4i * (drl[i, k] + abs(w[i]) * dJl[i, k]) + 0.5 * hd * a4n * (drl[i + 1, k] + abs(w[i + 1]) * dJl[i + 1, k])
            dM[i + 1, k] = num / den
    dw = np.zeros((n, K)); dsp = np.zeros((n, K))
    for j in range(1, n):
        if M[j] > 0:
            a4 = 4.0 * np.pi * y[j] * y[j]
            for k in range(K):
                dw[j, k] = abs(w[j]) * dM[j, k] / (2.0 * M[j])
                dsp[j, k] = a4 * abs(w[j]) * dJl[j, k] / (2.0 * M[j]) + abs(sp_[j]) * dM[j, k] / (2.0 * M[j])
    dlam = np.zeros((n + 1, K))
    denI = 1.0 - 0.5 * hd * sp_[I]
    for k in range(K):
        dTp = 0.0
        if MI > 0:
            dTp = np.sqrt(2.0 * G) / 2.0 * (dtail[k] / np.sqrt(MI) + abs(tail) * dM[I, k] / (2.0 * MI ** 1.5))
        dlam[I, k] = dTp / denI + abs(1.0 + Tp) * 0.5 * hd * dsp[I, k] / (denI * denI)
    for j in range(I - 1, 0, -1):
        den = 1.0 - 0.5 * hd * sp_[j]; rho = abs((1.0 + 0.5 * hd * sp_[j]) / den); c1 = abs(lam[j + 1]) * hd / (den * den)
        for k in range(K):
            dlam[j, k] = dlam[j + 1, k] * rho + c1 * dsp[j, k]
    dN = np.zeros((n, K))
    for i in range(1, I):
        for k in range(K):
            dN[i, k] = 0.5 * (dlam[i, k] + dlam[i + 1, k])
    for k in range(K):
        dN[I, k] = dlam[I, k]
        dN[0, k] = (4.0 * dN[1, k] + dN[2, k]) / 3.0
    return dM, dN, dw
