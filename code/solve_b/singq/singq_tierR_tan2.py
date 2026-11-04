# singq_tierR_tan2.py -- = singq_tierR_tan.py (6682fe2e) plus maj_pass copied verbatim from singq_tierR_v1.py (v2 exec-ed it and numba could not cache). Module for singq_tierR_v3: the SIGNED linear tangent of route 2's discrete Fields map (exact first variation of the
# m3r2_rows_v1 forward M recursion, the lambda adjoint recursion, v = w N, back-averaging with tails), numba.
import numpy as np
import numba as nb


@nb.njit(cache=True)
def tan_pass(drl, dJl, y, hd, M, w, sp_, lam, Tp, tail, MI, G, dtail):
    n = y.shape[0]; I = n - 1
    dM = np.zeros(n)
    for i in range(I):
        a4i = 4.0 * np.pi * y[i] * y[i]; a4n = 4.0 * np.pi * y[i + 1] * y[i + 1]
        num = (1.0 + 0.5 * hd * sp_[i]) * dM[i] + 0.5 * hd * a4i * (drl[i] + w[i] * dJl[i]) + 0.5 * hd * a4n * (drl[i + 1] + w[i + 1] * dJl[i + 1])
        dM[i + 1] = num / (1.0 - 0.5 * hd * sp_[i + 1])
    dw = np.zeros(n); dsp = np.zeros(n)
    for j in range(1, n):
        if M[j] > 0:
            a4 = 4.0 * np.pi * y[j] * y[j]
            dw[j] = w[j] * dM[j] / (2.0 * M[j])
            dsp[j] = a4 * w[j] * dJl[j] / (2.0 * M[j]) - sp_[j] * dM[j] / (2.0 * M[j])
    dlam = np.zeros(n + 1)
    denI = 1.0 - 0.5 * hd * sp_[I]
    dTp = 0.0
    if MI > 0:
        dTp = -np.sqrt(2.0 * G) / 2.0 * (dtail / np.sqrt(MI) - tail * dM[I] / (2.0 * MI ** 1.5))
    dlam[I] = dTp / denI + (1.0 + Tp) * 0.5 * hd * dsp[I] / (denI * denI)
    for j in range(I - 1, 0, -1):
        den = 1.0 - 0.5 * hd * sp_[j]
        dlam[j] = dlam[j + 1] * (1.0 + 0.5 * hd * sp_[j]) / den + lam[j + 1] * hd * dsp[j] / (den * den)
    dN = np.zeros(n)
    for i in range(1, I):
        dN[i] = 0.5 * (dlam[i] + dlam[i + 1])
    dN[I] = dlam[I]
    dN[0] = (4.0 * dN[1] - dN[2]) / 3.0
    return dM, dN, dw


@nb.njit(cache=True)
def maj_pass(drl, dJl, y, hd, M, w, sp_, lam, Tp, tail, MI, G, dtail):
    n = y.shape[0]; I = n - 1
    dM = np.zeros(n)
    for i in range(I):
        a4i = 4.0 * np.pi * y[i] * y[i]; a4n = 4.0 * np.pi * y[i + 1] * y[i + 1]
        num = abs(1.0 + 0.5 * hd * sp_[i]) * dM[i] + 0.5 * hd * a4i * (drl[i] + abs(w[i]) * dJl[i]) \
            + 0.5 * hd * a4n * (drl[i + 1] + abs(w[i + 1]) * dJl[i + 1])
        den = 1.0 - 0.5 * hd * sp_[i + 1]
        dM[i + 1] = num / den
    dw = np.zeros(n); dsp = np.zeros(n)
    for j in range(1, n):
        if M[j] > 0:
            a4 = 4.0 * np.pi * y[j] * y[j]
            dw[j] = abs(w[j]) * dM[j] / (2.0 * M[j])
            dsp[j] = a4 * abs(w[j]) * dJl[j] / (2.0 * M[j]) + abs(sp_[j]) * dM[j] / (2.0 * M[j])
    dlam = np.zeros(n + 1)
    denI = 1.0 - 0.5 * hd * sp_[I]
    dTp = 0.0
    if MI > 0:
        dTp = np.sqrt(2.0 * G) / 2.0 * (dtail / np.sqrt(MI) + abs(tail) * dM[I] / (2.0 * MI ** 1.5))
    dlam[I] = dTp / denI + abs(1.0 + Tp) * 0.5 * hd * dsp[I] / (denI * denI)
    for j in range(I - 1, 0, -1):
        den = 1.0 - 0.5 * hd * sp_[j]
        rho = (1.0 + 0.5 * hd * sp_[j]) / den
        dlam[j] = dlam[j + 1] * abs(rho) + abs(lam[j + 1]) * hd * dsp[j] / (den * den)
    dN = np.zeros(n)
    for i in range(1, I):
        dN[i] = 0.5 * (dlam[i] + dlam[i + 1])
    dN[I] = dlam[I]
    dN[0] = (4.0 * dN[1] + dN[2]) / 3.0
    return dM, dN, dw
