# Diagnostic of GAP08's CF4 stepper on the self-test's 32-site chain: the error against a fine midpoint reference
# as a function of dt for both steppers, and the spectral bound used by the Chebyshev series against the actual
# norm at the CF4 effective envelopes.
import sys, math
sys.path.insert(0, '<internal>/j6')
import torch
torch.set_num_threads(2)
import j6_transverse_chain as J
dev = 'cpu'
ch = J.Chain(32, dev); M = 0.2
P0, _ = J.flat_sea(ch, M); Mb = torch.tensor([M], dtype=J.RD, device=dev)
H0 = ch.dense(M, 0.0); Hv = ch.dense(M, 1.0) - H0
print("ebound", ch.ebound(M), " norm h(e=1)", float(torch.linalg.eigvalsh(ch.dense(M, 1.0)).abs().max()),
      " norm h(e=1.08)", float(torch.linalg.eigvalsh(H0 + 1.08*Hv).abs().max()))
# a cheap reference: midpoint at dt 0.25 by dense exponentials
dtr = 0.25; Ns = int(round(J.T_RAMP/dtr))
Phi = P0.reshape(64, -1).clone()
for n in range(Ns):
    Phi = torch.linalg.matrix_exp(-1j*dtr*(H0 + J.envelope((n + 0.5)*dtr)*Hv)) @ Phi
Cref = Phi @ Phi.conj().T
for stp in ('midpoint', 'cf4'):
    for dt in (64.0, 16.0, 4.0, 1.0):
        PhiT = J.transport(ch, P0.unsqueeze(0), Mb, stp, dt, ch.ebound(M))[0].reshape(64, -1)
        d = float((PhiT @ PhiT.conj().T - Cref).abs().max())
        print(f"{stp:8s} dt {dt:5.1f}: max|C - C_ref(midpoint 0.25)| = {d:.3e}", flush=True)
# CF4 by dense exponentials (no Chebyshev), to separate the scheme from the series
for dt in (16.0, 4.0):
    Ns = int(round(J.T_RAMP/dt)); dt = J.T_RAMP/Ns
    c1, c2 = 0.5 - math.sqrt(3)/6, 0.5 + math.sqrt(3)/6
    a1, a2 = (3 - 2*math.sqrt(3))/12, (3 + 2*math.sqrt(3))/12
    Phi = P0.reshape(64, -1).clone()
    for n in range(Ns):
        e1, e2 = J.envelope(n*dt + c1*dt), J.envelope(n*dt + c2*dt)
        Phi = torch.linalg.matrix_exp(-1j*0.5*dt*(H0 + 2*(a2*e1 + a1*e2)*Hv)) @ Phi
        Phi = torch.linalg.matrix_exp(-1j*0.5*dt*(H0 + 2*(a1*e1 + a2*e2)*Hv)) @ Phi
    d = float((Phi @ Phi.conj().T - Cref).abs().max())
    print(f"cf4 dense dt {dt:5.1f}: max|C - C_ref| = {d:.3e}", flush=True)
