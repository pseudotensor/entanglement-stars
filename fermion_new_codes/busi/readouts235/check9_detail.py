# Item 1 detail: CHECK 9's minimum per u_r (box numpy) and where it is attained
import numpy as np
print("numpy", np.__version__)
for ur in (0.3, 0.7, 1.0, 1.5, 3.0, 10.0):
    cz = np.linspace(0, min(1, 1/ur), 20001); f = -2*ur*cz**2 + cz + ur
    i = f.argmin(); print("u_r=%.1f min=%r at cos z=%r (endpoint cz=min(1,1/u_r): f=%r)" % (ur, f[i], cz[i], f[-1]))
