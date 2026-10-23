# Item 1 (c) detail: CHECK 11's 1D frozen symbol gap at |w| = 1, mu = 0.2: the minimum and where it sits, against the small-mu asymptote
import numpy as np
mu = 0.2
ks = np.linspace(-np.pi, np.pi, 400001); f = np.sqrt(mu**2 + np.sin(ks)**2) - np.abs(np.sin(2*ks)/2)
i = f.argmin()
print("1D minimum %.5f at sin^2 k = %.4f; mu/sqrt3 = %.4f (sin^2 k = %.4f there: f = %.5f); asymptote 0.877 mu^1.5 = %.5f" % (
    f[i], np.sin(ks[i])**2, mu/np.sqrt(3), mu**2/3, np.sqrt(mu**2 + mu**2/3) - np.sqrt(mu**2/3)*np.sqrt(1 - mu**2/3), 0.877*mu**1.5))
