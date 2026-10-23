#!/usr/bin/env python
"""engine_mixing_v1.py (builder analyst): Anderson mixing (type II, Walker and Ni 2011) for the self-consistency loops of the engine
(the static star's fields, the collapse's initial data).  The fixed point is that of the loop's map G (the fields re-solved from the sources
of the eigenstates at the current fields); mixing changes only the path to it, so a converged member is the same member (checked by the
outer residual of engine_star_v2's final recompute).  Safeguard: when the new residual norm exceeds 2 times the best so far, the history is
cleared and a damped linear step is taken."""
import numpy as np


class Anderson:
    def __init__(self, depth=6, beta=0.5, alpha_linear=0.3):
        self.depth, self.beta, self.alpha = depth, beta, alpha_linear
        self.X, self.F = [], []
        self.best = np.inf

    def step(self, x, gx):
        """x: the current iterate (1-D array), gx: G(x).  Returns the next iterate."""
        f = gx - x
        nf = float(np.linalg.norm(f))
        if nf > 2.0 * self.best and self.X:
            self.X, self.F = [], []
            return x + self.alpha * f
        self.best = min(self.best, nf)
        self.X.append(x.copy()); self.F.append(f.copy())
        if len(self.X) > self.depth + 1:
            self.X.pop(0); self.F.pop(0)
        if len(self.X) < 2:
            return x + self.beta * f
        dX = np.array([self.X[i + 1] - self.X[i] for i in range(len(self.X) - 1)]).T
        dF = np.array([self.F[i + 1] - self.F[i] for i in range(len(self.F) - 1)]).T
        gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
        return x + self.beta * f - (dX + self.beta * dF) @ gamma
