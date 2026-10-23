# o5_mjm_image_v1: is the m_j -1/2 lift the lattice image of the m_j +1/2 lift (O5_controls_reading_v1 section 4)?
# Read on the box's CPU before any control-4 cube run. Needs o5_b4_restate_v1.py beside it.
import argparse, itertools, sys
import numpy as np
from o5_b4_restate_v1 import load_src

ROTS = (("pi about x", lambda x: x * np.array([1.0, -1.0, -1.0])),
        ("pi about y", lambda x: x * np.array([-1.0, 1.0, -1.0])),
        ("pi about x+y", lambda x: np.column_stack([x[:, 1], x[:, 0], -x[:, 2]])),
        ("pi about x-y", lambda x: np.column_stack([-x[:, 1], -x[:, 0], -x[:, 2]])))
LIMIT_MOD, LIMIT_PHASE = 1e-12, 1e-10   # this note's rules

def centred(c): return c - 0.5 * (c.max(axis=0) + c.min(axis=0))

def finish(out, L, status):
    open(out + ".md", "w").write("# o5_mjm_image_v1\n\n" + "\n".join("- " + l for l in L) + "\n"); print("\n".join(L)); sys.exit(status)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plus", required=True); ap.add_argument("--minus", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cp, vp = load_src(a.plus); cm, vm = load_src(a.minus)
    xp, xm = centred(cp), centred(cm); L = []
    if np.abs(xp - np.rint(xp)).max() > 1e-9 or np.abs(xm - np.rint(xm)).max() > 1e-9:
        L.append("STOP: centred coordinates are not integers"); finish(a.out, L, 3)
    key = lambda arr: [tuple(t) for t in np.rint(arr).astype(int)]
    index = {k: i for i, k in enumerate(key(xm))}
    scale = np.abs(vp).max(); best = None
    for name, R in ROTS:
        idx = [index.get(k, -1) for k in key(R(xp))]
        missing = sum(1 for i in idx if i < 0)
        if missing:
            L.append("%s: %d image sites absent from the -1/2 file; skipped" % (name, missing)); continue
        vimg = vm[np.array(idx)]
        for perm in itertools.permutations(range(4)):
            Dm = float(np.abs(np.abs(vimg[:, list(perm)]) - np.abs(vp)).max() / scale)
            if best is None or Dm < best[0]: best = (Dm, name, perm, vimg[:, list(perm)])
    if best is None:
        L.append("STOP: no lattice rotation maps the +1/2 sites onto the -1/2 sites"); finish(a.out, L, 2)
    Dm, name, perm, vimg = best
    L.append("best lattice rotation %s with component permutation %s: largest modulus difference %.3e of the largest modulus" % (name, perm, Dm))
    phase_dev = 0.0
    for c in range(4):
        mc = np.abs(vp[:, c]) > 1e-8 * scale
        if mc.sum() < 2: L.append("component %d: no sites above 1e-8 of the largest modulus" % c); continue
        rho = vimg[mc, c] / vp[mc, c]; mean = rho.mean(); dev = float(np.abs(np.angle(rho / mean)).max())
        phase_dev = max(phase_dev, dev)
        L.append("component %d: constant factor modulus %.12f, phase %+.6f rad, largest phase deviation %.2e" % (c, abs(mean), np.angle(mean), dev))
    ph = np.array([np.exp(-1j * np.angle((vimg[:, c] * np.conj(vp[:, c])).sum())) for c in range(4)])
    nrm = np.linalg.norm(vimg) * np.linalg.norm(vp)
    ov = abs((np.conj(vimg) * vp).sum()) / nrm; ov2 = abs((np.conj(vimg * ph) * vp).sum()) / nrm
    L.append("overlap modulus of the image with the +1/2 vector: %.15f as mapped, %.15f with the 4 per-component phases removed" % (ov, ov2))
    image = Dm <= LIMIT_MOD and phase_dev <= LIMIT_PHASE and abs(ov2 - 1.0) <= LIMIT_MOD
    L.append("VERDICT: %s" % ("IMAGE: the -1/2 lift is the lattice image of the +1/2 lift; convention M is dropped, no cube run" if image else "NOT THE IMAGE within the limits: read before any cube run"))
    finish(a.out, L, 0 if image else 2)

if __name__ == "__main__":
    main()
