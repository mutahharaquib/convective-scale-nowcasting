"""Verification: POD / FAR / CSI / FSS per lead time (section 9)."""
import numpy as np
from scipy.ndimage import uniform_filter

THRESHOLDS = [16, 74, 133, 160, 181, 219]   # SEVIR VIL pixel thresholds (paper)


def contingency(pred, obs, thr):
    p, o = pred >= thr, obs >= thr
    return (p & o).sum(), (p & ~o).sum(), (~p & o).sum()   # hits, false alarms, misses


def scores(h, f, m):
    pod = h / (h + m) if h + m else np.nan
    far = f / (h + f) if h + f else np.nan
    csi = h / (h + f + m) if h + f + m else np.nan
    return pod, far, csi


def fss(pred, obs, thr, scale=9):
    pf = uniform_filter((pred >= thr).astype(float), scale)
    of = uniform_filter((obs >= thr).astype(float), scale)
    num, den = ((pf - of) ** 2).mean(), (pf ** 2).mean() + (of ** 2).mean()
    return 1 - num / den if den > 0 else np.nan
