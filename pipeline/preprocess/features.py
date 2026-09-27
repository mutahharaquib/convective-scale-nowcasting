"""Interpretable feature extraction shared by CI + hazard classifiers."""
import numpy as np

FEATURES = ["vil_max", "vil_mean", "area", "ir_min", "cooling_rate",
            "lght_rate", "lght_jump", "vil_growth"]


def region_features(ev, t, mask, area=None):
    """Features for a region (bool mask) at frame t. 3-frame (15 min) trends."""
    vil, ir, lg = ev["vil"], ev["ir"], ev["lght"]
    tp = max(t - 3, 0)
    rate = lg[t][mask].sum()
    prev = np.mean([lg[k][mask].sum() for k in range(max(t - 4, 0), max(t - 1, 1))])
    return dict(
        vil_max=float(vil[t][mask].max()), vil_mean=float(vil[t][mask].mean()),
        area=float(area if area is not None else mask.sum()),
        ir_min=float(ir[t][mask].min()),
        cooling_rate=float(ir[t][mask].min() - ir[tp][mask].min()),   # K / 15 min (neg = cooling)
        lght_rate=float(rate), lght_jump=float(rate - prev),
        vil_growth=float(vil[t][mask].max() - vil[tp][mask].max()))


def vil_to_kgm2(p):
    """SEVIR VIL pixel -> kg/m^2 (SEVIR decoding)."""
    p = np.asarray(p, np.float32)
    return np.where(p <= 5, 0, np.where(p <= 18, (p - 2) / 90.66, np.exp((p - 83.9) / 38.9)))


def rain_rate(p):
    """Rain-rate proxy (mm/h) from VIL. Empirical R ~ 7*VIL for deep convection; a proxy, not QPE."""
    return 7.0 * vil_to_kgm2(p)
