"""Hazard scoring: lightning density, hail probability (classifier + SHAP), cloudburst.
Downburst is velocity-gated (needs DWR radial velocity) -> roadmap, returns None."""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import GradientBoostingClassifier
from pipeline.preprocess.features import FEATURES, region_features, rain_rate
from pipeline.detect.cells import detect

LABEL = {"vil_max": "radar VIL core", "vil_mean": "mean VIL", "area": "storm size",
         "ir_min": "cloud-top temperature", "cooling_rate": "cloud-top cooling",
         "lght_rate": "lightning rate", "lght_jump": "lightning jump", "vil_growth": "VIL growth"}


def lightning_density(lght_t):
    """Flashes / 5 min, smoothed to ~density per 10 km cell -> 0..1."""
    d = gaussian_filter(lght_t, 1.5) * 20
    return np.clip(d / 5.0, 0, 1)


def cloudburst_prob(vil_fcst):
    """P(rain > 100 mm/h) from the nowcast max over the horizon (logistic around threshold)."""
    r = rain_rate(vil_fcst.max(0))
    return 1 / (1 + np.exp(-(r - 100) / 15))


def downburst(*_):
    return None   # ROADMAP: requires Doppler radial velocity (MOSDAC DWR)


def hail_label(ev, t, mask, horizon=6):
    """Proxy hail truth: severe VIL core (>=200 px, ~40 kg/m^2) under cold tops (<225 K)
    within 30 min. SEVIR has no hail reports; swap for SPC/IMD reports when joined."""
    fut = slice(t, min(t + horizon, len(ev["vil"])))
    return int(ev["vil"][fut][:, mask].max() >= 200 and ev["ir"][fut][:, mask].min() < 225)


def cell_table(events, step=2):
    X, y = [], []
    for ev in events:
        for t in range(3, ev["vil"].shape[0] - 6, step):
            lab, cells = detect(ev["vil"][t])
            for c in cells:
                m = lab == c["label"]
                f = region_features(ev, t, m)
                X.append([f[k] for k in FEATURES]); y.append(hail_label(ev, t, m))
    return np.array(X), np.array(y)


def train_hail(events):
    X, y = cell_table(events)
    return GradientBoostingClassifier(n_estimators=200, max_depth=3).fit(X, y), X


class Explainer:
    """SHAP TreeExplainer -> plain-language top-3 reasons."""

    def __init__(self, model, background):
        import shap
        self.e = shap.TreeExplainer(model, background[:200])

    def shap(self, feats):
        x = np.array([[feats[f] for f in FEATURES]])
        return np.array(self.e.shap_values(x)).reshape(-1)[-len(FEATURES):]

    def reasons(self, feats, k=3, sv=None):
        sv = self.shap(feats) if sv is None else sv
        order = np.argsort(-sv)[:k]
        return [dict(feature=FEATURES[i], label=LABEL[FEATURES[i]], value=round(feats[FEATURES[i]], 1),
                     contribution=round(float(sv[i]), 3)) for i in order if sv[i] > 0]
