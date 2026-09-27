"""Convective initiation: P(new cell within 30 min) for 8x8 blocks with no echo yet.
Signals: cloud-top cooling + lightning onset - catches storms before radar sees them."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

B, HORIZON = 8, 6   # block px, frames (30 min)
CI_FEATS = ["ir_min", "cooling_rate", "lght", "vil_max"]


def block_samples(ev, t):
    vil, ir, lg = ev["vil"], ev["ir"], ev["lght"]
    H, W = vil.shape[1:]
    X, y, pos = [], [], []
    for i in range(0, H - B + 1, B):
        for j in range(0, W - B + 1, B):
            s = np.s_[i:i + B, j:j + B]
            if vil[t][s].max() >= 74:
                continue                             # already convective -> tracker's job
            X.append([ir[t][s].min(), ir[t][s].min() - ir[max(t - 3, 0)][s].min(),
                      lg[t][s].sum(), vil[t][s].max()])
            fut = vil[t + 1:t + 1 + HORIZON][(slice(None),) + s] if t + 1 < len(vil) else np.zeros(1)
            y.append(int(fut.size and fut.max() >= 74)); pos.append((i + B // 2, j + B // 2))
    return np.array(X), np.array(y), pos


def train(events):
    X, y = [], []
    for ev in events:
        for t in range(3, ev["vil"].shape[0] - HORIZON, 2):
            a, b, _ = block_samples(ev, t); X.append(a); y.append(b)
    X, y = np.concatenate(X), np.concatenate(y)
    return GradientBoostingClassifier(n_estimators=150, max_depth=3).fit(X, y)
