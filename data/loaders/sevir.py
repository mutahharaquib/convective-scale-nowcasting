"""SEVIR loader + synthetic SEVIR-like generator.

Every event is a dict of aligned arrays on a common grid, shape (T, H, W):
  vil  : VIL in SEVIR pixel units 0..255 (paper thresholds 16/74/133/160/181/219)
  ir   : IR 10.7um brightness temperature (K)  (GOES C13)
  lght : GLM flash count per cell per 5-min frame
Real SEVIR is used when SEVIR_ROOT points at the AWS download; otherwise a
physically-motivated synthetic generator stands in (labelled as such everywhere).
"""
import os, glob
import numpy as np
from scipy.ndimage import zoom

SEVIR_ROOT = os.environ.get("SEVIR_ROOT", "")
GRID = int(os.environ.get("GRID", 128))   # CPU-friendly; real SEVIR 384 -> 128 (3 km)
T = 49                                     # 4 h at 5 min


# ---------------- real SEVIR ----------------
def _sevir_available():
    return bool(SEVIR_ROOT) and os.path.exists(os.path.join(SEVIR_ROOT, "CATALOG.csv"))


def load_sevir_events(n=50, split="train"):
    """Load up to n events that have vil+ir107+lght. split by 2019-06-01 (published split)."""
    import h5py, pandas as pd
    cat = pd.read_csv(os.path.join(SEVIR_ROOT, "CATALOG.csv"), parse_dates=["time_utc"], low_memory=False)
    cut = pd.Timestamp("2019-06-01")
    cat = cat[cat.time_utc < cut] if split == "train" else cat[cat.time_utc >= cut]
    ids = set.intersection(*[set(cat[cat.img_type == t].id) for t in ("vil", "ir107", "lght")])
    out = []
    for eid in list(sorted(ids))[:n]:
        ev = {}
        for t in ("vil", "ir107"):
            r = cat[(cat.id == eid) & (cat.img_type == t)].iloc[0]
            with h5py.File(os.path.join(SEVIR_ROOT, "data", r.file_name), "r") as f:
                a = f[t][int(r.file_index)].astype(np.float32)  # (H,W,T)
            a = np.moveaxis(a, -1, 0)
            a = zoom(a, (1, GRID / a.shape[1], GRID / a.shape[2]), order=1)
            ev["vil" if t == "vil" else "ir"] = a
        if ev["ir"].max() < 100:  # SEVIR stores IR in degC; convert to K
            ev["ir"] = ev["ir"] + 273.15
        r = cat[(cat.id == eid) & (cat.img_type == "lght")].iloc[0]
        with h5py.File(os.path.join(SEVIR_ROOT, "data", r.file_name), "r") as f:
            fl = f[eid][:]  # N x 5: t(s from start), lat, lon, x, y (0..48 km-ish in 4km px)
        l = np.zeros((T, GRID, GRID), np.float32)
        ti = np.clip((fl[:, 0] // 300).astype(int) + 36, 0, T - 1)  # lght times are relative to frame 36? clip safely
        xi = np.clip((fl[:, 3] / 48 * GRID).astype(int), 0, GRID - 1)
        yi = np.clip((fl[:, 4] / 48 * GRID).astype(int), 0, GRID - 1)
        np.add.at(l, (ti, GRID - 1 - yi, xi), 1)
        ev["lght"] = l
        ev["id"], ev["source"] = eid, "SEVIR"
        out.append(ev)
    return out


# ---------------- synthetic stand-in ----------------
def synthetic_event(seed=0, n_cells=None, grid=GRID):
    """Moving convective cells with a life cycle. IR cooling precedes radar echo
    by ~20 min (so CI is learnable); lightning jumps before VIL peak (so hail/jump
    features carry signal). Not real data - a stand-in until SEVIR is downloaded."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)
    steer = rng.normal(0, 0.6, 2)                     # px / frame steering flow
    n_cells = n_cells or rng.integers(4, 10)
    cells = []
    for _ in range(n_cells):
        t0 = rng.integers(-20, 30)                    # initiation frame (negative = already mature)
        cells.append(dict(
            x=rng.uniform(0, grid), y=rng.uniform(0, grid), t0=t0,
            life=rng.integers(18, 40), peak=rng.uniform(80, 250),
            r=rng.uniform(3, 9), v=steer + rng.normal(0, 0.25, 2)))
    vil = np.zeros((T, grid, grid), np.float32)
    ir = np.full((T, grid, grid), 290.0, np.float32)
    lg = np.zeros((T, grid, grid), np.float32)
    for c in cells:
        for t in range(T):
            age = t - c["t0"]
            cx, cy = c["x"] + c["v"][0] * t, c["y"] + c["v"][1] * t
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            # cloud-top cooling starts 4 frames (20 min) before echo
            if -4 <= age <= c["life"]:
                cool = np.clip((age + 4) / 8, 0, 1) * (50 + c["peak"] / 6)   # coldest tops ~198 K (-75 C)
                ir[t] = np.minimum(ir[t], 290 - cool * np.exp(-d2 / (2 * (c["r"] * 1.8) ** 2)))
            if 0 <= age <= c["life"]:
                phase = max(np.sin(np.pi * age / c["life"]), 0) ** 1.5
                amp = c["peak"] * phase
                vil[t] += amp * np.exp(-d2 / (2 * c["r"] ** 2))
                # lightning: rate ~ ice mass; jump leads the VIL peak
                lead = max(np.sin(np.pi * min(age + 3, c["life"]) / c["life"]), 0) ** 3
                rate = max(0, (c["peak"] - 90) / 12) * lead
                if rate > 0:
                    k = rng.poisson(rate)
                    if k:
                        px = np.clip(rng.normal(cx, c["r"] / 2, k).astype(int), 0, grid - 1)
                        py = np.clip(rng.normal(cy, c["r"] / 2, k).astype(int), 0, grid - 1)
                        np.add.at(lg[t], (py, px), 1)
    vil = np.clip(vil + rng.normal(0, 1.5, vil.shape).clip(0), 0, 255)
    ir += rng.normal(0, 0.8, ir.shape)
    return dict(vil=vil, ir=ir, lght=lg, id=f"SYN-{seed:04d}", source="synthetic")


def load_events(n=50, split="train"):
    if _sevir_available():
        return load_sevir_events(n, split)
    base = 0 if split == "train" else 10_000   # disjoint seeds = held-out
    return [synthetic_event(base + i) for i in range(n)]
