"""Replay engine (D5): steps a recorded storm frame-by-frame 'as if live' and runs the full
pipeline per frame: detect/track -> CI -> nowcast (U-Net + optical-flow baseline) -> hazards -> alerts.
Always labelled REPLAY - never implies a live feed."""
import base64, os, pickle, time
import numpy as np, cv2, torch
from data.loaders.sevir import load_events
from pipeline.nowcast.models.unet import UNet, predict, N_IN, N_OUT
from pipeline.nowcast.baseline.extrap import optical_flow
from pipeline.detect.cells import detect, Tracker
from pipeline.preprocess.features import region_features, FEATURES
from pipeline.initiation import ci
from pipeline.hazards import scorers
from alerts.engine import ASSETS

LAT0, LON0 = float(os.getenv("CENTER_LAT", 17.385)), float(os.getenv("CENTER_LON", 78.487))
KM_PX = 384 / 128   # SEVIR 384 km footprint on the model grid
MAX_KMH = 100       # physical cap on cell motion


def _lut(stops):
    lut = np.zeros((256, 4), np.uint8); xs = [s[0] for s in stops]
    for ch in range(4):
        lut[:, ch] = np.interp(np.arange(256), xs, [s[1][ch] for s in stops])
    return lut


VIL_LUT = _lut([(0, (0, 0, 0, 0)), (15, (0, 0, 0, 0)), (16, (40, 200, 90, 140)), (74, (250, 230, 40, 190)),
                (133, (250, 140, 20, 215)), (160, (230, 30, 30, 230)), (219, (200, 0, 200, 240)), (255, (255, 255, 255, 250))])
IR_LUT = _lut([(0, (20, 24, 30, 0)), (60, (110, 115, 125, 110)), (150, (215, 220, 228, 190)),
               (200, (60, 190, 255, 225)), (255, (255, 60, 200, 240))])   # cold cloud tops = bright / coloured
GROUPS = {"radar": ("vil_max", "vil_mean", "area", "vil_growth"), "satellite": ("ir_min", "cooling_rate"),
          "lightning": ("lght_rate", "lght_jump")}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
PROB_LUT = {k: _lut([(0, (*c, 0)), (25, (*c, 0)), (26, (*c, 70)), (255, (*c, 230))])
            for k, c in {"hail": (0, 220, 255), "cloudburst": (40, 90, 255), "lightning": (255, 255, 90), "ci": (255, 80, 200)}.items()}


def png(a, lut, scale=1.0):
    idx = np.clip(a * scale, 0, 255).astype(np.uint8)
    rgba = lut[idx]
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


class Replay:
    def __init__(self, n_events=5):
        ck = torch.load("models/unet.pt"); self.model = UNet(w=ck["width"]); self.model.load_state_dict(ck["state"])
        c = pickle.load(open("models/classifiers.pkl", "rb"))
        self.ci, self.hail = c["ci"], c["hail"]
        self.explainer = scorers.Explainer(self.hail, c["background"])
        pool = load_events(20, "test")   # held-out events only
        pool.sort(key=lambda e: -(e["vil"] >= 160).sum())
        self.events = pool[:n_events]
        self._build_library(pool[n_events:])   # analogue storms from held-out events not replayed
        self.H = self.events[0]["vil"].shape[1]
        self.dlat = self.H * KM_PX / 111.0
        self.dlon = self.dlat / np.cos(np.radians(LAT0))
        self.bounds = [[LON0 - self.dlon / 2, LAT0 + self.dlat / 2], [LON0 + self.dlon / 2, LAT0 + self.dlat / 2],
                       [LON0 + self.dlon / 2, LAT0 - self.dlat / 2], [LON0 - self.dlon / 2, LAT0 - self.dlat / 2]]
        self.select(0)

    def _build_library(self, events):
        """Past-storm analogue library: per-cell features + what the storm went on to do."""
        rows, meta = [], []
        for ev in events:
            for t in range(6, ev["vil"].shape[0] - 6, 4):
                lab, cells = detect(ev["vil"][t])
                for c in cells:
                    m = lab == c["label"]
                    f = region_features(ev, t, m)
                    rows.append([f[k] for k in FEATURES])
                    meta.append(dict(event=ev["id"], minutes=t * 5, vil_max=round(f["vil_max"]),
                                     hail=bool(scorers.hail_label(ev, t, m))))
        self.lib = np.array(rows, np.float32) if rows else np.zeros((0, len(FEATURES)), np.float32)
        self.lib_meta = meta
        self.lib_mu, self.lib_sd = (self.lib.mean(0), self.lib.std(0) + 1e-6) if rows else (0, 1)

    def similar(self, f, k=3):
        if not len(self.lib): return []
        z = (np.array([f[x] for x in FEATURES]) - self.lib_mu) / self.lib_sd
        d = np.sqrt((((self.lib - self.lib_mu) / self.lib_sd - z) ** 2).mean(1))
        return [dict(self.lib_meta[i], similarity=round(float(np.exp(-d[i])), 2)) for i in np.argsort(d)[:k]]

    def impacts(self, c):
        """9-member motion ensemble (speed x0.8/1/1.2, heading -15/0/+15 deg) -> arrival window + probability."""
        out = []
        for a in ASSETS:
            x, y, hits = a["fx"] * self.H, a["fy"] * self.H, []
            for sc in (0.8, 1.0, 1.2):
                for da in (-15, 0, 15):
                    r = np.radians(da)
                    u = (c["u"] * np.cos(r) - c["v"] * np.sin(r)) * sc
                    v = (c["u"] * np.sin(r) + c["v"] * np.cos(r)) * sc
                    for k in range(N_OUT + 1):
                        if np.hypot(c["cx"] + u * k - x, c["cy"] + v * k - y) < 8:
                            hits.append(k * 5); break
            if hits:
                out.append(dict(asset=a["name"], window=[min(hits), max(hits)], prob=round(len(hits) / 9, 2)))
        return sorted(out, key=lambda d: d["window"][0])

    def geo(self, x, y):
        return LAT0 + self.dlat / 2 - y / self.H * self.dlat, LON0 - self.dlon / 2 + x / self.H * self.dlon

    def select(self, i):
        self.ei, self.ev, self.cache = i, self.events[i], {}
        self.t0, self.T = N_IN - 1, self.ev["vil"].shape[0] - 1
        self._run_tracker()

    def _run_tracker(self):
        self.tracks, tr, self.vilhist = {}, Tracker(), {}
        for t in range(self.T + 1):
            lab, cells = detect(self.ev["vil"][t]); tr.update(t, lab, cells)
            for c in cells: self.vilhist.setdefault(c["sid"], []).append((t, round(c["vil_max"])))
            self.tracks[t] = (lab, cells, {c["sid"]: tr.motion(c["sid"]) for c in cells})
        self.history = tr.history

    def frame(self, t):
        if t in self.cache: return self.cache[t]
        ev, H = self.ev, self.H
        s = slice(t - N_IN + 1, t + 1)
        tm, t_a = {}, time.perf_counter()
        unet = predict(self.model, ev["vil"][s], ev["ir"][s], ev["lght"][s])
        tm["nowcast_unet"], t_a = (time.perf_counter() - t_a) * 1000, time.perf_counter()
        of = optical_flow(ev["vil"][s], N_OUT)
        tm["baseline_of"], t_a = (time.perf_counter() - t_a) * 1000, time.perf_counter()
        lab, cells, motion = self.tracks[t]
        hail_grid = np.zeros((H, H), np.float32)
        out_cells = []
        for c in cells:
            m = lab == c["label"]
            f = region_features(ev, t, m)
            p = float(self.hail.predict_proba([[f[k] for k in FEATURES]])[0, 1])
            hail_grid[m] = p
            u, v = motion[c["sid"]]
            sp = np.hypot(u, v) * KM_PX * 12
            if sp > MAX_KMH: u, v = u * MAX_KMH / sp, v * MAX_KMH / sp   # centroid jumps on merge/split
            lat, lon = self.geo(c["cx"], c["cy"])
            trk = [self.geo(x, y)[::-1] for tt, x, y in self.history[c["sid"]] if tt <= t]
            sv = self.explainer.shap(f)
            pos = {g: float(sum(max(sv[FEATURES.index(k)], 0) for k in ks)) for g, ks in GROUPS.items()}
            tot = sum(pos.values()) or 1.0
            vh = [x for tt, x in self.vilhist[c["sid"]] if t - 12 <= tt <= t]
            trend = vh[-1] - vh[max(-4, -len(vh))]
            hd = (np.degrees(np.arctan2(u, -v)) + 360) % 360
            out_cells.append(dict(sid=c["sid"], event=c["event"], parents=c["parents"], lat=lat, lon=lon,
                                  cx=c["cx"], cy=c["cy"], u=u, v=v, vil_max=round(c["vil_max"]), area_km2=c["area"] * KM_PX ** 2,
                                  speed_kmh=round(np.hypot(u, v) * KM_PX * 12, 1), hail_p=round(p, 3),
                                  heading_deg=round(float(hd)), heading=COMPASS[int((hd + 22.5) // 45) % 8],
                                  growth=int(trend), trend="intensifying" if trend > 8 else "weakening" if trend < -8 else "steady",
                                  vil_hist=vh, features={k: round(f[k], 1) for k in FEATURES},
                                  source_share={g: round(x / tot, 2) for g, x in pos.items()},
                                  shap={k: round(float(sv[i]), 3) for i, k in enumerate(FEATURES)},
                                  similar=self.similar(f), impacts=self.impacts(dict(cx=c["cx"], cy=c["cy"], u=u, v=v)),
                                  lght_rate=f["lght_rate"], reasons=self.explainer.reasons(f, sv=sv), track=trk))
        tm["detect_hazards"], t_a = (time.perf_counter() - t_a) * 1000, time.perf_counter()
        cb = scorers.cloudburst_prob(unet)
        X, _, pos = ci.block_samples(ev, t)
        ci_grid = np.zeros((H, H), np.float32)
        if len(X):
            for (i, j), pr in zip(pos, self.ci.predict_proba(X)[:, 1]):
                ci_grid[i - 4:i + 4, j - 4:j + 4] = pr
        tm["initiation"] = (time.perf_counter() - t_a) * 1000
        cand = self._asset_candidates(t, unet, cb, out_cells)
        fr = dict(event=ev["id"], source=ev["source"], t=t, t0=self.t0, T=self.T, minutes=t * 5,
                  obs=png(ev["vil"][t], VIL_LUT), ir=png(np.clip((290 - ev["ir"][t]) * 2.9, 0, 255), IR_LUT),
                  timing={k: round(v) for k, v in tm.items()}, unet=[png(x, VIL_LUT) for x in unet], of=[png(x, VIL_LUT) for x in of],
                  lightning=png(scorers.lightning_density(ev["lght"][t]), PROB_LUT["lightning"], 255),
                  hail=png(hail_grid, PROB_LUT["hail"], 255), cloudburst=png(cb, PROB_LUT["cloudburst"], 255),
                  ci=png(ci_grid, PROB_LUT["ci"], 255),
                  ci_hotspots=[dict(zip(("lat", "lon"), self.geo(j, i)), p=round(float(pr), 2))
                               for (i, j), pr in zip(pos, self.ci.predict_proba(X)[:, 1] if len(X) else []) if pr > 0.5],
                  cells=[{k: v for k, v in c.items() if k not in ("cx", "cy", "u", "v")} for c in out_cells],
                  assets=cand["assets"], candidates=cand["candidates"], downburst=None)
        self.cache[t] = fr
        return fr

    def _asset_candidates(self, t, unet, cb, cells):
        H, assets, cands = self.H, [], []
        for a in ASSETS:
            x, y = int(a["fx"] * H), int(a["fy"] * H)
            nb = np.s_[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
            hit = [k for k in range(N_OUT) if unet[k][nb].max() >= 74]
            arrival = 0 if self.ev["vil"][t][nb].max() >= 74 else ((hit[0] + 1) * 5 if hit else None)
            # nearest cell whose projected path passes the asset in the 2 h window
            best, near_c = None, None
            for c in cells:
                for k in range(0, N_OUT + 1, 2):
                    d = np.hypot(c["cx"] + c["u"] * k - x, c["cy"] + c["v"] * k - y)
                    if d < 8 and (best is None or k * 5 < best):
                        best, near_c = k * 5, c; break
            lat, lon = self.geo(x, y)
            base = dict(asset=a["name"], lat=lat, lon=lon, exposure=a["exposure"], audience=a["audience"],
                        sid=near_c["sid"] if near_c else None)
            eta = arrival if arrival is not None else best
            reasons = near_c["reasons"] if near_c else []
            if near_c:
                cands.append(dict(base, hazard="hail", prob=near_c["hail_p"], arrival_min=eta, reasons=reasons))
                cands.append(dict(base, hazard="lightning", prob=round(1 - np.exp(-near_c["lght_rate"] / 5), 3),
                                  arrival_min=eta, reasons=[r for r in reasons if r["feature"].startswith("lght")] or reasons))
            pcb = float(cb[nb].max())
            if pcb > 0.2:
                cands.append(dict(base, hazard="cloudburst", prob=round(pcb, 3), arrival_min=eta, reasons=reasons))
            assets.append(dict(base, arrival_min=eta, hail_p=near_c["hail_p"] if near_c else 0.0, cloudburst_p=round(pcb, 3)))
        return dict(assets=assets, candidates=cands)
