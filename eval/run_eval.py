"""Held-out verification: U-Net vs optical flow vs persistence, by lead time.
Writes eval/results.json + eval/csi_by_lead.png. Usage: python -m eval.run_eval"""
import json, os, pickle
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data.loaders.sevir import load_events
from pipeline.nowcast.models.unet import UNet, predict, N_IN, N_OUT
from pipeline.nowcast.baseline.extrap import persistence, optical_flow
from pipeline.initiation import ci
from eval.metrics import contingency, scores, fss

N_TEST = int(os.getenv("N_TEST", 40))
THR = [16, 74, 133, 160]
ck = torch.load("models/unet.pt"); model = UNet(w=ck["width"]); model.load_state_dict(ck["state"])
events = load_events(N_TEST, "test")
methods = {"persistence": lambda e, s: persistence(e["vil"][s], N_OUT),
           "optical_flow": lambda e, s: optical_flow(e["vil"][s], N_OUT),
           "unet": lambda e, s: predict(model, e["vil"][s], e["ir"][s], e["lght"][s])}
acc = {m: np.zeros((len(THR), N_OUT, 3)) for m in methods}
fs = {m: np.zeros((len(THR), N_OUT)) for m in methods}; nf = 0
peak = {m: [] for m in methods}; peak["obs"] = []
for ev in events:
    for t in (13, 24):
        s = slice(t - N_IN, t); obs = ev["vil"][t:t + N_OUT]; nf += 1
        peak["obs"].append(obs[-1].max())
        for m, fn in methods.items():
            p = fn(ev, s); peak[m].append(p[-1].max())
            for i, th in enumerate(THR):
                for k in range(N_OUT):
                    acc[m][i, k] += contingency(p[k], obs[k], th)
                    fs[m][i, k] += fss(p[k], obs[k], th) if (obs[k] >= th).any() else 0

leads = [(k + 1) * 5 for k in range(N_OUT)]
res = {"leads_min": leads, "thresholds": THR, "n_events": len(events), "source": events[0]["source"], "methods": {}}
for m in methods:
    r = {}
    for i, th in enumerate(THR):
        pod, far, csi = zip(*[scores(*acc[m][i, k]) for k in range(N_OUT)])
        r[str(th)] = dict(pod=np.nan_to_num(pod).round(3).tolist(), far=np.nan_to_num(far).round(3).tolist(),
                          csi=np.nan_to_num(csi).round(3).tolist(), fss=(fs[m][i] / nf).round(3).tolist())
    r["peak_ratio_120min"] = round(float(np.mean(peak[m]) / np.mean(peak["obs"])), 3)
    res["methods"][m] = r

# CI skill on held-out
clf = pickle.load(open("models/classifiers.pkl", "rb"))["ci"]
X, y = zip(*[ci.block_samples(e, t)[:2] for e in events[:15] for t in range(6, 40, 4)])
X, y = np.concatenate(X), np.concatenate(y); p = clf.predict_proba(X)[:, 1] >= 0.5
res["ci"] = dict(zip(["pod", "far", "csi"], np.round(scores((p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()), 3).tolist()))
json.dump(res, open("eval/results.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m, c in zip(methods, ["#999", "#e67e22", "#2471a3"]):
    ax[0].plot(leads, res["methods"][m]["74"]["csi"], label=m, color=c, lw=2)
    ax[1].plot(leads, res["methods"][m]["133"]["fss"], label=m, color=c, lw=2)
ax[0].set(title="CSI @ VIL 74 by lead", xlabel="lead (min)"); ax[1].set(title="FSS @ VIL 133 (9 km)", xlabel="lead (min)")
for a in ax: a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("eval/csi_by_lead.png", dpi=120)
for m in methods:
    c = res["methods"][m]["74"]["csi"]
    print(f"{m:13s} CSI@74  30m {c[5]:.3f}  60m {c[11]:.3f}  120m {c[23]:.3f}  peak_ratio {res['methods'][m]['peak_ratio_120min']}")
print("CI:", res["ci"])
