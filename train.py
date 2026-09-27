"""Train all learned components: U-Net nowcast, CI classifier, hail classifier.
Usage: python train.py  [env: N_TRAIN=200 STEPS=1500 WIDTH=16]"""
import os, pickle, time
import numpy as np, torch
from data.loaders.sevir import load_events
from pipeline.nowcast.models.unet import UNet, make_input, N_IN, N_OUT
from pipeline.initiation import ci
from pipeline.hazards import scorers

N_TRAIN, STEPS, WIDTH = int(os.getenv("N_TRAIN", 200)), int(os.getenv("STEPS", 1500)), int(os.getenv("WIDTH", 16))
os.makedirs("models", exist_ok=True)
torch.manual_seed(0); rng = np.random.default_rng(0)

t0 = time.time()
events = load_events(N_TRAIN, "train")
print(f"loaded {len(events)} train events ({events[0]['source']}) in {time.time()-t0:.0f}s")


def batch(bs=8):
    X, Y = [], []
    for _ in range(bs):
        ev = events[rng.integers(len(events))]
        t = rng.integers(N_IN, ev["vil"].shape[0] - N_OUT + 1)
        s = slice(t - N_IN, t)
        X.append(make_input(ev["vil"][s], ev["ir"][s], ev["lght"][s]))
        Y.append(torch.as_tensor(ev["vil"][t:t + N_OUT] / 255.0).float())
    return torch.stack(X), torch.stack(Y)


model = UNet(w=WIDTH)
opt = torch.optim.AdamW(model.parameters(), 2e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
for step in range(STEPS):
    model.train(); x, y = batch()
    p = model(x)
    w = 1 + 4 * (y > 74 / 255) + 8 * (y > 160 / 255)   # upweight convective cores -> fights peak-flattening
    loss = (w * (p - y).abs()).mean() + (w * (p - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 100 == 0:
        print(f"unet step {step}/{STEPS} loss {loss.item():.4f}  [{time.time()-t0:.0f}s]")
torch.save({"state": model.state_dict(), "width": WIDTH}, "models/unet.pt")

print("training CI classifier...")
ci_model = ci.train(events[:60])
print("training hail classifier...")
hail, bg = scorers.train_hail(events[:80])
pickle.dump(dict(ci=ci_model, hail=hail, background=bg), open("models/classifiers.pkl", "wb"))
print(f"done in {time.time()-t0:.0f}s -> models/")
