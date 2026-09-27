# Deadlock Nowcast — 0–2 h convective nowcasting (SIH 26084)

Radar + satellite + lightning → learned nowcast (U-Net) → cell tracking + convective initiation →
hazard scoring (lightning / hail / cloudburst) with SHAP reasons → severity-scored, audience-calibrated alerts
→ MapLibre dashboard driven by a **replay** of a recorded storm. Spec: [claude.md](claude.md).

> **Honesty labels.** Everything runs on replayed historical data, never a live feed. Without a SEVIR download the
> system uses a **synthetic SEVIR-like stand-in** generator (clearly labelled in the UI). Map placement over India
> is illustrative. Downburst is velocity-gated (needs Doppler radial velocity) and shown as roadmap.

## Run it

**Local (no Docker):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python train.py                 # U-Net + CI + hail classifiers  (env: N_TRAIN, STEPS, WIDTH)
python -m eval.run_eval         # held-out CSI/POD/FAR/FSS -> eval/results.json, eval/csi_by_lead.png
uvicorn services.api.main:app --port 8000   # open http://localhost:8000
```
Local runs use in-memory state + SQLite. **Docker:** `docker compose up --build` adds Redis Streams + PostGIS.

**Real SEVIR:** download from `s3://sevir` (`aws s3 sync --no-sign-request s3://sevir/ ./sevir --exclude "*" --include "CATALOG.csv" --include "data/vil/2018/*" --include "data/ir107/2018/*" --include "data/lght/2018/*"` plus 2019 files for test),
then `SEVIR_ROOT=./sevir python train.py`. Train/test split is the published one (before/after 2019-06-01).

## Map of the code
| Stage | File |
|---|---|
| SEVIR loader + synthetic stand-in | `data/loaders/sevir.py` (MOSDAC stub: `mosdac.py`) |
| Replay engine (full pipeline per frame) | `data/replay/engine.py` |
| Features (cooling rate, lightning jump, VIL→rain proxy) | `pipeline/preprocess/features.py` |
| Cell detection, Storm IDs, merge/split lineage | `pipeline/detect/cells.py` |
| Baselines: persistence, optical flow (pysteps if installed, else OpenCV) | `pipeline/nowcast/baseline/extrap.py` |
| **Primary nowcast: U-Net** (VIL+IR+GLM in, 24×5 min out) | `pipeline/nowcast/models/unet.py` |
| Convective-initiation classifier | `pipeline/initiation/ci.py` |
| Hazards + SHAP explainer (downburst = roadmap stub) | `pipeline/hazards/scorers.py` |
| Severity, audience thresholds, feedback recalibration, CAP, webhook/Twilio | `alerts/engine.py` |
| FastAPI + WebSocket | `services/api/main.py` |
| Redis Streams state / PostGIS history | `services/state/store.py`, `services/db/store.py` |
| Dashboard | `frontend/index.html` |
| Verification | `eval/metrics.py`, `eval/run_eval.py` |

## Dashboard (MVP of the architecture poster)
`/` walks the **forecaster flow** from `convective-nowcasting-architecture.pdf`:
1. **Secure login** — email + OTP, roles Forecaster / Admin / Viewer (viewer is read-only). Demo mode shows the OTP on screen.
2. **Live dashboard** — map, active-alert count, subscribed locations, data-latency indicator.
3. **Select storm / location** — click a cell or search a location / storm ID.
4. **Storm detail** — position, speed, direction, growth trend, arrival window + probability (9-member motion ensemble), radar + satellite loop with past/forecast scrubber.
5. **AI analysis** — per-source contribution (radar / satellite / lightning, from SHAP), top indicators, most-similar past storms.
6. **Genuine threat?** — AI-proposed alerts wait for review: YES → issue, NO → dismiss with reason.
7. **Issue + notify** — severity, area, validity, operator notes; SMS / WhatsApp / email / webhook / in-app (mock unless Twilio / `WEBHOOK_URL` set), CAP 1.2 per alert.
8. **Field response** — audience-specific action checklist.
9. **Outcome** — hit / false alarm / miss + notes + photo → per-audience thresholds recalibrate.

Other tabs: **Alerts & learning** (records, delivery log, threshold history), **Pipeline** (live status of stages 2A–2G and 3A–3F),
**Validation** (CSI / FSS by lead vs baselines). The previous single-page dashboard is at `/classic`.

## Known limits
- Hail truth is a proxy (severe VIL core under cold tops); join SPC/IMD hail reports for real labels.
- Rain rate from VIL is an empirical proxy, not QPE.
- Exposure layer (assets) is a stand-in; SEVIR has no population data.
- Upgrade path: ConvLSTM / DGMR for amplitude preservation (tracked as `peak_ratio` in eval).
