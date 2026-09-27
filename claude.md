# SIH 26084 — Convective Nowcasting Prototype Spec
**Thunderstorms, Hail & Cloudbursts (0–6 h) · MoES / NCMRWF · Team Deadlock**

> Purpose of this doc: a buildable prototype plan for the idea stage and a clear line to the finale. It fixes one flaw carried over from the deck (ML kept getting demoted to optional) and grounds the data plan in sources that are actually reachable now.

---

## 0. TL;DR — what we are building

A **0–6 h convective nowcasting system** that ingests aligned radar + satellite + lightning, predicts where severe convection will be in the next few hours at ~1 km resolution, scores four hazards, and drives an explainable, human-in-the-loop alerting dashboard.

**For the prototype we build the whole vertical slice on the public SEVIR dataset**, prove skill against an optical-flow baseline, and wire it into a live-looking replay demo. The Indian data (INSAT-3D/3DR, DWR) is registered for and treated as the *port target*, not a blocker.

**Two decisions that differ from the pitch deck — read these first:**
1. **A learned model owns the 0–2 h window, not optical flow.** Optical flow / pySTEPS is the *baseline we beat*, not the workhorse. (The deck had ML only at 2–6 h, which meant the most important warning window was just the baseline — a losing answer in a NCMRWF Q&A.)
2. **Downburst is explicitly a finale/roadmap hazard.** It needs Doppler radial velocity, which SEVIR does not contain. We build the other three hazards fully on SEVIR and show downburst as a velocity-gated extension.

---

## 1. Scope

### In scope (prototype / idea stage)
- Data pipeline: load SEVIR events → common 1 km grid → feature extraction.
- Baseline nowcast: persistence + optical-flow extrapolation (pySTEPS).
- **Primary nowcast: a learned spatiotemporal model** predicting the next 0–2 h (extendable to 6 h) of radar (VIL) fields.
- Convective-initiation (CI) detection from satellite cooling + lightning onset.
- Hazard scoring: **lightning density, hail probability, cloudburst threshold** (downburst deferred).
- Explainability: SHAP-based per-alert reasons on the hazard classifier.
- Backend: FastAPI + WebSocket, Redis (live/replay state), PostGIS (history).
- Frontend: React + MapLibre dashboard — hazard layers, storm cells, replay scrubber, arrival countdown, alert panel.
- Alerting: severity scoring, plain-language reason, mock multi-channel delivery (log/webhook; Twilio optional).
- Validation: POD / FAR / CSI / FSS by lead time vs baselines.
- One-command `docker compose up` demo on a replayed historical storm.

### Out of scope (prototype) — stated so we don't overclaim
- Live streaming ingestion from IMD/ISRO (needs formal agency agreement).
- Downburst detection (needs DWR radial velocity).
- Full B+9 / operational deployment, authentication hardening, multi-tenant scaling.
- Beating operational NWP/NCMRWF products (we beat *extrapolation baselines*, which is the honest and defensible claim).

---

## 2. Key design decisions & rationale

| # | Decision | Why |
|---|----------|-----|
| D1 | Learned model is primary at **all** lead times incl. 0–2 h; optical flow is the scored baseline | Avoids "our core is pySTEPS." The scientific claim must be that ML beats extrapolation *in the window that matters*. |
| D2 | Prototype on **SEVIR**; port to INSAT/DWR | SEVIR is public, aligned, restriction-free, and convective-event-centric. Removes the data blocker entirely for the build. |
| D3 | Compete on **operational intelligence**, not raw forecast skill | Merge/split-aware tracking, audience-specific alert calibration, alert-fatigue reduction, explainability — winnable ground for a student team vs national labs. |
| D4 | **Downburst = roadmap**, velocity-gated | SEVIR has no radial velocity; claiming downburst on it is indefensible. Honesty reads as credibility to domain judges. |
| D5 | **Replay engine** streams a recorded storm "as if live" | Demonstrates real-time UX without a live feed we cannot legally get by the finale. Must be labelled as replay, not implied live. |
| D6 | Validation is a **first-class deliverable**, its own slide/section | For a MoES jury the verification numbers *are* the credibility. Operational story alone reads as compensating for a thin core. |

---

## 3. Data plan (verified)

### 3.1 Primary: SEVIR (build + demo)
- **Source:** AWS Open Data — `registry.opendata.aws/sevir`. Free, **no usage restrictions**.
- **Tooling:** `MIT-AI-Accelerator/eie-sevir` (loaders, generators, baselines).
- **Contents:** 10,000+ events, 384×384 km, 4 h sequences at 5-min steps (49 frames), aligned across 5 modalities:
  - GOES-16 ABI channels **C02 (visible), C09, C13 (IR)**
  - **NEXRAD VIL** (vertically integrated liquid) — radar proxy, 1 km
  - **GOES-16 GLM** lightning flashes (the only non-image modality)
- **Standard nowcast task:** input 13 frames (≈1 h) → predict next 12 frames (≈1 h); extend horizon for 0–2 h. Train on pre-2019-06-01, test on/after — matches the published split so our numbers are comparable.
- **What SEVIR does NOT have:** Doppler radial velocity, terrain/DEM, population. → downburst out; terrain/exposure layers added separately for the Indian port.

### 3.2 Port target: Indian data (register now, use later)
- **INSAT-3D / 3DR** (cloud-top IR, the direct INSAT analogue of SEVIR's C09/C13): via **MOSDAC** (`mosdac.gov.in`). Requires a **free Single Sign-On registration**; available through Order Data, API-based Access, and Open Data sections.
- **DWR (Doppler Weather Radar):** MOSDAC "RADAR (DWR)" section incl. 3D volumetric products; reachable after registration (order/API). This is where **radial velocity** for downburst comes from — finale only.
- **Lightning:** Indian networks (IITM/Damini) — access likely needs a request; GLM-trained CI logic ports with recalibration.
- **Verification ground truth (India):** IMD 0.25° gridded rainfall via the `imdlib` Python package (open).
- **Reanalysis baseline (optional):** ERA5 (Copernicus, open) for climatology/normals.

### 3.3 Action this week
1. Create a **MOSDAC SSO account today** — registration lead time is the one thing we can't compress. Request INSAT-3D/3DR + DWR products.
2. Pull a handful of SEVIR events from AWS and run the `eie-sevir` tutorial end to end.
3. Pick **2–3 documented Indian severe events** (e.g. a known cloudburst/hailstorm) to target for the port narrative.

---

## 4. Architecture (maps to the 3-column poster)

```
INGEST → PROCESS → DETECT/INITIATE → NOWCAST → HAZARD SCORE → STORE/SERVE → ALERT → FEEDBACK
```

### A. Ingestion & replay (Platform)
- One **adapter per source** normalising to a common schema `{time, grid, var, values}`.
- **Replay archive**: reads a stored event and emits frames on a timer over Redis Streams to imitate a 5-min live cadence.
- QC: clutter removal, de-duplication, missing-frame handling.

### B. Processing & enrichment
- Georeference + regrid to common **1 km / 5-min** grid (SEVIR already ~this).
- Feature layers: reflectivity/VIL, echo area, **cloud-top temp & cooling rate**, lightning rate & **lightning jump**, motion vector, per-cell climatology (normals).

### C. Cell detection & convective initiation — **ML, primary**
- Threshold detect existing cells (VIL/≈35–40 dBZ equivalent) → connected-component labelling → **persistent Storm IDs** with explicit **merge/split inheritance** (child IDs keep lineage).
- **CI model:** learned classifier over patch features (cooling rate + lightning jump + echo-top growth) → probability a *new* cell initiates within N minutes. This is the differentiator: catches storms **before** they show on radar — the thing optical flow cannot do.

### D. Nowcast engine — **ML, primary (D1)**
- **Baseline (scored against):** persistence + pySTEPS optical-flow / S-PROG.
- **Primary model:** start with a **U-Net** (frames-in → frames-out on VIL); upgrade path to **ConvLSTM**, then **DGMR-style** generative nowcast for amplitude preservation (stretch).
- Ensemble of perturbed runs → arrival window (distance ÷ motion) + probability.
- **Amplitude preservation** is a design goal (avoid the spectral-smoothing peak-flattening); track it explicitly in validation.

### E. Hazard scoring (fusion) — 3 now, 1 roadmap
| Hazard | Method (prototype) | Data |
|--------|--------------------|------|
| Lightning density | strike counts / jump on grid | GLM (SEVIR) |
| Hail probability | high-VIL + cold cloud tops + lightning rate → classifier | VIL + IR + GLM |
| Cloudburst | nowcast VIL → rain-rate proxy vs intense-rain threshold (P(rain > 100 mm/hr)) | VIL |
| **Downburst** | **requires radial velocity — ROADMAP** | DWR velocity (MOSDAC, finale) |
- Fusion is **auditable**: each source's contribution is shown, not hidden.

### F. Store & serve
- **Redis** (live/replay state) → **PostGIS** (history, tracks, verified outcomes) → **FastAPI REST + WebSocket** → **React + MapLibre**.

### G. Monitoring & validation (continuous)
- POD / FAR / CSI / **FSS** by lead time + reliability diagram.
- Pipeline health: data-in→alert latency, missing-feed alarms.
- Drift checks → retrain trigger → versioned models + audit logs.

### Alerts & feedback (purple column)
- **Severity = hazard × arrival probability × exposure** (people/roads/farms).
- Plain-language reason generated from top SHAP features → route to authority → alert record.
- Multi-channel delivery (prototype: log + webhook; optional Twilio SMS/WhatsApp), per-location subscriptions, de-dup, escalation if unacknowledged.
- Outcome feedback: hit / false alarm / miss + evidence → recalibrate thresholds & retrain. **Audience-specific thresholds** (aviation low-miss vs farmer low-false-alarm) that self-adjust.

---

## 5. Explainability
- Hazard decision is a **tabular classifier** over interpretable features (VIL stats, cooling rate, lightning jump, echo-top growth).
- **SHAP** per prediction → "why this alert": top contributing signals, in language an officer can verify against their own judgment.
- CI score is a single interpretable index (cooling + jump + growth), not a black box.

---

## 6. Repo structure

```
deadlock-nowcast/
├── docker-compose.yml
├── README.md
├── data/
│   ├── loaders/        # SEVIR loader (eie-sevir wrapper), MOSDAC adapter (stub)
│   └── replay/         # replay engine → Redis Streams
├── pipeline/
│   ├── preprocess/     # regrid, QC, feature extraction
│   ├── detect/         # cell detection, Storm ID, merge/split
│   ├── nowcast/
│   │   ├── baseline/   # persistence, pysteps
│   │   └── models/     # unet.py, convlstm.py, (dgmr.py stretch)
│   ├── initiation/     # CI classifier
│   └── hazards/        # lightning, hail, cloudburst scorers + SHAP
├── services/
│   ├── api/            # FastAPI + WebSocket
│   ├── state/          # Redis client
│   └── db/             # PostGIS models, migrations
├── frontend/           # React + MapLibre (layers, scrubber, countdown, alerts)
├── alerts/             # severity, reason-gen, channel adapters
├── eval/               # POD/FAR/CSI/FSS, reliability, plots
└── notebooks/          # EDA, baseline-vs-model comparison
```

---

## 7. Tech stack

- **ML:** Python, PyTorch, pysteps, xarray, numpy, scikit-learn, SHAP, h5py (SEVIR is HDF5).
- **Backend:** FastAPI, Uvicorn, Redis (+ Streams), PostgreSQL/PostGIS, SQLAlchemy.
- **Frontend:** React, MapLibre GL, deck.gl (optional for grid layers), Vite.
- **Alerts:** webhook + optional Twilio; CAP-formatted output for realism.
- **Infra:** Docker Compose; single-GPU inference target (cloud GPU node).

---

## 8. Milestones

### Idea stage (now → submission) — MVP demo
1. SEVIR loaded, one event replaying over Redis. *(data+replay)*
2. Baseline nowcast (persistence + pysteps) with CSI numbers. *(gives us the bar)*
3. U-Net nowcast trained on VIL; **beats baseline CSI at 0–2 h** on held-out events. *(the core claim)*
4. Hazard scoring (lightning/hail/cloudburst) + SHAP reasons.
5. Dashboard: map + hazard layer + replay scrubber + one alert with plain-language reason.
6. Eval slide: POD/FAR/CSI/FSS vs baseline; downburst + Indian data as roadmap.

### Finale (if selected)
- ConvLSTM/DGMR upgrade for amplitude preservation.
- Merge/split-aware tracking hardened.
- INSAT-3D/3DR + DWR ingestion from MOSDAC; downburst via velocity.
- Multi-channel alerts live; audience-specific self-calibrating thresholds.
- CAP integration with IMD/NDMA-style dissemination format.

---

## 9. Validation plan (the credibility section)

- **Task:** predict next 0–2 h VIL; threshold to hazard masks.
- **Baselines:** persistence, pysteps optical flow.
- **Metrics:** CSI, POD, FAR at multiple VIL thresholds; **FSS** (fractions skill score) for spatial credit; reliability diagram for probabilities.
- **Headline target:** *"Learned nowcast beats optical-flow CSI at 0–3 h lead on N held-out SEVIR severe events, with FSS gains at convective thresholds."*
- **Operational metrics:** false-alarm-rate reduction and (simulated) alert-fatigue reduction from audience-specific thresholds vs a single fixed threshold.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| ML demoted to optional again (the recurring failure) | D1 is non-negotiable: a learned model owns 0–2 h; baseline is only for comparison. |
| Data blocker on Indian sources | Build entirely on SEVIR; register MOSDAC now; port is additive, not required for the demo. |
| "Real-time" overclaim | Label the demo as **replay** everywhere; never imply a live feed. |
| Downburst indefensible on SEVIR | Explicitly velocity-gated roadmap (D4). |
| Rare-event scarcity (severe cases few) | Oversample severe events; synthetic augmentation; **validate skill per lead-time bucket**, not pooled. |
| Thin core hidden by operational polish | Validation is its own deliverable (D6) with concrete numbers. |
| Compute for high-res | Cheap optical-flow at very short range where tolerable; heavier ML where latency budget allows — but the *scored* claim rests on ML skill, not on where it runs. |

---

## 11. Do-this-week checklist
- [ ] Register on MOSDAC (SSO); request INSAT-3D/3DR + DWR products.
- [ ] Pull SEVIR sample events from AWS; run `eie-sevir` tutorial.
- [ ] Stand up baseline (persistence + pysteps) and record CSI — this is the bar to beat.
- [ ] Pick 2–3 documented Indian severe events for the port narrative.
- [ ] Scaffold repo + `docker compose` skeleton (Redis, PostGIS, FastAPI, React).

---

## 12. References
- SEVIR dataset (AWS Open Data): https://registry.opendata.aws/sevir/
- SEVIR tooling & baselines: https://github.com/MIT-AI-Accelerator/eie-sevir
- SEVIR paper (Veillette et al., 2020, NeurIPS)
- MOSDAC (INSAT-3D/3DR, DWR): https://www.mosdac.gov.in/ (registration required)
- MOSDAC Open Data: https://www.mosdac.gov.in/open-data
- pysteps (optical-flow nowcasting baseline): https://pysteps.github.io/
- IMD gridded rainfall: `imdlib` (PyPI)
- ERA5 reanalysis: Copernicus Climate Data Store

---
*Scope note for judges: prototype runs on replayed historical SEVIR storms; Indian INSAT/DWR is the port target pending MOSDAC access; downburst requires Doppler velocity and is a roadmap item. Everything shown is labelled real vs stand-in.*
