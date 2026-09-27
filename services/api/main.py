"""FastAPI + WebSocket. Replay loop pushes one pipeline frame per tick.
Run: uvicorn services.api.main:app --port 8000"""
import asyncio, base64, json, os, secrets, time, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from data.replay.engine import Replay
from alerts.engine import AlertEngine, ASSETS, CHANNELS
from services.state import store as state
from services.db import store as db

TICK = float(os.getenv("REPLAY_SECONDS", 2.0))   # wall seconds per 5-min frame
app = FastAPI(title="Deadlock Nowcast (REPLAY)")
R, A = Replay(), AlertEngine()
S = {"t": R.t0, "playing": True, "clients": {}, "last_frame": 0.0, "frames": 0}
LOCK = asyncio.Lock()   # build() mutates shared engine state - never run two at once
ROLES = ("forecaster", "admin", "viewer")
OTP, SESS = {}, {}      # demo auth: in-memory OTP + session tokens
EVAL = json.load(open("eval/results.json")) if os.path.exists("eval/results.json") else {}


class LoginReq(BaseModel):
    email: str
    role: str


class VerifyReq(BaseModel):
    email: str
    otp: str


@app.post("/api/login")
def login(r: LoginReq):
    if "@" not in r.email or r.role not in ROLES: raise HTTPException(400, "valid email and role required")
    code = f"{secrets.randbelow(10 ** 6):06d}"
    OTP[r.email.lower()] = (code, r.role, time.time())
    # Demo mode: no SMS/email gateway, so the OTP is returned to the UI and shown on screen.
    return {"sent": True, "demo_otp": code}


@app.post("/api/verify")
def verify(r: VerifyReq):
    code, role, ts = OTP.get(r.email.lower(), (None, None, 0))
    if code is None or r.otp.strip() != code or time.time() - ts > 300: raise HTTPException(401, "invalid or expired OTP")
    OTP.pop(r.email.lower(), None)
    tok = secrets.token_urlsafe(24)
    SESS[tok] = {"email": r.email.lower(), "role": role, "subs": [a["name"] for a in ASSETS]}
    return {"token": tok, **SESS[tok]}


def pipeline_status(fr, build_ms):
    tm = fr.get("timing", {})
    m = EVAL.get("methods", {})
    csi = lambda k: m.get(k, {}).get("74", {}).get("csi", [None] * 12)[11]
    return dict(
        ingest=dict(source=fr["source"], cadence_min=5, tick_s=TICK, feeds=[
            dict(name="Radar VIL (NEXRAD)", ok=True), dict(name="Satellite IR 10.7 µm (GOES)", ok=True),
            dict(name="Lightning (GLM)", ok=True), dict(name="Doppler velocity (DWR)", ok=False, note="roadmap"),
            dict(name="Terrain + population", ok=None, note="stand-in"), dict(name="Replay archive", ok=True)]),
        process=dict(grid="3 km / 5 min (128×128)", features=8),
        detect=dict(cells=len(fr["cells"]), ci_hotspots=len(fr["ci_hotspots"]), ms=tm.get("detect_hazards"),
                    ci_ms=tm.get("initiation")),
        nowcast=dict(model="U-Net", horizon_min=120, members=9, ms=tm.get("nowcast_unet"), baseline_ms=tm.get("baseline_of")),
        hazards=dict(active=["lightning", "hail", "cloudburst"], roadmap=["downburst"]),
        store=dict(state=state.BACKEND, history="postgis" if db.URL.startswith("postgresql") else "sqlite"),
        monitor=dict(build_ms=round(build_ms), csi60_unet=csi("unet"), csi60_of=csi("optical_flow"),
                     outcomes=A.stats["hit"] + A.stats["false_alarm"] + A.stats["miss"], frames=S["frames"]))


def build(t):
    t_b = time.perf_counter()
    fr = dict(R.frame(t))
    new = A.evaluate(fr["event"], t, fr.pop("candidates"))
    for a in new: db.save_alert(a)
    db.save_cells(fr["event"], t, fr["cells"])
    alerts = sorted(A.all.values(), key=lambda a: ({"review": 0, "issued": 1, "responded": 2}.get(a["status"], 3), -a["severity"]))
    fr.update(alerts=alerts, new_alerts=[a["id"] for a in new],
              thresholds=A.thresholds, threshold_log=A.threshold_log[-30:], alert_stats=A.stats,
              deliveries=A.deliveries[-40:], playing=S["playing"], mode="REPLAY", tick_s=TICK,
              events=[e["id"] for e in R.events], event_index=R.ei, bounds=R.bounds, state_backend=state.BACKEND)
    fr["pipeline"] = pipeline_status(fr, (time.perf_counter() - t_b) * 1000)
    fr["sent_at"] = time.time()
    state.publish_frame(fr)
    S["last_frame"], S["frames"] = time.time(), S["frames"] + 1
    return fr


async def build_async(t):
    async with LOCK:
        return await asyncio.to_thread(build, t)


async def broadcast(fr):
    msg = json.dumps(fr, default=float)
    for ws in list(S["clients"]):
        try: await ws.send_text(msg)
        except Exception: S["clients"].pop(ws, None)


async def loop():
    while True:
        await asyncio.sleep(TICK)
        if S["playing"] and S["clients"]:
            try:
                S["t"] = S["t"] + 1 if S["t"] < R.T else R.t0
                await broadcast(await build_async(S["t"]))
            except Exception:
                traceback.print_exc()   # keep the replay alive


@app.on_event("startup")
async def _start():
    asyncio.create_task(loop())


def save_evidence(aid, data_url):
    if not data_url or not data_url.startswith("data:image/"): return None
    head, b64 = data_url.split(",", 1)
    ext = head.split("/")[1].split(";")[0][:4]
    os.makedirs("alerts/outbox/evidence", exist_ok=True)
    path = f"evidence/{aid}.{ext}"
    with open(f"alerts/outbox/{path}", "wb") as f: f.write(base64.b64decode(b64))
    return f"/outbox/{path}"


WRITE_OPS = {"issue", "dismiss", "respond", "outcome", "ack", "feedback"}


def handle(cmd, user):
    op, who = cmd["op"], user["email"]
    if op in WRITE_OPS and user["role"] == "viewer": return "viewers are read-only"
    if op == "play": S["playing"] = True
    elif op == "pause": S["playing"] = False
    elif op == "seek": S["t"] = max(R.t0, min(R.T, int(cmd["t"])))
    elif op == "event":
        R.select(int(cmd["i"])); A.active.clear(); S["t"] = R.t0
    elif op == "issue":
        A.issue(cmd["id"], who, cmd.get("level"), int(cmd.get("radius_km", 10)), int(cmd.get("valid_min", 60)),
                cmd.get("notes", ""), [c for c in cmd.get("channels", CHANNELS) if c in CHANNELS])
        db.set_alert(cmd["id"], status="issued")
    elif op == "dismiss":
        A.dismiss(cmd["id"], who, cmd.get("reason", "")); db.set_alert(cmd["id"], status="dismissed")
    elif op == "respond":
        A.respond(cmd["id"], who, cmd.get("actions", [])); db.set_alert(cmd["id"], status="responded")
    elif op in ("outcome", "feedback"):
        ev = save_evidence(cmd["id"], cmd.get("evidence"))
        A.feedback(cmd["id"], cmd["outcome"], who, cmd.get("notes", ""), ev)
        db.set_alert(cmd["id"], status="closed", outcome=cmd["outcome"])
    elif op == "ack":
        A.ack(cmd["id"]); db.set_alert(cmd["id"], status="issued")
    elif op == "subs":
        user["subs"] = [n for n in cmd.get("subs", []) if n in {a["name"] for a in ASSETS}]


@app.websocket("/ws")
async def ws(sock: WebSocket):
    user = SESS.get(sock.query_params.get("token", ""))
    if os.getenv("AUTH_DISABLED") and not user: user = {"email": "demo@local", "role": "forecaster", "subs": []}
    if not user:
        await sock.close(code=4401); return
    await sock.accept(); S["clients"][sock] = user
    await sock.send_text(json.dumps(await build_async(S["t"]), default=float))
    try:
        while True:
            cmd = json.loads(await sock.receive_text())
            async with LOCK:
                err = await asyncio.to_thread(handle, cmd, user)
            if err: await sock.send_text(json.dumps({"error": err})); continue
            await broadcast(await build_async(S["t"]))
    except (WebSocketDisconnect, RuntimeError):   # client went away mid-broadcast
        pass
    except Exception:
        traceback.print_exc()
    finally:
        S["clients"].pop(sock, None)


@app.get("/api/eval")
def evaluation():
    return JSONResponse(EVAL)


@app.get("/api/alerts")
def alerts():
    return db.list_alerts()


@app.get("/api/latest")
def latest():
    return state.latest()


@app.get("/api/health")
def health():
    return dict(ok=True, mode="REPLAY", clients=len(S["clients"]), frames=S["frames"],
                last_frame_age_s=round(time.time() - S["last_frame"], 1) if S["last_frame"] else None)


@app.get("/")
def index():
    return FileResponse("frontend/app.html")


@app.get("/classic")
def classic():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
os.makedirs("alerts/outbox", exist_ok=True)
app.mount("/outbox", StaticFiles(directory="alerts/outbox"), name="outbox")
