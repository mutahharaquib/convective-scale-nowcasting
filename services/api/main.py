"""FastAPI + WebSocket. Replay loop pushes one pipeline frame per tick.
Run: uvicorn services.api.main:app --port 8000"""
import asyncio, json, os, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from data.replay.engine import Replay
from alerts.engine import AlertEngine
from services.state import store as state
from services.db import store as db

TICK = float(os.getenv("REPLAY_SECONDS", 2.0))   # wall seconds per 5-min frame
app = FastAPI(title="Deadlock Nowcast (REPLAY)")
R, A = Replay(), AlertEngine()
S = {"t": R.t0, "playing": True, "clients": set()}
LOCK = asyncio.Lock()   # build() mutates shared engine state - never run two at once


async def build_async(t):
    async with LOCK:
        return await asyncio.to_thread(build, t)


def build(t):
    fr = dict(R.frame(t))
    new = A.evaluate(fr["event"], t, fr.pop("candidates"))
    for a in new: db.save_alert(a)
    db.save_cells(fr["event"], t, fr["cells"])
    fr.update(alerts=sorted(A.active.values(), key=lambda a: -a["severity"]), new_alerts=[a["id"] for a in new],
              thresholds=A.thresholds, alert_stats=A.stats, playing=S["playing"], mode="REPLAY",
              events=[e["id"] for e in R.events], event_index=R.ei, bounds=R.bounds, state_backend=state.BACKEND)
    state.publish_frame(fr)
    return fr


async def broadcast(fr):
    msg = json.dumps(fr, default=float)
    for ws in list(S["clients"]):
        try: await ws.send_text(msg)
        except Exception: S["clients"].discard(ws)


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


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept(); S["clients"].add(sock)
    await sock.send_text(json.dumps(await build_async(S["t"]), default=float))
    try:
        while True:
            cmd = json.loads(await sock.receive_text())
            if cmd["op"] == "play": S["playing"] = True
            elif cmd["op"] == "pause": S["playing"] = False
            elif cmd["op"] == "seek": S["t"] = max(R.t0, min(R.T, int(cmd["t"])))
            elif cmd["op"] == "event":
                async with LOCK:
                    await asyncio.to_thread(R.select, int(cmd["i"])); A.active.clear(); S["t"] = R.t0
            elif cmd["op"] == "ack": A.ack(cmd["id"]); db.set_alert(cmd["id"], status="acked")
            elif cmd["op"] == "feedback":
                A.feedback(cmd["id"], cmd["outcome"]); db.set_alert(cmd["id"], outcome=cmd["outcome"])
            await broadcast(await build_async(S["t"]))
    except (WebSocketDisconnect, RuntimeError):   # client went away mid-broadcast
        pass
    finally:
        S["clients"].discard(sock)


@app.get("/api/eval")
def evaluation():
    return JSONResponse(json.load(open("eval/results.json"))) if os.path.exists("eval/results.json") else {}


@app.get("/api/alerts")
def alerts():
    return db.list_alerts()


@app.get("/api/latest")
def latest():
    return state.latest()


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
os.makedirs("alerts/outbox", exist_ok=True)
app.mount("/outbox", StaticFiles(directory="alerts/outbox"), name="outbox")
