"""Live/replay state. Redis Streams when REDIS_URL is set, in-memory otherwise."""
import json, os

try:
    import redis
    _r = redis.from_url(os.environ["REDIS_URL"]) if os.getenv("REDIS_URL") else None
    if _r: _r.ping()
except Exception:
    _r = None

_mem = {}


def publish_frame(frame: dict):
    """Append a replay frame summary to the 'nowcast:frames' stream (capped)."""
    slim = {k: frame[k] for k in ("event", "t", "time", "cells", "alerts") if k in frame}
    if _r:
        _r.xadd("nowcast:frames", {"data": json.dumps(slim)}, maxlen=500)
        _r.set("nowcast:latest", json.dumps(slim))
    _mem["latest"] = slim


def latest():
    if _r:
        v = _r.get("nowcast:latest")
        return json.loads(v) if v else None
    return _mem.get("latest")


BACKEND = "redis" if _r else "memory"
