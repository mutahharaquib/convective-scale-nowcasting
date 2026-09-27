"""History store: alerts, storm tracks, verified outcomes.
PostGIS via DATABASE_URL (postgresql://...), SQLite fallback for local runs."""
import json, os, time
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, Text, text, select, update

URL = os.getenv("DATABASE_URL", "sqlite:///nowcast.db")
eng = create_engine(URL, future=True)
md = MetaData()
alerts = Table("alerts", md,
               Column("id", String, primary_key=True), Column("created", Float), Column("event", String),
               Column("t", Integer), Column("asset", String), Column("hazard", String), Column("audience", String),
               Column("severity", Float), Column("lat", Float), Column("lon", Float), Column("arrival_min", Float),
               Column("reason", Text), Column("status", String, default="open"), Column("outcome", String))
tracks = Table("tracks", md, Column("sid", String), Column("event", String), Column("t", Integer),
               Column("lat", Float), Column("lon", Float), Column("vil_max", Float))
md.create_all(eng)
if URL.startswith("postgresql"):
    with eng.begin() as c:   # geometry columns for spatial queries
        c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        for t in ("alerts", "tracks"):
            c.execute(text(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS geom geometry(Point,4326) "
                           f"GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lon,lat),4326)) STORED"))


def save_alert(a):
    with eng.begin() as c:
        if not c.execute(select(alerts.c.id).where(alerts.c.id == a["id"])).first():
            c.execute(alerts.insert().values(created=time.time(), **{k: a[k] for k in (
                "id", "event", "t", "asset", "hazard", "audience", "severity", "lat", "lon", "arrival_min")},
                reason=json.dumps(a["reason"]), status="open"))


def save_cells(event, t, cells):
    with eng.begin() as c:
        c.execute(tracks.insert(), [dict(sid=x["sid"], event=event, t=t, lat=x["lat"], lon=x["lon"],
                                         vil_max=x["vil_max"]) for x in cells]) if cells else None


def set_alert(aid, **kw):
    with eng.begin() as c:
        c.execute(update(alerts).where(alerts.c.id == aid).values(**kw))


def list_alerts(limit=100):
    with eng.begin() as c:
        rows = c.execute(select(alerts).order_by(alerts.c.created.desc()).limit(limit)).mappings().all()
    return [dict(r, reason=json.loads(r["reason"])) for r in rows]
