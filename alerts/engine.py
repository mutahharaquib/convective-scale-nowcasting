"""Severity = hazard x arrival probability x exposure; audience-specific self-calibrating
thresholds; plain-language reasons from SHAP; CAP output; log/webhook/Twilio delivery."""
import json, os, time
from xml.sax.saxutils import escape
import httpx

# Stand-in exposure layer (SEVIR has no population/assets) - illustrative placements.
ASSETS = [
    dict(name="Airport", fx=0.62, fy=0.70, exposure=0.9, audience="aviation"),
    dict(name="City centre", fx=0.50, fy=0.50, exposure=1.0, audience="disaster_mgmt"),
    dict(name="Farm belt N", fx=0.30, fy=0.25, exposure=0.5, audience="farmer"),
    dict(name="Highway junction", fx=0.75, fy=0.35, exposure=0.7, audience="disaster_mgmt"),
    dict(name="Farm belt S", fx=0.25, fy=0.80, exposure=0.5, audience="farmer"),
]
# aviation = low-miss (low threshold); farmer = low-false-alarm (high threshold)
THRESHOLDS = {"aviation": 0.25, "disaster_mgmt": 0.40, "farmer": 0.60}
ESCALATE_FRAMES = 3

PHRASE = {
    "lght_jump": lambda v: f"lightning jumped by {v:+.0f} flashes/5 min",
    "lght_rate": lambda v: f"{v:.0f} flashes in the last 5 min",
    "cooling_rate": lambda v: f"cloud tops cooled {abs(v):.0f} K in 15 min",
    "ir_min": lambda v: f"cloud tops at {v - 273.15:.0f} °C",
    "vil_max": lambda v: f"radar core VIL {v:.0f} (severe >160)",
    "vil_mean": lambda v: f"mean VIL {v:.0f}",
    "vil_growth": lambda v: f"core intensified by {v:+.0f} VIL in 15 min",
    "area": lambda v: f"storm spans ~{v * 9:.0f} km²",
}
HAZ_NAME = {"hail": "Hail", "cloudburst": "Cloudburst (>100 mm/h)", "lightning": "Lightning"}


def reason_text(asset, hazard, prob, arrival, shap_reasons):
    why = "; ".join(PHRASE[r["feature"]](r["value"]) for r in shap_reasons) or "nowcast intensity above threshold"
    eta = f"in ~{arrival:.0f} min" if arrival else "now"
    return f"{HAZ_NAME[hazard]} expected at {asset} {eta} (p={prob:.2f}). Why: {why}."


def to_cap(a):
    sev = "Extreme" if a["severity"] > 0.7 else "Severe" if a["severity"] > 0.45 else "Moderate"
    return (f'<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2"><identifier>{a["id"]}</identifier>'
            f'<sender>deadlock-nowcast</sender><sent>{time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}</sent>'
            f'<status>Exercise</status><msgType>Alert</msgType><scope>Public</scope><info><category>Met</category>'
            f'<event>{HAZ_NAME[a["hazard"]]}</event><urgency>Immediate</urgency><severity>{sev}</severity>'
            f'<certainty>Likely</certainty><description>{escape(a["text"])}</description>'
            f'<area><areaDesc>{escape(a["asset"])}</areaDesc><circle>{a["lat"]:.3f},{a["lon"]:.3f} 10</circle></area>'
            f'</info></alert>')


def deliver(a):
    os.makedirs("alerts/outbox", exist_ok=True)
    with open("alerts/outbox/alerts.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: a[k] for k in ("id", "asset", "hazard", "severity", "text")}) + "\n")
    with open(f"alerts/outbox/{a['id']}.cap.xml", "w", encoding="utf-8") as f:
        f.write(to_cap(a))
    if url := os.getenv("WEBHOOK_URL"):
        try: httpx.post(url, json=a, timeout=3)
        except Exception as e: print("webhook failed:", e)
    if (sid := os.getenv("TWILIO_SID")) and (to := os.getenv("TWILIO_TO")):
        try:
            httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                       auth=(sid, os.environ["TWILIO_TOKEN"]),
                       data=dict(From=os.environ["TWILIO_FROM"], To=to, Body=a["text"][:300]), timeout=5)
        except Exception as e: print("twilio failed:", e)


class AlertEngine:
    def __init__(self):
        self.thresholds = dict(THRESHOLDS)
        self.active = {}          # dedup key -> alert (current event)
        self.all = {}             # id -> alert, survives event switches so late feedback still counts
        self.stats = {"raised": 0, "suppressed_dedup": 0, "suppressed_threshold": 0,
                      "fixed_threshold_would_raise": 0}

    def evaluate(self, event, t, candidates):
        """candidates: dicts with asset, hazard, prob, arrival_min, reasons, lat, lon, exposure, audience."""
        new = []
        for c in candidates:
            arr_p = 1.0 if c["arrival_min"] is None else max(0.3, 1 - c["arrival_min"] / 150)
            sev = round(c["prob"] * arr_p * c["exposure"], 3)
            if sev >= 0.40: self.stats["fixed_threshold_would_raise"] += 1
            if sev < self.thresholds[c["audience"]]:
                self.stats["suppressed_threshold"] += 1; continue
            key = (event, c["asset"], c["hazard"])
            if key in self.active:
                self.active[key]["severity"] = max(self.active[key]["severity"], sev)
                self.stats["suppressed_dedup"] += 1; continue
            a = dict(id=f"{event}-{c['asset'].replace(' ', '')}-{c['hazard']}-{t}", event=event, t=t,
                     severity=sev, text=reason_text(c["asset"], c["hazard"], c["prob"], c["arrival_min"], c["reasons"]),
                     reason=c["reasons"], escalated=False, acked=False,
                     **{k: c[k] for k in ("asset", "hazard", "audience", "lat", "lon", "arrival_min")})
            self.active[key] = a; self.all[a["id"]] = a; new.append(a); self.stats["raised"] += 1
            deliver(a)
        for a in self.active.values():   # escalate unacknowledged alerts
            if not a["acked"] and t - a["t"] >= ESCALATE_FRAMES and not a["escalated"]:
                a["escalated"] = True
        return new

    def feedback(self, aid, outcome):
        """hit / false_alarm / miss -> nudge that audience's threshold (self-calibration)."""
        if a := self.all.get(aid):
            aud = a["audience"]
            step = {"false_alarm": +0.03, "miss": -0.03, "hit": 0.0}[outcome]
            self.thresholds[aud] = round(min(0.9, max(0.1, self.thresholds[aud] + step)), 3)
            return self.thresholds

    def ack(self, aid):
        if aid in self.all: self.all[aid]["acked"] = True
