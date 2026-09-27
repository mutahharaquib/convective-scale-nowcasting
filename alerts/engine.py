"""Severity = hazard x arrival probability x exposure; audience-specific self-calibrating
thresholds; plain-language reasons from SHAP; CAP output; multi-channel delivery.
Lifecycle (forecaster flow steps 6-9): review (AI-proposed) -> issued -> responded -> closed,
or review -> dismissed. Nothing reaches the public until a forecaster issues it."""
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
RECIPIENTS = {"aviation": "Airport ops duty officer", "disaster_mgmt": "District EOC control room",
              "farmer": "Farmer advisory list (KVK)"}   # stand-in contact lists
CHANNELS = ("sms", "whatsapp", "email", "webhook", "in_app")

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


def level(sev):
    return "Extreme" if sev > 0.7 else "Severe" if sev > 0.45 else "Moderate"


def reason_text(asset, hazard, prob, arrival, shap_reasons):
    why = "; ".join(PHRASE[r["feature"]](r["value"]) for r in shap_reasons) or "nowcast intensity above threshold"
    eta = f"in ~{arrival:.0f} min" if arrival else "now"
    return f"{HAZ_NAME[hazard]} expected at {asset} {eta} (p={prob:.2f}). Why: {why}."


def to_cap(a):
    return (f'<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2"><identifier>{a["id"]}</identifier>'
            f'<sender>deadlock-nowcast</sender><sent>{time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}</sent>'
            f'<status>Exercise</status><msgType>Alert</msgType><scope>Public</scope><info><category>Met</category>'
            f'<event>{HAZ_NAME[a["hazard"]]}</event><urgency>Immediate</urgency><severity>{a["level"]}</severity>'
            f'<certainty>Likely</certainty><expires>{a.get("valid_min", 60)} min</expires>'
            f'<description>{escape(a["text"])}</description>'
            f'<instruction>{escape(a.get("notes") or "")}</instruction>'
            f'<area><areaDesc>{escape(a["asset"])}</areaDesc>'
            f'<circle>{a["lat"]:.3f},{a["lon"]:.3f} {a.get("radius_km", 10)}</circle></area>'
            f'</info></alert>')


def deliver(a, channels=CHANNELS):
    """Write alert record + CAP, then fan out per channel. Real send where configured, else a logged mock."""
    os.makedirs("alerts/outbox", exist_ok=True)
    with open("alerts/outbox/alerts.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: a[k] for k in ("id", "asset", "hazard", "severity", "level", "text")}) + "\n")
    with open(f"alerts/outbox/{a['id']}.cap.xml", "w", encoding="utf-8") as f:
        f.write(to_cap(a))
    body, to, out = f"[{a['level'].upper()}] {a['text']}"[:300], RECIPIENTS[a["audience"]], []
    for ch in channels:
        status = "sent (mock)"
        if ch == "webhook" and (url := os.getenv("WEBHOOK_URL")):
            try: httpx.post(url, json=a, timeout=3); status = "sent"
            except Exception as e: status = f"failed: {e}"
        if ch in ("sms", "whatsapp") and (sid := os.getenv("TWILIO_SID")) and (num := os.getenv("TWILIO_TO")):
            pre = "whatsapp:" if ch == "whatsapp" else ""
            try:
                httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                           auth=(sid, os.environ["TWILIO_TOKEN"]),
                           data=dict(From=pre + os.environ["TWILIO_FROM"], To=pre + num, Body=body), timeout=5)
                status = "sent"
            except Exception as e: status = f"failed: {e}"
        if ch == "in_app": status = "delivered"
        out.append(dict(ts=time.time(), alert=a["id"], channel=ch, to=to, status=status))
    with open("alerts/outbox/deliveries.log", "a", encoding="utf-8") as f:
        for d in out: f.write(json.dumps(d) + "\n")
    return out


class AlertEngine:
    def __init__(self):
        self.thresholds = dict(THRESHOLDS)
        self.active = {}          # dedup key -> alert (current event)
        self.all = {}             # id -> alert, survives event switches so late feedback still counts
        self.deliveries = []
        self.threshold_log = []
        self.stats = {"raised": 0, "suppressed_dedup": 0, "suppressed_threshold": 0,
                      "fixed_threshold_would_raise": 0, "issued": 0, "dismissed": 0,
                      "hit": 0, "false_alarm": 0, "miss": 0}

    def evaluate(self, event, t, candidates):
        """candidates: dicts with asset, hazard, prob, arrival_min, reasons, lat, lon, exposure, audience, sid."""
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
                     created=time.time(), severity=sev, level=level(sev), prob=c["prob"],
                     text=reason_text(c["asset"], c["hazard"], c["prob"], c["arrival_min"], c["reasons"]),
                     reason=c["reasons"], escalated=False, status="review", outcome=None, notes="", actions=[],
                     audit=[dict(ts=time.time(), user="AI", action="proposed", detail=f"severity {sev}")],
                     **{k: c.get(k) for k in ("asset", "hazard", "audience", "lat", "lon", "arrival_min", "sid")})
            self.active[key] = a; self.all[a["id"]] = a; new.append(a); self.stats["raised"] += 1
        for a in self.active.values():   # escalate AI proposals nobody has reviewed
            if a["status"] == "review" and t - a["t"] >= ESCALATE_FRAMES and not a["escalated"]:
                a["escalated"] = True
                self._log(a, "system", "escalated", "unreviewed for 15 min of storm time")
        return new

    def _log(self, a, user, action, detail=""):
        a["audit"].append(dict(ts=time.time(), user=user, action=action, detail=detail))

    def _nudge(self, a, step, cause):
        aud, old = a["audience"], self.thresholds[a["audience"]]
        self.thresholds[aud] = round(min(0.9, max(0.1, old + step)), 3)
        if self.thresholds[aud] != old:
            self.threshold_log.append(dict(ts=time.time(), alert=a["id"], audience=aud, old=old,
                                           new=self.thresholds[aud], cause=cause))

    def issue(self, aid, user, level=None, radius_km=10, valid_min=60, notes="", channels=CHANNELS):
        a = self.all[aid]
        a.update(status="issued", level=level or a["level"], radius_km=radius_km, valid_min=valid_min, notes=notes)
        self._log(a, user, "issued", f"{a['level']} · {radius_km} km · valid {valid_min} min" + (f" · {notes}" if notes else ""))
        d = deliver(a, channels); self.deliveries += d; a["deliveries"] = d
        self._log(a, "system", "notified", ", ".join(f"{x['channel']}: {x['status']}" for x in d))
        self.stats["issued"] += 1
        return d

    def ack(self, aid):
        """Classic dashboard 'Ack' = forecaster confirms the AI proposal and issues it on all channels."""
        if (a := self.all.get(aid)) and a["status"] == "review":
            self.issue(aid, "forecaster")
        if a: a["acked"] = True

    def dismiss(self, aid, user, reason):
        a = self.all[aid]; a.update(status="dismissed", dismiss_reason=reason)
        self._log(a, user, "dismissed", reason)
        self._nudge(a, +0.015, "forecaster dismissal")   # soft false alarm
        self.stats["dismissed"] += 1

    def respond(self, aid, user, actions):
        a = self.all[aid]; a.update(actions=actions, status="responded")
        self._log(a, user, "field response", ", ".join(actions) or "none")

    def feedback(self, aid, outcome, user="forecaster", notes="", evidence=None):
        """hit / false_alarm / miss -> nudge that audience's threshold (self-calibration), close the alert."""
        if a := self.all.get(aid):
            a.update(outcome=outcome, status="closed", outcome_notes=notes, evidence=evidence)
            self._log(a, user, f"outcome: {outcome}", notes + (" · photo attached" if evidence else ""))
            self._nudge(a, {"false_alarm": +0.03, "miss": -0.03, "hit": 0.0}[outcome], outcome)
            self.stats[outcome] += 1
            return self.thresholds
