import json, threading, hashlib, hmac
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd
import pytest
import report as rp, webhook as wh, notifier as nt
from analytics import analyze
from zones import SpaceConfig, EntryLine, Zone

SIZE = (1000, 500)
PINFO = {"frames_procesados": 20, "stride": 1, "fps_video": 1.0, "segundos_analizados": 20.0,
         "errores_detector": 0, "tamano_procesamiento": [1000, 500], "detenido_manual": False}


def tracks():
    ys = [100 + 30 * t for t in range(10)] + [370 - 30 * (t - 9) for t in range(10, 20)]
    rows = [(1, t, 500, y) for t, y in enumerate(ys)]
    return pd.DataFrame(rows, columns=["id", "t", "x", "y"])


def make_report(cfg):
    a = analyze(tracks(), cfg, SIZE, 20, 1)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return rp.build_report("abc12345" * 4, t0, t0 + timedelta(seconds=30), a, PINFO, cfg)


CFG = SpaceConfig(EntryLine((0, .5), (1, .5), 1), [Zone("Z", [(0, 0), (1, 0), (1, 1), (0, 1)])], SIZE)


def test_report_real_values_and_null_when_unknown():
    r = make_report(CFG)
    assert r["entries"] == 1 and r["exits"] == 1 and r["average_dwell_seconds"] == pytest.approx(8.0)
    assert r["unique_people_estimated"] == 1 and r["zones"][0]["name"] == "Z"
    r2 = make_report(SpaceConfig())
    assert r2["entries"] is None and r2["exits"] is None and r2["average_dwell_seconds"] is None
    assert any("línea" in w for w in r2["warnings"])
    json.dumps(r2)


def test_select_fields_and_csv():
    r = make_report(CFG)
    p = rp.select_fields(r, ["entries"])
    assert set(p) == {"event", "schema_version", "session_id", "entries"}
    assert "unique_people_estimated" in rp.report_csv(r)


# ---- webhook con servidor local real ----
class H(BaseHTTPRequestHandler):
    log = []; codes = []
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        H.log.append((dict(self.headers), body))
        code = H.codes.pop(0) if H.codes else 200
        self.send_response(code); self.end_headers()
    def log_message(self, *a): pass


@pytest.fixture
def server():
    H.log, H.codes = [], []
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/hook?token=SECRETO"
    srv.shutdown()


OPT = dict(allow_private=True, allow_http=True, sleep=lambda s: None)


def test_webhook_signature_valid(server):
    r = wh.send(server, {"session_id": "s1", "event": "x"}, "shh", **OPT)
    assert r["ok"] and r["attempts"] == 1
    headers, body = H.log[0]
    expect = "sha256=" + hmac.new(b"shh", headers["X-Timestamp"].encode() + b"." + body, hashlib.sha256).hexdigest()
    assert headers["X-Signature"] == expect and json.loads(body)["session_id"] == "s1"


def test_webhook_retries_5xx_but_not_4xx(server):
    H.codes = [500, 503, 200]
    r = wh.send(server, {"a": 1}, None, **OPT)
    assert r["ok"] and r["attempts"] == 3
    H.codes, H.log[:] = [400], []
    r = wh.send(server, {"a": 1}, None, **OPT)
    assert not r["ok"] and r["attempts"] == 1 and not r["retryable"] and len(H.log) == 1


def test_webhook_gives_up_after_max_attempts(server):
    H.codes = [500] * 5
    r = wh.send(server, {}, None, max_attempts=2, **OPT)
    assert not r["ok"] and r["attempts"] == 2 and r["retryable"]


def test_webhook_url_safety_and_masking():
    assert "no configurado" in wh.send(None, {})["error"]
    assert "HTTPS" in wh.send("http://example.com/x", {})["error"]
    assert "privada" in wh.send("https://127.0.0.1/x", {})["error"]
    m = wh.mask_url("https://hooks.example.com/abc/SECRETO?k=1")
    assert "SECRETO" not in m and "hooks.example.com" in m
    out = wh.send("https://127.0.0.1/x?t=SECRETO", {})
    assert "SECRETO" not in json.dumps(out)


# ---- notificaciones (sin red: cliente simulado) ----
class Resp:
    def __init__(self, code, js=None, headers=None): self.status_code, self._j, self.headers = code, js or {}, headers or {}
    def json(self): return self._j


class FakeHttp:
    def __init__(self, resp): self.resp, self.calls = resp, []
    def post(self, url, **kw): self.calls.append((url, kw)); return self.resp


def test_email_confirmed_only_with_provider_id(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "k"); monkeypatch.setenv("EMAIL_FROM", "a@b.com")
    ok = nt.send_email("x@y.com", "s", "t", FakeHttp(Resp(200, {"id": "em_1"})))
    assert ok["ok"] and ok["provider_id"] == "em_1"
    bad = nt.send_email("x@y.com", "s", "t", FakeHttp(Resp(403, {"message": "no"})))
    assert not bad["ok"] and bad["status"] == "no enviado"
    assert not nt.send_email("no-es-correo", "s", "t")["ok"]


def test_not_configured_is_not_reported_as_sent(monkeypatch):
    for k in ["RESEND_API_KEY", "SENDGRID_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"]:
        monkeypatch.delenv(k, raising=False)
    assert not nt.send_email("x@y.com", "s", "t")["ok"]
    assert not nt.send_sms("+521234567890", "t")["ok"]
    assert not nt.send_sms("123", "t")["ok"]


def test_sms_twilio(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC1"); monkeypatch.setenv("TWILIO_AUTH_TOKEN", "t"); monkeypatch.setenv("TWILIO_FROM", "+15550001111")
    h = FakeHttp(Resp(201, {"sid": "SM1", "status": "queued"}))
    r = nt.send_sms("+521234567890", "hola", h)
    assert r["ok"] and "no garantiza" in r["status"] and h.calls[0][1]["data"]["To"] == "+521234567890"


def test_summary_has_no_pii_and_handles_null():
    _, text = nt.build_summary(make_report(SpaceConfig()))
    assert "no disponible" in text


def test_deliver_all_end_to_end(server, monkeypatch):
    import jobs
    monkeypatch.setenv("WEBHOOK_URL", server); monkeypatch.setenv("WEBHOOK_SECRET", "s")
    monkeypatch.setenv("WEBHOOK_ALLOW_PRIVATE", "1"); monkeypatch.setenv("WEBHOOK_ALLOW_HTTP", "1")
    s = jobs.SessionSettings(webhook_auto=True, webhook_fields=["entries", "unique_people_estimated"])
    payload, out = jobs.deliver_all(make_report(CFG), s)
    assert out[0]["ok"] and out[0]["channel"] == "webhook" and "SECRETO" not in json.dumps(out)
    sent = json.loads(H.log[0][1])
    assert sent["entries"] == 1 and "warnings" not in sent and sent["event"] == "people_flow.analysis.completed"
    H.codes = [500, 500, 500]
    _, out = jobs.deliver_all(make_report(CFG), s, sleep=lambda x: None)
    assert not out[0]["ok"] and out[0]["retryable"] and out[0]["attempts"] == 3
