"""Notificaciones opcionales por correo (Resend/SendGrid) o SMS (Twilio). Solo confirma lo que el proveedor confirma."""
from __future__ import annotations
import re
import requests
from config import get_secret

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")


def available():
    return {"email": "resend" if get_secret("RESEND_API_KEY") else ("sendgrid" if get_secret("SENDGRID_API_KEY") else None),
            "sms": "twilio" if (get_secret("TWILIO_ACCOUNT_SID") and get_secret("TWILIO_AUTH_TOKEN")
                                and get_secret("TWILIO_FROM")) else None}


def build_summary(report, report_url=None):
    def f(v, unit=""):
        return "no disponible" if v is None else f"{v}{unit}"
    lines = [f"Análisis de flujo de personas finalizado (sesión {report['session_id'][:8]}).",
             f"Inicio: {report['started_at']}", f"Fin: {report['finished_at']}",
             f"Duración analizada: {f(report['duration_seconds'], ' s')}",
             f"Personas únicas (estimado): {f(report['unique_people_estimated'])}",
             f"Entradas: {f(report['entries'])} · Salidas: {f(report['exits'])}",
             f"Calidad del análisis: {report['quality']['level']}"]
    if report_url:
        lines.append(f"Reporte: {report_url}")
    return "Reporte de flujo de personas", "\n".join(lines)


def _res(ok, provider, status, provider_id=None, error=None):
    return {"ok": ok, "provider": provider, "status": status, "provider_id": provider_id, "error": error}


def send_email(to, subject, text, http=None):
    http = http or requests
    if not EMAIL_RE.match(to or ""):
        return _res(False, None, "no enviado", error="Correo inválido")
    sender = get_secret("EMAIL_FROM")
    prov = available()["email"]
    if not prov or not sender:
        return _res(False, None, "no enviado", error="Proveedor de correo no configurado (RESEND_API_KEY o SENDGRID_API_KEY, y EMAIL_FROM)")
    try:
        if prov == "resend":
            r = http.post("https://api.resend.com/emails", timeout=15,
                          headers={"Authorization": f"Bearer {get_secret('RESEND_API_KEY')}"},
                          json={"from": sender, "to": [to], "subject": subject, "text": text})
            pid = r.json().get("id") if r.status_code == 200 else None
            return _res(bool(pid), prov, "aceptado por el proveedor" if pid else "no enviado", pid,
                        None if pid else f"HTTP {r.status_code}")
        r = http.post("https://api.sendgrid.com/v3/mail/send", timeout=15,
                      headers={"Authorization": f"Bearer {get_secret('SENDGRID_API_KEY')}"},
                      json={"personalizations": [{"to": [{"email": to}]}], "from": {"email": sender},
                            "subject": subject, "content": [{"type": "text/plain", "value": text}]})
        ok = r.status_code == 202
        return _res(ok, prov, "aceptado por el proveedor" if ok else "no enviado",
                    r.headers.get("X-Message-Id") if ok else None, None if ok else f"HTTP {r.status_code}")
    except (requests.RequestException, ValueError) as e:
        return _res(False, prov, "no enviado", error=type(e).__name__)


def send_sms(to, text, http=None):
    http = http or requests
    if not PHONE_RE.match(to or ""):
        return _res(False, None, "no enviado", error="Teléfono inválido (usa formato internacional, ej. +521234567890)")
    if not available()["sms"]:
        return _res(False, None, "no enviado", error="Twilio no configurado (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM)")
    sid = get_secret("TWILIO_ACCOUNT_SID")
    try:
        r = http.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", timeout=15,
                      auth=(sid, get_secret("TWILIO_AUTH_TOKEN")),
                      data={"To": to, "From": get_secret("TWILIO_FROM"), "Body": text[:1500]})
        msid = r.json().get("sid") if r.status_code in (200, 201) else None
        return _res(bool(msid), "twilio", "aceptado por el proveedor (no garantiza entrega)" if msid else "no enviado",
                    msid, None if msid else f"HTTP {r.status_code}")
    except (requests.RequestException, ValueError) as e:
        return _res(False, "twilio", "no enviado", error=type(e).__name__)
