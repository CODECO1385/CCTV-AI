"""Envío de reportes por webhook firmado (HMAC-SHA256). La URL y el secreto viven en el servidor."""
from __future__ import annotations
import hashlib
import hmac
import ipaddress
import json
import socket
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests
from config import get_secret


def mask_url(url):
    """Muestra solo esquema y host: la ruta/query suelen contener el secreto."""
    p = urlparse(url or "")
    return f"{p.scheme}://{p.hostname}/…" if p.hostname else "(no configurada)"


def validate_url(url, allow_private=False, allow_http=False):
    p = urlparse(url or "")
    if p.scheme not in ("https", "http") or not p.hostname:
        raise ValueError("URL de webhook inválida")
    if p.scheme == "http" and not allow_http:
        raise ValueError("El webhook debe usar HTTPS")
    if not allow_private:
        try:
            infos = socket.getaddrinfo(p.hostname, None)
        except socket.gaierror:
            raise ValueError("No se pudo resolver el host del webhook")
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError("El host del webhook apunta a una red privada/local (bloqueado)")


def sign(secret, timestamp, body):
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return "sha256=" + mac


def env_settings(url_override=None):
    return {"url": url_override or get_secret("WEBHOOK_URL"), "secret": get_secret("WEBHOOK_SECRET"),
            "allow_private": get_secret("WEBHOOK_ALLOW_PRIVATE") == "1",
            "allow_http": get_secret("WEBHOOK_ALLOW_HTTP") == "1"}


def send(url, payload, secret=None, *, max_attempts=3, backoff=(1.0, 3.0), timeout=10,
         allow_private=False, allow_http=False, http=None, sleep=time.sleep):
    """Devuelve {ok, attempts, status_code, error, retryable, sent_at}. Reintenta solo errores
    transitorios (red, 429, 5xx) y un número limitado de veces."""
    out = {"ok": False, "attempts": 0, "status_code": None, "error": None, "retryable": False, "sent_at": None}
    if not url:
        out["error"] = "Webhook no configurado"
        return out
    try:
        validate_url(url, allow_private, allow_http)
    except ValueError as e:
        out["error"] = str(e)
        return out
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    http = http or requests
    for attempt in range(1, max_attempts + 1):
        ts = str(int(time.time()))
        headers = {"Content-Type": "application/json", "User-Agent": "flujo-personas/1.0",
                   "X-Timestamp": ts, "X-Session-Id": str(payload.get("session_id", ""))}
        if secret:
            headers["X-Signature"] = sign(secret, ts, body)
        out["attempts"] = attempt
        try:
            r = http.post(url, data=body, headers=headers, timeout=timeout, allow_redirects=False)
            out["status_code"] = r.status_code
            if 200 <= r.status_code < 300:
                out.update(ok=True, error=None, retryable=False, sent_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                return out
            out["error"] = f"HTTP {r.status_code}"
            out["retryable"] = r.status_code == 429 or r.status_code >= 500
        except requests.RequestException as e:
            out["error"], out["retryable"] = type(e).__name__, True  # sin detalles: pueden incluir la URL
        if not out["retryable"] or attempt == max_attempts:
            break
        sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    return out
