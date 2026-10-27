"""Sesión de análisis: procesar -> analizar -> reporte -> entregas. Se ejecuta en un hilo para no bloquear la UI."""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
import analytics as an
import notifier
import report as rp
import video_processor as vp
import webhook as wh

log = logging.getLogger("flujo")  # nunca registrar URLs, secretos ni destinatarios


@dataclass
class SessionSettings:
    model: str = "yolov8n.pt"
    conf: float = 0.35
    target_fps: float = 5.0
    max_width: int = 960
    max_seconds: float | None = None
    write_video: bool = True
    test_mode: bool = False
    webhook_auto: bool = False
    webhook_fields: list = field(default_factory=lambda: list(rp.SELECTABLE_FIELDS))
    webhook_url: str | None = None   # override solo de esta sesión; por defecto WEBHOOK_URL del entorno
    email_to: str | None = None
    sms_to: str | None = None


def _delivery(channel, **kw):
    return {"channel": channel, "at": rp.utc_iso(rp.utc_now()), **kw}


def deliver_webhook(report, fields, url_override=None, http=None, sleep=None):
    cfg = wh.env_settings(url_override)
    payload = rp.select_fields(report, fields)
    kwargs = {"http": http} if http else {}
    if sleep:
        kwargs["sleep"] = sleep
    r = wh.send(cfg["url"], payload, cfg["secret"], allow_private=cfg["allow_private"],
                allow_http=cfg["allow_http"], **kwargs)
    return payload, _delivery("webhook", ok=r["ok"], attempts=r["attempts"], status_code=r["status_code"],
                              error=r["error"], retryable=r["retryable"], destino=wh.mask_url(cfg["url"]))


def deliver_all(report, s, http=None, sleep=None):
    out, payload = [], None
    if s.webhook_auto:
        payload, d = deliver_webhook(report, s.webhook_fields, s.webhook_url, http, sleep)
        out.append(d)
    if s.email_to or s.sms_to:
        subject, text = notifier.build_summary(report)
        if s.email_to:
            r = notifier.send_email(s.email_to, subject, text, http)
            out.append(_delivery("email", ok=r["ok"], provider=r["provider"], status=r["status"], error=r["error"]))
        if s.sms_to:
            r = notifier.send_sms(s.sms_to, text, http)
            out.append(_delivery("sms", ok=r["ok"], provider=r["provider"], status=r["status"], error=r["error"]))
    return payload, out


def run_session(video_path, tracker_factory, cfg, size, settings, session_id, started_at, out_video=None,
                progress_cb=None, should_stop=None, http=None, sleep=None):
    tracker = tracker_factory()
    ps = vp.ProcSettings(target_fps=settings.target_fps, max_width=settings.max_width,
                         max_seconds=settings.max_seconds, write_video=settings.write_video)
    tracks, pinfo = vp.process_video(video_path, tracker, ps, cfg, out_video if settings.write_video else None,
                                     progress_cb, should_stop)
    return finalize(tracks, pinfo, cfg, size, settings, session_id, started_at, vp.read_frame_at(video_path, 0.0),
                    out_video, "video_upload", http, sleep)


def finalize(tracks, pinfo, cfg, size, settings, session_id, started_at, ref, out_video=None,
             source="video_upload", http=None, sleep=None):
    """Tracks -> análisis -> reporte -> entregas. Común a video cargado y cámara en vivo."""
    res = an.analyze(tracks, cfg, size, pinfo["segundos_analizados"], pinfo["sample_dt"])
    finished = rp.utc_now()
    report = rp.build_report(session_id, started_at, finished, res, pinfo, cfg, source=source,
                             test_mode=settings.test_mode)
    payload, deliveries = deliver_all(report, settings, http, sleep)
    res.update(pinfo=pinfo, cfg=cfg, report=report, deliveries=deliveries, webhook_payload=payload,
               settings={"modelo": settings.model, "confianza": settings.conf, "fps_procesados": settings.target_fps,
                         "ancho": settings.max_width, "modo_prueba": settings.test_mode, "origen": source},
               video=out_video if (settings.write_video and out_video and not pinfo.get("video_error")
                                   and source == "video_upload") else None,
               ref=ref)
    return res


class Job:
    def __init__(self, **kw):
        self.kw = kw
        self.stop = threading.Event()
        self.progress, self.message = 0.0, "Iniciando…"
        self.state, self.error, self.result, self.consumed = "running", None, None, False
        self.thread = None

    def _cb(self, p, m):
        self.progress, self.message = p, m

    def _run(self):
        try:
            self.result = run_session(**self.kw, progress_cb=self._cb, should_stop=self.stop.is_set)
            self.state = "done"
        except vp.VideoError as e:
            self.error, self.state = str(e), "error"
        except Exception as e:  # noqa: BLE001
            log.exception("Fallo inesperado en la sesión")
            self.error, self.state = f"Error inesperado ({type(e).__name__}): {e}", "error"

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
