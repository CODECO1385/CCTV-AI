"""CSV, JSON y ZIP de resultados."""
from __future__ import annotations
import io
import json
import zipfile


def people_csv(people):
    return people.to_csv(index=False, float_format="%.3f")


def events_csv(events):
    return events.to_csv(index=False, float_format="%.3f")


def summary_json(summary, cfg, settings=None, video_info=None):
    doc = {"resumen": summary, "configuracion_espacio": cfg.to_dict() if cfg else None,
           "parametros": settings, "video": video_info,
           "aviso": "Los identificadores son temporales (por video) y no identifican a personas reales."}
    return json.dumps(doc, ensure_ascii=False, indent=2, default=str)


def build_zip(people, events, summary, cfg, settings=None, video_info=None, video_path=None, report=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("personas.csv", people_csv(people))
        z.writestr("eventos.csv", events_csv(events))
        z.writestr("resumen.json", summary_json(summary, cfg, settings, video_info))
        if report:
            import report as rp
            z.writestr("reporte.json", json.dumps(report, ensure_ascii=False, indent=2))
            z.writestr("reporte.csv", rp.report_csv(report))
        if video_path:
            z.write(video_path, "video_procesado.mp4")
    return buf.getvalue()
