"""Construcción del reporte estructurado de una sesión (sin I/O externo)."""
from __future__ import annotations
import csv
import io
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"
EVENT = "people_flow.analysis.completed"
ALWAYS = ("event", "schema_version", "session_id")
SELECTABLE_FIELDS = [
    "started_at", "finished_at", "duration_seconds", "processing_seconds", "source",
    "unique_people_estimated", "detections_total", "detections_per_frame_avg",
    "entries", "exits", "present_at_start", "present_at_end",
    "average_dwell_seconds", "median_dwell_seconds", "complete_dwell_records", "average_observed_seconds",
    "activity_by_interval", "interval_seconds", "zones", "zone_transitions", "most_active_zone",
    "quality", "warnings", "definitions",
]


def utc_now():
    return datetime.now(timezone.utc)


def utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def new_session_id():
    return uuid.uuid4().hex


def build_report(session_id, started_at, finished_at, analysis, pinfo, cfg, source="video_upload",
                 test_mode=False):
    S, P, T, tl = analysis["summary"], analysis["people"], analysis["tracks"], analysis["timeline"]
    has_line = bool(cfg and cfg.line)
    n_ids = S["ids_detectados"]
    n = max(len(P), 1)
    low = int((P["calidad"] == "Baja").sum()) if len(P) else 0
    switches = len(S["posibles_cambios_id"])
    errors = pinfo.get("errores_detector", 0)
    eff_fps = pinfo["fps_video"] / pinfo["stride"]
    if n_ids == 0:
        level = "insuficiente"
    elif low / n > 0.5 or switches > 0.3 * n or errors > 0:
        level = "baja"
    elif low / n > 0.2 or switches > 0 or eff_fps < 2:
        level = "media"
    else:
        level = "alta"

    w = []
    if pinfo.get("modo") == "live":
        w.append("Análisis en vivo: el detector procesa solo algunos fotogramas por segundo según la capacidad del servidor; "
                 "pueden perderse cruces rápidos.")
    if pinfo.get("detenido_manual"):
        w.append("La sesión se finalizó manualmente: el reporte es parcial.")
    if test_mode:
        w.append("Modo de prueba: solo se analizó el inicio del video.")
    if n_ids == 0:
        w.append("No se detectaron personas con los parámetros usados.")
    if n_ids:
        w.append("«Personas únicas» es una estimación basada en IDs del rastreador; si el rastreador pierde a alguien "
                 "y lo recupera con otro ID, se cuenta más de una vez (suele ser una cota superior).")
    if not has_line:
        w.append("No se configuró línea de entrada/salida: entradas, salidas y permanencia no están disponibles.")
    if not (cfg and cfg.zones):
        w.append("No se configuraron zonas.")
    if switches:
        w.append(f"{switches} posible(s) cambio(s) de ID detectado(s); no se unieron automáticamente.")
    if has_line and (S["presentes_al_inicio"] or S["presentes_al_final"]):
        w.append("Hay personas presentes al inicio o final del video; su permanencia real no se puede calcular.")
    if errors:
        w.append(f"El detector falló en {errors} fotograma(s).")
    if eff_fps < 2:
        w.append("Procesamiento a menos de 2 fps efectivos: puede perder cruces rápidos.")

    step = float(tl["inicio_s"].iloc[1] - tl["inicio_s"].iloc[0]) if len(tl) > 1 else 5.0
    zt = S["tiempo_por_zona_s"]
    return {
        "event": EVENT, "schema_version": SCHEMA_VERSION, "session_id": session_id,
        "started_at": utc_iso(started_at), "finished_at": utc_iso(finished_at),
        "duration_seconds": round(pinfo["segundos_analizados"], 2),
        "processing_seconds": round((finished_at - started_at).total_seconds(), 1),
        "source": source,
        "unique_people_estimated": n_ids,
        "detections_total": int(len(T)),
        "detections_per_frame_avg": round(len(T) / max(pinfo["frames_procesados"], 1), 3),
        "entries": S["entradas_validas"], "exits": S["salidas_validas"],
        "present_at_start": S["presentes_al_inicio"], "present_at_end": S["presentes_al_final"],
        "average_dwell_seconds": S["permanencia_media_s"], "median_dwell_seconds": S["permanencia_mediana_s"],
        "complete_dwell_records": S["registros_completos"],
        "average_observed_seconds": S["tiempo_observado_medio_s"],
        "interval_seconds": step,
        "activity_by_interval": [
            {"start_s": r["inicio_s"], "people_observed": r["personas_observadas"],
             "entries": r["entradas"] if has_line else None, "exits": r["salidas"] if has_line else None}
            for r in tl.to_dict("records")],
        "zones": [{"name": z.name, "observed_seconds": zt.get(z.name, 0.0)} for z in (cfg.zones if cfg else [])],
        "zone_transitions": S["transiciones"], "most_active_zone": S["zona_mayor_actividad"],
        "quality": {"level": level, "low_quality_records": low, "possible_id_switches": switches,
                    "detector_errors": errors, "effective_fps": round(eff_fps, 2),
                    "processing_resolution": pinfo["tamano_procesamiento"]},
        "warnings": w,
        "definitions": {
            "unique_people_estimated": "IDs de seguimiento distintos (estimación; no es identidad real)",
            "detections_total": "Suma de detecciones en todos los fotogramas procesados (misma persona cuenta muchas veces)",
            "entries_exits": "Cruces válidos de la línea configurada; null si no hay línea",
            "average_dwell_seconds": "Solo registros con entrada y salida válidas; null si no hay ninguno",
            "duration_seconds": "Segundos de video analizados"},
    }


def select_fields(report, fields):
    keep = set(ALWAYS) | set(fields or [])
    return {k: v for k, v in report.items() if k in keep}


def report_csv(report):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seccion", "clave", "valor"])
    for k, v in report.items():
        if isinstance(v, (dict, list)):
            continue
        w.writerow(["resumen", k, "" if v is None else v])
    for k, v in report["quality"].items():
        w.writerow(["calidad", k, v])
    for z in report["zones"]:
        w.writerow(["zona", z["name"], z["observed_seconds"]])
    for k, v in report["zone_transitions"].items():
        w.writerow(["transicion", k, v])
    for r in report["activity_by_interval"]:
        w.writerow(["intervalo", r["start_s"],
                    f"personas={r['people_observed']};entradas={r['entries']};salidas={r['exits']}"])
    for x in report["warnings"]:
        w.writerow(["advertencia", "", x])
    return buf.getvalue()
