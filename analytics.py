"""Tiempos, cruces de línea, zonas, transiciones y estadísticas."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from zones import line_px, zones_px, zone_of, signed_distance

OUTSIDE = "Fuera de zonas"


def frame_to_time(frame_idx, fps):
    if fps <= 0:
        raise ValueError("fps debe ser positivo")
    return frame_idx / fps


def time_to_frame(t, fps):
    if fps <= 0:
        raise ValueError("fps debe ser positivo")
    return int(round(t * fps))


def line_events(obs, lpx, inside_side=1, margin_px=10.0, min_gap_s=1.5):
    """obs: iterable de (t, x, y). Cruces con histéresis (banda muerta de margin_px) y
    cancelación de rebotes: un cruce inverso en menos de min_gap_s anula el anterior."""
    (x1, y1), (x2, y2) = lpx
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return []
    events, stable = [], None
    for t, x, y in obs:
        d = (dx * (y - y1) - dy * (x - x1)) / length
        if abs(d) < margin_px:
            continue
        s = 1 if d > 0 else -1
        if stable is not None and s != stable[0]:
            _, t0, x0, y0, d0 = stable
            u = abs(d0) / (abs(d0) + abs(d))
            cx, cy, ct = x0 + u * (x - x0), y0 + u * (y - y0), t0 + u * (t - t0)
            proj = ((cx - x1) * dx + (cy - y1) * dy) / length
            if 0 <= proj <= length:  # el cruce ocurre sobre el segmento dibujado
                kind = "entrada" if s == inside_side else "salida"
                if events and events[-1]["kind"] != kind and ct - events[-1]["t"] < min_gap_s:
                    events.pop()  # rebote/jitter
                else:
                    events.append({"t": ct, "kind": kind, "x": cx, "y": cy})
        stable = (s, t, x, y, d)
    return events


def zone_dwell(t, z, max_gap):
    """Segundos observados por zona. Intervalos mayores que max_gap (detección perdida) no cuentan."""
    out = {}
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if 0 < dt <= max_gap:
            k = z[i] or OUTSIDE
            out[k] = out.get(k, 0.0) + dt
    return out


def zone_transitions(t, z, min_dwell_s=0.5, max_gap_s=15.0):
    """Transiciones entre zonas consecutivas visitadas. Una zona solo cuenta si la persona
    permaneció >= min_dwell_s (evita contar roces en el límite)."""
    runs = []
    for ti, zi in zip(t, z):
        if zi is None:
            continue
        if runs and runs[-1][0] == zi:
            runs[-1][2] = ti
        else:
            runs.append([zi, ti, ti])
    kept = []
    for r in runs:
        if r[2] - r[1] < min_dwell_s:
            continue
        if kept and kept[-1][0] == r[0]:
            kept[-1][2] = r[2]
        else:
            kept.append(r)
    return [(b[1], a[0], b[0]) for a, b in zip(kept, kept[1:]) if b[1] - a[2] <= max_gap_s]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o != o) else round(float(o), 3)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def analyze(tracks, cfg, size, duration, sample_dt, min_obs=3, margin_px=10.0, min_gap_s=1.5,
            min_zone_dwell_s=0.5, max_gap_s=None, bin_s=5.0):
    """tracks: DataFrame [id, t, x, y] (px del video original, punto inferior-centro)."""
    max_gap = max_gap_s if max_gap_s is not None else max(3 * sample_dt, 1.0)
    lpx = line_px(cfg, size)
    zpx = zones_px(cfg, size)
    inside = cfg.line.inside_side if (cfg and cfg.line) else 1
    tracks = tracks.copy()
    tracks["zone"] = [zone_of((x, y), zpx) for x, y in zip(tracks["x"], tracks["y"])] if len(tracks) else []
    start_win = max(1.0, 2 * sample_dt)

    people, events, zone_tot, trans_count = [], [], {}, {}
    dropped, ends = 0, {}
    for pid, g in tracks.groupby("id"):
        g = g.sort_values("t")
        if len(g) < min_obs:
            dropped += 1
            continue
        t, x, y, z = g["t"].to_numpy(float), g["x"].to_numpy(float), g["y"].to_numpy(float), g["zone"].tolist()
        first, last = float(t[0]), float(t[-1])
        expected = (last - first) / sample_dt + 1
        cov = min(1.0, len(t) / expected) if expected > 0 else 1.0
        gaps = int(np.sum(np.diff(t) > max_gap))
        quality = "Baja" if (last - first < 1.0 or cov < 0.5) else ("Alta" if cov >= 0.8 and gaps == 0 else "Media")
        notes = []
        if gaps:
            notes.append(f"{gaps} hueco(s) de detección")
        row = {"persona": f"Persona {pid}", "id": int(pid), "primera_aparicion_s": first,
               "ultima_observacion_s": last, "tiempo_observado_s": last - first, "n_observaciones": len(t),
               "entrada_s": None, "salida_s": None, "permanencia_estimada_s": None,
               "presente_al_inicio": None, "presente_al_final": None}
        if lpx:
            ev = line_events(zip(t, x, y), lpx, inside, margin_px, min_gap_s)
            ins = [e for e in ev if e["kind"] == "entrada"]
            outs = [e for e in ev if e["kind"] == "salida"]
            t_in = ins[0]["t"] if ins else None
            t_out = None
            if t_in is not None:
                later = [e["t"] for e in outs if e["t"] > t_in]
                t_out = later[0] if later else None
            elif outs:
                t_out = outs[0]["t"]
            row["entrada_s"], row["salida_s"] = t_in, t_out
            side_first = signed_distance((x[0], y[0]), lpx) * inside > 0
            side_last = signed_distance((x[-1], y[-1]), lpx) * inside > 0
            row["presente_al_inicio"] = bool(first <= start_win and side_first and not (ev and ev[0]["kind"] == "entrada"))
            row["presente_al_final"] = bool(last >= duration - start_win and side_last and not (ev and ev[-1]["kind"] == "salida"))
            if t_in is not None and t_out is not None:
                row["permanencia_estimada_s"] = t_out - t_in
                state = "Completo"
            elif t_in is not None:
                state = "Incompleto: entrada detectada, salida no detectada (sigue dentro o se perdió)"
            elif t_out is not None:
                state = "Incompleto: salida detectada, entrada no detectada (ya estaba dentro al inicio)"
            elif row["presente_al_inicio"] or row["presente_al_final"]:
                state = "Incompleto: presente al inicio y/o al final del video"
            else:
                state = "Incompleto: sin cruces de la línea"
            if len(ins) > 1 or len(outs) > 1:
                notes.append("múltiples cruces de línea")
            for e in ev:
                events.append({"tipo": e["kind"], "persona": row["persona"], "t_s": e["t"],
                               "origen": None, "destino": None, "x": e["x"], "y": e["y"]})
        else:
            state = "Sin línea configurada: solo tiempo observado"
        row["estado"], row["calidad"], row["cobertura"] = state, quality, cov
        row["notas"] = "; ".join(notes)
        for k, v in zone_dwell(t, z, max_gap).items():
            row["tz_" + k] = v
            zone_tot[k] = zone_tot.get(k, 0.0) + v
        for tt, a, b in zone_transitions(t, z, min_zone_dwell_s):
            k = f"{a} → {b}"
            trans_count[k] = trans_count.get(k, 0) + 1
            events.append({"tipo": "transicion", "persona": row["persona"], "t_s": tt,
                           "origen": a, "destino": b, "x": None, "y": None})
        ends[row["id"]] = (first, last, (x[0], y[0]), (x[-1], y[-1]))
        people.append(row)

    # Posibles cambios de ID (NO se unen automáticamente): un ID termina y otro empieza cerca en espacio y tiempo
    diag = math.hypot(*size)
    switches = []
    for a, (_, la, _, pa) in ends.items():
        for b, (fb, _, pb, _) in ends.items():
            if a != b and 0 < fb - la <= 2.0 and la < duration - 1 and math.dist(pa, pb) < 0.08 * diag:
                switches.append((f"Persona {a}", f"Persona {b}"))
    sw = {}
    for a, b in switches:
        sw.setdefault(a, []).append(b)
        sw.setdefault(b, []).append(a)
    for r in people:
        if r["persona"] in sw:
            r["notas"] = (r["notas"] + "; " if r["notas"] else "") + "posible cambio de ID con " + ", ".join(sw[r["persona"]])

    pdf = pd.DataFrame(people)
    edf = pd.DataFrame(events, columns=["tipo", "persona", "t_s", "origen", "destino", "x", "y"])
    if len(edf):
        edf = edf.sort_values("t_s").reset_index(drop=True)
    has_line = lpx is not None
    complete = [r["permanencia_estimada_s"] for r in people if r["permanencia_estimada_s"] is not None]
    real_zones = {k: v for k, v in zone_tot.items() if k != OUTSIDE}
    summary = {
        "ids_detectados": len(people),
        "ids_descartados_por_ruido": dropped,
        "duracion_analizada_s": duration,
        "entradas_validas": int((edf["tipo"] == "entrada").sum()) if has_line and len(edf) else (0 if has_line else None),
        "salidas_validas": int((edf["tipo"] == "salida").sum()) if has_line and len(edf) else (0 if has_line else None),
        "presentes_al_inicio": sum(1 for r in people if r["presente_al_inicio"]) if has_line else None,
        "presentes_al_final": sum(1 for r in people if r["presente_al_final"]) if has_line else None,
        "registros_completos": len(complete),
        "permanencia_media_s": float(np.mean(complete)) if complete else None,
        "permanencia_mediana_s": float(np.median(complete)) if complete else None,
        "tiempo_observado_medio_s": float(np.mean([r["tiempo_observado_s"] for r in people])) if people else None,
        "tiempo_por_zona_s": zone_tot,
        "zona_mayor_actividad": max(real_zones, key=real_zones.get) if real_zones else None,
        "transiciones": dict(sorted(trans_count.items(), key=lambda kv: -kv[1])),
        "posibles_cambios_id": switches,
    }
    # Distribución temporal
    bins = np.arange(0, max(duration, bin_s) + bin_s, bin_s)
    tl = []
    for b in bins[:-1]:
        sel = tracks[(tracks["t"] >= b) & (tracks["t"] < b + bin_s)] if len(tracks) else tracks
        ids = set(sel["id"]) & {r["id"] for r in people} if len(sel) else set()
        n_in = n_out = 0
        if len(edf):
            m = (edf["t_s"] >= b) & (edf["t_s"] < b + bin_s)
            n_in, n_out = int(((edf["tipo"] == "entrada") & m).sum()), int(((edf["tipo"] == "salida") & m).sum())
        tl.append({"inicio_s": float(b), "personas_observadas": len(ids), "entradas": n_in, "salidas": n_out})
    return {"people": pdf, "events": edf, "summary": _clean(summary), "timeline": pd.DataFrame(tl),
            "tracks": tracks, "max_gap": max_gap}
