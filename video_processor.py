"""Lectura de video fotograma a fotograma (sin cargarlo completo en memoria) y seguimiento."""
from __future__ import annotations
import time
from collections import defaultdict, deque
from dataclasses import dataclass
import cv2
import pandas as pd
from visualizations import draw_space_overlay, color_for_id

SUPPORTED = (".mp4", ".mov", ".avi")
COLUMNS = ["frame", "t", "id", "x", "y", "conf"]


class VideoError(Exception):
    pass


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass
class ProcSettings:
    target_fps: float = 5.0
    max_width: int = 960       # 0 = resolución original
    max_seconds: float | None = None
    write_video: bool = True


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("No se pudo abrir el video. Verifica que sea MP4, MOV o AVI compatible con OpenCV.")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or w <= 0 or h <= 0:
            raise VideoError("El video está vacío o no contiene fotogramas legibles.")
        if fps <= 0 or n <= 0:
            raise VideoError("No se pudo determinar la duración o los fotogramas por segundo del video.")
        return VideoInfo(str(path), w, h, fps, n, n / fps)
    finally:
        cap.release()


def read_frame_at(path, t_sec):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ok, frame = cap.read()
        if not ok:
            raise VideoError("No se pudo leer el fotograma solicitado.")
        return frame
    finally:
        cap.release()


def process_video(path, tracker, settings=None, space_cfg=None, out_video=None, progress_cb=None, should_stop=None):
    """Devuelve (DataFrame[frame,t,id,x,y,conf], info). x,y = centro inferior del cuadro en px del
    video ORIGINAL; t = frame/fps (marca temporal original aunque se saltaran fotogramas)."""
    s = settings or ProcSettings()
    info = probe_video(path)
    stride = max(1, int(round(info.fps / max(s.target_fps, 0.1))))
    scale = min(1.0, s.max_width / info.width) if s.max_width else 1.0
    pw, ph = int(round(info.width * scale)), int(round(info.height * scale))
    limit = min(info.frame_count, int(s.max_seconds * info.fps)) if s.max_seconds else info.frame_count
    cap = cv2.VideoCapture(str(path))
    writer, video_error = None, None
    if out_video and s.write_video:
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), info.fps / stride, (pw, ph))
        if not writer.isOpened():
            writer, video_error = None, "No se pudo crear el video procesado."
    rows, trails = [], defaultdict(lambda: deque(maxlen=60))
    idx = processed = errors = consec = 0
    stopped = False
    t0 = time.time()
    try:
        while idx < limit:
            if should_stop is not None and should_stop():
                stopped = True
                break
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    if scale < 1.0:
                        frame = cv2.resize(frame, (pw, ph))
                    try:
                        dets = tracker.update(frame)
                        consec = 0
                    except MemoryError:
                        raise VideoError("Memoria insuficiente. Reduce la resolución de procesamiento.")
                    except Exception as e:
                        errors += 1
                        consec += 1
                        dets = []
                        if consec >= 5:
                            raise VideoError(f"El detector falló repetidamente: {e}")
                    t = idx / info.fps
                    for d in dets:
                        x1, y1, x2, y2 = d["box"]
                        fx, fy = (x1 + x2) / 2, y2
                        rows.append((idx, t, d["id"], fx / scale, fy / scale, d["conf"]))
                        trails[d["id"]].append((int(fx), int(fy)))
                    processed += 1
                    if writer is not None:
                        draw_space_overlay(frame, space_cfg)
                        for d in dets:
                            x1, y1, x2, y2 = map(int, d["box"])
                            c = color_for_id(d["id"])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                            cv2.putText(frame, f"Persona {d['id']}", (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                            pts = list(trails[d["id"]])
                            for a, b in zip(pts, pts[1:]):
                                cv2.line(frame, a, b, c, 2)
                        writer.write(frame)
                    if progress_cb:
                        progress_cb(min(1.0, idx / max(limit, 1)), f"Fotograma {idx}/{limit}")
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if processed == 0:
        raise VideoError("La sesión terminó antes de procesar fotogramas." if stopped else "El video no contiene fotogramas legibles.")
    df = pd.DataFrame(rows, columns=COLUMNS)
    pinfo = {"frames_procesados": processed, "stride": stride, "sample_dt": stride / info.fps,
             "tamano_procesamiento": [pw, ph], "errores_detector": errors, "video_error": video_error,
             "segundos_analizados": min(idx, limit) / info.fps, "tiempo_procesamiento_s": time.time() - t0,
             "fps_video": info.fps, "detenido_manual": stopped, "tamano_original": [info.width, info.height]}
    return df, pinfo
