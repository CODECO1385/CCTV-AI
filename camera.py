"""Análisis en vivo de la cámara del navegador (streamlit-webrtc).
El video fluye en tiempo real; la detección corre en un hilo aparte a un ritmo limitado para no bloquear la imagen.
No se guarda video: solo coordenadas de cajas (y un fotograma de referencia en memoria para los gráficos)."""
from __future__ import annotations
import threading
import time
from collections import defaultdict, deque
import cv2
import pandas as pd
from config import get_secret
from video_processor import COLUMNS, VideoError
from visualizations import draw_space_overlay, color_for_id


def rtc_config():
    ice = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if get_secret("TURN_URL"):  # opcional: necesario en redes restrictivas
        ice.append({"urls": [get_secret("TURN_URL")], "username": get_secret("TURN_USERNAME"),
                    "credential": get_secret("TURN_CREDENTIAL")})
    return {"iceServers": ice}


class LiveSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.ready = False
        self.error = None
        self.rows, self.dets = [], []
        self.frames_seen = self.frames_processed = self.errors = 0
        self.t0 = self.t_last = self.size = self.ref_frame = None
        self.trails = defaultdict(lambda: deque(maxlen=40))
        self._thread = None

    # ---- ciclo de vida ----
    def begin(self, cfg, tracker_factory, max_width=480, min_interval=0.5):
        self.end_worker()
        with self.lock:
            self.cfg, self.max_width, self.min_interval = cfg, max_width, min_interval
            self.rows, self.dets = [], []
            self.frames_seen = self.frames_processed = self.errors = 0
            self.t0 = self.t_last = self.size = self.ref_frame = self.error = None
            self.trails.clear()
            self._latest, self.ready, self.active = None, False, True
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._worker, args=(tracker_factory,), daemon=True)
        self._thread.start()

    def end_worker(self):
        if self._thread is not None:
            self._stop.set()
            self._wake.set()
            self._thread.join(timeout=10)
            self._thread = None

    def end(self):
        """Cierra la sesión y devuelve (tracks, pinfo, size, ref_frame)."""
        self.active = False
        self.end_worker()
        with self.lock:
            if self.error and not self.rows:
                raise VideoError(self.error)
            if not self.frames_processed or self.t0 is None:
                raise VideoError("No se procesó ningún fotograma. Verifica el permiso de la cámara y que la sesión haya estado activa.")
            elapsed = max(self.t_last - self.t0, 1e-3)
            w, h = self.size
            scale = min(1.0, self.max_width / w)
            tracks = pd.DataFrame(self.rows, columns=COLUMNS)
            pinfo = {"modo": "live", "frames_procesados": self.frames_processed, "stride": 1,
                     "sample_dt": elapsed / self.frames_processed, "fps_video": self.frames_processed / elapsed,
                     "tamano_procesamiento": [int(w * scale), int(h * scale)], "tamano_original": [w, h],
                     "errores_detector": self.errors, "video_error": self.error, "detenido_manual": False,
                     "segundos_analizados": elapsed, "tiempo_procesamiento_s": elapsed}
            return tracks, pinfo, (w, h), self.ref_frame

    # ---- hilo de detección ----
    def _worker(self, factory):
        try:
            tracker = factory()
        except Exception as e:  # noqa: BLE001
            self.error, self.active = f"No se pudo cargar el detector: {e}", False
            return
        self.ready = True
        last, consec = 0.0, 0
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            with self.lock:
                item, self._latest = self._latest, None
            if item is None or time.monotonic() - last < self.min_interval:
                continue
            last = time.monotonic()
            frame, t = item
            h, w = frame.shape[:2]
            scale = min(1.0, self.max_width / w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
            try:
                dets = tracker.update(small)
                consec = 0
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                consec += 1
                if consec >= 5:
                    self.error, self.active = f"El detector falló repetidamente: {type(e).__name__}", False
                    return
                continue
            with self.lock:
                self.frames_processed += 1
                cur = []
                for d in dets:
                    x1, y1, x2, y2 = [v / scale for v in d["box"]]
                    self.rows.append((self.frames_processed, t, d["id"], (x1 + x2) / 2, y2, d["conf"]))
                    self.trails[d["id"]].append((int((x1 + x2) / 2), int(y2)))
                    cur.append((d["id"], (int(x1), int(y1), int(x2), int(y2))))
                self.dets = cur

    # ---- entrada de fotogramas ----
    def process_image(self, img):
        """Recibe un fotograma BGR, lo ofrece al detector y devuelve el fotograma anotado."""
        if not self.active:
            return img
        now = time.monotonic()
        if self.ready:
            with self.lock:
                if self.t0 is None:
                    self.t0, self.size, self.ref_frame = now, (img.shape[1], img.shape[0]), img.copy()
                self.t_last = now
                self._latest = (img.copy(), now - self.t0)
                self.frames_seen += 1
                dets, trails = list(self.dets), {k: list(v) for k, v in self.trails.items()}
            self._wake.set()
        else:
            dets, trails = [], {}
            cv2.putText(img, "Cargando modelo...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        draw_space_overlay(img, self.cfg)
        for pid, (x1, y1, x2, y2) in dets:
            c = color_for_id(pid)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img, f"Persona {pid}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            for a, b in zip(trails.get(pid, []), trails.get(pid, [])[1:]):
                cv2.line(img, a, b, c, 2)
        return img

    def callback(self, frame):  # streamlit-webrtc video_frame_callback
        import av
        img = self.process_image(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(img, format="bgr24")
