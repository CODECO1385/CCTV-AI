"""Carga del modelo YOLO (solo clase persona)."""
from __future__ import annotations


def load_model(name="yolov8n.pt", device=None):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Falta 'ultralytics'. Instala con: pip install -r requirements.txt") from e
    model = YOLO(name)  # la primera vez descarga los pesos (el video nunca sale de tu equipo)
    if device:
        model.to(device)
    return model


def detect_people(model, frame, conf=0.35):
    """Detección sin seguimiento (útil para pruebas). Devuelve [(x1,y1,x2,y2,conf)]."""
    r = model.predict(frame, classes=[0], conf=conf, verbose=False)[0]
    return [tuple(map(float, b.xyxy[0].tolist())) + (float(b.conf[0]),) for b in r.boxes]
