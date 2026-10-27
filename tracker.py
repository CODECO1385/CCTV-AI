"""Seguimiento con ByteTrack (integrado en Ultralytics). Los IDs son temporales, no identidades reales."""
from __future__ import annotations
from detector import load_model


class PersonTracker:
    def __init__(self, model_name="yolov8n.pt", conf=0.35, iou=0.5, device=None):
        self.model = load_model(model_name, device)
        self.conf, self.iou = conf, iou

    def update(self, frame):
        """Devuelve [{'id': int, 'box': (x1,y1,x2,y2), 'conf': float}] para el fotograma."""
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                             conf=self.conf, iou=self.iou, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None:
            return []
        return [{"id": int(i), "box": tuple(map(float, xy)), "conf": float(c)}
                for i, xy, c in zip(b.id.tolist(), b.xyxy.tolist(), b.conf.tolist())]
