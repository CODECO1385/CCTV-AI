"""Línea de entrada/salida y zonas, en coordenadas normalizadas (0-1)."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EntryLine:
    p1: tuple
    p2: tuple
    # Lado donde está el INTERIOR del local, mirando de p1 a p2 en coordenadas de imagen:
    # +1 = derecha, -1 = izquierda.
    inside_side: int = 1


@dataclass
class Zone:
    name: str
    points: list


@dataclass
class SpaceConfig:
    line: EntryLine | None = None
    zones: list = field(default_factory=list)
    ref_size: tuple | None = None  # (ancho, alto) del video de referencia

    def to_dict(self):
        return {
            "line": None if self.line is None else {
                "p1": list(self.line.p1), "p2": list(self.line.p2), "inside_side": self.line.inside_side},
            "zones": [{"name": z.name, "points": [list(p) for p in z.points]} for z in self.zones],
            "ref_size": list(self.ref_size) if self.ref_size else None,
        }

    @classmethod
    def from_dict(cls, d):
        ln = d.get("line")
        line = None if not ln else EntryLine(tuple(ln["p1"]), tuple(ln["p2"]), int(ln.get("inside_side", 1)))
        zones = [Zone(z["name"], [tuple(p) for p in z["points"]]) for z in d.get("zones", [])]
        rs = d.get("ref_size")
        return cls(line, zones, tuple(rs) if rs else None)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


def line_px(cfg, size):
    if cfg is None or cfg.line is None:
        return None
    w, h = size
    return ((cfg.line.p1[0] * w, cfg.line.p1[1] * h), (cfg.line.p2[0] * w, cfg.line.p2[1] * h))


def zones_px(cfg, size):
    if cfg is None:
        return []
    w, h = size
    return [(z.name, np.array([(x * w, y * h) for x, y in z.points], dtype=float)) for z in cfg.zones]


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def zone_of(pt, zpx):
    """Primera zona (por orden de definición) que contiene el punto, o None."""
    for name, poly in zpx:
        if point_in_polygon(pt, poly):
            return name
    return None


def signed_distance(pt, lpx):
    """Distancia con signo (px) a la línea. Positivo = derecha de p1->p2 (coords de imagen)."""
    (x1, y1), (x2, y2) = lpx
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    return (dx * (pt[1] - y1) - dy * (pt[0] - x1)) / length


def check_compatibility(cfg, size):
    """Devuelve un aviso si la proporción del video difiere de la de referencia."""
    if not cfg or not cfg.ref_size:
        return None
    rw, rh = cfg.ref_size
    if abs(rw / rh - size[0] / size[1]) / (rw / rh) > 0.02:
        return (f"La proporción del video ({size[0]}x{size[1]}) no coincide con la de la configuración "
                f"({rw}x{rh}). Las zonas podrían quedar desalineadas: revísalas antes de analizar.")
    return None


def parse_points(text):
    """'10,20; 40,20; 40,60' (porcentajes 0-100) -> [(0.1,0.2), ...]"""
    pts = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            x, y = [float(v) for v in chunk.split(",")]
        except ValueError:
            raise ValueError(f"Punto inválido: '{chunk}'. Usa el formato x,y")
        if not (0 <= x <= 100 and 0 <= y <= 100):
            raise ValueError(f"Punto fuera de rango (0-100): '{chunk}'")
        pts.append((x / 100, y / 100))
    if len(pts) < 3:
        raise ValueError("Una zona necesita al menos 3 puntos")
    return pts


def points_to_text(points):
    return "; ".join(f"{x * 100:.1f},{y * 100:.1f}" for x, y in points)
