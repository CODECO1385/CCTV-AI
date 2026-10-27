"""Superposiciones, trayectorias, mapas de calor y gráficas."""
from __future__ import annotations
import colorsys
import unicodedata
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ascii(s):  # cv2.putText no dibuja acentos
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def color_for_id(i):
    r, g, b = colorsys.hsv_to_rgb((int(i) * 0.618034) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_space_overlay(img, cfg):
    """Dibuja zonas y línea (coordenadas normalizadas) sobre img BGR, in place."""
    if cfg is None:
        return img
    h, w = img.shape[:2]
    layer = img.copy()
    for k, z in enumerate(cfg.zones):
        pts = np.array([(x * w, y * h) for x, y in z.points], np.int32)
        cv2.fillPoly(layer, [pts], color_for_id(k + 3))
    cv2.addWeighted(layer, 0.25, img, 0.75, 0, img)
    for k, z in enumerate(cfg.zones):
        pts = np.array([(x * w, y * h) for x, y in z.points], np.int32)
        cv2.polylines(img, [pts], True, color_for_id(k + 3), 2)
        cv2.putText(img, _ascii(z.name), tuple(pts[0] + [4, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if cfg.line:
        p1 = (int(cfg.line.p1[0] * w), int(cfg.line.p1[1] * h))
        p2 = (int(cfg.line.p2[0] * w), int(cfg.line.p2[1] * h))
        cv2.line(img, p1, p2, (0, 255, 255), 3)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        n = max(np.hypot(dx, dy), 1e-6)
        nx, ny = -dy / n * cfg.line.inside_side, dx / n * cfg.line.inside_side
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.arrowedLine(img, mid, (int(mid[0] + nx * 45), int(mid[1] + ny * 45)), (0, 255, 255), 3, tipLength=0.35)
        cv2.putText(img, "interior", (int(mid[0] + nx * 50), int(mid[1] + ny * 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, "P1", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(img, "P2", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return img


def trajectories_image(frame_bgr, tracks, ids, cfg=None):
    img = frame_bgr.copy()
    draw_space_overlay(img, cfg)
    for pid in ids:
        g = tracks[tracks["id"] == pid].sort_values("t")
        if len(g) < 2:
            continue
        pts = np.array(list(zip(g["x"], g["y"])), np.int32)
        c = color_for_id(pid)
        cv2.polylines(img, [pts], False, c, 2)
        cv2.circle(img, tuple(pts[0]), 5, (0, 255, 0), -1)
        cv2.circle(img, tuple(pts[-1]), 5, (0, 0, 255), -1)
        cv2.putText(img, f"P{pid}", tuple(pts[-1] + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def heatmap_image(frame_bgr, tracks, mode="posiciones", max_gap=1.0):
    """mode='posiciones': detecciones acumuladas. mode='tiempo': segundos acumulados (pondera por
    el intervalo real entre observaciones, así que distingue paso rápido de permanencia)."""
    h, w = frame_bgr.shape[:2]
    s = 4
    grid = np.zeros((h // s + 1, w // s + 1), np.float32)
    for _, g in tracks.groupby("id"):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        wts = np.append(np.where(np.diff(t) <= max_gap, np.diff(t), 0.0), 0.0) if mode == "tiempo" else np.ones(len(t))
        xs = np.clip((g["x"].to_numpy() / s).astype(int), 0, grid.shape[1] - 1)
        ys = np.clip((g["y"].to_numpy() / s).astype(int), 0, grid.shape[0] - 1)
        np.add.at(grid, (ys, xs), wts)
    if grid.sum() <= 0:
        return None
    grid = cv2.GaussianBlur(grid, (0, 0), max(3, grid.shape[1] * 0.03))
    grid = cv2.resize(grid / grid.max(), (w, h))
    color = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(grid * 1.5, 0, 0.65)[..., None]
    out = (frame_bgr * (1 - alpha) + color * alpha).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def flow_fig(transitions):
    fig, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(transitions) + 1)))
    ks = list(transitions)[::-1]
    ax.barh(ks, [transitions[k] for k in ks], color="#4C78A8")
    ax.set_xlabel("Transiciones observadas")
    fig.tight_layout()
    return fig


def timeline_fig(tl):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(tl["inicio_s"], tl["personas_observadas"], marker="o", label="Personas observadas")
    if tl["entradas"].sum() or tl["salidas"].sum():
        ax.bar(tl["inicio_s"] - 0.8, tl["entradas"], width=1.5, alpha=0.5, label="Entradas")
        ax.bar(tl["inicio_s"] + 0.8, tl["salidas"], width=1.5, alpha=0.5, label="Salidas")
    ax.set_xlabel("Tiempo (s)")
    ax.legend()
    fig.tight_layout()
    return fig
