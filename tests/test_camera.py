import time
import numpy as np
import pytest
from camera import LiveSession
from jobs import finalize, SessionSettings
from report import utc_now
from video_processor import VideoError
from zones import SpaceConfig, EntryLine


class Walker:
    def __init__(self): self.n = 0
    def update(self, frame):
        self.n += 1
        y = 10 + self.n * 4
        return [{"id": 1, "box": (90, y - 30, 110, y), "conf": .9}]


CFG = SpaceConfig(EntryLine((0, .5), (1, .5), 1), [], (200, 100))


def feed(live, n=60):
    for _ in range(n):
        live.process_image(np.zeros((100, 200, 3), np.uint8))
        time.sleep(0.02)


def test_live_session_end_to_end():
    live = LiveSession()
    live.begin(CFG, lambda: Walker(), max_width=200, min_interval=0.0)
    t0 = time.time()
    while not live.ready and time.time() - t0 < 5: time.sleep(0.05)
    feed(live)
    tracks, pinfo, size, ref = live.end()
    assert size == (200, 100) and pinfo["modo"] == "live" and pinfo["frames_procesados"] >= 5
    assert ref is not None and not live.active
    res = finalize(tracks, pinfo, CFG, size, SessionSettings(write_video=False), "sid", utc_now(), ref, source="camera_live")
    r = res["report"]
    assert r["source"] == "camera_live" and r["unique_people_estimated"] == 1 and r["entries"] == 1
    assert any("en vivo" in w for w in r["warnings"]) and not any("manualmente" in w for w in r["warnings"])
    assert res["video"] is None and res["deliveries"] == []


def test_end_without_frames_is_clear_error():
    live = LiveSession()
    live.begin(CFG, lambda: Walker(), 200, 0.0)
    time.sleep(0.3)
    with pytest.raises(VideoError):
        live.end()


def test_detector_load_failure_reported():
    def bad(): raise RuntimeError("sin modelo")
    live = LiveSession()
    live.begin(CFG, bad)
    time.sleep(0.3)
    assert "No se pudo cargar" in live.error
    with pytest.raises(VideoError):
        live.end()


def test_inactive_session_passes_frame_through():
    img = np.zeros((10, 10, 3), np.uint8)
    assert LiveSession().process_image(img) is img
