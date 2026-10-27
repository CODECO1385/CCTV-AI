import cv2
import numpy as np
import pytest
from video_processor import probe_video, process_video, ProcSettings, VideoError


def make_video(path, n=50, fps=10, w=160, h=120):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), i * 4 % 255, np.uint8))
    wr.release()


class FakeTracker:
    def update(self, frame):
        return [{"id": 1, "box": (40, 20, 60, 100), "conf": 0.9}]


def test_invalid_files(tmp_path):
    with pytest.raises(VideoError):
        probe_video(tmp_path / "no_existe.mp4")
    empty = tmp_path / "vacio.mp4"; empty.write_bytes(b"")
    with pytest.raises(VideoError):
        probe_video(empty)
    junk = tmp_path / "x.avi"; junk.write_bytes(b"esto no es un video" * 50)
    with pytest.raises(VideoError):
        probe_video(junk)


def test_process_with_fake_tracker(tmp_path):
    v = tmp_path / "v.avi"; make_video(v)
    info = probe_video(v)
    assert (info.width, info.height) == (160, 120) and info.duration == pytest.approx(5.0, abs=0.2)
    out = tmp_path / "o.mp4"
    df, pinfo = process_video(str(v), FakeTracker(), ProcSettings(target_fps=5, max_width=0), out_video=out)
    assert pinfo["stride"] == 2 and len(df) == 25
    assert df["t"].iloc[1] == pytest.approx(0.2)          # marcas temporales originales
    assert df["x"].iloc[0] == 50 and df["y"].iloc[0] == 100  # centro inferior
    assert out.exists() and out.stat().st_size > 0


def test_downscale_maps_back_to_original(tmp_path):
    v = tmp_path / "v.avi"; make_video(v, w=200, h=100)
    df, _ = process_video(str(v), FakeTracker(), ProcSettings(target_fps=10, max_width=100, write_video=False))
    assert df["x"].iloc[0] == 100  # caja a 50 px en 100 de ancho -> 100 en el original


def test_detector_failure_is_reported(tmp_path):
    class Bad:
        def update(self, f): raise RuntimeError("boom")
    v = tmp_path / "v.avi"; make_video(v)
    with pytest.raises(VideoError):
        process_video(str(v), Bad(), ProcSettings(write_video=False))
