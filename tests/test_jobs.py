import cv2, numpy as np, pytest
from jobs import run_session, Job, SessionSettings
from zones import SpaceConfig, EntryLine


def make_video(path, n=60, fps=10, w=200, h=100):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n): wr.write(np.full((h, w, 3), i, np.uint8))
    wr.release()


class Walker:
    """Persona que cruza la línea y=50 de arriba (afuera) hacia abajo (adentro)."""
    def __init__(self, stop=None, after=None): self.n, self.stop, self.after = 0, stop, after
    def update(self, frame):
        self.n += 1
        if self.stop and self.n == self.after: self.stop.set()
        y = 10 + self.n * 3
        return [{"id": 1, "box": (90, y - 40, 110, y), "conf": .9}]


CFG = SpaceConfig(EntryLine((0, .5), (1, .5), 1), [], (200, 100))


def run(tmp_path, tracker, **kw):
    v = tmp_path / "v.avi"; make_video(v)
    s = SessionSettings(target_fps=5, max_width=0, write_video=False)
    from report import utc_now, new_session_id
    return run_session(str(v), lambda: tracker, CFG, (200, 100), s, new_session_id(), utc_now(), **kw)


def test_full_session_report_and_no_delivery_when_disabled(tmp_path):
    res = run(tmp_path, Walker())
    r = res["report"]
    assert r["entries"] == 1 and r["unique_people_estimated"] == 1
    assert res["deliveries"] == []          # nada se "envía" si no está activado
    assert r["duration_seconds"] == pytest.approx(5.8, abs=0.3)


def test_manual_stop_gives_partial_report(tmp_path):
    import threading
    ev = threading.Event()
    res = run(tmp_path, Walker(ev, after=8), should_stop=ev.is_set)
    assert res["pinfo"]["detenido_manual"] is True
    assert any("manual" in w for w in res["report"]["warnings"])
    assert res["pinfo"]["frames_procesados"] < 30


def test_job_thread_runs_and_reports_errors(tmp_path):
    v = tmp_path / "v.avi"; make_video(v)
    from report import utc_now
    from jobs import SessionSettings
    kw = dict(video_path=str(v), tracker_factory=lambda: Walker(), cfg=CFG, size=(200, 100),
              settings=SessionSettings(max_width=0, write_video=False), session_id="s", started_at=utc_now())
    j = Job(**kw); j.start(); j.thread.join(30)
    assert j.state == "done" and j.result["report"]["session_id"] == "s"
    bad = Job(**{**kw, "video_path": str(tmp_path / "nada.mp4")}); bad.start(); bad.thread.join(10)
    assert bad.state == "error" and bad.error
