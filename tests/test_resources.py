"""Detectors and windows must be released on every exit path."""
from app.camera import Camera
from app.languages import ASL
from app.modes.practice import PracticeScreen
from app.modes.recognize import RecognizeScreen
from cli import calibrate


class FakeTracker:
    instances = []

    def __init__(self, *args, **kwargs):
        self.closed = 0
        FakeTracker.instances.append(self)

    def close(self):
        self.closed += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, frame):
        return []


class FakeCamera(Camera):
    def open(self):
        return True


def setup_function(function):
    FakeTracker.instances.clear()


def make(screen_cls, monkeypatch):
    module = __import__(screen_cls.__module__, fromlist=["x"])
    monkeypatch.setattr(module, "HandTracker", FakeTracker)
    monkeypatch.setattr(module, "mediapipe_available", lambda: True)
    return screen_cls(ASL, FakeCamera(0))


def test_recognize_leave_closes_tracker(monkeypatch):
    screen = make(RecognizeScreen, monkeypatch)
    screen.enter()
    tracker = FakeTracker.instances[0]
    screen.leave()
    assert tracker.closed == 1
    assert screen._tracker is None


def test_practice_leave_closes_tracker(monkeypatch):
    screen = make(PracticeScreen, monkeypatch)
    screen.enter()
    tracker = FakeTracker.instances[0]
    screen.leave()
    assert tracker.closed == 1
    assert screen._tracker is None


def test_reentering_creates_a_fresh_tracker(monkeypatch):
    screen = make(RecognizeScreen, monkeypatch)
    screen.enter()
    screen.leave()
    screen.enter()
    assert len(FakeTracker.instances) == 2
    assert FakeTracker.instances[0].closed == 1
    assert FakeTracker.instances[1].closed == 0


def test_leave_without_enter_is_harmless(monkeypatch):
    screen = make(RecognizeScreen, monkeypatch)
    screen.leave()


def test_calibrate_closes_tracker_when_camera_is_unavailable(monkeypatch, capsys):
    class NoCamera:
        opened = False
        error = "camera busy"

        def __init__(self, index):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    windows = []
    monkeypatch.setattr(calibrate, "HandTracker", FakeTracker)
    monkeypatch.setattr(calibrate, "Camera", NoCamera)
    monkeypatch.setattr(calibrate.cv2, "namedWindow", lambda *a, **k: windows.append(a))

    assert calibrate.main(["A"]) == 1
    assert "camera busy" in capsys.readouterr().out
    assert FakeTracker.instances[0].closed == 1
    assert windows == [], "no window should be opened when the camera failed"
