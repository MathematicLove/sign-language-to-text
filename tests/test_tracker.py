import pytest

from app.recognition import model as model_module
from app.recognition.model import ModelUnavailable, hand_landmarker_model
from app.recognition.tracker import HandTracker, TrackerUnavailable, backend

class TestModelLookup:
    def test_env_path_wins(self, monkeypatch, tmp_path):
        path = tmp_path / "custom.task"
        path.write_bytes(b"model")
        monkeypatch.setenv("SLT_HAND_MODEL", str(path))
        assert model_module.find() == path

    def test_missing_model_is_reported_not_downloaded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLT_HAND_MODEL", str(tmp_path / "absent.task"))
        monkeypatch.setattr(model_module, "PROJECT_MODELS", tmp_path)
        monkeypatch.setattr(model_module, "CACHE_DIR", tmp_path)

        with pytest.raises(ModelUnavailable) as exc:
            hand_landmarker_model()
        assert "SLT_NO_DOWNLOAD" in str(exc.value)

    def test_download_failure_explains_manual_step(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SLT_NO_DOWNLOAD")
        monkeypatch.setattr(model_module, "CACHE_DIR", tmp_path)

        def boom(*args, **kwargs):
            raise OSError("no network")

        monkeypatch.setattr(model_module.urllib.request, "urlopen", boom)
        with pytest.raises(ModelUnavailable) as exc:
            model_module.download(tmp_path / "m.task")
        assert model_module.MODEL_URL in str(exc.value)

    def test_candidates_are_ordered(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLT_HAND_MODEL", str(tmp_path / "first.task"))
        paths = model_module.candidates()
        assert paths[0].name == "first.task"
        assert paths[-1].name == model_module.MODEL_NAME

class TestObservations:

    def points(self):
        from app.languages import ASL
        from app.synthesis.hand_model import pose_from_spec, pose_to_points

        pts = pose_to_points(pose_from_spec(ASL.spec_for("B")))

        return [(0.5 + x * 0.2, 0.5 - y * 0.2, 0.0) for x, y in pts]

    def test_pixels_are_scaled_to_frame(self):
        obs = HandTracker._observations([self.points()], [("Right", 0.9)], 640, 480)
        assert len(obs) == 1
        assert all(0 <= x <= 640 and 0 <= y <= 480 for x, y in obs[0].pixels)
        assert obs[0].confidence == pytest.approx(0.9)

    def test_handedness_is_mirrored(self):
        right = HandTracker._observations([self.points()], [("Right", 1.0)], 640, 480)
        left = HandTracker._observations([self.points()], [("Left", 1.0)], 640, 480)
        assert right[0].features.handedness == "Left"
        assert left[0].features.handedness == "Right"

    def test_missing_handedness_defaults_to_right(self):
        obs = HandTracker._observations([self.points()], [], 640, 480)
        assert obs[0].features.handedness == "Left"

    def test_no_hands_gives_no_observations(self):
        assert HandTracker._observations([], [], 640, 480) == []

def test_backend_is_known():
    assert backend() in ("solutions", "tasks", "")

def test_tracker_without_model_raises_tracker_unavailable(monkeypatch, tmp_path):
    if backend() != "tasks":
        pytest.skip("needs the new mediapipe tasks API")
    monkeypatch.setenv("SLT_HAND_MODEL", str(tmp_path / "absent.task"))
    monkeypatch.setattr(model_module, "PROJECT_MODELS", tmp_path)
    monkeypatch.setattr(model_module, "CACHE_DIR", tmp_path)
    with pytest.raises(TrackerUnavailable):
        HandTracker()