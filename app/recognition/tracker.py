import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.recognition.geometry import HandFeatures, extract
from app.recognition.model import ModelUnavailable, hand_landmarker_model

_IMPORT_ERROR: Optional[Exception] = None
try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    _IMPORT_ERROR = exc

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

SOLUTIONS = "solutions"
TASKS = "tasks"

@dataclass
class HandObservation:
    features: HandFeatures
    pixels: List[Tuple[int, int]]
    confidence: float

class TrackerUnavailable(RuntimeError):
    pass

def backend() -> str:
    if mp is None:
        return ""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return SOLUTIONS
    if hasattr(mp, "tasks"):
        return TASKS
    return ""

def mediapipe_available() -> bool:
    return backend() != ""

def mediapipe_error() -> str:
    if _IMPORT_ERROR is not None:
        return str(_IMPORT_ERROR)
    if not mediapipe_available():
        return "installed mediapipe has neither solutions.hands nor tasks"
    return ""

class HandTracker:

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        model_complexity: int = 0,
    ) -> None:
        self.backend = backend()
        if not self.backend:
            raise TrackerUnavailable(
                f"MediaPipe is not available: {mediapipe_error()}\n"
                "Run inside Docker or the conda environment (see README)."
            )
        self.max_hands = max_hands
        self._t0 = time.monotonic()
        try:
            if self.backend == SOLUTIONS:
                self._impl = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    model_complexity=model_complexity,
                    min_detection_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence,
                )
            else:
                self._impl = self._make_landmarker(detection_confidence,
                                                   tracking_confidence)
        except TrackerUnavailable:
            raise
        except Exception as exc:
            raise TrackerUnavailable(
                f"MediaPipe ({self.backend}) failed to create a detector: {exc}"
            ) from exc

    def _make_landmarker(self, detection_confidence: float, tracking_confidence: float):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        try:
            model_path = hand_landmarker_model()
        except ModelUnavailable as exc:
            raise TrackerUnavailable(str(exc)) from exc
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> List[HandObservation]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        if self.backend == SOLUTIONS:
            rgb.flags.writeable = False
            result = self._impl.process(rgb)
            hands = result.multi_hand_landmarks or []
            handedness = [
                (c.classification[0].label, float(c.classification[0].score))
                for c in (result.multi_handedness or [])
            ]
            landmarks = [[(p.x, p.y, p.z) for p in lm.landmark] for lm in hands]
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - self._t0) * 1000)
            result = self._impl.detect_for_video(image, timestamp_ms)
            landmarks = [[(p.x, p.y, p.z) for p in lm] for lm in result.hand_landmarks]
            handedness = [(c[0].category_name, float(c[0].score))
                          for c in result.handedness]
        return self._observations(landmarks, handedness, w, h)

    @staticmethod
    def _observations(landmarks: Sequence[Sequence[Tuple[float, float, float]]],
                      handedness: Sequence[Tuple[str, float]],
                      w: int, h: int) -> List[HandObservation]:
        out: List[HandObservation] = []
        for i, pts in enumerate(landmarks):
            label, score = handedness[i] if i < len(handedness) else ("Right", 1.0)
            label = "Left" if label == "Right" else "Right"
            pixels = [(int(x * w), int(y * h)) for x, y, _ in pts]
            out.append(HandObservation(extract(pts, label), pixels, score))
        return out

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()