import math
from collections import deque
from typing import Deque, List, Tuple

from app.alphabet.spec import M_CIRCLE, M_DOWN, M_HOOK, M_NONE, M_SHAKE, M_ZIGZAG
from app.recognition.geometry import HandFeatures

MIN_TRAVEL = 0.55

class MotionDetector:

    def __init__(self, window: int = 24) -> None:
        self._track: Deque[Tuple[float, float, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self._track.clear()

    def update(self, f: HandFeatures) -> None:
        tip = f.raw[20] if (f.extended["pinky"] and not f.extended["index"]) else f.raw[8]
        s = max(1e-6, f.scale)
        self._track.append((tip[0] / s, tip[1] / s, tip[2] / s))

    def _travel(self) -> float:
        pts = list(self._track)
        if len(pts) < 4:
            return 0.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def _direction_changes(self, axis: int) -> int:
        pts = list(self._track)
        signs: List[int] = []
        for a, b in zip(pts, pts[1:]):
            d = b[axis] - a[axis]
            if abs(d) > 0.03:
                signs.append(1 if d > 0 else -1)
        return sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    def classify(self) -> Tuple[str, float]:
        pts = list(self._track)
        if len(pts) < 8:
            return M_NONE, 0.0
        travel = self._travel()
        if travel < MIN_TRAVEL:
            return M_NONE, 0.0
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        flips_x = self._direction_changes(0)
        flips_y = self._direction_changes(1)
        conf = min(1.0, travel / 1.4)
        if flips_x >= 2 and abs(dx) > 0.3:
            return M_ZIGZAG, conf
        if (flips_x + flips_y) >= 3 and math.hypot(dx, dy) < 0.35 * travel:
            return M_SHAKE, conf
        mid = len(pts) // 2
        first_dy = pts[mid][1] - pts[0][1]
        second_dx = pts[-1][0] - pts[mid][0]
        if first_dy > 0.3 and abs(second_dx) > 0.25:
            return M_HOOK, conf
        if flips_x >= 1 and flips_y >= 1 and math.hypot(dx, dy) < 0.3 * travel:
            return M_CIRCLE, conf * 0.8
        if dy > 0.4 and abs(dx) < 0.5 * dy:
            return M_DOWN, conf
        return M_NONE, 0.0