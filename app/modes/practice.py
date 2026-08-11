import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.camera import Camera
from app.languages import Language
from app.modes.base import MENU, Go, Screen
from app.modes.recognize import blit_fit
from app.recognition.classifier import LetterClassifier, Prediction
from app.recognition.tracker import HandTracker, mediapipe_available, mediapipe_error
from app.synthesis.animator import Animator
from app.synthesis.hand_model import pose_to_points, to_pixels
from app.ui import renderer as r
from app.ui.keys import Key

SIDE_W = 340
SUCCESS_S = 1.1
GIVE_UP_S = 25.0

@dataclass
class Stats:
    asked: int = 0
    solved: int = 0
    skipped: int = 0
    total_time: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.solved / self.asked if self.asked else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.solved if self.solved else 0.0

class PracticeScreen(Screen):

    needs_camera = True

    def __init__(self, language: Language, camera: Camera) -> None:
        self.lang = language
        self.camera = camera
        self.stats = Stats()
        self.classifier = LetterClassifier(language.specs)
        self.animator = Animator(language.spec_for, speed=0.9)
        self.animator.loop = True

        self._tracker: Optional[HandTracker] = None
        self._error = "" if mediapipe_available() else (
            f"MediaPipe is not available: {mediapipe_error()}"
        )
        self._queue: List[str] = []
        self._target = ""
        self._pred = Prediction()
        self._pixels: List[Tuple[int, int]] = []
        self._frame: Optional[np.ndarray] = None
        self._solved_at = 0.0
        self._asked_at = time.monotonic()
        self._anim_frame = self.animator.update(0.0)
        self.next_letter()

    def enter(self) -> None:
        if self._error:
            return
        if self._tracker is None:
            try:
                self._tracker = HandTracker()
            except RuntimeError as exc:
                self._error = str(exc)
        if not self.camera.open():
            self._error = self.camera.error

    def leave(self) -> None:
        self.camera.release()
        self.classifier.reset()

    def next_letter(self, skipped: bool = False) -> None:
        if skipped and self._target:
            self.stats.skipped += 1
        if not self._queue:
            self._queue = list(self.lang.letters)
            random.shuffle(self._queue)
            if self._queue[0] == self._target and len(self._queue) > 1:
                self._queue[0], self._queue[-1] = self._queue[-1], self._queue[0]

        self._target = self._queue.pop(0)
        self.stats.asked += 1
        self._solved_at = 0.0
        self._asked_at = time.monotonic()
        self.classifier.reset()
        self.animator.set_text(self._target)
        self._anim_frame = self.animator.update(0.0)

    def update(self, dt: float) -> Optional[Go]:
        self._anim_frame = self.animator.update(dt)

        if self._solved_at:
            if time.monotonic() - self._solved_at >= SUCCESS_S:
                self.next_letter()
            return None
        if self._error or self._tracker is None:
            return None

        frame = self.camera.read()
        if frame is None:
            return None
        self._frame = frame

        hands = self._tracker.process(frame)
        self._pixels = hands[0].pixels if hands else []
        self._pred = self.classifier.update(hands[0].features if hands else None)

        if self._pred.stable and self._pred.letter == self._target:
            self._solved_at = time.monotonic()
            self.stats.solved += 1
            self.stats.total_time += self._solved_at - self._asked_at
        return None

    def on_key(self, key: Key) -> Optional[Go]:
        if key.is_("esc"):
            return MENU
        if key.is_("space") or key.lower() == "n":
            self.next_letter(skipped=True)
        elif key.is_("tab"):
            return Go("spell", {"code": self.lang.code, "text": self._target})
        elif key.char in "+=":
            self.animator.faster()
        elif key.char in "-_":
            self.animator.slower()
        return None

    def draw(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        canvas[:, :] = r.BG
        video_box = (0, 46, w - SIDE_W, h - 46 - 34)

        if self._frame is not None:
            dx, dy, k = blit_fit(canvas, self._frame, video_box)
            if self._pixels:
                pts = [(int(dx + px * k), int(dy + py * k)) for px, py in self._pixels]
                ok = bool(self._solved_at)
                r.draw_hand(canvas, pts, bone=r.ACCENT if ok else r.BONE)

        with r.TextLayer(canvas) as text:
            r.title_bar(canvas, f"Sign test - {self.lang.name}",
                        "repeat the sign shown by the animation", layer=text)
            self._draw_target(canvas, text, (w - SIDE_W, 46, SIDE_W, h - 46 - 34))
            self._draw_feedback(canvas, text, video_box)
            r.hotkeys(canvas, [
                ("Space", "another letter"), ("+/-", "playback speed"),
                ("Tab", "show as text"), ("Esc", "menu"),
            ], layer=text)

    def _draw_target(self, canvas: np.ndarray, text: r.TextLayer,
                     box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        r.panel(canvas, x, y, w, h, r.PANEL, 0.95)
        text.text("show the letter", (x + w // 2, y + 14), 16, r.MUTED, anchor="mt")
        text.text(self._target, (x + w // 2, y + 36), 64, r.ACCENT2, anchor="mt")

        frame = self._anim_frame
        size = min(w, h) * 0.32
        cx = x + w // 2 + int(frame.offset[0] * size)
        cy = y + h // 2 + 30 + int(-frame.offset[1] * size)
        if frame.trail:
            trail = [(x + w // 2 + int(ox * size), y + h // 2 + 30 - int(oy * size))
                     for ox, oy in frame.trail]
            r.draw_trail(canvas, trail)
        r.draw_hand(canvas, to_pixels(pose_to_points(frame.pose), (cx, cy), size),
                    bone=(120, 120, 120), joint=r.ACCENT2, thickness=2, radius=3)

        spec = self.lang.spec_for(self._target)
        if spec is not None and spec.hint:
            for i, chunk in enumerate(_wrap(spec.hint, 30)[:3]):
                text.text(chunk, (x + w // 2, y + h - 118 + i * 20), 14,
                          r.MUTED, anchor="mt")

        s = self.stats
        text.text(f"solved {s.solved} - skipped {s.skipped}",
                  (x + w // 2, y + h - 52), 16, r.FG, anchor="mt")
        text.text(f"accuracy {s.accuracy * 100:.0f}%"
                  + (f" - {s.avg_time:.1f}s per letter" if s.solved else ""),
                  (x + w // 2, y + h - 28), 14, r.MUTED, anchor="mt")

    def _draw_feedback(self, canvas: np.ndarray, text: r.TextLayer,
                       box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        if self._error:
            r.panel(canvas, x + 30, y + h // 2 - 40, w - 60, 80, r.PANEL, 0.95)
            text.text(self._error.split("\n")[0][:70], (x + w // 2, y + h // 2 - 10),
                      16, r.DANGER, anchor="mt")
            return

        if self._solved_at:
            r.panel(canvas, x + 16, y + 16, 260, 76, r.PANEL, 0.9)
            text.text("correct!", (x + 146, y + 30), 40, r.ACCENT, anchor="mt")
            return

        pred = self._pred
        r.panel(canvas, x + 16, y + 16, 260, 92, r.PANEL, 0.85)
        text.text("seen:", (x + 30, y + 24), 15, r.MUTED)
        text.text(pred.letter or "-", (x + 116, y + 22), 44,
                  r.FG if pred.letter else r.MUTED, anchor="mt")
        text.text(f"{pred.score * 100:.0f}%", (x + 180, y + 40), 18, r.MUTED)
        r.progress_bar(canvas, x + 30, y + 86, 232, 8, pred.progress)

        if time.monotonic() - self._asked_at > GIVE_UP_S:
            text.text("stuck? [Space] for another letter",
                      (x + 30, y + 122), 15, r.ACCENT2)

def _wrap(text: str, width: int) -> List[str]:
    lines: List[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines