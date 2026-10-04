from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.camera import Camera
from app.languages import Language
from app.modes.base import MENU, Go, Screen
from app.recognition.classifier import LetterClassifier, Prediction, TextBuilder
from app.recognition.tracker import HandTracker, mediapipe_available, mediapipe_error
from app.synthesis.hand_model import pose_from_spec, pose_to_points, to_pixels
from app.ui import renderer as r
from app.ui.keys import Key

SIDE_W = 300

def blit_fit(canvas: np.ndarray, frame: np.ndarray,
             box: Tuple[int, int, int, int]) -> Tuple[int, int, float]:
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    k = min(w / fw, h / fh)
    nw, nh = int(fw * k), int(fh * k)
    dx, dy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return dx, dy, k

class RecognizeScreen(Screen):

    needs_camera = True

    def __init__(self, language: Language, camera: Camera) -> None:
        self.lang = language
        self.camera = camera
        self.text = TextBuilder()
        self.classifier = LetterClassifier(language.specs)
        self._tracker: Optional[HandTracker] = None
        self._error = "" if mediapipe_available() else (
            f"MediaPipe is not available: {mediapipe_error()}"
        )
        self._pred = Prediction()
        self._pixels: List[Tuple[int, int]] = []
        self._frame: Optional[np.ndarray] = None
        self._show_skeleton = True

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
        # Screens are rebuilt on every visit, so the detector must be freed here
        # or each visit leaks a MediaPipe graph.
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    def update(self, dt: float) -> Optional[Go]:
        if self._error or self._tracker is None:
            return None
        frame = self.camera.read()
        if frame is None:
            return None
        self._frame = frame

        hands = self._tracker.process(frame)
        if hands:
            self._pixels = hands[0].pixels
            self._pred = self.classifier.update(hands[0].features)
        else:
            self._pixels = []
            self._pred = self.classifier.update(None)

        letter = self.classifier.accept(self._pred)
        if letter is not None:
            self.text.push(letter)
        return None

    def on_key(self, key: Key) -> Optional[Go]:
        if key.is_("esc"):
            return MENU
        if key.is_("space"):
            self.text.space()
        elif key.is_("backspace"):
            self.text.backspace()
        elif key.lower() == "c":
            self.text.clear()
            self.classifier.reset()
        elif key.lower() == "r":

            return Go("spell", {"code": self.lang.code, "text": self.text.text})
        elif key.lower() == "h":
            self._show_skeleton = not self._show_skeleton
        return None

    def draw(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        canvas[:, :] = r.BG
        video_box = (0, 46, w - SIDE_W, h - 46 - 34)

        if self._frame is not None:
            dx, dy, k = blit_fit(canvas, self._frame, video_box)
            if self._pixels and self._show_skeleton:
                pts = [(int(dx + px * k), int(dy + py * k)) for px, py in self._pixels]
                r.draw_hand(canvas, pts)

        with r.TextLayer(canvas) as text:
            r.title_bar(canvas, self.lang.to_text,
                        "R - reverse direction (text to signs)", layer=text)
            if self._error:
                self._draw_error(canvas, text, video_box)
            else:
                self._draw_prediction(canvas, text, video_box)
            self._draw_side(canvas, text, (w - SIDE_W, 46, SIDE_W, h - 46 - 34))
            self._draw_text(canvas, text, video_box)
            r.hotkeys(canvas, [
                ("Space", "space"), ("Backspace", "erase"), ("C", "clear"),
                ("R", "text to signs"), ("H", "skeleton"), ("Esc", "menu"),
            ], layer=text)

    def _draw_error(self, canvas: np.ndarray, text: r.TextLayer,
                    box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        r.panel(canvas, x + 30, y + h // 2 - 60, w - 60, 120, r.PANEL, 0.95)
        text.text("Could not start recognition", (x + w // 2, y + h // 2 - 44),
                  22, r.DANGER, anchor="mt")
        for i, line in enumerate(self._error.split("\n")[:3]):
            text.text(line[:78], (x + w // 2, y + h // 2 - 8 + i * 22), 15,
                      r.MUTED, anchor="mt")

    def _draw_prediction(self, canvas: np.ndarray, text: r.TextLayer,
                         box: Tuple[int, int, int, int]) -> None:
        x, y, w, _ = box
        pred = self._pred
        r.panel(canvas, x + 16, y + 16, 210, 108, r.PANEL, 0.85)
        letter = pred.letter or "-"
        color = r.ACCENT if pred.stable else (r.FG if pred.score > 0.5 else r.MUTED)
        text.text(letter, (x + 68, y + 26), 62, color, anchor="mt")
        text.text(f"{pred.score * 100:4.0f}%", (x + 150, y + 40), 22, r.MUTED)
        if pred.motion != "none":
            text.text(pred.motion, (x + 150, y + 70), 14, r.ACCENT2)
        r.progress_bar(canvas, x + 30, y + 104, 182, 10, pred.progress,
                       r.ACCENT if pred.stable else r.ACCENT2)

        if pred.hint:
            r.panel(canvas, x + 16, y + 134, min(w - 32, 460), 30, r.PANEL, 0.8)
            text.text(pred.hint[:64], (x + 26, y + 140), 15, r.MUTED)

        for i, alt in enumerate(pred.alternatives[:3]):
            text.text(f"{alt.letter} {alt.score * 100:.0f}%",
                      (x + 26, y + 178 + i * 22), 15, r.MUTED)

    def _draw_side(self, canvas: np.ndarray, text: r.TextLayer,
                   box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        r.panel(canvas, x, y, w, h, r.PANEL, 0.95)
        text.text("reference", (x + w // 2, y + 14), 16, r.MUTED, anchor="mt")

        spec = self.lang.spec_for(self._pred.letter) if self._pred.letter else None
        if spec is None:
            text.text("show a sign", (x + w // 2, y + h // 2), 16, r.MUTED, anchor="mt")
            return
        pixels = to_pixels(pose_to_points(pose_from_spec(spec)),
                           (x + w // 2, y + h // 2 - 20), min(w, h) * 0.34)
        r.draw_hand(canvas, pixels, bone=(120, 120, 120), joint=r.ACCENT2,
                    thickness=2, radius=3)
        text.text(spec.letter, (x + w // 2, y + h - 74), 34, r.ACCENT2, anchor="mt")
        if spec.letter in self.lang.unverified:
            text.text("shape needs verification", (x + w // 2, y + h - 26), 13,
                      r.MUTED, anchor="mt")

    def _draw_text(self, canvas: np.ndarray, text: r.TextLayer,
                   box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        lines = self.text.wrapped(width=max(10, w // 20))
        bar_h = 30 + 30 * len(lines)
        top = y + h - bar_h - 12
        r.panel(canvas, x + 16, top, w - 32, bar_h, r.PANEL, 0.9)
        text.text("text", (x + 28, top + 6), 14, r.MUTED)
        for i, line in enumerate(lines):
            text.text(line or " ", (x + 28, top + 26 + i * 30), 26, r.FG)