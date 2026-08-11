from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.languages import ASL, RSL, Language
from app.modes.base import QUIT, Go, Screen
from app.recognition.tracker import mediapipe_available, mediapipe_error
from app.synthesis.animator import Animator
from app.synthesis.hand_model import pose_to_points, to_pixels
from app.ui import renderer as r
from app.ui.keys import Key

@dataclass(frozen=True)
class Item:
    title: str
    subtitle: str
    target: Go
    language: Language

def _items() -> List[Item]:
    return [
        Item(ASL.to_text, "camera -> letters of the English alphabet",
             Go("recognize", {"code": ASL.code}), ASL),
        Item(RSL.to_text, "camera -> letters of the Russian alphabet",
             Go("recognize", {"code": RSL.code}), RSL),
        Item("Sign test: ASL", "we play the animation, you repeat it",
             Go("practice", {"code": ASL.code}), ASL),
        Item("Sign test: RSL", "we play the animation, you repeat it",
             Go("practice", {"code": RSL.code}), RSL),
    ]

class MenuScreen(Screen):

    def __init__(self, selected: int = 0) -> None:
        self.items = _items()
        self.i = max(0, min(selected, len(self.items) - 1))
        self._preview_for: Optional[str] = None
        self._animator = Animator(self.language.spec_for)
        self._frame = self._animator.update(0.0)
        self._sync_preview()

    @property
    def language(self) -> Language:
        return self.items[self.i].language

    def _sync_preview(self) -> None:
        lang = self.language
        if self._preview_for == lang.code:
            return
        self._preview_for = lang.code
        self._animator = Animator(lang.spec_for, speed=1.15)
        self._animator.loop = True
        self._animator.set_text(lang.sample)
        self._frame = self._animator.update(0.0)

    def update(self, dt: float) -> Optional[Go]:
        self._frame = self._animator.update(dt)
        return None

    def on_key(self, key: Key) -> Optional[Go]:
        if key.is_("esc") or key.lower() == "q":
            return QUIT
        if key.is_("up"):
            self.i = (self.i - 1) % len(self.items)
            self._sync_preview()
        elif key.is_("down"):
            self.i = (self.i + 1) % len(self.items)
            self._sync_preview()
        elif key.is_("enter", "space"):
            return self.items[self.i].target
        elif key.char.isdigit() and 1 <= int(key.char) <= len(self.items):
            self.i = int(key.char) - 1
            self._sync_preview()
            return self.items[self.i].target
        return None

    def draw(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        canvas[:, :] = r.BG
        split = int(w * 0.58)
        self._draw_preview(canvas, (split, 96, w - split, h - 190))
        with r.TextLayer(canvas) as text:
            text.text("Sign Language -> Text", (40, 34), 34, r.FG)
            text.text("ASL and RSL: recognition and reverse playback",
                      (42, 74), 17, r.MUTED)
            y = 130
            for n, item in enumerate(self.items):
                self._draw_item(canvas, text, item, n, y, split - 72)
                y += 92
            if not mediapipe_available():
                r.panel(canvas, 40, h - 132, split - 72, 44, r.PANEL, 0.9)
                text.text("MediaPipe is not available, camera modes will not start",
                          (56, h - 124), 15, r.DANGER)
                text.text(mediapipe_error()[:70], (56, h - 106), 13, r.MUTED)
            r.hotkeys(canvas, [
                ("Up/Down", "select"), ("Enter", "open"), ("1-4", "quick select"),
                ("Esc", "quit"),
            ], layer=text)

    def _draw_item(self, canvas: np.ndarray, text: r.TextLayer, item: Item,
                   n: int, y: int, width: int) -> None:
        active = n == self.i
        r.panel(canvas, 40, y, width, 74, r.PANEL, 0.95 if active else 0.55)
        if active:
            canvas[y:y + 74, 40:46] = r.ACCENT
        text.text(f"{n + 1}", (62, y + 22), 26, r.ACCENT if active else r.MUTED)
        text.text(item.title, (98, y + 14), 23, r.FG if active else r.MUTED)
        text.text(item.subtitle, (99, y + 44), 15, r.MUTED)

    def _draw_preview(self, canvas: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        x, y, bw, bh = box
        r.panel(canvas, x, y, bw, bh, r.PANEL, 0.5)
        frame = self._frame
        size = min(bw, bh) * 0.42
        cx = x + bw // 2 + int(frame.offset[0] * size)
        cy = y + bh // 2 + int(-frame.offset[1] * size) + 20
        pixels = to_pixels(pose_to_points(frame.pose), (cx, cy), size)
        r.draw_hand(canvas, pixels, thickness=3, radius=4)
        with r.TextLayer(canvas) as text:
            text.text(self.language.name, (x + bw // 2, y + 16), 20, r.MUTED, anchor="mt")
            text.text(frame.letter or "-", (x + bw // 2, y + bh - 54), 40,
                      r.ACCENT2, anchor="mt")