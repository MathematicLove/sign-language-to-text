from typing import Optional, Tuple

import numpy as np

from app.languages import Language
from app.modes.base import MENU, Go, Screen
from app.synthesis.animator import Animator
from app.synthesis.hand_model import pose_to_points, to_pixels
from app.ui import renderer as r
from app.ui.keys import Key

class SpellScreen(Screen):

    def __init__(self, language: Language, text: str = "") -> None:
        self.lang = language
        self.animator = Animator(language.spec_for)
        self.editing = True
        self._raw = text or language.sample
        self._frame = self.animator.update(0.0)
        self._apply_text()

    @property
    def shown(self) -> str:
        return self.lang.normalize(self._raw)

    def _apply_text(self) -> None:
        self.animator.set_text(self.shown)
        self._frame = self.animator.update(0.0)

    def play(self) -> None:
        self.editing = False
        self.animator.paused = False
        self.animator.restart()

    def update(self, dt: float) -> Optional[Go]:
        self._frame = self.animator.update(0.0 if self.editing else dt)
        return None

    def on_key(self, key: Key) -> Optional[Go]:
        if key.is_("esc"):
            if not self.editing:
                self.editing = True
                return None
            return MENU
        if key.is_("tab"):
            return Go("recognize", {"code": self.lang.code})
        if key.is_("enter"):
            if self.editing:
                self._apply_text()
                self.play()
            else:
                self.animator.restart()
            return None

        if self.editing:
            self._edit(key)
        else:
            self._control(key)
        return None

    def _edit(self, key: Key) -> None:
        if key.is_("backspace"):
            self._raw = self._raw[:-1]
        elif key.char:
            self._raw += key.char
        else:
            return
        self._apply_text()

    def _control(self, key: Key) -> None:
        if key.is_("space"):
            self.animator.toggle_pause()
        elif key.is_("left"):
            self.animator.step_back()
        elif key.is_("right"):
            self.animator.step_forward()
        elif key.char in "+=":
            self.animator.faster()
        elif key.char in "-_":
            self.animator.slower()
        elif key.lower() == "l":
            self.animator.loop = not self.animator.loop
        elif key.lower() == "e":
            self.editing = True

    def draw(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        canvas[:, :] = r.BG
        stage = (0, 46, w, h - 46 - 150)
        self._draw_hand(canvas, stage)

        with r.TextLayer(canvas) as text:
            r.title_bar(canvas, self.lang.to_signs,
                        "Tab - recognize signs from the camera", layer=text)
            self._draw_letter(canvas, text, stage)
            self._draw_input(canvas, text, (0, h - 150, w, 116))
            if self.editing:
                keys = [("Enter", "play as signs"), ("Backspace", "erase"),
                        ("Tab", "camera"), ("Esc", "menu")]
            else:
                keys = [("Space", "pause"), ("Left/Right", "letter"), ("+/-", "speed"),
                        ("L", "loop"), ("E", "edit"), ("Esc", "back")]
            r.hotkeys(canvas, keys, layer=text)

    def _draw_hand(self, canvas: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        frame = self._frame
        size = min(w, h) * 0.42
        cx = x + w // 2 + int(frame.offset[0] * size)
        cy = y + h // 2 + int(-frame.offset[1] * size)
        pixels = to_pixels(pose_to_points(frame.pose), (cx, cy), size)

        if frame.trail:
            trail = [(x + w // 2 + int(ox * size), y + h // 2 - int(oy * size))
                     for ox, oy in frame.trail]
            r.draw_trail(canvas, trail)
        r.draw_hand(canvas, pixels, thickness=4, radius=5)

    def _draw_letter(self, canvas: np.ndarray, text: r.TextLayer,
                     box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        frame = self._frame
        done, total = self.animator.progress

        text.text(frame.letter if frame.letter.strip() else ".",
                  (x + 90, y + 30), 76, r.ACCENT2, anchor="mt")
        if frame.hint:
            text.text(frame.hint[:52], (x + 24, y + 130), 15, r.MUTED)
        if total:
            r.progress_bar(canvas, x + 24, y + h - 24, w - 48, 8,
                           done / total if total else 0.0)
            text.text(f"{done}/{total}", (x + w - 24, y + h - 52), 15,
                      r.MUTED, anchor="rt")
        if self.animator.paused and not self.editing:
            text.text("paused", (x + w // 2, y + 20), 20, r.DANGER, anchor="mt")
        if self.animator.speed != 1.0:
            text.text(f"x{self.animator.speed:.2f}", (x + w - 24, y + 20), 16,
                      r.MUTED, anchor="rt")

    def _draw_input(self, canvas: np.ndarray, text: r.TextLayer,
                    box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        r.panel(canvas, x + 16, y, w - 32, h, r.PANEL, 0.95)
        label = "input (Enter to play)" if self.editing else "text"
        text.text(label, (x + 30, y + 8), 14, r.MUTED)

        shown = self.shown
        size = 28 if len(shown) <= 34 else 20
        cx = x + 30
        for i, ch in enumerate(shown[:80]):
            active = (not self.editing) and i == self._frame.index
            text.text(ch, (cx, y + 30), size, r.ACCENT if active else r.FG)
            cx += r.text_size(ch if ch != " " else "n", size)[0] + 3
        if self.editing and cx < x + w - 34:

            canvas[y + 30:y + 30 + size, cx:cx + 2] = r.ACCENT

        if self.lang.cyrillic and self.editing:
            text.text("latin is transliterated to cyrillic: sh, ch, zh, ts, yu, ya, shch",
                      (x + 30, y + h - 26), 14, r.MUTED)
        unknown = self.animator.unknown
        if unknown:
            text.text(f"not in the alphabet: {' '.join(sorted(set(unknown)))[:40]}",
                      (x + w - 30, y + h - 26), 14, r.DANGER, anchor="rt")