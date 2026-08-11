import os
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.recognition.tracker import HAND_CONNECTIONS

BGR = Tuple[int, int, int]

BG: BGR = (24, 22, 20)
PANEL: BGR = (40, 36, 33)
FG: BGR = (240, 238, 235)
MUTED: BGR = (150, 145, 140)
ACCENT: BGR = (120, 190, 90)
ACCENT2: BGR = (230, 170, 60)
DANGER: BGR = (80, 90, 230)
BONE: BGR = (200, 200, 200)
JOINT: BGR = (90, 200, 250)

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
)

@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    env = os.environ.get("SLT_FONT")
    for path in ((env,) if env else ()) + _FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()

def text_size(text: str, size: int) -> Tuple[int, int]:
    box = _font(size).getbbox(text)
    return (box[2] - box[0], box[3] - box[1])

class TextLayer:

    def __init__(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        self._img = img
        self._overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._overlay)
        self._box: Optional[List[int]] = None

    def text(self, text: str, pos: Tuple[int, int], size: int = 20,
             color: BGR = FG, anchor: str = "lt") -> None:
        if not text:
            return
        font = _font(size)
        x, y = pos
        if anchor != "lt":
            w, _ = text_size(text, size)
            x = x - w // 2 if anchor == "mt" else x - w
        self._draw.text((x, y), text, font=font, fill=color[::-1] + (255,))
        self._grow(self._draw.textbbox((x, y), text, font=font))

    def _grow(self, bbox: Tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        pad = 2
        new = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
        if self._box is None:
            self._box = new
            return
        self._box = [min(self._box[0], new[0]), min(self._box[1], new[1]),
                     max(self._box[2], new[2]), max(self._box[3], new[3])]

    def flush(self) -> None:
        if self._box is None:
            return
        h, w = self._img.shape[:2]
        x0, y0 = max(0, self._box[0]), max(0, self._box[1])
        x1, y1 = min(w, self._box[2]), min(h, self._box[3])
        self._box = None
        if x1 <= x0 or y1 <= y0:
            return
        patch = np.asarray(self._overlay.crop((x0, y0, x1, y1)), dtype=np.float32)
        alpha = patch[:, :, 3:4] / 255.0
        rgb = patch[:, :, 2::-1]
        roi = self._img[y0:y1, x0:x1].astype(np.float32)
        self._img[y0:y1, x0:x1] = (rgb * alpha + roi * (1.0 - alpha)).astype(np.uint8)

    def __enter__(self) -> "TextLayer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()

def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], size: int = 20,
              color: BGR = FG, anchor: str = "lt") -> None:
    layer = TextLayer(img)
    layer.text(text, pos, size, color, anchor)
    layer.flush()

def new_canvas(w: int, h: int, color: BGR = BG) -> np.ndarray:
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:, :] = color
    return canvas

def panel(img: np.ndarray, x: int, y: int, w: int, h: int,
          color: BGR = PANEL, alpha: float = 0.88) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:, :] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

def progress_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float,
                 color: BGR = ACCENT) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), MUTED, 1)
    filled = int(w * max(0.0, min(1.0, value)))
    if filled > 1:
        cv2.rectangle(img, (x + 1, y + 1), (x + filled - 1, y + h - 1), color, -1)

def draw_hand(img: np.ndarray, pixels: Sequence[Tuple[int, int]],
              bone: BGR = BONE, joint: BGR = JOINT, thickness: int = 3,
              radius: int = 4) -> None:
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pixels[a], pixels[b], bone, thickness, cv2.LINE_AA)
    for i, p in enumerate(pixels):
        r = radius + 2 if i in (4, 8, 12, 16, 20) else radius
        cv2.circle(img, p, r, joint, -1, cv2.LINE_AA)

def draw_trail(img: np.ndarray, points: Iterable[Tuple[int, int]],
               color: BGR = ACCENT2) -> None:
    pts = list(points)
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        fade = 0.25 + 0.75 * (i / max(1, len(pts) - 1))
        c = tuple(int(v * fade) for v in color)
        cv2.line(img, a, b, c, 2, cv2.LINE_AA)

def hotkeys(img: np.ndarray, items: Sequence[Tuple[str, str]],
            y: Optional[int] = None, layer: Optional[TextLayer] = None) -> None:
    h, w = img.shape[:2]
    y = h - 34 if y is None else y
    panel(img, 0, y, w, 34, PANEL, 0.92)
    own = layer or TextLayer(img)
    x = 16
    for key, label in items:
        chunk = text_size(f"[{key}]", 16)[0] + 6 + text_size(label, 16)[0] + 18
        if x + chunk > w - 8:
            break
        own.text(f"[{key}]", (x, y + 8), 16, ACCENT2)
        x += text_size(f"[{key}]", 16)[0] + 6
        own.text(label, (x, y + 8), 16, MUTED)
        x += text_size(label, 16)[0] + 18
    if layer is None:
        own.flush()

def title_bar(img: np.ndarray, title: str, subtitle: str = "",
              layer: Optional[TextLayer] = None) -> None:
    w = img.shape[1]
    panel(img, 0, 0, w, 46, PANEL, 0.92)
    own = layer or TextLayer(img)
    own.text(title, (16, 11), 22, FG)
    own.text(subtitle, (w - 16, 15), 16, MUTED, anchor="rt")
    if layer is None:
        own.flush()

def fit_hand(pixels: Sequence[Tuple[int, int]], box: Tuple[int, int, int, int],
             margin: int = 24) -> List[Tuple[int, int]]:
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    src_w = max(1, max(xs) - min(xs))
    src_h = max(1, max(ys) - min(ys))
    x, y, w, h = box
    k = min((w - 2 * margin) / src_w, (h - 2 * margin) / src_h)
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    return [(int(x + w / 2 + (px - cx) * k), int(y + h / 2 + (py - cy) * k))
            for px, py in pixels]