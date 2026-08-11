import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from app.alphabet.spec import (
    ANY, EXT, FOLD, HALF,
    T_ACROSS, T_ANY, T_OUT, T_TUCKED, T_UP,
    LetterSpec,
)
from app.recognition.geometry import FINGER_JOINTS, LONG_FINGERS

Vec2 = Tuple[float, float]

MCP_POS: Dict[str, Vec2] = {
    "index": (-0.24, 0.80),
    "middle": (-0.02, 0.86),
    "ring": (0.19, 0.81),
    "pinky": (0.38, 0.71),
}
THUMB_CMC: Vec2 = (-0.34, 0.22)

SEG_LEN: Dict[str, Tuple[float, float, float]] = {
    "index": (0.34, 0.21, 0.16),
    "middle": (0.37, 0.23, 0.17),
    "ring": (0.34, 0.21, 0.16),
    "pinky": (0.26, 0.16, 0.14),
    "thumb": (0.30, 0.22, 0.17),
}

SPREAD_OPEN: Dict[str, float] = {"index": 36.0, "middle": 12.0, "ring": -14.0, "pinky": -38.0}
SPREAD_CLOSED: Dict[str, float] = {"index": 6.0, "middle": 2.0, "ring": -3.0, "pinky": -8.0}

BEND: Dict[str, Tuple[float, float, float]] = {
    EXT: (3.0, 4.0, 3.0),
    HALF: (42.0, 62.0, 22.0),
    FOLD: (88.0, 96.0, 34.0),
    ANY: (16.0, 20.0, 10.0),
}

THUMB_POSE: Dict[str, Tuple[float, Tuple[float, float]]] = {
    T_UP: (26.0, (6.0, 4.0)),
    T_OUT: (74.0, (10.0, 6.0)),
    T_ACROSS: (-95.0, (20.0, 8.0)),
    T_TUCKED: (-110.0, (75.0, 55.0)),
    T_ANY: (34.0, (26.0, 16.0)),
}

CROSS_ANGLE = 22.0

HAND_ROTATION: Dict[str, float] = {
    "up": 0.0, "down": 180.0, "left": 75.0, "right": -75.0, "forward": 25.0,
}

@dataclass
class HandPose:
    fingers: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    thumb: Tuple[float, float, float] = (-62.0, 28.0, 18.0)
    rotation: float = 0.0
    mirror: bool = False
    touch: Tuple[str, ...] = ()
    under: str = ""
    crossed: bool = False
    letter: str = ""

    def copy(self) -> "HandPose":
        return HandPose(dict(self.fingers), self.thumb, self.rotation, self.mirror,
                        self.touch, self.under, self.crossed, self.letter)

def pose_from_spec(spec: LetterSpec) -> HandPose:
    spread_open = bool(spec.spread) if spec.spread is not None else False
    table = SPREAD_OPEN if spread_open else SPREAD_CLOSED
    fingers: Dict[str, Tuple[float, float, float, float]] = {}
    for name in LONG_FINGERS:
        state = spec.fingers.get(name, ANY)
        mcp, pip, dip = BEND[state]
        base = table[name]
        if state == FOLD:
            base *= 0.35
        if spec.crossed:
            base += {"index": -CROSS_ANGLE, "middle": CROSS_ANGLE}.get(name, 0.0)
        fingers[name] = (base, mcp, pip, dip)
    base_angle, (ip, tip) = THUMB_POSE.get(spec.thumb, THUMB_POSE[T_ANY])
    rotation = HAND_ROTATION.get(spec.fingers_dir or "up", 0.0)
    return HandPose(
        fingers=fingers,
        thumb=(base_angle, ip, tip),
        rotation=rotation,
        mirror=(spec.palm_facing == "away"),
        touch=tuple(spec.touch),
        under=spec.thumb_under or "",
        crossed=bool(spec.crossed),
        letter=spec.letter,
    )

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def blend(a: HandPose, b: HandPose, t: float) -> HandPose:
    t = ease(t)
    fingers = {
        name: tuple(_lerp(x, y, t) for x, y in zip(a.fingers[name], b.fingers[name]))
        for name in LONG_FINGERS
    }
    delta = (b.rotation - a.rotation + 180.0) % 360.0 - 180.0
    return HandPose(
        fingers=fingers,
        thumb=tuple(_lerp(x, y, t) for x, y in zip(a.thumb, b.thumb)),
        rotation=a.rotation + delta * t,
        mirror=b.mirror if t > 0.5 else a.mirror,
        touch=b.touch if t > 0.5 else a.touch,
        under=b.under if t > 0.5 else a.under,
        crossed=b.crossed if t > 0.5 else a.crossed,
        letter=b.letter if t > 0.5 else a.letter,
    )

def _rotate(p: Vec2, deg: float) -> Vec2:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)

def _chain(origin: Vec2, base_angle: float, bends: Sequence[float],
           lengths: Sequence[float]) -> List[Vec2]:
    pts: List[Vec2] = []
    x, y = origin
    angle = base_angle
    for bend, length in zip(bends, lengths):
        angle += bend
        r = math.radians(angle)
        x += -math.sin(r) * length
        y += math.cos(r) * length
        pts.append((x, y))
    return pts

def pose_to_points(pose: HandPose) -> List[Vec2]:
    pts: List[Vec2] = [(0.0, 0.0)] * 21
    base, ip, tip_bend = pose.thumb
    pts[1] = THUMB_CMC
    thumb_chain = _chain(THUMB_CMC, base, (0.0, ip, tip_bend), SEG_LEN["thumb"])
    pts[2], pts[3], pts[4] = thumb_chain
    for name in LONG_FINGERS:
        mcp_i, pip_i, dip_i, tip_i = FINGER_JOINTS[name]
        base_angle, b1, b2, b3 = pose.fingers[name]
        origin = MCP_POS[name]
        pts[mcp_i] = origin
        chain = _chain(origin, base_angle, (b1, b2, b3), SEG_LEN[name])
        pts[pip_i], pts[dip_i], pts[tip_i] = chain
    targets: List[Vec2] = [pts[FINGER_JOINTS[n][3]] for n in pose.touch
                           if n in FINGER_JOINTS]
    if pose.under in FINGER_JOINTS:
        pip = pts[FINGER_JOINTS[pose.under][1]]
        palm_y = sum(MCP_POS[n][1] for n in LONG_FINGERS) / 4.0
        targets.append((pip[0], _lerp(pip[1], palm_y * 0.55, 0.55)))
    if targets:
        tx = sum(p[0] for p in targets) / len(targets)
        ty = sum(p[1] for p in targets) / len(targets)
        old = pts[4]
        pts[4] = (_lerp(old[0], tx, 0.85), _lerp(old[1], ty, 0.85))
        pts[3] = ((pts[2][0] + pts[4][0]) / 2.0, (pts[2][1] + pts[4][1]) / 2.0)
    if pose.mirror:
        pts = [(-x, y) for x, y in pts]
    if abs(pose.rotation) > 1e-6:
        pts = [_rotate(p, pose.rotation) for p in pts]
    return pts

def palm_center(pts: Sequence[Vec2]) -> Vec2:
    anchors = [pts[0], pts[5], pts[9], pts[17]]
    return (sum(p[0] for p in anchors) / 4.0, sum(p[1] for p in anchors) / 4.0)

def to_pixels(pts: Sequence[Vec2], center: Tuple[int, int], size: float) -> List[Tuple[int, int]]:
    cx, cy = center
    px, py = palm_center(pts)
    return [(int(cx + (x - px) * size), int(cy - (y - py) * size)) for x, y in pts]

def points_to_landmarks(pts: Sequence[Vec2]) -> List[Tuple[float, float, float]]:
    return [(x, -y, 0.0) for x, y in pts]