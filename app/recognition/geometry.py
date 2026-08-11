import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

WRIST = 0

FINGER_JOINTS: Dict[str, Tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FINGERS: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
LONG_FINGERS: Tuple[str, ...] = ("index", "middle", "ring", "pinky")

Point = Tuple[float, float, float]

CURL_EXTENDED = 0.35
CURL_FOLDED = 0.68

def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def scale(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k, a[2] * k)

def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

def norm(a: Point) -> float:
    return math.sqrt(dot(a, a))

def unit(a: Point) -> Point:
    n = norm(a)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return scale(a, 1.0 / n)

def angle_between(a: Point, b: Point) -> float:
    n = norm(a) * norm(b)
    if n < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, dot(a, b) / n))
    return math.degrees(math.acos(c))

def joint_angle(p_prev: Point, p_mid: Point, p_next: Point) -> float:
    return 180.0 - angle_between(sub(p_prev, p_mid), sub(p_next, p_mid))

@dataclass
class HandFeatures:
    handedness: str
    scale: float
    curl: Dict[str, float] = field(default_factory=dict)
    extended: Dict[str, bool] = field(default_factory=dict)
    folded: Dict[str, bool] = field(default_factory=dict)
    half: Dict[str, bool] = field(default_factory=dict)
    spread: Dict[str, float] = field(default_factory=dict)
    tip_dist: Dict[str, float] = field(default_factory=dict)
    thumb_to_tip: Dict[str, float] = field(default_factory=dict)
    thumb_to_pip: Dict[str, float] = field(default_factory=dict)
    thumb_pose: str = "out"
    thumb_under: str = "index"
    crossed: bool = False
    palm_dir: Point = (0.0, 0.0, 0.0)
    hand_dir: Point = (0.0, 0.0, 0.0)
    palm_facing: str = "camera"
    fingers_dir: str = "up"
    raw: List[Point] = field(default_factory=list)

    def is_fist(self) -> bool:
        return all(self.folded[f] for f in LONG_FINGERS)

    def n_extended(self) -> int:
        return sum(1 for f in LONG_FINGERS if self.extended[f])

def hand_scale(pts: Sequence[Point]) -> float:
    palm_width = norm(sub(pts[17], pts[5]))
    palm_len = norm(sub(pts[9], pts[WRIST]))
    return max(1e-6, 0.5 * (palm_width + palm_len))

def finger_curl(pts: Sequence[Point], finger: str) -> float:
    mcp, pip, dip, tip = FINGER_JOINTS[finger]
    if finger == "thumb":
        total = (joint_angle(pts[mcp], pts[pip], pts[dip])
                 + joint_angle(pts[pip], pts[dip], pts[tip]))
        max_total = 170.0
    else:
        total = (
            joint_angle(pts[WRIST], pts[mcp], pts[pip])
            + joint_angle(pts[mcp], pts[pip], pts[dip])
            + joint_angle(pts[pip], pts[dip], pts[tip])
        )
        max_total = 270.0
    by_angles = max(0.0, min(1.0, total / max_total))
    chain = (
        norm(sub(pts[pip], pts[mcp]))
        + norm(sub(pts[dip], pts[pip]))
        + norm(sub(pts[tip], pts[dip]))
    )
    straight = norm(sub(pts[tip], pts[mcp]))
    by_length = 1.0 - max(0.0, min(1.0, straight / max(1e-6, chain)))
    by_length = min(1.0, by_length * 1.9)
    return 0.55 * by_angles + 0.45 * by_length

def classify_thumb(pts: Sequence[Point], feats: HandFeatures) -> str:
    s = feats.scale
    tip = pts[4]
    along = dot(sub(tip, pts[WRIST]), feats.hand_dir) / max(1e-6, s)
    side_axis = unit(sub(pts[5], pts[17]))
    sideways = dot(sub(tip, pts[9]), side_axis) / max(1e-6, s)
    palm = scale(add(add(pts[5], pts[17]), add(pts[WRIST], pts[9])), 0.25)
    to_palm = norm(sub(tip, palm)) / max(1e-6, s)
    curl = feats.curl["thumb"]
    if curl < 0.5 and along > 0.6 and along > sideways:
        return "up"
    if sideways > 0.5 and curl < 0.65:
        return "out"
    if curl > 0.5 and to_palm < 0.75:
        return "tucked"
    return "across"

def fingers_crossed(pts: Sequence[Point], scale_: float) -> bool:
    axis = unit(sub(pts[5], pts[17]))
    lateral = dot(sub(pts[8], pts[12]), axis) / max(1e-6, scale_)
    return lateral < -0.08

def nearest_base(pts: Sequence[Point], scale_: float) -> str:
    return min(
        LONG_FINGERS,
        key=lambda n: norm(sub(pts[4], pts[FINGER_JOINTS[n][1]])) / max(1e-6, scale_),
    )

def orientation(pts: Sequence[Point], handedness: str) -> Tuple[Point, Point, str, str]:
    hand_dir = unit(sub(pts[9], pts[WRIST]))
    v1 = sub(pts[5], pts[WRIST])
    v2 = sub(pts[17], pts[WRIST])
    n = unit(cross(v1, v2))
    if handedness == "Left":
        n = scale(n, -1.0)
    if n[2] < -0.55:
        facing = "camera"
    elif n[2] > 0.55:
        facing = "away"
    else:
        facing = "side"
    x, y, z = hand_dir
    if abs(z) > 0.75:
        fdir = "forward"
    elif abs(y) >= abs(x):
        fdir = "up" if y < 0 else "down"
    else:
        fdir = "right" if x > 0 else "left"
    return n, hand_dir, facing, fdir

def extract(landmarks: Sequence[Point], handedness: str = "Right") -> HandFeatures:
    pts = [tuple(map(float, p)) for p in landmarks]
    if len(pts) != 21:
        raise ValueError(f"expected 21 points, got {len(pts)}")
    s = hand_scale(pts)
    n, hand_dir, facing, fdir = orientation(pts, handedness)
    f = HandFeatures(handedness=handedness, scale=s, raw=pts)
    f.palm_dir, f.hand_dir, f.palm_facing, f.fingers_dir = n, hand_dir, facing, fdir
    for name in FINGERS:
        c = finger_curl(pts, name)
        f.curl[name] = c
        f.extended[name] = c < CURL_EXTENDED
        f.folded[name] = c > CURL_FOLDED
        f.half[name] = CURL_EXTENDED <= c <= CURL_FOLDED
    pairs = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"), ("thumb", "index"))
    for a, b in pairs:
        ja, jb = FINGER_JOINTS[a], FINGER_JOINTS[b]
        va = sub(pts[ja[3]], pts[ja[0]])
        vb = sub(pts[jb[3]], pts[jb[0]])
        f.spread[f"{a}_{b}"] = angle_between(va, vb)
        f.tip_dist[f"{a}_{b}"] = norm(sub(pts[ja[3]], pts[jb[3]])) / s
    for name in LONG_FINGERS:
        j = FINGER_JOINTS[name]
        f.thumb_to_tip[name] = norm(sub(pts[4], pts[j[3]])) / s
        f.thumb_to_pip[name] = norm(sub(pts[4], pts[j[1]])) / s
    f.tip_dist["index_ring"] = norm(sub(pts[8], pts[16])) / s
    f.tip_dist["index_pinky"] = norm(sub(pts[8], pts[20])) / s
    f.tip_dist["middle_pinky"] = norm(sub(pts[12], pts[20])) / s
    f.thumb_pose = classify_thumb(pts, f)
    f.thumb_under = nearest_base(pts, s)
    f.crossed = fingers_crossed(pts, s)
    return f