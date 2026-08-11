from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.recognition.geometry import FINGERS, LONG_FINGERS, HandFeatures

EXT = "ext"
HALF = "half"
FOLD = "fold"
ANY = "any"

FINGER_STATES = (EXT, HALF, FOLD, ANY)

T_UP = "up"
T_OUT = "out"
T_ACROSS = "across"
T_TUCKED = "tucked"
T_ANY = "any"

M_NONE = "none"
M_HOOK = "hook"
M_ZIGZAG = "zigzag"
M_DOWN = "down"
M_SHAKE = "shake"
M_CIRCLE = "circle"

_NEIGHBOURS = {"index": ("middle",), "middle": ("index", "ring"),
               "ring": ("middle", "pinky"), "pinky": ("ring",)}

@dataclass
class LetterSpec:
    letter: str
    fingers: Dict[str, str]
    thumb: str = T_ANY
    spread: Optional[bool] = None
    touch: Sequence[str] = ()
    apart: Sequence[str] = ()
    thumb_under: Optional[str] = None
    crossed: Optional[bool] = None
    palm_facing: Optional[str] = None
    fingers_dir: Optional[str] = None
    motion: str = M_NONE
    hint: str = ""

    def __post_init__(self) -> None:
        self.fingers = {f: self.fingers.get(f, ANY) for f in FINGERS}

def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def _state_score(curl: float, state: str) -> float:
    if state == ANY:
        return 1.0
    if state == EXT:
        return 1.0 - _ramp(curl, 0.28, 0.52)
    if state == FOLD:
        return _ramp(curl, 0.55, 0.78)
    if state == HALF:
        return max(0.0, 1.0 - abs(curl - 0.5) / 0.28)
    raise ValueError(f"unknown finger state: {state}")

def _spread_score(f: HandFeatures, spec: LetterSpec, want: bool) -> float:
    active = [n for n in LONG_FINGERS if spec.fingers[n] == EXT]
    pairs = [
        (a, b)
        for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))
        if a in active and b in active
    ]
    if not pairs:
        return 1.0
    scores = []
    for a, b in pairs:
        ang = f.spread[f"{a}_{b}"]
        scores.append(_ramp(ang, 8.0, 25.0) if want else 1.0 - _ramp(ang, 10.0, 28.0))
    return sum(scores) / len(scores)

def _touch_score(f: HandFeatures, spec: LetterSpec) -> float:
    scores = [1.0 - _ramp(f.thumb_to_tip[n], 0.30, 0.60) for n in spec.touch]
    scores += [_ramp(f.thumb_to_tip[n], 0.35, 0.65) for n in spec.apart]
    return sum(scores) / len(scores) if scores else 1.0

def _wants_crossed(spec: LetterSpec) -> Optional[bool]:
    if spec.crossed is not None:
        return spec.crossed
    if spec.fingers["index"] == EXT and spec.fingers["middle"] == EXT:
        return False
    return None

def _detail_score(f: HandFeatures, spec: LetterSpec) -> float:
    scores = []
    if spec.thumb_under is not None:
        if f.thumb_under == spec.thumb_under:
            scores.append(1.0)
        elif f.thumb_under in _NEIGHBOURS.get(spec.thumb_under, ()):
            scores.append(0.45)
        else:
            scores.append(0.0)
    want_crossed = _wants_crossed(spec)
    if want_crossed is not None:
        scores.append(1.0 if f.crossed == want_crossed else 0.0)
    return sum(scores) / len(scores) if scores else 1.0

def _thumb_score(f: HandFeatures, spec: LetterSpec) -> float:
    if spec.thumb == T_ANY:
        return 1.0
    if f.thumb_pose == spec.thumb:
        return 1.0
    near = {
        (T_UP, T_OUT), (T_OUT, T_UP),
        (T_ACROSS, T_TUCKED), (T_TUCKED, T_ACROSS),
    }
    return 0.45 if (f.thumb_pose, spec.thumb) in near else 0.0

def _orientation_score(f: HandFeatures, spec: LetterSpec) -> float:
    score, n = 0.0, 0
    if spec.palm_facing is not None:
        n += 1
        score += 1.0 if f.palm_facing == spec.palm_facing else (
            0.5 if "side" in (f.palm_facing, spec.palm_facing) else 0.0
        )
    if spec.fingers_dir is not None:
        n += 1
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        if f.fingers_dir == spec.fingers_dir:
            score += 1.0
        elif opposite.get(spec.fingers_dir) == f.fingers_dir:
            score += 0.0
        else:
            score += 0.5
    return score / n if n else 1.0

@dataclass
class Match:
    letter: str
    score: float
    parts: Dict[str, float] = field(default_factory=dict)
    hint: str = ""

def match(f: HandFeatures, spec: LetterSpec) -> Match:
    finger_scores = {n: _state_score(f.curl[n], spec.fingers[n]) for n in FINGERS}
    shape = sum(finger_scores[n] for n in LONG_FINGERS) / 4.0
    weakest = min(finger_scores[n] for n in LONG_FINGERS)
    parts = {
        "shape": shape,
        "thumb": _thumb_score(f, spec),
        "touch": _touch_score(f, spec),
        "spread": (_spread_score(f, spec, spec.spread)
                   if spec.spread is not None and not f.crossed else 1.0),
        "orient": _orientation_score(f, spec),
        "detail": _detail_score(f, spec),
    }
    weights = {"shape": 3.0, "thumb": 1.2, "touch": 1.5, "spread": 1.0,
               "orient": 1.3, "detail": 1.2}
    total = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
    total *= 0.35 + 0.65 * min(1.0, weakest / 0.45)
    return Match(letter=spec.letter, score=total, parts=parts, hint=spec.hint)

def best_matches(f: HandFeatures, specs: Sequence[LetterSpec], top: int = 3) -> List[Match]:
    out = [match(f, s) for s in specs if s.motion == M_NONE]
    out.sort(key=lambda m: m.score, reverse=True)
    return out[:top]