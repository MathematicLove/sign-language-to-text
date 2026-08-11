import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from app.alphabet.spec import (
    M_CIRCLE, M_DOWN, M_HOOK, M_NONE, M_SHAKE, M_ZIGZAG,
    ANY, LetterSpec,
)
from app.synthesis.hand_model import HandPose, Vec2, blend, ease, pose_from_spec

NEUTRAL_SPEC = LetterSpec("", {f: ANY for f in ("index", "middle", "ring", "pinky")},
                          hint="pause")

TRANSITION_S = 0.26
HOLD_S = 0.5
SPACE_S = 0.4

def _motion_offset(kind: str, phase: float) -> Vec2:
    p = phase
    if kind == M_ZIGZAG:
        seg = min(2, int(p * 3))
        local = p * 3 - seg
        if seg == 0:
            return (-0.25 + 0.5 * local, 0.3)
        if seg == 1:
            return (0.25 - 0.5 * local, 0.3 - 0.6 * local)
        return (-0.25 + 0.5 * local, -0.3)
    if kind == M_HOOK:
        if p < 0.6:
            return (0.15, 0.25 - 0.7 * (p / 0.6))
        local = (p - 0.6) / 0.4
        return (0.15 - 0.45 * local, -0.45 + 0.12 * local)
    if kind == M_DOWN:
        return (0.0, 0.25 - 0.55 * ease(p))
    if kind == M_SHAKE:
        return (0.16 * math.sin(p * math.pi * 4.0), 0.0)
    if kind == M_CIRCLE:
        a = p * math.tau
        return (0.22 * math.sin(a), 0.22 * (math.cos(a) - 1.0))
    return (0.0, 0.0)

@dataclass
class Frame:
    pose: HandPose
    offset: Vec2 = (0.0, 0.0)
    letter: str = ""
    hint: str = ""
    index: int = -1
    trail: List[Vec2] = field(default_factory=list)
    finished: bool = False

@dataclass
class _Step:
    letter: str
    pose: HandPose
    motion: str
    hint: str
    index: int
    hold: float

class Animator:

    def __init__(self, spec_lookup: Callable[[str], Optional[LetterSpec]],
                 speed: float = 1.0) -> None:
        self._lookup = spec_lookup
        self.speed = speed
        self.loop = False
        self.paused = False
        self._steps: List[_Step] = []
        self._i = 0
        self._t = 0.0
        self._neutral = pose_from_spec(NEUTRAL_SPEC)
        self._prev_pose = self._neutral
        self._trail: List[Vec2] = []
        self.unknown: List[str] = []

    def set_text(self, text: str) -> None:
        self._steps.clear()
        self.unknown.clear()
        for i, ch in enumerate(text.upper()):
            if ch.isspace():
                self._steps.append(_Step(" ", self._neutral, M_NONE, "space", i, SPACE_S))
                continue
            spec = self._lookup(ch)
            if spec is None:
                self.unknown.append(ch)
                continue
            hold = HOLD_S * (1.6 if spec.motion != M_NONE else 1.0)
            self._steps.append(
                _Step(ch, pose_from_spec(spec), spec.motion, spec.hint, i, hold)
            )
        self.restart()

    def restart(self) -> None:
        self._i = 0
        self._t = 0.0
        self._prev_pose = self._neutral
        self._trail.clear()

    @property
    def empty(self) -> bool:
        return not self._steps

    @property
    def progress(self) -> Tuple[int, int]:
        return (min(self._i + 1, len(self._steps)), len(self._steps))

    def update(self, dt: float) -> Frame:
        if not self._steps:
            return Frame(pose=self._neutral, finished=True)
        if self._i >= len(self._steps):
            return Frame(pose=self._neutral, letter="", index=-1, finished=True)
        if not self.paused:
            self._t += dt * max(0.1, self.speed)
        step = self._steps[self._i]
        total = TRANSITION_S + step.hold
        if self._t >= total:
            self._prev_pose = step.pose
            self._t = 0.0
            self._i += 1
            self._trail.clear()
            if self._i >= len(self._steps):
                if self.loop:
                    self.restart()
                else:
                    return Frame(pose=step.pose, letter=step.letter,
                                 index=step.index, hint=step.hint, finished=True)
            step = self._steps[min(self._i, len(self._steps) - 1)]
            total = TRANSITION_S + step.hold
        if self._t < TRANSITION_S:
            pose = blend(self._prev_pose, step.pose, self._t / TRANSITION_S)
            offset: Vec2 = (0.0, 0.0)
        else:
            pose = step.pose
            phase = (self._t - TRANSITION_S) / max(1e-6, step.hold)
            offset = _motion_offset(step.motion, min(1.0, phase))
            if step.motion != M_NONE:
                self._trail.append(offset)
                if len(self._trail) > 40:
                    self._trail.pop(0)
        return Frame(pose=pose, offset=offset, letter=step.letter,
                     hint=step.hint, index=step.index, trail=list(self._trail))

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def faster(self) -> None:
        self.speed = min(3.0, self.speed + 0.25)

    def slower(self) -> None:
        self.speed = max(0.25, self.speed - 0.25)

    def step_back(self) -> None:
        self._i = max(0, self._i - 1)
        self._t = 0.0
        self._trail.clear()

    def step_forward(self) -> None:
        self._i = min(len(self._steps), self._i + 1)
        self._t = 0.0
        self._trail.clear()