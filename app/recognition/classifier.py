import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

from app.alphabet.spec import M_NONE, LetterSpec, Match, match
from app.recognition.geometry import HandFeatures
from app.recognition.motion import MotionDetector

@dataclass
class Prediction:
    letter: Optional[str] = None
    score: float = 0.0
    hint: str = ""
    alternatives: List[Match] = field(default_factory=list)
    stable: bool = False
    progress: float = 0.0
    motion: str = M_NONE
    locked: bool = False
    released: bool = False

class LetterClassifier:

    def __init__(
        self,
        specs: Sequence[LetterSpec],
        min_score: float = 0.62,
        margin: float = 0.05,
        hold_frames: int = 8,
        cooldown_s: float = 0.7,
        release_frames: int = 3,
    ) -> None:
        self.specs = list(specs)
        self.min_score = min_score
        self.margin = margin
        self.hold_frames = hold_frames
        self.cooldown_s = cooldown_s
        self.release_frames = release_frames
        self._history: Deque[Optional[str]] = deque(maxlen=hold_frames)
        self._motion = MotionDetector()
        self._last_emitted: Optional[str] = None
        self._last_emit_time = 0.0
        self._armed = True
        self._release_run = 0

    @property
    def armed(self) -> bool:
        return self._armed

    def _observe_release(self, released: bool) -> None:
        if not released:
            self._release_run = 0
            return
        self._release_run += 1
        if self._release_run >= self.release_frames and not self._armed:
            self._armed = True
            self._history.clear()

    def update(self, f: Optional[HandFeatures]) -> Prediction:
        if f is None:
            self._observe_release(True)
            self._history.append(None)
            self._motion.reset()
            return Prediction(locked=not self._armed, released=True)
        released = f.is_open_palm()
        self._observe_release(released)
        self._motion.update(f)
        motion, motion_conf = self._motion.classify()
        hypotheses = [match(f, s) for s in self.specs if s.motion == M_NONE]
        if motion != M_NONE:
            for s in self.specs:
                if s.motion == motion:
                    m = match(f, s)
                    m.score = min(1.0, m.score * (1.0 + 0.25 * motion_conf))
                    hypotheses.append(m)
        hypotheses.sort(key=lambda m: m.score, reverse=True)
        top = hypotheses[0] if hypotheses else None
        runner_up = hypotheses[1].score if len(hypotheses) > 1 else 0.0
        confident = (
            top is not None
            and top.score >= self.min_score
            and (top.score - runner_up) >= self.margin
        )
        self._history.append(top.letter if confident and not released else None)
        held = Counter(x for x in self._history if x is not None)
        pred = Prediction(
            letter=top.letter if top else None,
            score=top.score if top else 0.0,
            hint=top.hint if top else "",
            alternatives=hypotheses[1:4],
            motion=motion,
            locked=not self._armed,
            released=released,
        )
        if top is not None:
            pred.progress = held.get(top.letter, 0) / self.hold_frames
            pred.stable = held.get(top.letter, 0) >= self.hold_frames
        return pred

    def accept(self, pred: Prediction) -> Optional[str]:
        if not self._armed or not pred.stable or pred.letter is None:
            return None
        now = time.monotonic()
        if (now - self._last_emit_time) < self.cooldown_s:
            return None
        self._armed = False
        self._release_run = 0
        self._last_emitted = pred.letter
        self._last_emit_time = now
        self._history.clear()
        return pred.letter

    def reset(self) -> None:
        self._history.clear()
        self._motion.reset()
        self._last_emitted = None
        self._last_emit_time = 0.0
        self._armed = True
        self._release_run = 0

class TextBuilder:

    def __init__(self) -> None:
        self.text = ""

    def push(self, letter: str) -> None:
        self.text += letter

    def space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""

    def wrapped(self, width: int) -> List[str]:
        lines: List[str] = []
        line = ""
        for word in self.text.split(" "):
            candidate = f"{line} {word}".strip()
            if len(candidate) > width and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        return lines[-4:]