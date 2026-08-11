from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from app.alphabet import asl, rsl
from app.alphabet.spec import LetterSpec
from app.ui import translit

@dataclass(frozen=True)
class Language:
    code: str
    name: str
    to_text: str
    to_signs: str
    specs: Sequence[LetterSpec]
    letters: str
    sample: str
    cyrillic: bool = False
    unverified: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {s.letter: s for s in self.specs})

    def spec_for(self, letter: str) -> Optional[LetterSpec]:
        return self._index.get(letter.upper())

    def normalize(self, text: str) -> str:
        text = translit.normalize(text) if self.cyrillic else text
        return text.upper()

    def spellable(self, text: str) -> Tuple[str, List[str]]:
        known, unknown = [], []
        for ch in self.normalize(text):
            if ch.isspace():
                known.append(" ")
            elif self.spec_for(ch) is not None:
                known.append(ch)
            else:
                unknown.append(ch)
        return "".join(known), unknown

ASL = Language(
    code="asl",
    name="ASL",
    to_text="ASL -> English text",
    to_signs="English text -> ASL",
    specs=asl.ASL,
    letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    sample="HELLO WORLD",
)

RSL = Language(
    code="rsl",
    name="RSL",
    to_text="RSL -> Russian text",
    to_signs="Russian text -> RSL",
    specs=rsl.RSL,
    letters="АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    sample="ПРИВЕТ МИР",
    cyrillic=True,
    unverified=frozenset(rsl.UNVERIFIED),
)

LANGUAGES: Tuple[Language, ...] = (ASL, RSL)
BY_CODE: Dict[str, Language] = {lang.code: lang for lang in LANGUAGES}

def get(code: str) -> Language:
    try:
        return BY_CODE[code.lower()]
    except KeyError:
        raise SystemExit(f"unknown language: {code}. Available: {', '.join(BY_CODE)}")

__all__ = ["Language", "ASL", "RSL", "LANGUAGES", "BY_CODE", "get"]