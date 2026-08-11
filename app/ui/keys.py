from dataclasses import dataclass

_ARROWS = {
    65361: "left", 65362: "up", 65363: "right", 65364: "down",
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",
    63232: "up", 63233: "down", 63234: "left", 63235: "right",
}

_NAMED = {
    27: "esc", 13: "enter", 10: "enter", 9: "tab",
    8: "backspace", 127: "backspace", 32: "space",
}

NO_KEY = -1

@dataclass(frozen=True)
class Key:
    code: int
    name: str = ""
    char: str = ""

    @property
    def pressed(self) -> bool:
        return self.code != NO_KEY

    def is_(self, *names: str) -> bool:
        return self.name in names

    def lower(self) -> str:
        return self.char.lower()

NONE = Key(NO_KEY)

def read(code: int) -> Key:
    if code == NO_KEY:
        return NONE
    if code in _ARROWS:
        return Key(code, _ARROWS[code])
    if code in _NAMED:
        name = _NAMED[code]
        return Key(code, name, " " if name == "space" else "")
    if 32 <= code < 127:
        return Key(code, "", chr(code))
    low = code & 0xFF
    if 32 <= low < 127 and code > 255:
        return Key(code, "", chr(low))
    return Key(code)