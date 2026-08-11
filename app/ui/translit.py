from typing import Dict, List

DIGRAPHS: Dict[str, str] = {
    "shch": "щ", "shh": "щ", "sch": "щ",
    "sh": "ш", "ch": "ч", "zh": "ж", "ts": "ц", "yu": "ю", "ya": "я",
    "yo": "ё", "ye": "э", "je": "э", "kh": "х", "ph": "ф",
}

SINGLES: Dict[str, str] = {
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "z": "з",
    "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "r": "р", "s": "с", "t": "т", "u": "у", "f": "ф", "h": "х",
    "c": "ц", "y": "ы", "w": "в", "q": "к", "x": "х",
    "'": "ь", '"': "ъ",
}

MAX_LEN = max(len(k) for k in DIGRAPHS)

def to_cyrillic(latin: str) -> str:
    src = latin.lower()
    out: List[str] = []
    i = 0
    while i < len(src):
        for size in range(MAX_LEN, 1, -1):
            chunk = src[i:i + size]
            if chunk in DIGRAPHS:
                out.append(DIGRAPHS[chunk])
                i += size
                break
        else:
            ch = src[i]
            out.append(SINGLES.get(ch, ch))
            i += 1
    return "".join(out)

def is_cyrillic(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)

def normalize(text: str) -> str:
    return to_cyrillic(text)