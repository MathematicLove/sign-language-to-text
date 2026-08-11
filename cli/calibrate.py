import argparse
from typing import List, Optional

import cv2

from app.alphabet.spec import (
    ANY, EXT, FOLD, HALF, LetterSpec, match,
)
from app.camera import Camera
from app.languages import BY_CODE, get
from app.recognition.geometry import FINGERS, HandFeatures, LONG_FINGERS
from app.recognition.tracker import HandTracker, TrackerUnavailable
from app.ui import keys, renderer as r

WINDOW = "calibrate"

_CONST = {EXT: "EXT", HALF: "HALF", FOLD: "FOLD", ANY: "ANY"}

def state_of(curl: float) -> str:
    if curl < 0.40:
        return EXT
    if curl > 0.62:
        return FOLD
    return HALF

def suggest(f: HandFeatures, letter: str) -> str:
    fingers = ", ".join(f'"{n}": {_CONST[state_of(f.curl[n])]}' for n in LONG_FINGERS)
    spread = sum(f.spread[k] for k in ("index_middle", "middle_ring", "ring_pinky")) / 3.0
    touch = [n for n in LONG_FINGERS if f.thumb_to_tip[n] < 0.38]
    parts = [f'thumb=T_{f.thumb_pose.upper()}']
    if any(f.extended[n] for n in LONG_FINGERS):
        parts.append(f"spread={spread > 16.0}")
    if touch:
        parts.append("touch=(" + ", ".join(f'"{n}"' for n in touch) + ",)")
    if f.crossed:
        parts.append("crossed=True")
    if f.thumb_pose == "tucked":
        parts.append(f'thumb_under="{f.thumb_under}"')
    if f.fingers_dir != "up":
        parts.append(f'fingers_dir="{f.fingers_dir}"')
    return (f'LetterSpec("{letter}", {{{fingers}}},\n'
            f'           {", ".join(parts)},\n'
            f'           hint="..."),')

def report(f: HandFeatures, spec: Optional[LetterSpec]) -> List[str]:
    lines = [f"{n:<7}{f.curl[n]:.2f}  {state_of(f.curl[n])}" for n in FINGERS]
    lines.append(f"thumb_pose {f.thumb_pose} (under {f.thumb_under})")
    lines.append(f"crossed {'yes' if f.crossed else 'no'}")
    lines.append(f"palm {f.palm_facing} / {f.fingers_dir}")
    lines.append("spread" + " ".join(
        f"{f.spread[k]:.0f}" for k in ("index_middle", "middle_ring", "ring_pinky")))
    lines.append("thumb-tip   " + " ".join(
        f"{f.thumb_to_tip[n]:.2f}" for n in LONG_FINGERS))
    if spec is not None:
        m = match(f, spec)
        lines.append("")
        lines.append(f"score {spec.letter}: {m.score:.2f}")
        lines.append("  " + "  ".join(f"{k}={v:.2f}" for k, v in m.parts.items()))
    return lines

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Calibrate a letter shape for your own hand.")
    p.add_argument("letter", help="letter you are showing")
    p.add_argument("--lang", choices=tuple(BY_CODE), default="asl")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args(argv)

    lang = get(args.lang)
    letter = lang.normalize(args.letter)[:1]
    spec = lang.spec_for(letter)
    if spec is None:
        print(f"letter '{letter}' is not in the {lang.name} alphabet; available: {lang.letters}")
        return 2

    try:
        tracker = HandTracker()
    except TrackerUnavailable as exc:
        print(exc)
        return 1
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    suggestion = ""
    with Camera(args.camera) as cam:
        if not cam.opened:
            print(cam.error)
            return 1
        while True:
            frame = cam.read()
            if frame is None:
                break
            hands = tracker.process(frame)
            lines = ["no hand visible"]
            if hands:
                obs = hands[0]
                r.draw_hand(frame, obs.pixels)
                lines = report(obs.features, spec)
                suggestion = suggest(obs.features, letter)

            r.panel(frame, 10, 10, 330, 26 * len(lines) + 60)
            with r.TextLayer(frame) as text:
                text.text(f"{lang.name} - {letter} - {spec.hint[:34]}", (24, 22), 17, r.ACCENT2)
                for i, line in enumerate(lines):
                    text.text(line, (24, 52 + i * 24), 17, r.FG)
                r.hotkeys(frame, [("S", "print LetterSpec"), ("Esc", "quit")],
                          layer=text)

            cv2.imshow(WINDOW, frame)
            key = keys.read(cv2.waitKeyEx(1))
            if key.is_("esc") or key.lower() == "q":
                break
            if key.lower() == "s" and suggestion:
                print(suggestion)

    tracker.close()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())