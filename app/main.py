import argparse
import time
from typing import Optional

import cv2

from app.camera import Camera
from app.languages import BY_CODE, get
from app.modes.base import Go, Screen
from app.modes.menu import MenuScreen
from app.modes.practice import PracticeScreen
from app.modes.recognize import RecognizeScreen
from app.modes.spell import SpellScreen
from app.ui import keys, renderer as r

WINDOW = "Sign Language <-> Text"
TARGET_FPS = 60

class App:

    def __init__(self, camera_index: int = 0, width: int = 1180, height: int = 760) -> None:
        self.width = width
        self.height = height
        self.camera = Camera(camera_index)
        self.canvas = r.new_canvas(width, height)
        self.screen: Screen = MenuScreen()
        self._running = True

    def build(self, go: Go) -> Optional[Screen]:
        params = go.params
        if go.screen == "quit":
            return None
        if go.screen == "menu":
            return MenuScreen(selected=params.get("selected", 0))
        lang = get(params.get("code", "asl"))
        if go.screen == "recognize":
            return RecognizeScreen(lang, self.camera)
        if go.screen == "spell":
            return SpellScreen(lang, params.get("text", ""))
        if go.screen == "practice":
            return PracticeScreen(lang, self.camera)
        raise ValueError(f"unknown screen: {go.screen}")

    def goto(self, go: Go) -> None:
        self.screen.leave()
        nxt = self.build(go)
        if nxt is None:
            self._running = False
            return
        self.screen = nxt
        self.screen.enter()

    def run(self, start: Optional[Go] = None) -> int:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        if start is not None:
            self.goto(start)
        else:
            self.screen.enter()

        last = time.monotonic()
        try:
            while self._running:
                now = time.monotonic()
                dt, last = now - last, now

                go = self.screen.update(min(dt, 0.1))
                self.screen.draw(self.canvas)
                cv2.imshow(WINDOW, self.canvas)

                spent = time.monotonic() - now
                delay = max(1, int((1.0 / TARGET_FPS - spent) * 1000))
                key = keys.read(cv2.waitKeyEx(delay))
                if key.pressed:
                    go = self.screen.on_key(key) or go
                if self._closed():
                    break
                if go is not None:
                    self.goto(go)
        finally:
            self.screen.leave()
            self.camera.release()
            cv2.destroyAllWindows()
        return 0

    def _closed(self) -> bool:
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sign-language-to-text",
        description="Fingerspelling (ASL / RSL) to text and back.",
    )
    p.add_argument("--mode", choices=("menu", "recognize", "spell", "practice"),
                   default="menu", help="screen to start from")
    p.add_argument("--lang", choices=tuple(BY_CODE), default="asl",
                   help="sign language")
    p.add_argument("--text", default="", help="text for the spell mode")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=1180, help="window width")
    p.add_argument("--height", type=int, default=760, help="window height")
    return p.parse_args(argv)

def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    app = App(camera_index=args.camera, width=args.width, height=args.height)
    start = None
    if args.mode != "menu":
        start = Go(args.mode, {"code": args.lang, "text": args.text})
    return app.run(start)

if __name__ == "__main__":
    raise SystemExit(main())