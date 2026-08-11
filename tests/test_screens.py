import numpy as np
import pytest

from app.camera import Camera
from app.languages import ASL, RSL
from app.main import App
from app.modes.base import Go
from app.modes.menu import MenuScreen
from app.modes.practice import PracticeScreen
from app.modes.recognize import RecognizeScreen
from app.modes.spell import SpellScreen
from app.ui import keys, renderer as r

SIZE = (1180, 760)

@pytest.fixture
def canvas():
    return r.new_canvas(*SIZE)

def press(screen, code_or_char):
    table = {"esc": 27, "enter": 13, "tab": 9, "space": 32, "backspace": 8,
             "up": 65362, "down": 65364, "left": 65361, "right": 65363}
    code = table.get(code_or_char, ord(code_or_char[0]) if code_or_char else -1)
    return screen.on_key(keys.read(code))

class TestKeys:
    def test_printable_char(self):
        assert keys.read(ord("a")).char == "a"

    def test_named_keys(self):
        assert keys.read(27).is_("esc")
        assert keys.read(13).is_("enter")
        assert keys.read(8).is_("backspace")

    def test_space_has_both_name_and_char(self):
        key = keys.read(32)
        assert key.is_("space") and key.char == " "

    @pytest.mark.parametrize("code,name", [
        (65361, "left"), (2424832, "left"), (63234, "left"),
        (65364, "down"), (2621440, "down"), (63233, "down"),
    ])
    def test_arrows_on_every_platform(self, code, name):
        assert keys.read(code).is_(name)

    def test_no_key(self):
        assert not keys.read(-1).pressed

    def test_unknown_code_is_ignored(self):
        key = keys.read(0xF0000)
        assert key.pressed and not key.char and not key.name

class TestMenu:
    def test_draws(self, canvas):
        MenuScreen().draw(canvas)
        assert canvas.any()

    def test_arrows_move_selection(self):
        menu = MenuScreen()
        press(menu, "down")
        assert menu.i == 1
        press(menu, "up")
        assert menu.i == 0

    def test_selection_wraps(self):
        menu = MenuScreen()
        press(menu, "up")
        assert menu.i == len(menu.items) - 1

    def test_enter_opens_selected_mode(self):
        menu = MenuScreen()
        go = press(menu, "enter")
        assert go.screen == "recognize" and go.params["code"] == "asl"

    def test_digit_opens_mode_directly(self):
        go = press(MenuScreen(), "2")
        assert go.screen == "recognize" and go.params["code"] == "rsl"

    def test_escape_quits(self):
        assert press(MenuScreen(), "esc").screen == "quit"

    def test_preview_follows_selected_language(self):
        menu = MenuScreen()
        press(menu, "down")
        press(menu, "down")
        menu.update(0.5)
        assert menu.language is ASL

class TestSpell:
    def test_typing_builds_text(self):
        screen = SpellScreen(ASL, "")
        for ch in "cab":
            press(screen, ch)
        assert screen.shown.endswith("CAB")

    def test_backspace_removes_last_char(self):
        screen = SpellScreen(ASL, "AB")
        press(screen, "backspace")
        assert screen.shown == "A"

    def test_cyrillic_is_typed_in_translit(self):
        screen = SpellScreen(RSL, "")
        for ch in "privet":
            press(screen, ch)
        assert screen.shown.endswith("ПРИВЕТ")

    def test_enter_starts_playback(self):
        screen = SpellScreen(ASL, "AB")
        press(screen, "enter")
        assert not screen.editing
        screen.update(0.2)
        assert screen.animator.progress == (1, 2)

    def test_escape_returns_to_editing_then_to_menu(self):
        screen = SpellScreen(ASL, "AB")
        press(screen, "enter")
        assert press(screen, "esc") is None and screen.editing
        assert press(screen, "esc").screen == "menu"

    def test_space_pauses_only_during_playback(self):
        screen = SpellScreen(ASL, "AB")
        press(screen, "space")
        assert screen.shown.endswith(" ")
        press(screen, "enter")
        press(screen, "space")
        assert screen.animator.paused

    def test_tab_switches_to_camera(self):
        assert press(SpellScreen(ASL, "A"), "tab").screen == "recognize"

    def test_draws_in_both_states(self, canvas):
        screen = SpellScreen(RSL, "ПРИВЕТ")
        screen.draw(canvas)
        press(screen, "enter")
        screen.update(0.3)
        screen.draw(canvas)
        assert canvas.any()

    def test_unknown_symbols_are_reported(self, canvas):
        screen = SpellScreen(ASL, "A@B")
        screen.draw(canvas)
        assert screen.animator.unknown == ["@"]

class TestRecognize:
    def screen(self):
        return RecognizeScreen(ASL, Camera(0))

    def test_draws_without_camera(self, canvas):
        self.screen().draw(canvas)
        assert canvas.any()

    def test_text_editing_keys(self):
        screen = self.screen()
        screen.text.push("A")
        press(screen, "space")
        screen.text.push("B")
        assert screen.text.text == "A B"
        press(screen, "backspace")
        assert screen.text.text == "A "
        press(screen, "c")
        assert screen.text.text == ""

    def test_r_switches_to_reverse_direction_with_text(self):
        screen = self.screen()
        screen.text.push("H")
        go = press(screen, "r")
        assert go == Go("spell", {"code": "asl", "text": "H"})

    def test_escape_returns_to_menu(self):
        assert press(self.screen(), "esc").screen == "menu"

class TestPractice:
    def screen(self):
        return PracticeScreen(RSL, Camera(0))

    def test_starts_with_a_target_letter(self):
        screen = self.screen()
        assert screen._target in RSL.letters

    def test_space_asks_another_letter(self):
        screen = self.screen()
        first = screen._target
        press(screen, "space")
        assert screen._target != first
        assert screen.stats.skipped == 1

    def test_queue_covers_the_whole_alphabet(self):
        screen = self.screen()
        seen = {screen._target}
        for _ in range(len(RSL.letters) - 1):
            press(screen, "space")
            seen.add(screen._target)
        assert seen == set(RSL.letters)

    def test_draws_without_camera(self, canvas):
        screen = self.screen()
        screen.update(0.1)
        screen.draw(canvas)
        assert canvas.any()

class TestRouting:
    def test_builds_every_screen(self):
        app = App()
        assert isinstance(app.build(Go("menu")), MenuScreen)
        assert isinstance(app.build(Go("recognize", {"code": "asl"})), RecognizeScreen)
        assert isinstance(app.build(Go("spell", {"code": "rsl"})), SpellScreen)
        assert isinstance(app.build(Go("practice", {"code": "asl"})), PracticeScreen)
        assert app.build(Go("quit")) is None

    def test_unknown_screen_is_an_error(self):
        with pytest.raises(ValueError):
            App().build(Go("nowhere"))

    def test_screens_share_one_camera(self):
        app = App()
        first = app.build(Go("recognize", {"code": "asl"}))
        second = app.build(Go("practice", {"code": "asl"}))
        assert first.camera is second.camera is app.camera

    def test_cli_arguments(self):
        from app.main import parse_args
        args = parse_args(["--mode", "spell", "--lang", "rsl", "--text", "da"])
        assert (args.mode, args.lang, args.text) == ("spell", "rsl", "da")

class TestRenderer:
    def test_text_layer_only_touches_drawn_area(self, canvas):
        before = canvas.copy()
        with r.TextLayer(canvas) as text:
            text.text("HELLO", (40, 40), 20, r.FG)
        changed = np.argwhere((canvas != before).any(axis=2))
        assert len(changed) > 0
        assert changed[:, 0].max() < 120 and changed[:, 1].max() < 300

    def test_text_outside_canvas_does_not_crash(self, canvas):
        r.draw_text(canvas, "edge", (-500, -500), 20)
        r.draw_text(canvas, "edge", (5000, 5000), 20)

    def test_panel_clips_to_canvas(self, canvas):
        r.panel(canvas, -100, -100, 50, 50)
        r.panel(canvas, 0, 0, 10_000, 10_000)

    def test_progress_bar_clamps_value(self, canvas):
        r.progress_bar(canvas, 10, 10, 100, 8, 5.0)
        r.progress_bar(canvas, 10, 30, 100, 8, -1.0)

    def test_fit_hand_keeps_points_inside_box(self):
        pixels = [(x, y) for x in range(0, 100, 20) for y in range(0, 100, 25)]
        fitted = r.fit_hand(pixels, (10, 10, 200, 100), margin=10)
        assert all(10 <= x <= 210 and 10 <= y <= 110 for x, y in fitted)

class TestMainLoop:

    def run_with_keys(self, monkeypatch, presses):
        import cv2

        from app import main as main_module

        queue = list(presses) + [27] * 10
        shown = []
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.append(img.shape))
        monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: queue.pop(0) if queue else 27)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
        app = main_module.App(width=640, height=480)
        assert app.run() == 0
        return app, shown

    def test_menu_to_spell_and_back_to_quit(self, monkeypatch):

        app, shown = self.run_with_keys(monkeypatch, [ord("2"), ord("r"), 27, 27, 27])
        assert shown and shown[0] == (480, 640, 3)
        assert not app._running

    def test_practice_screen_runs(self, monkeypatch):
        app, shown = self.run_with_keys(monkeypatch, [ord("4"), 32, 32, 27, 27])
        assert len(shown) >= 5

    def test_closing_window_stops_the_loop(self, monkeypatch):
        import cv2

        from app import main as main_module

        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(cv2, "imshow", lambda *a: None)
        monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: -1)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 0.0)
        assert main_module.App(width=320, height=240).run() == 0