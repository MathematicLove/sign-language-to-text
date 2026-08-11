import time

from app.languages import ASL
from app.recognition.classifier import LetterClassifier, TextBuilder
from tests.test_roundtrip import features_for

def feed(clf: LetterClassifier, letter: str, frames: int):
    f = features_for(ASL.spec_for(letter))
    pred = None
    for _ in range(frames):
        pred = clf.update(f)
    return pred

def test_letter_needs_to_be_held():
    clf = LetterClassifier(ASL.specs, hold_frames=8)
    assert not feed(clf, "B", 3).stable
    assert feed(clf, "B", 5).stable

def test_progress_grows_while_holding():
    clf = LetterClassifier(ASL.specs, hold_frames=8)
    assert feed(clf, "B", 2).progress < feed(clf, "B", 4).progress

def test_lost_hand_resets_hypothesis():
    clf = LetterClassifier(ASL.specs, hold_frames=4)
    feed(clf, "B", 4)
    pred = clf.update(None)
    assert pred.letter is None and not pred.stable

def test_accept_emits_letter_once():
    clf = LetterClassifier(ASL.specs, hold_frames=4, cooldown_s=10.0)
    pred = feed(clf, "B", 6)
    assert clf.accept(pred) == "B"

    assert clf.accept(feed(clf, "B", 6)) is None

def test_cooldown_expires():
    clf = LetterClassifier(ASL.specs, hold_frames=3, cooldown_s=0.01)
    assert clf.accept(feed(clf, "B", 5)) == "B"
    time.sleep(0.02)
    assert clf.accept(feed(clf, "B", 5)) == "B"

def test_different_letters_pass_without_waiting():
    clf = LetterClassifier(ASL.specs, hold_frames=3, cooldown_s=10.0)
    assert clf.accept(feed(clf, "B", 4)) == "B"
    assert clf.accept(feed(clf, "V", 4)) == "V"

def test_reset_clears_state():
    clf = LetterClassifier(ASL.specs, hold_frames=3, cooldown_s=10.0)
    clf.accept(feed(clf, "B", 4))
    clf.reset()
    assert clf.accept(feed(clf, "B", 4)) == "B"

def test_alternatives_are_reported():
    clf = LetterClassifier(ASL.specs)
    pred = feed(clf, "V", 2)
    assert pred.alternatives and all(a.letter != pred.letter for a in pred.alternatives)

class TestTextBuilder:
    def test_push_and_backspace(self):
        t = TextBuilder()
        t.push("A")
        t.push("B")
        t.backspace()
        assert t.text == "A"

    def test_space_is_not_doubled(self):
        t = TextBuilder()
        t.push("A")
        t.space()
        t.space()
        assert t.text == "A "

    def test_leading_space_is_ignored(self):
        t = TextBuilder()
        t.space()
        assert t.text == ""

    def test_wrapped_splits_by_words(self):
        t = TextBuilder()
        t.text = "HELLO BIG WORLD"
        assert t.wrapped(width=9) == ["HELLO BIG", "WORLD"]

    def test_wrapped_keeps_last_lines_only(self):
        t = TextBuilder()
        t.text = " ".join(["WORD"] * 12)
        assert len(t.wrapped(width=8)) == 4

    def test_clear(self):
        t = TextBuilder()
        t.push("A")
        t.clear()
        assert t.text == ""