import pytest

from app.alphabet.spec import M_NONE
from app.languages import ASL, RSL
from app.synthesis.animator import HOLD_S, TRANSITION_S, Animator
from app.synthesis.hand_model import blend, pose_from_spec, pose_to_points

def play(animator: Animator, seconds: float, step: float = 1 / 60):
    seen = []
    for _ in range(int(seconds / step)):
        frame = animator.update(step)
        if frame.letter and (not seen or seen[-1] != frame.letter):
            seen.append(frame.letter)
        if frame.finished:
            break
    return seen

def test_empty_text_finishes_immediately():
    a = Animator(ASL.spec_for)
    a.set_text("")
    assert a.empty and a.update(0.1).finished

def test_plays_letters_in_order():
    a = Animator(ASL.spec_for)
    a.set_text("CAB")
    assert play(a, 10.0) == ["C", "A", "B"]

def test_unknown_symbols_are_collected_not_shown():
    a = Animator(ASL.spec_for)
    a.set_text("A1B!")
    assert a.unknown == ["1", "!"]
    assert play(a, 10.0) == ["A", "B"]

def test_space_is_a_pause_step():
    a = Animator(ASL.spec_for)
    a.set_text("A B")
    assert play(a, 10.0) == ["A", " ", "B"]

def test_index_points_into_source_text():
    a = Animator(ASL.spec_for)
    a.set_text("AB")
    indexes = {a.update(1 / 60).index for _ in range(120)}
    assert indexes <= {0, 1, -1}

def test_progress_counts_steps():
    a = Animator(ASL.spec_for)
    a.set_text("ABC")
    assert a.progress == (1, 3)
    play(a, TRANSITION_S + HOLD_S + 0.05)
    assert a.progress[0] >= 2

def test_loop_restarts():
    a = Animator(ASL.spec_for)
    a.loop = True
    a.set_text("A")
    letters = play(a, 4.0)
    assert not a.update(0.0).finished
    assert letters == ["A"]

def test_pause_freezes_progress():
    a = Animator(ASL.spec_for)
    a.set_text("ABC")
    a.toggle_pause()
    play(a, 3.0)
    assert a.progress == (1, 3)

def test_speed_is_clamped():
    a = Animator(ASL.spec_for)
    for _ in range(40):
        a.faster()
    assert a.speed <= 3.0
    for _ in range(40):
        a.slower()
    assert a.speed >= 0.25

def test_step_controls_move_between_letters():
    a = Animator(ASL.spec_for)
    a.set_text("ABC")
    a.step_forward()
    a.step_forward()
    assert a.progress[0] == 3
    a.step_back()
    assert a.progress[0] == 2
    a.step_back()
    a.step_back()
    assert a.progress[0] == 1

def test_motion_letters_leave_a_trail():
    a = Animator(ASL.spec_for)
    a.set_text("Z")
    trails = [len(a.update(1 / 60).trail) for _ in range(60)]
    assert max(trails) > 1

def test_static_letters_have_no_trail():
    a = Animator(ASL.spec_for)
    a.set_text("B")
    assert all(not a.update(1 / 60).trail for _ in range(40))

def test_cyrillic_text_plays():
    a = Animator(RSL.spec_for)
    a.set_text("ДА")
    assert play(a, 10.0) == ["Д", "А"]

class TestPose:
    def test_blend_endpoints(self):
        a = pose_from_spec(ASL.spec_for("A"))
        b = pose_from_spec(ASL.spec_for("B"))
        assert blend(a, b, 0.0).fingers["index"] == pytest.approx(a.fingers["index"])
        assert blend(a, b, 1.0).fingers["index"] == pytest.approx(b.fingers["index"])

    def test_blend_takes_short_way_around(self):
        a = pose_from_spec(ASL.spec_for("G"))
        b = pose_from_spec(ASL.spec_for("P"))
        mid = blend(a, b, 0.5).rotation
        assert min(a.rotation, b.rotation) - 1 <= mid <= max(a.rotation, b.rotation) + 1

    def test_pose_has_21_points(self):
        for spec in ASL.specs:
            assert len(pose_to_points(pose_from_spec(spec))) == 21

    def test_static_letters_do_not_move(self):
        from app.synthesis.animator import _motion_offset
        assert _motion_offset(M_NONE, 0.5) == (0.0, 0.0)