import math

import pytest

from app.alphabet.spec import EXT, FOLD, LetterSpec
from app.recognition.geometry import (
    FINGERS, LONG_FINGERS, angle_between, extract, joint_angle,
)
from app.synthesis.hand_model import (
    pose_from_spec, points_to_landmarks, pose_to_points,
)

def landmarks(**kwargs) -> list:
    spec = LetterSpec("X", kwargs.pop("fingers", {}), **kwargs)
    return points_to_landmarks(pose_to_points(pose_from_spec(spec)))

def test_joint_angle_straight_is_zero():
    assert joint_angle((0, 0, 0), (0, 1, 0), (0, 2, 0)) == pytest.approx(0.0)

def test_joint_angle_right_angle():
    assert joint_angle((0, 0, 0), (0, 1, 0), (1, 1, 0)) == pytest.approx(90.0)

def test_angle_between_ignores_length():
    assert angle_between((0, 3, 0), (4, 0, 0)) == pytest.approx(90.0)

def test_extended_fingers_have_low_curl():
    f = extract(landmarks(fingers={n: EXT for n in LONG_FINGERS}))
    for name in LONG_FINGERS:
        assert f.curl[name] < 0.3, name
        assert f.extended[name]

def test_folded_fingers_have_high_curl():
    f = extract(landmarks(fingers={n: FOLD for n in LONG_FINGERS}))
    for name in LONG_FINGERS:
        assert f.curl[name] > 0.7, name
        assert f.folded[name]
    assert f.is_fist()

def test_features_are_scale_invariant():
    pts = landmarks(fingers={n: EXT for n in LONG_FINGERS}, spread=True)
    small = extract(pts)
    big = extract([(x * 2, y * 2, z * 2) for x, y, z in pts])

    assert big.scale == pytest.approx(small.scale * 2)
    for name in FINGERS:
        assert big.curl[name] == pytest.approx(small.curl[name], abs=1e-6)
    for key, value in small.spread.items():
        assert big.spread[key] == pytest.approx(value, abs=1e-6)

def test_spread_flag_changes_measured_angles():
    open_ = extract(landmarks(fingers={n: EXT for n in LONG_FINGERS}, spread=True))
    closed = extract(landmarks(fingers={n: EXT for n in LONG_FINGERS}, spread=False))
    assert open_.spread["index_middle"] > 18.0
    assert closed.spread["index_middle"] < 8.0

@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_fingers_direction_matches_spec(direction):
    f = extract(landmarks(fingers={n: EXT for n in LONG_FINGERS},
                          fingers_dir=direction))
    assert f.fingers_dir == direction

@pytest.mark.parametrize("thumb", ["up", "out", "across", "tucked"])
def test_thumb_pose_round_trips(thumb):
    f = extract(landmarks(fingers={n: FOLD for n in LONG_FINGERS}, thumb=thumb))
    assert f.thumb_pose == thumb

def test_crossed_fingers_detected():
    crossed = extract(landmarks(fingers={"index": EXT, "middle": EXT}, crossed=True))
    normal = extract(landmarks(fingers={"index": EXT, "middle": EXT}, crossed=False))
    assert crossed.crossed
    assert not normal.crossed

def test_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        extract([(0.0, 0.0, 0.0)] * 20)

def test_rotation_does_not_change_curl():
    base = extract(landmarks(fingers={n: EXT for n in LONG_FINGERS}))
    a = math.radians(37.0)
    turned = extract([(x * math.cos(a) - y * math.sin(a),
                       x * math.sin(a) + y * math.cos(a), z)
                      for x, y, z in landmarks(fingers={n: EXT for n in LONG_FINGERS})])
    for name in LONG_FINGERS:
        assert turned.curl[name] == pytest.approx(base.curl[name], abs=1e-6)