import pytest

from app.alphabet.spec import M_NONE, best_matches, match
from app.languages import LANGUAGES, Language
from app.recognition.geometry import extract
from app.synthesis.hand_model import (
    points_to_landmarks, pose_from_spec, pose_to_points,
)

def features_for(spec):
    return extract(points_to_landmarks(pose_to_points(pose_from_spec(spec))), "Right")

def static_specs(lang: Language):
    return [s for s in lang.specs if s.motion == M_NONE]

def ids(lang: Language):
    return [f"{lang.code}-{s.letter}" for s in static_specs(lang)]

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_every_static_letter_is_recognized_from_its_own_pose(lang):
    specs = static_specs(lang)
    wrong = []
    for spec in specs:
        best = best_matches(features_for(spec), specs, top=1)[0]
        if best.letter != spec.letter:
            wrong.append(f"{spec.letter} -> {best.letter}")
    assert not wrong, "letters are indistinguishable: " + ", ".join(wrong)

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_self_match_is_confident(lang):
    for spec in static_specs(lang):
        score = match(features_for(spec), spec).score
        assert score > 0.75, f"{lang.code} {spec.letter}: {score:.2f}"

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_letters_are_not_duplicated(lang):
    seen = {}
    for spec in lang.specs:
        key = (tuple(sorted(spec.fingers.items())), spec.thumb, spec.spread,
               tuple(spec.touch), tuple(spec.apart), spec.thumb_under,
               spec.crossed, spec.fingers_dir, spec.palm_facing, spec.motion)
        assert key not in seen, f"{spec.letter} is described the same as {seen.get(key)}"
        seen[key] = spec.letter

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_alphabet_is_complete(lang):
    table = {s.letter for s in lang.specs}
    assert table == set(lang.letters)
    assert len(lang.specs) == len(lang.letters)

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_every_letter_has_hint(lang):
    for spec in lang.specs:
        assert spec.hint, spec.letter

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_motion_letters_share_shape_with_a_static_letter(lang):
    specs = static_specs(lang)
    for spec in (s for s in lang.specs if s.motion != M_NONE):
        top = best_matches(features_for(spec), specs, top=2)
        assert top[0].score > 0.7, f"{spec.letter}: shape does not match anything"