import pytest

from app.languages import ASL, BY_CODE, LANGUAGES, RSL, get
from app.ui import translit

@pytest.mark.parametrize("lang", LANGUAGES, ids=lambda l: l.code)
def test_sample_is_fully_spellable(lang):
    known, unknown = lang.spellable(lang.sample)
    assert not unknown
    assert known == lang.sample

def test_get_by_code_is_case_insensitive():
    assert get("ASL") is ASL
    assert get("rsl") is RSL

def test_get_rejects_unknown_language():
    with pytest.raises(SystemExit):
        get("klingon")

def test_codes_are_unique():
    assert len(BY_CODE) == len(LANGUAGES)

def test_asl_normalizes_case_only():
    assert ASL.normalize("hello") == "HELLO"

def test_rsl_accepts_latin_and_cyrillic():
    assert RSL.normalize("privet") == "ПРИВЕТ"
    assert RSL.normalize("Привет") == "ПРИВЕТ"

def test_spellable_reports_unknown_symbols():
    known, unknown = ASL.spellable("A1B")
    assert known == "AB" and unknown == ["1"]

def test_spellable_keeps_spaces():
    known, _ = ASL.spellable("a b")
    assert known == "A B"

def test_spec_lookup_is_case_insensitive():
    assert ASL.spec_for("a").letter == "A"
    assert RSL.spec_for("я").letter == "Я"

def test_unknown_letter_has_no_spec():
    assert ASL.spec_for("Ж") is None

class TestTranslit:
    @pytest.mark.parametrize("latin,cyrillic", [
        ("privet", "привет"),
        ("shar", "шар"),
        ("chas", "час"),
        ("zhuk", "жук"),
        ("yabloko", "яблоко"),
        ("shchi", "щи"),
        ("yozh", "ёж"),
    ])
    def test_digraphs_win_over_single_letters(self, latin, cyrillic):
        assert translit.to_cyrillic(latin) == cyrillic

    def test_unknown_symbols_pass_through(self):
        assert translit.to_cyrillic("da 1!") == "да 1!"

    def test_cyrillic_is_detected(self):
        assert translit.is_cyrillic("мир")
        assert not translit.is_cyrillic("mir")

    def test_normalize_leaves_cyrillic_alone(self):
        assert translit.normalize("мир") == "мир"