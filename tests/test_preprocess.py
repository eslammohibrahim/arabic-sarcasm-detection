import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import normalize_arabic, normalize_with_signals  # noqa: E402


def test_removes_diacritics():
    assert normalize_arabic("جَمِيلٌ") == "جميل"


def test_removes_tatweel():
    assert normalize_arabic("رائعــــة") == normalize_arabic("رائعة")


def test_normalizes_alef_forms():
    assert normalize_arabic("أحمد إسلام آمن") == "احمد اسلام امن"


def test_normalizes_ya_and_ta_marbuta():
    assert normalize_arabic("مستشفى جميلة") == "مستشفي جميله"


def test_collapses_repeated_letters():
    assert normalize_arabic("حلوووووو جداااا") == "حلو جدا"


def test_removes_urls_mentions_and_punctuation():
    assert normalize_arabic("ممتاز!!! @user https://t.co/x") == "ممتاز"


def test_converts_arabic_digits():
    assert normalize_arabic("٥ نجوم") == "5 نجوم"


def test_same_word_written_differently_matches():
    variants = ["رائعة", "رآئعه", "رائعــة", "رَائِعَة"]
    assert len({normalize_arabic(v) for v in variants}) == 1


def test_signals_keep_emojis_and_punctuation():
    out = normalize_with_signals("حلو اوي يا عم!!! 😂")
    assert out.startswith("حلو اوي يا عم")
    assert "emo1f602" in out and "xmanyexclaim" in out and "xlaugh" in out


def test_signals_without_any_signal_equal_plain_normalization():
    assert normalize_with_signals("الجو حلو النهارده") == normalize_arabic("الجو حلو النهارده")


def test_signals_drop_retweet_prefix():
    assert not normalize_with_signals("RT @someone كلام عادي").startswith("rt")
