"""Arabic text normalization.

Arabic text has many ways to write the same word: optional diacritics,
different alef forms, stretched letters, and repeated letters for emphasis.
Normalizing these makes "رائعــــة", "رائعة" and "رآئعه" look the same to the model.
"""

import re

DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")  # tashkeel + dagger alef
TATWEEL = "ـ"  # the stretching character: جميــــل
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
REPEATED = re.compile(r"(.)\1{2,}")  # 3+ of the same character: جداااا
NOT_ALLOWED = re.compile(r"[^ء-غف-يa-z0-9\s]")  # keep Arabic letters, latin, digits
SPACES = re.compile(r"\s+")

ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
YA_TA = str.maketrans({"ى": "ي", "ة": "ه"})
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(text: str) -> str:
    """Return a normalized version of `text` for bag-of-words models."""
    text = str(text).lower()
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = text.translate(ALEF).translate(YA_TA).translate(ARABIC_DIGITS)
    text = REPEATED.sub(r"\1", text)
    text = NOT_ALLOWED.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def basic_clean(text: str) -> str:
    """Minimal cleaning only (used as the 'no normalization' baseline)."""
    text = URL.sub(" ", str(text).lower())
    return SPACES.sub(" ", text).strip()


# Sarcasm lives partly outside the words: "!!!!", "؟؟", 😂 🙄 🌚. normalize_arabic() throws all of
# that away, so this version turns them into word-like tokens the model can count.
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
RT = re.compile(r"^rt\s+")


def _signal_tokens(text: str) -> str:
    out = []
    for e in EMOJI.findall(text):
        out.append(f"emo{ord(e):x}")
    if re.search(r"!{2,}", text):
        out.append("xmanyexclaim")
    elif "!" in text:
        out.append("xexclaim")
    if re.search(r"[?؟]{2,}", text):
        out.append("xmanyquestion")
    if REPEATED.search(re.sub(r"[!?؟.\s]", "", text)):
        out.append("xstretched")  # حلووووو, ههههه
    if re.search(r"ه{3,}|ha{2,}|😂|🤣", text):
        out.append("xlaugh")
    return " ".join(out)


def normalize_with_signals(text: str) -> str:
    """normalize_arabic() plus tokens for emojis, !!!, ??? and stretched letters."""
    raw = RT.sub("", str(text).lower())
    return (normalize_arabic(raw) + " " + _signal_tokens(raw)).strip()
