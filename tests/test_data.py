import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402


def test_clean_drops_empty_conflicting_and_duplicate_tweets():
    raw = pd.DataFrame({"tweet": ["يا سلام على النظام", "يا سلام على النظام", "@someone يا سلام على النظام",
                                  "https://t.co/x", "كلام عادي جدا", "كلام عادي جدا!"],
                        "sarcasm": [1, 1, 1, 0, 0, 1], "dialect": [0] * 6})
    df = data.clean(raw)
    # duplicates and the mention-only variant collapse into one; the link-only tweet is empty;
    # "كلام عادي جدا" is marked both ways, so both copies go
    assert df["clean"].tolist() == ["يا سلام علي النظام"]


def test_clean_keeps_both_text_versions():
    df = data.clean(pd.DataFrame({"tweet": ["حلو اوي 😂"], "sarcasm": [1], "dialect": [0]}))
    assert df.loc[0, "clean"] == "حلو اوي"
    assert "emo1f602" in df.loc[0, "signals"]
