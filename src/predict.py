"""Check Arabic sentences for sarcasm with the trained model.

    python src/predict.py                          # a few example sentences
    python src/predict.py "يا سلام على النظام"      # your own sentence
"""

import sys
from pathlib import Path

import joblib

from preprocess import normalize_arabic, normalize_with_signals

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sarcasm_model.joblib"
CLEANERS = {"clean": normalize_arabic, "signals": normalize_with_signals}

EXAMPLES = [
    "يا سلام على الخدمة، استنينا ساعتين بس 🙄",
    "الخدمة كانت سريعة والناس محترمة",
    "عبقري والله، نسي المفتاح جوه العربية تاني !!",
    "النهارده الجو حلو ورايحين النادي",
]


def main():
    saved = joblib.load(MODEL_PATH)
    model, clean = saved["model"], CLEANERS[saved["text"]]
    texts = sys.argv[1:] or EXAMPLES
    cleaned = [clean(t) for t in texts]
    for text, label, margin in zip(texts, model.predict(cleaned), model.decision_function(cleaned)):
        print(f"{'sarcastic' if label else 'not sarcastic':<14}{margin:+.2f}  | {text}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
