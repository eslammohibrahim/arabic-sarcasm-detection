"""Train and compare sarcasm classifiers.

1. Official ArSarcasm train / test split, 10% of train held out for validation.
2. Baseline: always say "not sarcastic" (most tweets aren't).
3. Each model is tried on two versions of the text: words only, and words plus
   emojis / "!!!" / stretched letters. One hyperparameter is searched on validation.
4. The best one (by validation F1 on the sarcastic class) is retrained on train + validation
   and scored on the test set once.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from data import load_splits

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
TEXT_COLUMNS = {"words only": "clean", "words + emojis/punctuation": "signals"}


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, dtype=np.float32)


def features(kind):
    if kind == "words":
        return word_tfidf()
    if kind == "characters":
        return char_tfidf()
    return FeatureUnion([("words", word_tfidf()), ("characters", char_tfidf())])


# name: (features, classifier factory, values to try). Sarcasm is ~16% of tweets, so the
# linear models weight the classes; otherwise they mostly learn to say "not sarcastic".
MODELS = {
    "Naive Bayes (words)": ("words", lambda a: ComplementNB(alpha=a), [0.1, 0.3, 1.0]),
    "Logistic Regression (words)": ("words", lambda c: LogisticRegression(C=c, class_weight="balanced", max_iter=3000), [0.3, 1, 3, 10]),
    "Linear SVM (characters)": ("characters", lambda c: LinearSVC(C=c, class_weight="balanced"), [0.03, 0.1, 0.3, 1.0]),
    "Linear SVM (words + characters)": ("both", lambda c: LinearSVC(C=c, class_weight="balanced"), [0.03, 0.1, 0.3, 1.0]),
}


def score(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "f1_sarcastic": round(float(f1_score(y, p, zero_division=0)), 4),
            "precision_sarcastic": round(float(precision_score(y, p, zero_division=0)), 4),
            "recall_sarcastic": round(float(recall_score(y, p, zero_division=0)), 4)}


def plot_confusion(y, p, path):
    cm = confusion_matrix(y, p)
    names = ["not sarcastic", "sarcastic"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n{share:.0%}", ha="center", va="center", color="white" if share > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(verbose=True)
    results = {"sizes": {"train": len(train), "val": len(val), "test": len(test)},
               "sarcastic_share_test": round(float(test["sarcasm"].mean()), 4), "models": {}}

    zeros = np.zeros(len(val), dtype=int)
    results["models"]["Always not sarcastic"] = {"val": score(val["sarcasm"], zeros)}
    print("Always not sarcastic", results["models"]["Always not sarcastic"])

    best = (None, None, None, -1)
    for text_name, col in TEXT_COLUMNS.items():
        cache = {}
        for name, (kind, make, values) in MODELS.items():
            if kind not in cache:
                vec = features(kind)
                cache[kind] = (vec.fit_transform(train[col]), vec.transform(val[col]))
            Xtr, Xva = cache[kind]
            tried = {}
            for v in values:
                t = time.time()
                clf = make(v).fit(Xtr, train["sarcasm"])
                tried[v] = score(val["sarcasm"], clf.predict(Xva))
                print(f"  [{text_name}] {name} {v}: {tried[v]} ({time.time() - t:.1f}s)")
            v_best = max(tried, key=lambda v: tried[v]["f1_sarcastic"])
            key = f"{name} | {text_name}"
            results["models"][key] = {"best_param": v_best, "val": tried[v_best]}
            if tried[v_best]["f1_sarcastic"] > best[3]:
                best = (name, text_name, v_best, tried[v_best]["f1_sarcastic"])

    name, text_name, v, _ = best
    kind, make, _ = MODELS[name]
    col = TEXT_COLUMNS[text_name]
    print(f"best: {name} on {text_name} ({v}); retraining on train + val")
    full = pd.concat([train, val])
    model = make_pipeline(features(kind), make(v)).fit(full[col], full["sarcasm"])
    p = model.predict(test[col])
    results["picked"] = {"model": name, "text": text_name, "param": v}
    results["test"] = score(test["sarcasm"], p)
    results["test_always_not_sarcastic"] = score(test["sarcasm"], np.zeros(len(test), dtype=int))
    results["test_confusion"] = confusion_matrix(test["sarcasm"], p).tolist()
    results["test_by_dialect_recall"] = {
        d: round(float(recall_score(g["sarcasm"], p[g.index], zero_division=0)), 4)
        for d, g in test.groupby("dialect_name") if g["sarcasm"].sum() >= 20}
    plot_confusion(test["sarcasm"], p, FIG_DIR / "confusion_matrix.png")
    joblib.dump({"model": model, "text": col}, MODEL_DIR / "sarcasm_model.joblib")
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in ["picked", "test", "test_always_not_sarcastic", "test_confusion", "test_by_dialect_recall"]}, indent=2))


if __name__ == "__main__":
    main()
