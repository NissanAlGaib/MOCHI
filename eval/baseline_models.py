"""Classical statistical baselines for Stage II - no neural network required.

    python eval/baseline_models.py --data data/clean
    python eval/baseline_models.py --data data/clean --save models/linear

Before committing to a fine-tuned transformer, establish what a linear model on
n-gram counts already achieves. If the gap is small, the transformer is not
earning its cost - and the cost is considerable: ~2.5 GB of dependencies, a GPU
to train, and 30-55 ms per request against a budget that a linear model meets in
microseconds.

Four models, all from scikit-learn, all trained on CPU in seconds:

* **TF-IDF (word) + logistic regression** - the standard text-classification
  baseline. Coefficients are directly readable as per-token evidence.
* **TF-IDF (character n-gram) + logistic regression** - robust to the character
  tricks Phase 3 normalizes, and to misspellings that dodge word features.
* **TF-IDF (word) + linear SVM** - usually the strongest linear text classifier.
* **Multinomial naive Bayes** - the weakest, included as a floor. If a model
  cannot beat naive Bayes it is not learning anything interesting.

Beyond cost, a linear model answers the "which token contributed" question
*exactly*: the decision is a weighted sum of feature values, so a token's
contribution is its TF-IDF value times its coefficient. No approximation, no
attention-weight debate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import DATA_DIR, DatasetError  # noqa: E402


@dataclass
class Scored:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    train_seconds: float
    predict_us_per_sample: float
    n_features: int
    confusion: dict = field(default_factory=dict)


def engineered_transformer():
    """Sklearn transformer over ``mochi.preprocess.features``.

    Calls the runtime extractor rather than recomputing anything, so the
    ablation measures the features the gateway would actually see. Results are
    memoised per text: fit and predict pass over overlapping data, and a Stage I
    scan per row is the expensive part.

    Each string column is encoded according to what it actually is:

    * ``stage1_max_severity`` - **ordinal** (none < low < medium < high). One-hot
      would discard the ordering, which here is real.
    * ``payload_region`` - **one-hot** over five values. head/middle/tail/
      decoded_only/none have no ordering, so an ordinal code would invent one.
    * ``dominant_script`` - binary (latin vs not).
    * ``stage1_detector_ids`` - **dropped, explicitly.** It is a semicolon-joined
      string that duplicates the nine ``hit_*`` booleans exactly. Listed in
      ``DROPPED_COLUMNS`` rather than left to fall through the type checks,
      because a column that disappears by accident is indistinguishable from one
      that disappears by design when someone later counts the features.

    ``EXPECTED_WIDTH`` pins the resulting matrix width so a column added to
    ``FeatureVector`` without a decision here fails loudly instead of silently
    vanishing.
    """
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin

    from mochi.detect.stage1_syntactic import get_detector
    from mochi.preprocess import normalize
    from mochi.preprocess.features import extract

    severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    regions = ("none", "head", "middle", "tail", "decoded_only")
    languages = ("unknown", "english", "tagalog", "mixed")
    DROPPED_COLUMNS = {"stage1_detector_ids"}

    class EngineeredFeatures(BaseEstimator, TransformerMixin):
        def __init__(self) -> None:
            self._cache: dict[str, list[float]] = {}
            self._detector = None

        def fit(self, X, y=None):  # noqa: N803
            return self

        def _row(self, text: str) -> list[float]:
            cached = self._cache.get(text)
            if cached is not None:
                return cached
            if self._detector is None:
                self._detector = get_detector()
            result = normalize(text)
            stage1 = self._detector.scan(result.scannable, result.flags)
            row = extract(text, norm=result, stage1=stage1).as_dict()

            values: list[float] = []
            for name, value in row.items():
                if name in DROPPED_COLUMNS:
                    continue
                if name == "stage1_max_severity":
                    values.append(float(severity_rank.get(value, 0)))
                elif name == "payload_region":
                    values.extend(float(value == r) for r in regions)
                elif name == "dominant_script":
                    values.append(float(value == "latin"))
                elif name == "detected_language":
                    # One-hot, not ordinal: english/tagalog/mixed have no order.
                    values.extend(float(value == lang) for lang in languages)
                elif isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float)):
                    values.append(float(value))
                else:
                    raise ValueError(
                        f"No encoding decided for feature column {name!r} "
                        f"(type {type(value).__name__}). Add it to "
                        f"DROPPED_COLUMNS or give it an encoding - do not let "
                        f"it fall through, or the feature count silently shifts."
                    )
            self._cache[text] = values
            return values

        def transform(self, X):  # noqa: N803
            return np.asarray([self._row(t) for t in X], dtype=np.float64)

    return EngineeredFeatures()


def build_models(max_features: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    def word_tfidf(**kwargs):
        return TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=max_features,
            min_df=2, sublinear_tf=True, strip_accents=None, **kwargs
        )

    return {
        "tfidf_word + logreg": Pipeline([
            ("vec", word_tfidf()),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)),
        ]),
        "tfidf_char + logreg": Pipeline([
            ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                    max_features=max_features, min_df=2,
                                    sublinear_tf=True, lowercase=True)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)),
        ]),
        "tfidf_word + linearSVC": Pipeline([
            ("vec", word_tfidf()),
            ("clf", LinearSVC(C=0.5, max_iter=5000)),
        ]),
        "tfidf_word + naive bayes": Pipeline([
            ("vec", word_tfidf()),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
    }


def build_hybrid_models(max_features: int):
    """TF-IDF alone vs TF-IDF + engineered features - the Phase 6.5 ablation.

    The question is narrow: do the engineered columns add anything a bag of
    n-grams does not already capture? The obfuscation family is the reason to
    expect they might - it describes the *envelope* (was this base64-wrapped, did
    homoglyphs need folding, how much did normalization change) rather than the
    words, and no TF-IDF vectoriser can see that.

    Features are scaled before they meet the linear model: raw ``char_count``
    runs to five figures while every boolean is 0 or 1, and an unscaled
    regularised model would spend its entire penalty budget on the large columns.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import StandardScaler

    def word_tfidf():
        return TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=max_features,
            min_df=2, sublinear_tf=True,
        )

    def logreg():
        return LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)

    engineered = Pipeline([
        ("extract", engineered_transformer()),
        ("scale", StandardScaler()),
    ])

    return {
        "tfidf only (control)": Pipeline([
            ("vec", word_tfidf()),
            ("clf", logreg()),
        ]),
        "engineered only": Pipeline([
            ("vec", engineered),
            ("clf", logreg()),
        ]),
        "tfidf + engineered": Pipeline([
            ("vec", FeatureUnion([
                ("tfidf", word_tfidf()),
                ("engineered", engineered),
            ])),
            ("clf", logreg()),
        ]),
    }


def score_model(name: str, model, train, validation, test) -> Scored:
    started = time.perf_counter()
    model.fit(train.texts, train.labels)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predicted = model.predict(test.texts)
    predict_seconds = time.perf_counter() - started

    tp = sum(1 for p, y in zip(predicted, test.labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(predicted, test.labels) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(predicted, test.labels) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(predicted, test.labels) if p == 0 and y == 1)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return Scored(
        name=name,
        accuracy=(tp + tn) / max(len(test.labels), 1),
        precision=precision,
        recall=recall,
        f1=(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        fpr=fp / (fp + tn) if fp + tn else 0.0,
        train_seconds=train_seconds,
        predict_us_per_sample=predict_seconds / max(len(test.texts), 1) * 1e6,
        n_features=_feature_count(model),
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    )


def _feature_count(model) -> int:
    """Width of the fitted feature space.

    ``vocabulary_`` only exists on a bare vectoriser; the hybrid pipelines put a
    FeatureUnion or a scaler in that slot, so fall back to the fitted
    coefficient width, which is correct for every linear model here.
    """
    vectoriser = model.named_steps.get("vec")
    vocabulary = getattr(vectoriser, "vocabulary_", None)
    if vocabulary is not None:
        return len(vocabulary)
    classifier = model.named_steps.get("clf")
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is not None:
        return int(coefficients.shape[1])
    return 0


def top_features(model, k: int = 15) -> tuple[list, list]:
    """The tokens the linear model leans on, in both directions."""
    vec = model.named_steps["vec"]
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return [], []
    names = vec.get_feature_names_out()
    weights = clf.coef_[0]
    order = weights.argsort()
    benign = [(names[i], float(weights[i])) for i in order[:k]]
    malicious = [(names[i], float(weights[i])) for i in order[-k:][::-1]]
    return malicious, benign


def print_report(results: list[Scored], stage1: dict | None = None) -> None:
    print()
    print("=" * 100)
    print("  Classical baselines for Stage II  -  scikit-learn, CPU, no torch")
    print("=" * 100)
    header = (f"  {'model':<28}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}"
              f"{'FPR':>9}{'train':>9}{'us/pred':>10}{'features':>10}")
    print(header)
    print("  " + "-" * 96)
    for r in sorted(results, key=lambda r: r.f1, reverse=True):
        print(f"  {r.name:<28}{r.accuracy:>8.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.f1:>8.4f}{r.fpr:>9.4f}{r.train_seconds:>8.1f}s"
              f"{r.predict_us_per_sample:>10.1f}{r.n_features:>10,}")
    if stage1:
        print("  " + "-" * 96)
        print(f"  {'Stage I (regex) for scale':<28}{stage1['accuracy']:>8.4f}"
              f"{stage1['precision']:>8.4f}{stage1['recall']:>8.4f}"
              f"{stage1['f1']:>8.4f}{stage1['fpr']:>9.4f}"
              f"{'n/a':>9}{550.0:>10.1f}{'60 rules':>10}")
    print("=" * 100)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None,
                        help="persist the best model with joblib")
    parser.add_argument("--hybrid", action="store_true",
                        help="run the Phase 6.5 ablation instead: TF-IDF alone "
                             "vs engineered features alone vs both")
    args = parser.parse_args()

    from training.finetune_e5 import build_splits

    try:
        train, validation, test = build_splits(args.data)
    except (DatasetError, SystemExit) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\n  train {len(train):,}   validation {len(validation):,}   "
          f"test {len(test):,}   ({test.positive_rate:.1%} malicious in test)\n")

    if args.hybrid:
        results = []
        for name, model in build_hybrid_models(args.max_features).items():
            print(f"  training {name} ...", flush=True)
            results.append(score_model(name, model, train, validation, test))

        print()
        print("=" * 100)
        print("  Phase 6.5 ablation  -  do the engineered features add anything?")
        print("=" * 100)
        print(f"  {'variant':<28}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}"
              f"{'FPR':>9}{'train':>9}{'features':>11}")
        print("  " + "-" * 96)
        for r in results:
            print(f"  {r.name:<28}{r.accuracy:>8.4f}{r.precision:>8.4f}"
                  f"{r.recall:>8.4f}{r.f1:>8.4f}{r.fpr:>9.4f}"
                  f"{r.train_seconds:>8.1f}s{r.n_features:>11,}")
        print("  " + "-" * 96)

        control = next((r for r in results if r.name.startswith("tfidf only")), None)
        combined = next((r for r in results if r.name.startswith("tfidf +")), None)
        if control and combined:
            delta_f1 = combined.f1 - control.f1
            delta_fpr = combined.fpr - control.fpr
            print(f"  Delta from adding engineered features: "
                  f"F1 {delta_f1:+.4f}, FPR {delta_fpr:+.4f}")
            if abs(delta_f1) < 0.005:
                print("  The features do not earn a place in the runtime path on this")
                print("  evidence. Keep them as dataset columns for analysis only.")
            elif delta_f1 > 0:
                print("  The features add signal TF-IDF does not already capture.")
            else:
                print("  The features hurt. Do not wire them into the decision path.")
        print("=" * 100)
        print()

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps([vars(r) for r in results], indent=2),
                                 encoding="utf-8")
            print(f"  Wrote {args.json}\n")
        return 0

    results = []
    for name, model in build_models(args.max_features).items():
        print(f"  training {name} ...", flush=True)
        results.append(score_model(name, model, train, validation, test))

    print_report(results, stage1={
        "accuracy": 0.5918, "precision": 0.9728, "recall": 0.0590,
        "f1": 0.1112, "fpr": 0.0013,
    })

    best_name = max(results, key=lambda r: r.f1).name
    best_model = build_models(args.max_features)[best_name]
    best_model.fit(train.texts, train.labels)
    malicious, benign = top_features(best_model)
    if malicious:
        print(f"  What '{best_name}' learned")
        print(f"    {'toward MALICIOUS':<34}{'toward BENIGN':<34}")
        for (mt, mw), (bt, bw) in zip(malicious, benign):
            print(f"    {mt[:22]:<24}{mw:>+7.2f}   {bt[:22]:<24}{bw:>+7.2f}")
        print()

    if args.save:
        import joblib
        args.save.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, args.save / "model.joblib")
        (args.save / "meta.json").write_text(json.dumps(
            {"name": best_name, "metrics": vars(max(results, key=lambda r: r.f1))},
            indent=2, default=str), encoding="utf-8")
        print(f"  Saved {best_name} -> {args.save / 'model.joblib'}\n")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([vars(r) for r in results], indent=2),
                             encoding="utf-8")
        print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
