"""Classical statistical baselines for Stage II - no neural network required.

    python eval/baseline_models.py --data data/clean
    python eval/baseline_models.py --data data/clean --save models/linear
    python eval/baseline_models.py --track-a          # the classification study
    python eval/baseline_models.py --hybrid           # the Phase 6.5 ablation

Three modes live here, and only one of them is part of the classification study:

* **default** - four TF-IDF baselines, below. A gateway engineering question
  ("is a transformer worth its cost?"), not a thesis result.
* ``--hybrid`` - the Phase 6.5 ablation. Also not a thesis result: its pipelines
  read raw text, which Track A's never do.
* ``--track-a`` - SVM, decision tree and random forest on the 18 engineered
  columns alone. **This is Track A of the two-track comparison** in
  ``docs/CLASSIFICATION_PLAN.md``; the numbers it prints are the ones that go in
  the comparison table, against Track B's models on the identical test rows.

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

A linear model also answers "which token contributed" exactly: the decision is
a weighted sum, so a token's contribution is its TF-IDF value times its
coefficient.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import (  # noqa: E402
    CV_FOLDS,
    DATA_DIR,
    DatasetError,
)


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


#: Default location of the fit instruction-verb weight table, matching
#: ``eval/build_features.py`` and ``eval/fit_malicious_word_weights.py``.
INSTRUCTION_VERB_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "features" / "instruction_verb_weights.json"
)


def engineered_transformer(*, weights_path: Path = INSTRUCTION_VERB_WEIGHTS_PATH):
    """Sklearn transformer over the full 18-column Track A input.

    17 columns are pure functions of the text (``TRACK_A_FEATURES``); the 18th,
    ``malicious_word_weight_sum``, is corpus-fitted and therefore lives in
    ``TRACK_A_FITTED_FEATURES`` instead. Both lists are imported, not
    duplicated, so this transformer follows any change to either.

    **No Stage I scan runs** - ``extract(stage1=None)``. None of the 17 are
    Stage I-derived, so a Track A model cannot be rediscovering Stage I's own
    regexes. ``normalize()`` still runs, for ``is_code_switched``. The 18th
    column is pure lookup and fits nothing, so it carries no leakage risk.

    Results are memoised per text: fit and predict pass over overlapping data.
    """
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin

    from eval.token_association import score_malicious_word_weight
    from mochi.preprocess import normalize
    from mochi.preprocess.features import TRACK_A_FEATURES, extract

    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Run eval/fit_malicious_word_weights.py "
            f"first - malicious_word_weight_sum needs a weight table fit on "
            f"the train split before Track A can be built."
        )
    weights = json.loads(weights_path.read_text(encoding="utf-8"))["weights"]

    class EngineeredFeatures(BaseEstimator, TransformerMixin):
        def __init__(self) -> None:
            self._cache: dict[str, list[float]] = {}

        def fit(self, X, y=None):  # noqa: N803
            return self

        def _row(self, text: str) -> list[float]:
            cached = self._cache.get(text)
            if cached is not None:
                return cached
            row = extract(text, norm=normalize(text)).as_dict()
            values = [float(row[name]) for name in TRACK_A_FEATURES]
            values.append(float(score_malicious_word_weight(text, weights)))
            self._cache[text] = values
            return values

        def transform(self, X):  # noqa: N803
            return np.asarray([self._row(t) for t in X], dtype=np.float64)

    return EngineeredFeatures()


def subsampled_rbf_svc(n_samples: int | None = None, C: float = 1.0,  # noqa: N803
                       gamma: str = "scale", seed: int = 42):
    """An RBF-kernel SVM, optionally fit on a stratified subsample.

    **Subsampling is off by default, because Track A does not need it.** The
    reflex comes from kernel SVMs on sparse TF-IDF text; Track A rows are 18
    dense numbers, so the constant factor is tiny. Measured on this corpus:

        n= 1,000  0.01s     n= 8,000   0.83s
        n= 2,000  0.05s     n=16,000   3.98s
        n= 4,000  0.20s     n=52,144  28.8s   (the full training split)

    Growth is the expected ~O(n^2) but from a small enough base that the full
    fit is half a minute. ``n_samples`` stays available for a wider feature set
    or larger corpus; when set it is disclosed in the model's name
    (``svm (rbf, 15k)``) so a subsampled fit cannot read as a full one.

    Paired with ``LinearSVC``, this separates two questions a single SVM row
    would confound: whether the SVM *family* underperforms, or only linear
    models do. Any subsample is stratified and seeded.

    Defined in a factory so sklearn is imported lazily, as
    ``engineered_transformer`` does - sklearn 1.6 requires estimators to
    inherit ``BaseEstimator``.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin

    class SubsampledRBFSVC(ClassifierMixin, BaseEstimator):
        def __init__(self, n_samples: int | None = None, C: float = 1.0,  # noqa: N803
                     gamma: str = "scale", seed: int = 42) -> None:
            self.n_samples = n_samples
            self.C = C
            self.gamma = gamma
            self.seed = seed

        def fit(self, X, y):  # noqa: N803
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.svm import SVC

            X = np.asarray(X)
            y = np.asarray(y)
            if self.n_samples is not None and self.n_samples < len(y):
                X, _, y, _ = train_test_split(
                    X, y, train_size=self.n_samples, stratify=y,
                    random_state=self.seed,
                )
            self.estimator_ = SVC(kernel="rbf", C=self.C, gamma=self.gamma,
                                  random_state=self.seed)
            self.estimator_.fit(X, y)
            self.classes_ = self.estimator_.classes_
            self.n_features_in_ = X.shape[1]
            self.n_fit_rows_ = len(y)
            # Recorded so the subsample's composition is inspectable after the
            # fact - a results table naming a fit size should be checkable, and
            # a stratified draw that silently stopped stratifying would
            # otherwise be invisible. Aligned with ``classes_``: both come from
            # sorted unique labels.
            _, counts = np.unique(y, return_counts=True)
            self.fit_class_counts_ = counts.tolist()
            return self

        def predict(self, X):  # noqa: N803
            return self.estimator_.predict(X)

        def decision_function(self, X):  # noqa: N803
            # Needed for threshold tuning: SVC exposes no calibrated
            # probability without ``probability=True`` (an expensive internal
            # cross-validation), but signed distance to the margin orders rows
            # exactly as well, and ordering is all a threshold sweep needs.
            return self.estimator_.decision_function(X)

    return SubsampledRBFSVC(n_samples=n_samples, C=C, gamma=gamma, seed=seed)


def track_a_feature_names() -> list[str]:
    """The 18 column names, in the order ``engineered_transformer`` emits them.

    Imported from the two frozen lists rather than restated, so an importance
    table can never label a column with the wrong name after a set changes.
    """
    from eval.token_association import TRACK_A_FITTED_FEATURES
    from mochi.preprocess.features import TRACK_A_FEATURES

    return list(TRACK_A_FEATURES) + list(TRACK_A_FITTED_FEATURES)


def build_track_a_models(*, weights_path: Path = INSTRUCTION_VERB_WEIGHTS_PATH,
                         svm_subsample: int | None = None, seed: int = 42,
                         params: dict[str, dict] | None = None):
    """The classification study's classical track: SVM, decision tree, random forest.

    The three families the thesis commits to. Unlike ``build_hybrid_models``
    these are genuine Track A contenders: the only input is the 18-column table,
    never raw text.

    **Scaling branches by family.** ``char_count`` runs to five figures while
    most columns are 0/1, so both SVMs get a ``StandardScaler``; trees are
    invariant to monotone rescaling and get none, since a scaler in front of a
    tree implies a dependency that does not exist.

    All four pipelines share one ``engineered_transformer`` so its per-text
    cache is shared - extraction runs ``normalize()`` per row and should be paid
    once per run, not once per model.

    Four entries for three families - see :func:`subsampled_rbf_svc`.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    extractor = engineered_transformer(weights_path=weights_path)

    def scaled(classifier):
        return Pipeline([
            ("vec", Pipeline([("extract", extractor), ("scale", StandardScaler())])),
            ("clf", classifier),
        ])

    def unscaled(classifier):
        return Pipeline([("vec", extractor), ("clf", classifier)])

    # Named for what was actually fit: a subsampled run must never be readable
    # as a full-data one in the results table.
    rbf_name = "svm (rbf)" if svm_subsample is None else \
        f"svm (rbf, {svm_subsample // 1000}k)"

    models = {
        "svm (linear)": scaled(
            LinearSVC(C=1.0, max_iter=5000, random_state=seed)),
        rbf_name: scaled(
            subsampled_rbf_svc(n_samples=svm_subsample, C=1.0, seed=seed)),
        # max_depth bounds the tree at a size a person can actually read - the
        # point of including a single tree at all is that its splits are
        # inspectable, and an unbounded tree on 58k rows is neither readable
        # nor better (the forest already covers raw performance).
        "decision tree": unscaled(
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=25,
                                   random_state=seed)),
        "random forest": unscaled(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed)),
    }

    # Hyperparameters chosen by cross_validate_track_a, applied to the final
    # estimator. Without this the CV result would be a number in a report that
    # nothing acts on - the models scored on test would still carry the
    # hand-picked defaults the search was run to replace.
    for name, overrides in (params or {}).items():
        if name in models:
            models[name].named_steps["clf"].set_params(**overrides)

    return models


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
    """TF-IDF alone vs TF-IDF + the frozen Track A features - the Phase 6.5 ablation.

    The question is narrow: do the 17 engineered columns
    (``mochi.preprocess.features.TRACK_A_FEATURES``) add anything a bag of
    n-grams does not already capture? The imperative and length families are the
    reason to expect they might - a count or a boolean over the whole text is a
    different kind of signal than a token weight, even where the same words are
    driving both.

    Note this pipeline is neither Track A nor Track B under the two-track
    separation (`docs/CLASSIFICATION_PLAN.md`): it reads raw text through
    TF-IDF, which Track A's models never do. It is a standing reference
    ablation, not a contender in the classification study's comparison table.

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

    Track A's trees and forests have no ``coef_`` at all - a tree's decision is
    a path, not a weighted sum - so ``n_features_in_`` is the last resort. It is
    the input width every fitted sklearn estimator records, which for those
    models is exactly the 18 engineered columns.
    """
    vectoriser = model.named_steps.get("vec")
    vocabulary = getattr(vectoriser, "vocabulary_", None)
    if vocabulary is not None:
        return len(vocabulary)
    classifier = model.named_steps.get("clf")
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is not None:
        return int(coefficients.shape[1])
    n_features_in = getattr(classifier, "n_features_in_", None)
    if n_features_in is not None:
        return int(n_features_in)
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


#: FPR budgets the threshold report selects against, loosest first. Chosen to
#: bracket the two enforcement tiers in ``mochi.mitigate.sanitizer``: a
#: SANITIZE-tier false positive costs a stripped instruction, so it can sit near
#: 0.10, while a BLOCK-tier one refuses a legitimate request and belongs nearer
#: 0.01. The middle value is there so the shape of the trade-off is visible
#: rather than inferred from two endpoints.
FPR_BUDGETS = (0.10, 0.05, 0.01)


def decision_scores(model, texts):
    """A continuous "more malicious" score for every text.

    Trees expose ``predict_proba``, SVMs ``decision_function``. The two are on
    different scales, which is why every threshold is selected per model - a
    sweep needs the score to *order* rows, not to mean anything absolute.
    """
    classifier = model.named_steps["clf"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def default_threshold(model) -> float:
    """The cutoff ``.predict()`` uses implicitly, for the comparison row.

    0.5 for a probability, 0.0 for a signed margin. Neither was ever chosen for
    this problem - both are the arbitrary midpoint of their own scale, which is
    the entire reason this module tunes.
    """
    return 0.5 if hasattr(model.named_steps["clf"], "predict_proba") else 0.0


def threshold_sweep(scores, labels):
    """Every distinct operating point, most to least aggressive.

    Returns aligned ``(thresholds, recall, fpr, precision)``, where predicting
    ``score >= thresholds[i]`` yields ``recall[i]``.

    Rows sharing a score collapse to one point: no threshold can separate them,
    and keeping one point per row would invent unreachable operating points.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    positives = max(int((sorted_labels == 1).sum()), 1)
    negatives = max(int((sorted_labels == 0).sum()), 1)

    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)

    last_of_run = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp = true_positives[last_of_run]
    fp = false_positives[last_of_run]

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return sorted_scores[last_of_run], tp / positives, fp / negatives, precision


def select_threshold(scores, labels, *, max_fpr: float | None = None) -> float:
    """Choose a cutoff. **Call this on validation scores, never on test.**

    With ``max_fpr``, returns the highest-recall threshold inside that budget -
    the form the gateway needs, since "catch as much as possible" only means
    something once the tolerable false-positive rate is fixed. Without one,
    maximises F1.

    An unreachable budget returns a threshold above every score, classifying
    everything benign - visible as recall 0.0 rather than a silent fallback to
    a looser budget.
    """
    import numpy as np

    thresholds, recall, fpr, precision = threshold_sweep(scores, labels)

    if max_fpr is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(precision + recall > 0,
                          2 * precision * recall / np.maximum(precision + recall, 1e-12),
                          0.0)
        return float(thresholds[int(np.argmax(f1))])

    eligible = fpr <= max_fpr
    if not eligible.any():
        return float(thresholds[0]) + 1.0
    return float(thresholds[int(np.argmax(np.where(eligible, recall, -1.0)))])


def metrics_at(scores, labels, threshold: float) -> dict:
    """Confusion-matrix metrics for one fixed cutoff (``score >= threshold``)."""
    import numpy as np

    return metrics_from_predictions(
        np.asarray(scores, dtype=float) >= threshold, labels, threshold)


def metrics_from_predictions(predicted, labels, threshold: float) -> dict:
    """Metrics for predictions already made, however they were made.

    Lets the report's ``default`` row come from ``.predict()`` rather than
    ``score >= 0.5``. The two disagree on exact ties - ``argmax`` awards a 50/50
    leaf to benign, ``>=`` to malicious - which is rare in a 300-tree forest but
    not in a shallow decision tree.
    """
    import numpy as np

    predicted = np.asarray(predicted, dtype=bool)
    labels = np.asarray(labels)

    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def tune_thresholds(models: dict, validation, test) -> list[dict]:
    """Select each model's cutoff on validation, then measure it on test.

    Every threshold is chosen from validation scores only; test is scored once,
    afterwards, at the committed cutoff. Picking the best of several cutoffs by
    their test scores would report the maximum of a search as a single
    measurement.
    """
    rows: list[dict] = []
    for name, model in models.items():
        validation_scores = decision_scores(model, validation.texts)
        test_scores = decision_scores(model, test.texts)

        # The untuned baseline, taken from the model's own decision rule so it
        # reproduces the fixed-threshold table exactly - see
        # ``metrics_from_predictions`` on why ``score >= 0.5`` would not.
        baseline = metrics_from_predictions(
            model.predict(test.texts), test.labels, default_threshold(model))
        baseline["model"] = name
        baseline["objective"] = f"default ({default_threshold(model):g})"
        baseline["validation"] = metrics_from_predictions(
            model.predict(validation.texts), validation.labels,
            default_threshold(model))
        rows.append(baseline)

        objectives: list[tuple[str, float]] = [
            ("max F1", select_threshold(validation_scores, validation.labels)),
        ]
        for budget in FPR_BUDGETS:
            objectives.append((
                f"recall @FPR<={budget:g}",
                select_threshold(validation_scores, validation.labels, max_fpr=budget),
            ))

        for objective, threshold in objectives:
            row = metrics_at(test_scores, test.labels, threshold)
            row["model"] = name
            row["objective"] = objective
            row["validation"] = metrics_at(
                validation_scores, validation.labels, threshold)
            rows.append(row)
    return rows


def print_threshold_report(rows: list[dict], validation_n: int, test_n: int) -> None:
    print("=" * 100)
    print(f"  Threshold tuning  -  chosen on validation ({validation_n:,} rows), "
          f"reported on test ({test_n:,})")
    print("=" * 100)
    print(f"  {'model':<18}{'objective':<20}{'thr':>9}{'recall':>9}{'FPR':>9}"
          f"{'prec':>9}{'F1':>9}{'missed':>9}")
    print("  " + "-" * 96)

    current = None
    for row in rows:
        if current is not None and row["model"] != current:
            print("  " + "-" * 96)
        current = row["model"]
        label = row["model"] if row["objective"].startswith("default") else ""
        print(f"  {label:<18}{row['objective']:<20}{row['threshold']:>9.3f}"
              f"{row['recall']:>9.4f}{row['fpr']:>9.4f}{row['precision']:>9.4f}"
              f"{row['f1']:>9.4f}{row['confusion']['fn']:>9,}")
    print("=" * 100)
    print("  'missed' = injections that reached the model and got through "
          "(false negatives).")
    print()


def track_a_search_space(seed: int = 42):
    """(pipeline, grid) per model, operating on the **18-column matrix**.

    Deliberately not the text pipelines: ``GridSearchCV`` clones its estimator
    per fold and grid point, and a cloned extractor starts with an empty cache,
    so a search over text would re-run ``normalize()`` on ~40,000 rows hundreds
    of times. The winning parameters transfer back unchanged.

    Grids are small on purpose - ten folds multiplies every point by ten, and
    the RBF SVM costs ~15s a fit.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    def scaled(classifier):
        return Pipeline([("scale", StandardScaler()), ("clf", classifier)])

    def unscaled(classifier):
        return Pipeline([("clf", classifier)])

    return {
        "svm (linear)": (
            scaled(LinearSVC(max_iter=5000, random_state=seed)),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "svm (rbf)": (
            scaled(SVC(kernel="rbf", random_state=seed)),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.1]},
        ),
        "decision tree": (
            unscaled(DecisionTreeClassifier(random_state=seed)),
            {"clf__max_depth": [4, 8, 12, None],
             "clf__min_samples_leaf": [5, 25, 100]},
        ),
        "random forest": (
            unscaled(RandomForestClassifier(n_jobs=-1, random_state=seed)),
            {"clf__n_estimators": [200, 500],
             "clf__max_depth": [None, 20],
             "clf__min_samples_leaf": [1, 2, 5]},
        ),
    }


def cross_validate_track_a(train, *, folds: int = CV_FOLDS,
                           weights_path: Path = INSTRUCTION_VERB_WEIGHTS_PATH,
                           seed: int = 42) -> dict:
    """Select hyperparameters by ``folds``-fold CV **inside the training split**.

    Searches the train tier only. Validation is reserved for the decision
    threshold; test is untouched until the final score.

    Folds are stratified, so a difference in F1 comes from the parameter rather
    than from a fold's class balance.

    Returns per model: winning parameters (``clf__`` stripped, ready for
    ``build_track_a_models(params=...)``), mean and standard deviation of F1
    across folds, and search time. The deviation is what a single holdout cannot
    give, and is what makes "A beat B" a claim about the models.
    """
    import numpy as np
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    matrix = engineered_transformer(weights_path=weights_path).transform(train.texts)
    labels = np.asarray(train.labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    results: dict[str, dict] = {}
    for name, (pipeline, grid) in track_a_search_space(seed).items():
        combinations = int(np.prod([len(v) for v in grid.values()]))
        print(f"  cross-validating {name} ... "
              f"({combinations} settings x {folds} folds)", flush=True)

        started = time.perf_counter()
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=splitter,
                              n_jobs=-1, refit=False)
        search.fit(matrix, labels)
        elapsed = time.perf_counter() - started

        best = int(search.best_index_)
        results[name] = {
            "params": {key.removeprefix("clf__"): value
                       for key, value in search.best_params_.items()},
            "mean_f1": float(search.cv_results_["mean_test_score"][best]),
            "std_f1": float(search.cv_results_["std_test_score"][best]),
            "seconds": elapsed,
            "n_settings": combinations,
            "folds": folds,
        }
    return results


def print_cv_report(results: dict, n_train: int) -> None:
    print()
    print("=" * 100)
    print(f"  {results[next(iter(results))]['folds']}-fold cross-validation  -  "
          f"inside the training split only ({n_train:,} rows)")
    print("=" * 100)
    print(f"  {'model':<18}{'CV F1 (mean +- sd)':>24}{'search':>10}   "
          f"{'chosen hyperparameters'}")
    print("  " + "-" * 96)
    for name, row in sorted(results.items(), key=lambda kv: -kv[1]["mean_f1"]):
        chosen = ", ".join(f"{k}={v}" for k, v in sorted(row["params"].items()))
        print(f"  {name:<18}{row['mean_f1']:>15.4f} +- {row['std_f1']:<6.4f}"
              f"{row['seconds']:>9.0f}s   {chosen}")
    print("=" * 100)
    print("  Standard deviation is across folds - a spread a single holdout "
          "cannot report.")
    print()


def print_track_a_report(results: list[Scored]) -> None:
    print()
    print("=" * 100)
    print("  Track A  -  classical models on the 18 engineered columns (no raw text)")
    print("=" * 100)
    print(f"  {'model':<28}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'FPR':>9}{'train':>9}{'us/pred':>10}{'features':>10}")
    print("  " + "-" * 96)
    for r in sorted(results, key=lambda r: r.f1, reverse=True):
        print(f"  {r.name:<28}{r.accuracy:>8.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.f1:>8.4f}{r.fpr:>9.4f}{r.train_seconds:>8.1f}s"
              f"{r.predict_us_per_sample:>10.1f}{r.n_features:>10,}")
    print("=" * 100)
    print()


def print_track_a_importances(models: dict) -> None:
    """Which of the 18 columns each fitted model actually leans on.

    The artifact Track A exists to produce: direct evidence about which
    engineered features carry signal, in a form no Track B model can supply.

    The three columns are **not** on a common scale - tree importances are
    normalised impurity reductions, the SVM's coefficients are signed and in
    standard deviations. Only the ordering within a column is comparable.
    """
    names = track_a_feature_names()

    tree = models.get("decision tree")
    forest = models.get("random forest")
    linear = models.get("svm (linear)")

    dt_importance = getattr(tree.named_steps["clf"], "feature_importances_", None) if tree else None
    rf_importance = getattr(forest.named_steps["clf"], "feature_importances_", None) if forest else None
    svm_coefficients = getattr(linear.named_steps["clf"], "coef_", None) if linear else None
    svm_weights = svm_coefficients[0] if svm_coefficients is not None else None

    ranking = rf_importance if rf_importance is not None else dt_importance
    if ranking is None:
        return

    print("=" * 100)
    print("  Feature importance  -  ordered by random forest (impurity reduction)")
    print("=" * 100)
    print(f"  {'feature':<34}{'forest':>10}{'tree':>10}{'svm coef':>12}")
    print("  " + "-" * 96)
    for index in sorted(range(len(names)), key=lambda i: -ranking[i]):
        rf_cell = f"{rf_importance[index]:.4f}" if rf_importance is not None else ""
        dt_cell = f"{dt_importance[index]:.4f}" if dt_importance is not None else ""
        svm_cell = f"{svm_weights[index]:+.3f}" if svm_weights is not None else ""
        print(f"  {names[index]:<34}{rf_cell:>10}{dt_cell:>10}{svm_cell:>12}")
    print("=" * 100)
    print()

    if tree is not None:
        from sklearn.tree import export_text

        print("  Decision tree, top 3 levels  (the full tree is 8 deep)")
        print("  " + "-" * 96)
        rendered = export_text(tree.named_steps["clf"], feature_names=names,
                               max_depth=2, decimals=1)
        for line in rendered.splitlines():
            print(f"    {line}")
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
    parser.add_argument("--track-a", action="store_true",
                        help="run the classification study's classical track: "
                             "SVM, decision tree and random forest on the 18 "
                             "engineered columns only")
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip hyperparameter cross-validation and use the "
                             "hand-picked defaults - a fast smoke run, not the "
                             "thesis protocol")
    parser.add_argument("--folds", type=int, default=CV_FOLDS,
                        help=f"cross-validation folds (default {CV_FOLDS})")
    parser.add_argument("--svm-subsample", type=int, default=None,
                        help="cap training rows for the RBF SVM (default: use "
                             "all of them - a full fit is ~30s at 18 columns). "
                             "Only needed if the feature set or corpus grows.")
    args = parser.parse_args()

    from training.finetune_e5 import build_splits

    try:
        train, validation, test = build_splits(args.data)
    except (DatasetError, SystemExit) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\n  train {len(train):,}   validation {len(validation):,}   "
          f"test {len(test):,}   ({test.positive_rate:.1%} malicious in test)\n")

    if args.track_a:
        cv_results: dict = {}
        try:
            if not args.skip_cv:
                cv_results = cross_validate_track_a(train, folds=args.folds)
                print_cv_report(cv_results, len(train))
            models = build_track_a_models(
                svm_subsample=args.svm_subsample,
                params={name: row["params"] for name, row in cv_results.items()},
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1

        results = []
        for name, model in models.items():
            print(f"  training {name} ...", flush=True)
            results.append(score_model(name, model, train, validation, test))

        print_track_a_report(results)
        tuned = tune_thresholds(models, validation, test)
        print_threshold_report(tuned, len(validation), len(test))
        print_track_a_importances(models)

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(
                {"cross_validation": cv_results,
                 "fixed_threshold": [vars(r) for r in results],
                 "tuned": tuned}, indent=2), encoding="utf-8")
            print(f"  Wrote {args.json}\n")
        return 0

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
