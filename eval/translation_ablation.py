"""Does the Tagalog translation filter recover the recall Track A loses on Taglish?

    python eval/translation_ablation.py
    python eval/translation_ablation.py --examples 6

Scores the same models on the same Taglish rows twice - once on the raw text,
once on the text after ``CodeSwitchTranslator`` has replaced Tagalog words with
their English equivalents - and reports the difference.

**This is the experiment that justifies the code-switching layer, or does not.**
Track A's features count English instruction verbs: ``imperative_verb_count``,
``instruction_verb_ratio``, ``malicious_word_weight_sum``. Swap ``ignore`` for
``balewalain`` and every one of them reads zero, which is why the classical
models lose 3-5 points of recall on code-switched text while the neural models
lose roughly nothing. Translation is supposed to hand those counters their
vocabulary back.

The gateway runs this filter at ``mochi/detect/pipeline.py`` and it is off by
default (``MOCHI_ENABLE_TAGALOG_TRANSLATION``), so every number reported
elsewhere in this project was measured *without* it. This script is the only
place the two conditions are compared directly.

Track B is not included. Its models read subwords and embeddings rather than a
lexicon, they already lose almost nothing on Taglish, and translating their
input would be testing a defense built for a problem they do not have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.baseline_models import (  # noqa: E402
    build_track_a_models,
    decision_scores,
    select_threshold,
)
from eval.data_loading import DATA_DIR  # noqa: E402
from eval.taglish_report import TAGLISH_MARKER, load_splits  # noqa: E402


def translate_all(texts: list[str]) -> tuple[list[str], dict]:
    """Run every text through the code-switch filter.

    Returns the filtered texts and a summary of how much actually changed -
    a translation pass that altered nothing would make the ablation look like
    a null result when it was really a no-op.
    """
    from mochi.preprocess.code_switch import CodeSwitchTranslator

    translator = CodeSwitchTranslator()
    out: list[str] = []
    changed = 0
    for text in texts:
        result = translator.filter_and_translate(text)
        filtered = result.text or text
        if filtered != text:
            changed += 1
        out.append(filtered)
    return out, {"texts": len(texts), "changed": changed,
                 "changed_share": changed / max(len(texts), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "clean")
    parser.add_argument("--cv", type=Path, default=REPO / "reports" / "track_a.json")
    parser.add_argument("--json", type=Path,
                        default=REPO / "reports" / "translation_ablation.json")
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()

    import numpy as np

    train, validation, test = load_splits(args.data)
    taglish_index = [i for i, s in enumerate(test.samples)
                     if TAGLISH_MARKER in s.dataset and s.label == 1]
    taglish_texts = [test.texts[i] for i in taglish_index]

    print(f"\n  {len(taglish_texts):,} malicious Taglish rows in the test split")
    print("  translating ...", flush=True)
    translated, summary = translate_all(taglish_texts)
    print(f"  filter changed {summary['changed']:,} of {summary['texts']:,} "
          f"({summary['changed_share']:.1%})\n")

    params = {}
    if args.cv.exists():
        saved = json.loads(args.cv.read_text(encoding="utf-8"))
        params = {name: row["params"]
                  for name, row in saved.get("cross_validation", {}).items()}

    rows = []
    for name, model in build_track_a_models(params=params).items():
        print(f"  fitting {name} ...", flush=True)
        model.fit(train.texts, train.labels)
        # The threshold is chosen on untranslated validation data and then held
        # fixed across both conditions. Re-tuning it per condition would let the
        # cutoff absorb the effect this script exists to measure.
        threshold = select_threshold(
            decision_scores(model, validation.texts), validation.labels)

        raw = np.asarray(decision_scores(model, taglish_texts))
        fixed = np.asarray(decision_scores(model, translated))
        rows.append({
            "model": name,
            "threshold": threshold,
            "recall_raw": float((raw >= threshold).mean()),
            "recall_translated": float((fixed >= threshold).mean()),
            "recovered": int(((fixed >= threshold) & (raw < threshold)).sum()),
            "lost": int(((fixed < threshold) & (raw >= threshold)).sum()),
            "n": len(taglish_texts),
        })

    print()
    print("=" * 88)
    print("  Tagalog translation filter  -  recall on malicious Taglish rows")
    print("=" * 88)
    print(f"  {'model':<18}{'raw':>10}{'translated':>13}{'delta':>10}"
          f"{'recovered':>12}{'lost':>8}")
    print("  " + "-" * 84)
    for row in sorted(rows, key=lambda r: -(r["recall_translated"] - r["recall_raw"])):
        delta = row["recall_translated"] - row["recall_raw"]
        print(f"  {row['model']:<18}{row['recall_raw']:>10.4f}"
              f"{row['recall_translated']:>13.4f}{delta:>+10.4f}"
              f"{row['recovered']:>12,}{row['lost']:>8,}")
    print("=" * 88)
    print("  recovered = injections the filter turned from a miss into a catch.")
    print("  lost      = the reverse, where translation destroyed the signal.")
    print(f"  n = {len(taglish_texts):,}; the threshold is the one tuned on")
    print("  untranslated validation data, held fixed across both conditions.")
    print()

    if args.examples:
        print(f"  Sample of what the filter does:\n")
        for original, fixed in list(zip(taglish_texts, translated))[:args.examples]:
            if original == fixed:
                continue
            print(f"    before: {original[:84]}")
            print(f"    after : {fixed[:84]}")
            print()

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"summary": summary, "models": rows}, indent=2),
                         encoding="utf-8")
    print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
