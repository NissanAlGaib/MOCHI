"""One table across the per-modality false-positive arms.

    python eval/benign_media_summary.py

Reads the rows each ``benign_media_run`` wrote and puts the three modalities
beside each other, which the per-arm reports cannot do. The point of the
comparison is the middle column: every arm sends the carrier prompt, so a
difference between arms is the modality and nothing else.

Reports the carrier effect pooled as well as per arm. Pooling is legitimate
there and only there - the carrier arm is plain text in every case, so the
three arms' carrier measurements differ only in which carrier string was used
and how many queries were drawn. The *modality* numbers are never pooled,
because an image, a recording and a video are not samples of one thing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.benign_media_run import ModalityResult  # noqa: E402
from eval.stats import mcnemar_test, wilson_interval  # noqa: E402

MODALITIES = ("image", "audio", "video")


def load(modality: str) -> list[ModalityResult]:
    path = REPO / "reports" / f"benign_{modality}_runs.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if rows and "carrier_score" not in rows[0]:
        print(f"  skipping {path.name}: rows predate the carrier arm")
        return []

    # One row per clip. Two runs of the same arm once appended to one file
    # concurrently, and the duplicates doubled every count and every interval
    # width - a rate stays right under duplication but its confidence
    # interval does not, which is the kind of error that reads as a stronger
    # result rather than a broken one.
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["clip_id"] in seen:
            continue
        seen.add(row["clip_id"])
        unique.append(row)
    if len(unique) != len(rows):
        print(f"  {path.name}: kept {len(unique)} of {len(rows)} rows "
              f"({len(rows) - len(unique)} duplicate clip_ids)")
    return [ModalityResult(**row) for row in unique]


def main() -> int:
    loaded = {m: load(m) for m in MODALITIES}
    missing = [m for m, rows in loaded.items() if not rows]
    if missing:
        print(f"\n  no rows for: {', '.join(missing)}"
              f"   (run eval/benign_media_run.py --modality <m>)")
    rows_by_modality = {m: r for m, r in loaded.items() if r}
    if not rows_by_modality:
        raise SystemExit("\n  Nothing to summarise.\n")

    print("\n" + "=" * 78)
    print("  FALSE POSITIVE RATE BY MODALITY - ordinary customer queries, "
          "no attacks")
    print("=" * 78)
    print("\n  Every row is a real BANKING77 or CLINC150 query, so any verdict"
          "\n  other than ALLOW is a false positive.\n")
    print(f"  {'arm':<10}{'n':>5}{'query alone':>22}"
          f"{'+ carrier, text':>22}{'+ carrier, as arm':>22}")
    print("  " + "-" * 74)

    for modality, rows in rows_by_modality.items():
        n = len(rows)
        alone = wilson_interval(sum(r.text_false_positive for r in rows), n)
        carrier = wilson_interval(
            sum(r.carrier_false_positive for r in rows), n)
        clip = wilson_interval(sum(r.clip_false_positive for r in rows), n)
        print(f"  {modality:<10}{n:>5}"
              f"{alone.rate:>14.2%} {'':>6}"
              f"{carrier.rate:>14.2%} {'':>6}"
              f"{clip.rate:>14.2%}")

    print("\n  with intervals:\n")
    for modality, rows in rows_by_modality.items():
        n = len(rows)
        print(f"    {modality}  (n = {n})")
        for label, flagged in (
            ("query alone      ", sum(r.text_false_positive for r in rows)),
            ("carrier + query  ", sum(r.carrier_false_positive for r in rows)),
            (f"as {modality:<14}", sum(r.clip_false_positive for r in rows)),
        ):
            print(f"      {label}{wilson_interval(flagged, n).summary()}")
        print()

    # --- the modality effect, per arm, never pooled ---
    print("  " + "-" * 74)
    print("  MODALITY EFFECT - carrier + query as text, versus as the arm")
    print("  " + "-" * 74)
    print("\n  Paired on the same queries, so the carrier is held constant and"
          "\n  only the delivery changes.\n")

    for modality, rows in rows_by_modality.items():
        both = sum(r.carrier_false_positive and r.clip_false_positive
                   for r in rows)
        text_only = sum(r.carrier_false_positive and not r.clip_false_positive
                        for r in rows)
        clip_only = sum(r.clip_false_positive and not r.carrier_false_positive
                        for r in rows)
        test = mcnemar_test(both, text_only, clip_only,
                            len(rows) - both - text_only - clip_only)
        caused = [r for r in rows
                  if r.clip_false_positive and not r.carrier_false_positive]
        print(f"    {modality:<8}fixed by the arm {text_only:>3}   "
              f"caused by the arm {clip_only:>3}   "
              f"p = {test.p_value:.4f}  "
              f"{'significant' if test.significant else 'n.s.'}")
        for row in caused:
            print(f"        {row.clip_id}  {row.carrier_score:.4f} -> "
                  f"{row.clip_score:.4f}")
            print(f"          query     {row.text[:76]!r}")
            print(f"          extracted {row.extracted_text[:76]!r}")

    # --- the carrier effect, pooled across arms ---
    print("\n  " + "-" * 74)
    print("  CARRIER EFFECT - query alone, versus carrier + query (both text)")
    print("  " + "-" * 74)

    pooled = [r for rows in rows_by_modality.values() for r in rows]
    both = sum(r.text_false_positive and r.carrier_false_positive
               for r in pooled)
    alone_only = sum(r.text_false_positive and not r.carrier_false_positive
                     for r in pooled)
    carrier_only = sum(r.carrier_false_positive and not r.text_false_positive
                       for r in pooled)
    test = mcnemar_test(both, alone_only, carrier_only,
                        len(pooled) - both - alone_only - carrier_only)
    print(f"\n    pooled n = {len(pooled)}")
    print(f"      query alone        "
          f"{wilson_interval(both + alone_only, len(pooled)).summary()}")
    print(f"      carrier + query    "
          f"{wilson_interval(both + carrier_only, len(pooled)).summary()}")
    print(f"      the carrier cleared {alone_only}, and caused "
          f"{carrier_only}")
    print(f"      McNemar p = {test.p_value:.6f}  "
          f"{'significant' if test.significant else 'not significant'} "
          f"at alpha = 0.05")

    # --- extraction health ---
    print("\n  " + "-" * 74)
    print("  EXTRACTION - whether there was anything to measure")
    print("  " + "-" * 74 + "\n")
    for modality, rows in rows_by_modality.items():
        ran = sum(r.extraction_ran for r in rows)
        empty = sum(1 for r in rows
                    if r.extraction_ran and not r.extracted_text.strip())
        # Containment, not equality. Video concatenates its soundtrack and its
        # frame text, so the extraction holds the query twice and can never
        # equal it; audio re-cases and re-punctuates everything. Equality
        # would report 0% for a channel that recovered the words perfectly.
        def recovered(row) -> bool:
            want = " ".join(row.text.lower().split())
            got = " ".join(row.extracted_text.lower().split())
            return bool(want) and want in got

        found = sum(recovered(r) for r in rows)
        print(f"    {modality:<8}ran {ran}/{len(rows)}   "
              f"empty {empty}   "
              f"query recovered verbatim {found}/{len(rows)} "
              f"({found / len(rows):.1%})")

    print("\n" + "=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
