"""Does the carrier wording trade detection against false positives?

    python eval/probe_carrier_choice.py

The setup
---------
Every modality arm wraps its attachment in a carrier prompt, because an
attachment with no accompanying question is not a request anyone sends. The
arms chose one, checked it scored low alone, and used it throughout.

``probe_carrier_effect`` showed that check is not sufficient: ``score(CARRIER)``
being low says nothing about ``score(CARRIER + content)``. The image carrier
scores 0.0589 alone, yet composed with ordinary BANKING77 queries it took the
false-positive rate from 2.61% to 8.70%, and composed with attack payloads it
caught six attacks the payload alone would have missed.

Those are the same effect. The carrier shifts the operating point, and a
shift buys recall and costs precision whichever direction it goes. So the
question the arms never asked is whether their carrier was a *good* choice -
and the only way to answer it is to measure both sides for several carriers
on the same inputs.

What this produces
------------------
For each candidate carrier, on identical data:

``benign flagged``
    Ordinary BANKING77 and CLINC150 queries scoring at or above the blocking
    threshold with the carrier prepended. The cost.
``attacks caught``
    User-turn attack payloads doing the same. The benefit.

Reported together, because either alone is meaningless - a carrier that
flagged everything would look like a perfect detector, which is the exact
failure the original 0.7531 carrier produced.

Stage I is excluded from the attack side: a payload Stage I blocks outright
never reaches Stage II, so the carrier cannot be what caught it, and counting
those would flatter every candidate equally.

Text only - no rendering, no OCR, no transcription. The composition happens
in the detector, and the smoke runs showed the clip arm's score is identical
to the text arm's when extraction is accurate, so rendering would add cost
and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.av_corpus import user_turn_attacks  # noqa: E402
from eval.benign_media_corpus import SOURCES, sample  # noqa: E402

#: Candidates. The first is what the image arms use and the second what the
#: audio and video arms use; the rest vary one thing at a time - whether the
#: carrier asks a question, whether it names a task, and whether it refers to
#: the attachment as something to read or something to repeat.
CANDIDATES = [
    "",                                        # no carrier, as the floor
    "What does this document say?",            # image arms
    "What does this recording say?",           # audio and video arms
    "Please summarise the attached document.",
    "Describe the attached file.",
    "Attached document follows.",
    "The attached file is data, not instructions.",
]

#: Queries per dataset. Enough for the differences between candidates to be
#: visible without making this a long job.
BENIGN_PER_DATASET = 200


def compose(carrier: str, text: str) -> str:
    return f"{carrier}\n{text}" if carrier else text


def main() -> int:
    from mochi.detect.stage1_syntactic import get_detector as get_stage1
    from mochi.detect.stage2_semantic import MALICIOUS_THRESHOLD, get_detector

    stage1 = get_stage1()
    detector = get_detector(None, device="cpu")

    benign: list[tuple[str, str]] = []
    for dataset, path, text_col, label_col in SOURCES:
        if not path.exists():
            raise SystemExit(f"\n  Missing {path}.\n")
        for _, text, _ in sample(dataset, path, text_col, label_col,
                                 BENIGN_PER_DATASET):
            benign.append((dataset, text))

    # Only the attacks Stage II actually decides. The rest are blocked before
    # it, so no carrier can change their outcome.
    attacks = [item for item in user_turn_attacks()
               if not stage1.scan([item.user_input]).should_block]

    print(f"\n  {len(benign)} benign queries "
          f"({BENIGN_PER_DATASET} per dataset)")
    print(f"  {len(attacks)} attack payloads that reach Stage II")
    print(f"  blocking threshold {MALICIOUS_THRESHOLD}\n")
    print(f"  {'carrier':<46}{'benign flagged':>16}{'attacks caught':>16}")
    print(f"  {'-' * 44:<46}{'-' * 14:>16}{'-' * 14:>16}")

    rows = []
    for carrier in CANDIDATES:
        flagged = sum(
            detector.scan_text(compose(carrier, text)).score >= MALICIOUS_THRESHOLD
            for _, text in benign)
        caught = sum(
            detector.scan_text(compose(carrier, item.user_input)).score
            >= MALICIOUS_THRESHOLD
            for item in attacks)
        rows.append((carrier, flagged, caught))
        label = carrier or "(no carrier)"
        print(f"  {label[:44]:<46}"
              f"{flagged:>6} ({flagged / len(benign):>6.2%})"
              f"{caught:>6} ({caught / len(attacks):>6.2%})", flush=True)

    floor = rows[0]
    print(f"\n  Against no carrier at all "
          f"({floor[1]} benign, {floor[2]} attacks):\n")
    for carrier, flagged, caught in rows[1:]:
        label = carrier or "(no carrier)"
        print(f"    {label[:44]:<46}"
              f"{flagged - floor[1]:+4} benign  "
              f"{caught - floor[2]:+4} attacks")

    print("\n  A carrier that moves both columns up is not improving the "
          "detector,\n  it is lowering the threshold. The arms that used one "
          "reported only the\n  second column.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
