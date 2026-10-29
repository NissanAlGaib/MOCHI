"""Does the carrier prompt do work the attack arms credited to the defence?

    python eval/probe_carrier_effect.py

Why ask
-------
The image, audio and video attack arms each wrap their attachment in a short
carrier prompt - "What does this document say?" - because an attachment with no
accompanying question is not a request anyone would send. Each harness guards
that choice by asserting the carrier scores below ``MAX_CARRIER_SCORE`` on its
own, after an earlier carrier scored 0.7531 and made every request refused,
reporting a perfect mitigation rate while measuring nothing.

The benign per-modality arms then found that guard is not sufficient. Over 115
ordinary BANKING77 queries, none of them attacks:

    ==============================  =======
    query alone                       2.61%
    carrier + query, as text          8.70%
    ==============================  =======

Both strings are benign and the composition blocks. ``score(CARRIER)`` being
low therefore says nothing about ``score(CARRIER + content)``, which is what
the arms actually submit.

So this asks the question that matters for the numbers already reported: is
there any attack the arms counted as mitigated which **would not have been
caught without the carrier**? Such an attack was stopped by the composition
rather than by the payload, and crediting the defence for it overstates the
result.

How it answers
--------------
For every attack the modality arms use, score the payload alone and with each
carrier, and look for the payloads that cross ``MALICIOUS_THRESHOLD`` only
with the carrier attached. Stage I is checked first: a payload Stage I blocks
outright never reaches Stage II, so the carrier cannot be doing the work there
and it is excluded from the count.

No gateway and no LLM - this is a property of the detector, so it is measured
directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.av_corpus import user_turn_attacks  # noqa: E402
from eval.benign_media_run import CARRIER  # noqa: E402


def main() -> int:
    from mochi.detect.stage1_syntactic import get_detector as get_stage1
    from mochi.detect.stage2_semantic import (MALICIOUS_THRESHOLD,
                                              get_detector)

    stage1 = get_stage1()
    detector = get_detector(None, device="cpu")
    attacks = user_turn_attacks()
    carriers = sorted(set(CARRIER.values()))

    print(f"\n  {len(attacks)} user-turn attacks, "
          f"{len(carriers)} distinct carrier(s)")
    print(f"  Stage II blocking threshold: {MALICIOUS_THRESHOLD}\n")

    for carrier in carriers:
        print(f"  carrier {carrier!r}")
        print(f"    scores {detector.scan_text(carrier).score:.4f} alone\n")

        stage1_caught = 0
        alone_caught = 0
        carrier_rescued: list[tuple[str, float, float]] = []
        missed_by_both: list[tuple[str, float, float]] = []

        for item in attacks:
            payload = item.user_input
            if stage1.scan([payload]).should_block:
                stage1_caught += 1
                continue

            alone = detector.scan_text(payload).score
            composed = detector.scan_text(f"{carrier}\n{payload}").score

            if alone >= MALICIOUS_THRESHOLD:
                alone_caught += 1
            elif composed >= MALICIOUS_THRESHOLD:
                carrier_rescued.append((item.id, alone, composed))
            else:
                missed_by_both.append((item.id, alone, composed))

        reaching_stage2 = len(attacks) - stage1_caught
        print(f"    blocked by Stage I alone      {stage1_caught:>3}"
              f"  (carrier is irrelevant to these)")
        print(f"    reaching Stage II             {reaching_stage2:>3}")
        print(f"      caught by the payload alone {alone_caught:>3}")
        print(f"      caught only with carrier    {len(carrier_rescued):>3}"
              f"  <-- credited to the defence, done by the composition"
              if carrier_rescued else
              f"      caught only with carrier    {len(carrier_rescued):>3}")
        print(f"      missed either way           {len(missed_by_both):>3}")

        for attack_id, alone, composed in carrier_rescued:
            print(f"        {attack_id:<10}{alone:.4f} -> {composed:.4f}")

        if not carrier_rescued:
            print("\n    No attack depended on the carrier to be caught, so "
                  "the mitigation\n    rates these arms report are not "
                  "inflated by it. The benign\n    false-positive rate still "
                  "is - see eval/benign_media_run.py.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
