"""Measure the false-positive rate on ordinary traffic.

The attack simulation answers "how many attacks get through". It cannot answer
"how many legitimate users get refused", because its corpus is attacks only.
This is the other half, and without it a mitigation rate means very little: a
gateway that blocks everything scores 100%.

**No LLM is called.** To know whether a legitimate request was refused you only
need MOCHI's verdict, not the model's answer. So this drives the detection
pipeline directly - normalization, Stage I, Stage II, trust rules, enforcement -
and skips the upstream call. That turns 8,580 requests from hours into minutes
and removes the target model as a confound entirely.

Two datasets, kept separate on purpose:

===========  =====  ==============================================
BANKING77    3,080  one domain, matching the Acme Bank scenario
CLINC150     5,500  ten domains, 150 intents - does it generalise?
===========  =====  ==============================================

Pooling them would weight the result 64/36 toward CLINC for no principled
reason, and would hide whichever one is worse. They are reported separately.

Each query is sent exactly as a customer would send it: the same system prompt
the attack simulation uses, the query as ``user_input``, no attached document.
Same scenario, same thresholds, same build - so the false-positive rate and the
attack success rate describe one operating point rather than two.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.attack_corpus import SYSTEM_PROMPT

REPO = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO / "reports" / "benign_runs.jsonl"

#: ``(dataset name, csv path, column holding the query, column holding a label)``
#: The label column is the intent, kept so a false positive can be attributed to
#: a kind of request rather than only counted.
SOURCES = [
    ("banking77", REPO / "data" / "external" / "banking77_test.csv", "text", "label"),
    ("clinc150", REPO / "data" / "external" / "clinc150_test.csv", "text", "intent"),
]


@dataclass
class BenignResult:
    """One ordinary request, and what MOCHI decided about it."""

    dataset: str
    row_id: int
    intent: str
    text: str

    decision: str
    """ALLOW | BLOCK | SANITIZE. Anything but ALLOW is a false positive: every
    row in these datasets is a real customer query."""

    false_positive: bool
    reason: str = ""

    stage1_outcome: str | None = None
    stage2_outcome: str | None = None
    semantic_score: float | None = None
    stopped_by: str | None = None
    spans_redacted: int = 0
    latency_ms: float | None = None


def load_rows(path: Path, text_col: str, label_col: str) -> list[tuple[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\n"
            "Fetch it first - see docs/NEXT_STEPS.md, Part B."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return [(r[text_col], str(r.get(label_col, ""))) for r in csv.DictReader(handle)]


def classify_stopped_by(result, verdict) -> str | None:
    """Which layer produced a non-ALLOW verdict.

    Stage I short-circuits Stage II, so this is first-responder attribution.
    A low Stage II count means Stage I matched first, not that Stage II was
    quiet.
    """
    if verdict.decision.value == "allow":
        return None
    for _, stage1 in result.stage1:
        if stage1.should_block:
            return "stage1"
    if getattr(result, "stage2_blocked", False):
        return "stage2"
    if getattr(result, "stage2_uncertain", False):
        return "stage2_band"
    if getattr(result, "session_escalated", False):
        return "session"
    return "other"


def run_one(text: str, intent: str, dataset: str, row_id: int, *, stage2) -> BenignResult:
    from mochi.detect.pipeline import inspect
    from mochi.gateway.models import ChatCompletionRequest
    from mochi.mitigate.sanitizer import decide
    from mochi.telemetry.schema import TelemetryRecord

    request = ChatCompletionRequest.model_validate({
        "model": "benign-probe",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "context": {"system_prompt": SYSTEM_PROMPT, "user_input": text},
    })
    record = TelemetryRecord()

    started = time.perf_counter()
    result = inspect(request, record, enable_stage1=True,
                     enable_stage2=stage2 is not None, stage2=stage2)
    verdict = decide(result)
    elapsed = (time.perf_counter() - started) * 1000

    detection = record.detection_results
    decision = verdict.decision.value.upper()
    return BenignResult(
        dataset=dataset, row_id=row_id, intent=intent, text=text[:400],
        decision=decision,
        false_positive=decision != "ALLOW",
        reason=verdict.reason[:200],
        stage1_outcome=detection.stage_1_syntactic,
        stage2_outcome=detection.stage_2_semantic,
        semantic_score=detection.semantic_score,
        stopped_by=classify_stopped_by(result, verdict),
        spans_redacted=len(verdict.targets),
        latency_ms=elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="first N rows per dataset, for a smoke test")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Stage II weights to evaluate (default: the "
                             "shipped models/e5-fine-tuned). Pass an "
                             "alternative to compare two models on identical "
                             "inputs.")
    parser.add_argument("--device", default="cpu",
                        help="torch device for Stage II (default cpu: the GPU "
                             "is usually holding a target model)")
    parser.add_argument("--no-stage2", action="store_true",
                        help="Stage I only, as an ablation arm")
    args = parser.parse_args()

    stage2 = None
    if not args.no_stage2:
        from mochi.detect.stage2_semantic import get_detector
        stage2 = get_detector(args.model_dir, device=args.device)
        stage2.scorer.score(["warmup"])  # surface load errors before the loop

    sources = [s for s in SOURCES
               if args.datasets is None or s[0] in args.datasets]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    totals: dict[str, list[int]] = {}
    with args.out.open("w", encoding="utf-8") as handle:
        for name, path, text_col, label_col in sources:
            rows = load_rows(path, text_col, label_col)
            if args.limit:
                rows = rows[:args.limit]
            print(f"\n  {name}: {len(rows):,} ordinary queries", flush=True)
            flagged = 0
            started = time.perf_counter()
            for index, (text, intent) in enumerate(rows):
                result = run_one(text, intent, name, index, stage2=stage2)
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                flagged += result.false_positive
                if (index + 1) % 250 == 0 or index + 1 == len(rows):
                    rate = flagged / (index + 1)
                    elapsed = time.perf_counter() - started
                    eta = (len(rows) - index - 1) * elapsed / (index + 1)
                    print(f"      {index + 1:>5,}/{len(rows):,}   "
                          f"flagged {flagged:>4}  ({rate:.2%})   "
                          f"{eta / 60:.0f}m left", flush=True)
            handle.flush()
            totals[name] = [flagged, len(rows)]

    print("\n" + "=" * 62)
    print("  FALSE POSITIVE RATE - ordinary customer queries, no attacks")
    print("=" * 62)
    for name, (flagged, total) in totals.items():
        print(f"  {name:<14}{flagged:>6,} of {total:>6,}   {flagged / total:>7.2%}")
    print("=" * 62)
    print(f"\n  written -> {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
