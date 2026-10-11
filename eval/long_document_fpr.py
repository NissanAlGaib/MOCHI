"""Do false positives grow with document length?

    python eval/long_document_fpr.py
    python eval/long_document_fpr.py --per-length 200

Every false-positive number in this evaluation comes from short utterances.
BANKING77 and CLINC150 are one-line customer queries; the sealed split is
single prompts. Nothing measures what happens to a *long* benign document, and
the architecture gives a specific reason to expect it is worse.

The mechanism
-------------
Stage II scores a segment as overlapping 2,048-character windows and keeps the
**maximum**, not the mean (``SemanticScore`` in stage2_semantic.py). Windows
stride 1,536 characters, so length buys windows: ~2 KB is one window, ~12.8 KB
is eight, ~49.7 KB is thirty-two, where ``MAX_WINDOWS`` caps the scan.

Max-pooling is right for detection - a payload buried in one paragraph of a
long document must not be averaged away by the surrounding text. But it is
one-sided. If any single window of a benign document trips the threshold, the
whole document is flagged, so the chance of a false positive compounds with
length rather than staying flat. At a per-window false-positive rate ``p``, a
document of ``n`` windows should flag at roughly ``1 - (1 - p)^n`` if windows
fail independently. At p = 0.013 that is 1.3% for one window and about 10% for
eight.

This measures whether that happens, and how closely the independence model
holds. Divergence is as informative as the rate: much lower means a benign
document's windows fail together rather than independently.

The documents
-------------
Real benign text, composed rather than written. Each document is customer
queries drawn from BANKING77 and CLINC150, formatted as a numbered FAQ or
ticket export, which is a genuine shape for a retrieved document and is how
such a page would actually read. Nothing is invented: inventing prose would
measure the generator, and these sources are the ones whose vocabulary already
drives false positives.

Documents are tagged ``retrieved_document`` so they travel the untrusted path,
which is where a long document arrives in a RAG application and where
enforcement would redact rather than refuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SOURCES = [
    REPO / "data" / "external" / "banking77_test.csv",
    REPO / "data" / "external" / "clinc150_test.csv",
]

#: Target sizes, chosen so each is about one more doubling of Stage II windows:
#: 1, 2, 4, 8, 16 and 32 windows at a 2,048-char window and 1,536-char stride.
#: 200 chars is the control - the length every existing FPR figure was measured
#: at.
TARGET_CHARS = [200, 2048, 3584, 6656, 12800, 25088, 49664]

HEADERS = [
    "Customer support knowledge base - frequently asked questions",
    "Support ticket export - resolved queries",
    "Help centre article index",
    "Branch enquiry log",
]


def load_utterances() -> list[str]:
    """Real benign one-liners from both sources, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for path in SOURCES:
        if not path.exists():
            raise SystemExit(f"ERROR: {path} not found.")
        with path.open(encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                text = (record.get("text") or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    out.append(text)
    return out


def compose(utterances: list[str], target: int, rng: random.Random) -> str:
    """A document of about ``target`` characters, built from real queries."""
    parts = [rng.choice(HEADERS), ""]
    size = len(parts[0]) + 1
    index = 1
    while size < target:
        line = f"{index}. {rng.choice(utterances)}"
        parts.append(line)
        size += len(line) + 1
        index += 1
    return "\n".join(parts)[:target]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-length", type=int, default=100,
                        help="documents per length (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dir", default=None,
                        help="Stage II model (default: models/e5-fine-tuned)")
    parser.add_argument("--out", type=Path,
                        default=REPO / "reports" / "long_document_fpr.json")
    args = parser.parse_args()

    from mochi.detect.stage2_semantic import (
        BENIGN_THRESHOLD, MALICIOUS_THRESHOLD, MAX_WINDOWS, WINDOW_CHARS,
        WINDOW_OVERLAP_CHARS, get_detector)

    utterances = load_utterances()
    print(f"\n{len(utterances):,} distinct benign utterances from "
          f"{len(SOURCES)} sources")

    # scan_text, not scorer.score. The windowing lives in Stage2Detector;
    # E5Scorer.score scores whatever single text it is handed and the tokenizer
    # truncates at 512 tokens. Calling the scorer directly would silently
    # measure only the first ~2 KB of every document and report a flat line.
    detector = get_detector(args.model_dir, device=args.device)
    stride = WINDOW_CHARS - WINDOW_OVERLAP_CHARS
    # 0.45 is the operative line for an untrusted document: at or above it the
    # score has left the clean band, and with resolve_band_by_trust an
    # untrusted segment in the 0.45-0.55 band is acted on rather than passed.
    # 0.55 blocks outright regardless of origin, so both are reported.
    print(f"window {WINDOW_CHARS} chars, stride {stride}; "
          f"flagged at >= {BENIGN_THRESHOLD}, blocked at >= {MALICIOUS_THRESHOLD}")

    rng = random.Random(args.seed)
    results = []

    print(f"\n{'chars':>7}{'windows':>9}{'docs':>6}{'flagged':>9}"
          f"{'flag':>7}{'docFPR':>8}{'perWindow':>11}{'pred':>8}"
          f"{'blockFPR':>9}{'mean':>8}")
    print("-" * 79)

    for target in TARGET_CHARS:
        docs = [compose(utterances, target, rng) for _ in range(args.per_length)]
        scores: list[float] = []
        window_counts: list[int] = []
        truncated = 0
        # Windows flagged individually, over all documents. This measures the
        # per-window rate directly instead of inferring it from the document
        # rate, which is what makes the independence question answerable.
        windows_total = 0
        windows_flagged = 0
        started = time.perf_counter()
        for doc in docs:
            result = detector.scan_text(doc)
            scores.append(float(result.score))
            window_counts.append(len(result.chunk_scores))
            truncated += 1 if result.truncated else 0
            windows_total += len(result.chunk_scores)
            windows_flagged += sum(1 for cs in result.chunk_scores
                                   if cs.score >= BENIGN_THRESHOLD)
        elapsed = time.perf_counter() - started

        # Observed, not derived - MAX_WINDOWS caps the scan and the last
        # window may be short.
        windows = round(statistics.fmean(window_counts))
        flagged = sum(1 for s in scores if s >= BENIGN_THRESHOLD)
        blocked = sum(1 for s in scores if s >= MALICIOUS_THRESHOLD)
        fpr = flagged / len(scores)
        per_window = windows_flagged / windows_total if windows_total else 0.0
        # What the document rate would be if a document's windows failed
        # independently at the rate just measured. Divergence says the windows
        # are correlated; agreement says the compounding is simply arithmetic.
        predicted = 1 - (1 - per_window) ** windows

        results.append({
            "target_chars": target, "windows": windows,
            "documents": len(docs), "flagged": flagged, "fpr": fpr,
            "blocked": blocked, "block_fpr": blocked / len(scores),
            "windows_total": windows_total, "windows_flagged": windows_flagged,
            "per_window_fpr": per_window,
            "predicted_fpr": predicted, "truncated_docs": truncated,
            "max_score": max(scores), "mean_score": statistics.fmean(scores),
            "seconds": round(elapsed, 1),
        })
        print(f"{target:>7,}{windows:>8}{len(docs):>6}{flagged:>7}{fpr:>8.3f}"
              f"{per_window:>11.4f}{predicted:>8.3f}"
              f"{blocked/len(scores):>9.3f}{statistics.fmean(scores):>8.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "benign_threshold": BENIGN_THRESHOLD,
        "malicious_threshold": MALICIOUS_THRESHOLD,
        "window_chars": WINDOW_CHARS, "stride": stride,
        "per_length": args.per_length, "seed": args.seed,
        "model_dir": args.model_dir or "models/e5-fine-tuned",
        "rows": results,
    }, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
