"""Dataset audit: what needs cleaning, and what it costs.

    python eval/audit_datasets.py --data data/
    python eval/audit_datasets.py --data data/ --json reports/audit.json

Reports the things that silently corrupt results if ignored:

* exact and near-duplicate counts, within and *across* datasets (train/test leakage)
* length distribution against the Stage II 512-token window
* where in the text the attack signal sits (head/middle/tail) - which decides
  whether head-truncation destroys the payload
* whitespace, control-character, and encoding artifacts

This is measurement only; it never modifies the data. Cleaning decisions belong
to :mod:`eval.clean_datasets`, which acts on what this reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data_loading import DATA_DIR, DatasetError, Sample, load_directory  # noqa: E402
from mochi.detect import get_detector  # noqa: E402

#: E5 models cap at 512 tokens. Chars-per-token is ~4 for English; this is a
#: heuristic until a real tokenizer is available in Phase 8.
CHARS_PER_TOKEN = 4
STAGE2_TOKEN_LIMIT = 512
STAGE2_CHAR_LIMIT = STAGE2_TOKEN_LIMIT * CHARS_PER_TOKEN

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalized_key(text: str) -> str:
    """Aggressive normalization for near-duplicate detection only.

    Never used on data that reaches a detector - it deliberately destroys
    signal (casing, punctuation) to make trivial variants collide.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = _PUNCT.sub(" ", folded)
    return _WS.sub(" ", folded).strip()


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def percentile(values: Sequence[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


@dataclass
class DatasetStats:
    name: str
    n: int = 0
    benign: int = 0
    malicious: int = 0
    lengths: list[int] = field(default_factory=list)
    exact_dupes: int = 0
    near_dupes: int = 0
    empty_or_tiny: int = 0
    over_limit: int = 0
    control_chars: int = 0
    excess_whitespace: int = 0
    html_entities: int = 0
    non_latin: int = 0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "benign": self.benign,
            "malicious": self.malicious,
            "median_chars": percentile(self.lengths, 0.50),
            "p95_chars": percentile(self.lengths, 0.95),
            "max_chars": max(self.lengths) if self.lengths else 0,
            "est_median_tokens": percentile(self.lengths, 0.50) // CHARS_PER_TOKEN,
            "exact_duplicates": self.exact_dupes,
            "near_duplicates": self.near_dupes,
            "empty_or_tiny": self.empty_or_tiny,
            "over_512_tokens": self.over_limit,
            "control_chars": self.control_chars,
            "excess_whitespace": self.excess_whitespace,
            "html_entities": self.html_entities,
            "non_latin_script": self.non_latin,
        }


HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{2,5});")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_WS = re.compile(r"\s{3,}")


def audit(samples: Sequence[Sample]) -> tuple[dict[str, DatasetStats], dict]:
    per_dataset: dict[str, DatasetStats] = defaultdict(lambda: DatasetStats(name=""))
    seen_exact: dict[str, str] = {}
    seen_near: dict[str, str] = {}
    cross_exact: Counter = Counter()
    cross_near: Counter = Counter()
    label_conflicts = 0
    near_label_map: dict[str, int] = {}

    for sample in samples:
        stats = per_dataset[sample.dataset]
        stats.name = sample.dataset
        stats.n += 1
        stats.benign += sample.label == 0
        stats.malicious += sample.label == 1

        text = sample.text
        length = len(text)
        stats.lengths.append(length)

        if length < 10:
            stats.empty_or_tiny += 1
        if length > STAGE2_CHAR_LIMIT:
            stats.over_limit += 1
        if CONTROL_CHARS.search(text):
            stats.control_chars += 1
        if MULTI_WS.search(text):
            stats.excess_whitespace += 1
        if HTML_ENTITY.search(text):
            stats.html_entities += 1
        if any(ord(c) > 0x24F for c in text[:400]):
            stats.non_latin += 1

        exact = digest(text)
        if exact in seen_exact:
            stats.exact_dupes += 1
            origin = seen_exact[exact]
            if origin != sample.dataset:
                cross_exact[tuple(sorted((origin, sample.dataset)))] += 1
        else:
            seen_exact[exact] = sample.dataset

        near = digest(normalized_key(text))
        if near in seen_near:
            stats.near_dupes += 1
            origin = seen_near[near]
            if origin != sample.dataset:
                cross_near[tuple(sorted((origin, sample.dataset)))] += 1
            if near_label_map.get(near) not in (None, sample.label):
                label_conflicts += 1
        else:
            seen_near[near] = sample.dataset
            near_label_map[near] = sample.label

    summary = {
        "cross_dataset_exact": {f"{a} <-> {b}": n for (a, b), n in cross_exact.items()},
        "cross_dataset_near": {f"{a} <-> {b}": n for (a, b), n in cross_near.items()},
        "label_conflicts": label_conflicts,
    }
    return dict(per_dataset), summary


def signal_position(samples: Sequence[Sample], limit: int = 4000) -> dict:
    """Where in a malicious sample does Stage I find the payload?

    Decides truncation strategy: if attacks cluster in the tail, keeping only
    the first 512 tokens discards them.
    """
    detector = get_detector()
    buckets = Counter()
    checked = matched = 0

    for sample in samples:
        if sample.label != 1 or len(sample.text) < 200:
            continue
        checked += 1
        if checked > limit:
            break
        result = detector.scan([sample.text])
        top = result.highest
        if top is None or not top.matched_text:
            continue
        index = sample.text.find(top.matched_text)
        if index < 0:
            continue
        matched += 1
        ratio = index / max(len(sample.text), 1)
        if ratio < 0.33:
            buckets["head"] += 1
        elif ratio < 0.66:
            buckets["middle"] += 1
        else:
            buckets["tail"] += 1

    return {
        "checked": checked,
        "matched": matched,
        "head": buckets["head"],
        "middle": buckets["middle"],
        "tail": buckets["tail"],
    }


def print_report(stats: dict[str, DatasetStats], summary: dict, positions: dict) -> None:
    print()
    print("=" * 96)
    print("  Dataset Audit")
    print("=" * 96)
    print()
    header = (f"{'Dataset':<26}{'N':>9}{'median':>9}{'p95':>9}{'>512tok':>10}"
              f"{'exact dup':>11}{'near dup':>10}{'tiny':>7}")
    print(header)
    print("-" * 96)
    for name in sorted(stats):
        s = stats[name]
        d = s.as_dict()
        print(f"{name:<26}{d['n']:>9,}{d['median_chars']:>9,}{d['p95_chars']:>9,}"
              f"{d['over_512_tokens']:>10,}{d['exact_duplicates']:>11,}"
              f"{d['near_duplicates']:>10,}{d['empty_or_tiny']:>7,}")
    print("-" * 96)

    print()
    print("  Text artifacts")
    print(f"  {'Dataset':<26}{'control chars':>15}{'excess ws':>12}{'html entities':>15}{'non-latin':>11}")
    for name in sorted(stats):
        d = stats[name].as_dict()
        print(f"  {name:<26}{d['control_chars']:>15,}{d['excess_whitespace']:>12,}"
              f"{d['html_entities']:>15,}{d['non_latin_script']:>11,}")

    print()
    print("  Cross-dataset overlap  (LEAKAGE RISK)")
    if not summary["cross_dataset_exact"] and not summary["cross_dataset_near"]:
        print("    none detected")
    for pair, n in summary["cross_dataset_exact"].items():
        print(f"    exact  {pair:<50}{n:>8,}")
    for pair, n in summary["cross_dataset_near"].items():
        print(f"    near   {pair:<50}{n:>8,}")

    print()
    print(f"  Label conflicts (same text, different label): {summary['label_conflicts']:,}")

    print()
    print("  Attack-signal position in long malicious samples")
    total = positions["matched"] or 1
    print(f"    sampled {positions['checked']:,}, located {positions['matched']:,}")
    for bucket in ("head", "middle", "tail"):
        n = positions[bucket]
        print(f"    {bucket:<8}{n:>8,}  ({n / total:>6.1%})")
    print()
    print("=" * 96)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    try:
        samples = load_directory(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    stats, summary = audit(samples)
    positions = signal_position(samples)
    print_report(stats, summary, positions)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "datasets": {k: v.as_dict() for k, v in stats.items()},
                    "overlap": summary,
                    "signal_position": positions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
