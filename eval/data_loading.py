"""Dataset loading and normalization.

Public prompt-injection datasets disagree on nearly everything: column names,
label encodings, file formats. This module normalizes whatever you downloaded
into one :class:`Sample` schema so the rest of the harness never has to care.

Supported out of the box: PromptShield, ``deepset/prompt-injections``,
``jayavibhav/prompt-injection``, and any CSV/JSONL/JSON/Parquet with a text
column and a binary label column - detected automatically where possible.

Label convention throughout MOCHI: **0 = benign, 1 = malicious**, matching the
thesis "Label Encoding" preprocessing step.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

#: Column names that plausibly hold the prompt text, best guess first.
TEXT_COLUMN_CANDIDATES = (
    "text", "prompt", "input", "content", "question", "instruction",
    "user_input", "query", "sentence", "message",
)

#: Column names that plausibly hold the label.
LABEL_COLUMN_CANDIDATES = (
    "label", "labels", "is_injection", "injection", "malicious", "target",
    "class", "category", "is_malicious", "toxic", "jailbreak", "y",
)

#: Every label spelling seen across public datasets, mapped to 0/1.
LABEL_ALIASES: dict[str, int] = {
    # benign
    "0": 0, "benign": 0, "safe": 0, "legitimate": 0, "clean": 0, "normal": 0,
    "false": 0, "no": 0, "negative": 0, "not_injection": 0, "valid": 0,
    # malicious
    "1": 1, "malicious": 1, "injection": 1, "prompt_injection": 1, "unsafe": 1,
    "attack": 1, "adversarial": 1, "jailbreak": 1, "true": 1, "yes": 1,
    "positive": 1, "harmful": 1,
}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be loaded or interpreted."""


@dataclass(frozen=True)
class Sample:
    """One labeled evaluation example."""

    text: str
    label: int
    """0 = benign, 1 = malicious."""
    dataset: str
    source_tag: str = "user_input"
    """Which source tag to present this as. Benchmark prompts are user-supplied
    by default; indirect-injection fixtures should be loaded as
    ``retrieved_document`` / ``web_content`` so they exercise the untrusted path."""
    attack_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_label(value: Any) -> int | None:
    """Map any known label spelling to 0/1, or ``None`` if unrecognized."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        as_int = int(value)
        return as_int if as_int in (0, 1) else None
    key = str(value).strip().lower()
    return LABEL_ALIASES.get(key)


def _detect_column(available: Sequence[str], candidates: Sequence[str],
                   kind: str, path: Path) -> str:
    lowered = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise DatasetError(
        f"Could not find a {kind} column in {path.name}. "
        f"Columns present: {list(available)}. "
        f"Pass {kind}_column=... explicitly, or rename the column to one of: "
        f"{', '.join(candidates[:5])}."
    )


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a data file into row dicts plus the column order."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text("utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("data", [])
    elif suffix in {".csv", ".tsv", ".parquet"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DatasetError(f"pandas is required to read {suffix} files") from exc
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        rows = frame.to_dict(orient="records")
    else:
        raise DatasetError(
            f"Unsupported file type {suffix!r} for {path.name}. "
            "Supported: .csv, .tsv, .jsonl, .json, .parquet"
        )

    if not rows:
        raise DatasetError(f"{path.name} contains no rows")
    return rows, list(rows[0].keys())


def load_file(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    source_tag: str = "user_input",
    invert_labels: bool = False,
) -> list[Sample]:
    """Load one dataset file into :class:`Sample` objects.

    Rows whose label cannot be interpreted are skipped rather than guessed at -
    silently mislabeling evaluation data would corrupt every downstream metric.

    Args:
        invert_labels: For datasets that encode 1 = safe (rare, but it happens).
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"No such dataset file: {path}")

    rows, columns = _read_rows(path)
    name = dataset_name or path.stem
    text_col = text_column or _detect_column(columns, TEXT_COLUMN_CANDIDATES, "text", path)
    label_col = label_column or _detect_column(columns, LABEL_COLUMN_CANDIDATES, "label", path)

    samples: list[Sample] = []
    skipped = 0
    for row in rows:
        text = row.get(text_col)
        label = normalize_label(row.get(label_col))
        if not isinstance(text, str) or not text.strip() or label is None:
            skipped += 1
            continue
        if invert_labels:
            label = 1 - label
        samples.append(
            Sample(
                text=text,
                label=label,
                dataset=name,
                source_tag=source_tag,
                attack_type=row.get("attack_type") or row.get("type"),
            )
        )

    if not samples:
        raise DatasetError(
            f"{path.name}: no usable rows (text column {text_col!r}, "
            f"label column {label_col!r}). Check the label values are recognizable: "
            f"{sorted(set(str(r.get(label_col)) for r in rows[:20]))}"
        )
    if skipped:
        print(f"  note: skipped {skipped} unusable row(s) in {path.name}")

    return samples


def load_directory(directory: str | Path = DATA_DIR, **kwargs) -> list[Sample]:
    """Load every supported data file in a directory.

    Each file becomes its own ``dataset`` name, so per-source breakdowns stay
    available in the report.
    """
    directory = Path(directory)
    if not directory.exists():
        raise DatasetError(
            f"Data directory not found: {directory}\n"
            f"Create it and add your dataset files:\n"
            f"  mkdir data\n"
            f"  # then place e.g. promptshield.csv, deepset.csv, jayavibhav.csv there\n"
            f"See docs/EVALUATION.md for where to download each one."
        )

    patterns = ("*.csv", "*.tsv", "*.jsonl", "*.json", "*.parquet")
    files = sorted(f for pattern in patterns for f in directory.glob(pattern))
    if not files:
        raise DatasetError(f"No dataset files found in {directory}")

    samples: list[Sample] = []
    for file in files:
        try:
            loaded = load_file(file, **kwargs)
        except DatasetError as exc:
            print(f"  WARNING: skipping {file.name}: {exc}")
            continue
        print(f"  loaded {len(loaded):>6,} samples from {file.name}")
        samples.extend(loaded)

    if not samples:
        raise DatasetError(f"No dataset file in {directory} could be loaded")
    return samples


def load_huggingface(name: str, *, split: str = "train", **kwargs) -> list[Sample]:
    """Load directly from the Hugging Face hub.

    Requires ``pip install datasets``. Offered as a convenience; downloading to
    ``data/`` and using :func:`load_file` is more reproducible for a thesis
    because the exact snapshot is preserved.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetError(
            "The 'datasets' package is required for hub loading. "
            "Either `pip install datasets`, or download the files to data/ "
            "and use load_file() - which is more reproducible."
        ) from exc

    hub_dataset = load_dataset(name, split=split)
    rows = [dict(row) for row in hub_dataset]
    columns = list(rows[0].keys()) if rows else []

    text_col = kwargs.get("text_column") or _detect_column(
        columns, TEXT_COLUMN_CANDIDATES, "text", Path(name)
    )
    label_col = kwargs.get("label_column") or _detect_column(
        columns, LABEL_COLUMN_CANDIDATES, "label", Path(name)
    )

    samples = []
    for row in rows:
        text, label = row.get(text_col), normalize_label(row.get(label_col))
        if isinstance(text, str) and text.strip() and label is not None:
            samples.append(Sample(text=text, label=label, dataset=name))
    return samples


def describe(samples: Iterable[Sample]) -> str:
    """Human-readable composition summary - the thesis Table 11 equivalent."""
    samples = list(samples)
    if not samples:
        return "(empty dataset)"

    by_dataset: dict[str, Counter] = {}
    for sample in samples:
        by_dataset.setdefault(sample.dataset, Counter())[sample.label] += 1

    lines = [
        f"{'Dataset':<28} {'Benign':>10} {'Malicious':>10} {'Total':>10}",
        "-" * 62,
    ]
    total_benign = total_malicious = 0
    for name, counts in sorted(by_dataset.items()):
        benign, malicious = counts[0], counts[1]
        total_benign += benign
        total_malicious += malicious
        lines.append(f"{name:<28} {benign:>10,} {malicious:>10,} {benign + malicious:>10,}")

    lines.append("-" * 62)
    lines.append(
        f"{'Total':<28} {total_benign:>10,} {total_malicious:>10,} "
        f"{total_benign + total_malicious:>10,}"
    )

    total = total_benign + total_malicious
    if total:
        balance = total_malicious / total
        lines.append("")
        lines.append(f"Class balance: {balance:.1%} malicious")
        if not 0.3 <= balance <= 0.7:
            lines.append(
                "  WARNING: classes are imbalanced. Accuracy will be misleading; "
                "lead with F1 and report the confusion matrix."
            )
    return "\n".join(lines)


def stratified_split(
    samples: Sequence[Sample],
    *,
    train: float = 0.6,
    validation: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split into train/validation/evaluation, preserving class balance.

    Mirrors the thesis Data Splits table. Seeded so runs are reproducible - a
    requirement of the Reliability section.
    """
    import random

    rng = random.Random(seed)
    benign = [s for s in samples if s.label == 0]
    malicious = [s for s in samples if s.label == 1]
    rng.shuffle(benign)
    rng.shuffle(malicious)

    def cut(items: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
        n_train = int(len(items) * train)
        n_val = int(len(items) * validation)
        return (
            items[:n_train],
            items[n_train:n_train + n_val],
            items[n_train + n_val:],
        )

    b_train, b_val, b_eval = cut(benign)
    m_train, m_val, m_eval = cut(malicious)

    out = ([*b_train, *m_train], [*b_val, *m_val], [*b_eval, *m_eval])
    for part in out:
        rng.shuffle(part)
    return out
