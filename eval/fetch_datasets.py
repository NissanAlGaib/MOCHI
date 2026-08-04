"""Download the public benchmark datasets into ``data/``.

    python eval/fetch_datasets.py            # fetch all
    python eval/fetch_datasets.py --list     # show what would be fetched
    python eval/fetch_datasets.py deepset    # fetch one

Requires ``datasets`` (``pip install datasets``). PromptShield is not listed -
it is not on the Hub, so place your copy in ``data/`` manually.

Downloading to disk rather than streaming from the Hub at evaluation time is
deliberate: it pins the exact snapshot the results were produced from, which
is what the thesis Reliability section commits to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

#: name -> (hub id, splits to fetch)
#:
#: ``hendzh/PromptShield`` is the dataset released with the PromptShield paper
#: (arXiv 2501.15145). Several later mirrors exist on the Hub under the same
#: name with byte-identical READMEs; this is the original - oldest upload and
#: by far the most downloads. Provenance matters when the thesis cites it.
SOURCES: dict[str, tuple[str, list[str]]] = {
    "deepset": ("deepset/prompt-injections", ["train"]),
    "jayavibhav": ("jayavibhav/prompt-injection", ["train"]),
    "promptshield": ("hendzh/PromptShield", ["train", "validation", "test"]),
}


def fetch_split(name: str, hub_id: str, split: str, *, multi: bool) -> Path | None:
    """Fetch one split. Multi-split datasets get ``name_split.csv`` filenames."""
    from datasets import load_dataset

    filename = f"{name}_{split}.csv" if multi else f"{name}.csv"
    target = DATA_DIR / filename
    label = f"{name}[{split}]" if multi else name

    if target.exists():
        print(f"  {label:<22} already present as {target.name}, skipping")
        return target

    print(f"  {label:<22} downloading {hub_id} ...")
    try:
        dataset = load_dataset(hub_id, split=split)
    except Exception as exc:
        print(f"  {label:<22} FAILED: {exc}")
        print(f"  {' ':<22} download manually from "
              f"https://huggingface.co/datasets/{hub_id}")
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(target)
    print(f"  {label:<22} wrote {len(dataset):,} rows -> {target.name}")
    return target


def fetch(name: str, hub_id: str, splits: list[str]) -> list[Path]:
    multi = len(splits) > 1
    results = [fetch_split(name, hub_id, s, multi=multi) for s in splits]
    return [r for r in results if r is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=None,
                        help=f"which to fetch (default: all of {', '.join(SOURCES)})")
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, (hub_id, splits) in SOURCES.items():
            print(f"  {name:<14} {hub_id} [{', '.join(splits)}]")
        print()
        return 0

    try:
        import datasets  # noqa: F401
    except ImportError:
        print(
            "\nThe 'datasets' package is required.\n"
            "  pip install datasets\n\n"
            "Or download the CSVs by hand from the dataset pages - see data/README.md.\n"
        )
        return 1

    selected = args.names or list(SOURCES)
    unknown = [n for n in selected if n not in SOURCES]
    if unknown:
        print(f"Unknown dataset(s): {', '.join(unknown)}. Known: {', '.join(SOURCES)}")
        return 1

    print(f"\nFetching into {DATA_DIR}\n")
    expected = sum(len(SOURCES[name][1]) for name in selected)
    written = sum(len(fetch(name, *SOURCES[name])) for name in selected)

    print(f"\n{written}/{expected} file(s) available.")
    if written:
        print("\nNext:\n  python eval/run_detection.py --data data/ --config baseline\n")
    return 0 if written == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
