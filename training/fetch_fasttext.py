"""Download the pretrained fastText vectors Track B's RNN baselines need.

    python training/fetch_fasttext.py              # English + Tagalog
    python training/fetch_fasttext.py --only tl    # just one
    python training/fetch_fasttext.py --list       # sizes, download nothing

Writes ``.vec`` files to ``data/vectors/``, which ``training/finetune_rnn.py``
reads. Run once.

**This downloads several gigabytes**, which is why it is a separate script and
not a step inside the trainer. The Wikipedia-trained vectors are used rather
than the Common Crawl ones: at ~2.2 GB versus ~4.5 GB for English they are less
than half the size, and the gap in downstream accuracy is small for a
classification task whose vocabulary is ordinary prose. The Tagalog set is small
either way.

The files are read once, streamed, and only the rows matching this corpus's
vocabulary are kept - see ``training.rnn_models.load_fasttext_matrix``. They can
be deleted after training, and only need re-downloading if the vocabulary
changes.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "data" / "vectors"

#: Wikipedia-trained fastText vectors, 300 dimensions.
#: Tagalog is the language the thesis scopes to alongside English (adviser
#: comment A14); ``tl`` is fastText's code for it.
VECTORS = {
    "en": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec",
        "filename": "wiki.en.vec",
        "approx_gb": 2.2,
    },
    "tl": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tl.vec",
        "filename": "wiki.tl.vec",
        "approx_gb": 0.2,
    },
}


def download(url: str, destination: Path) -> None:
    """Stream to a ``.part`` file, renaming only on success.

    An interrupted download must not leave a truncated file at the final path:
    ``load_fasttext_matrix`` would read it without complaint, find fewer
    vectors than it should, and the only symptom would be a coverage figure
    nobody thought to question.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  downloading {destination.name} ...", flush=True)

    def progress(block_number: int, block_size: int, total_size: int) -> None:
        if total_size <= 0 or block_number % 2000:
            return
        done = min(block_number * block_size, total_size)
        print(f"    {done / 1e9:5.2f} / {total_size / 1e9:5.2f} GB "
              f"({100 * done / total_size:5.1f}%)", flush=True)

    urllib.request.urlretrieve(url, partial, reporthook=progress)
    partial.replace(destination)
    print(f"    done -> {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--only", choices=sorted(VECTORS), action="append",
                        help="download just this language (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="show what would be downloaded and exit")
    args = parser.parse_args()

    wanted = args.only or sorted(VECTORS)
    total = sum(VECTORS[code]["approx_gb"] for code in wanted)

    print()
    print(f"  {'language':<12}{'file':<20}{'size':>10}   status")
    print("  " + "-" * 62)
    for code in wanted:
        spec = VECTORS[code]
        target = args.out / spec["filename"]
        state = "present" if target.exists() else "will download"
        print(f"  {code:<12}{spec['filename']:<20}{spec['approx_gb']:>8.1f} GB"
              f"   {state}")
    print("  " + "-" * 62)
    print(f"  {'total':<32}{total:>8.1f} GB")
    print()

    if args.list:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for code in wanted:
        spec = VECTORS[code]
        target = args.out / spec["filename"]
        if target.exists():
            print(f"  {spec['filename']} already present - skipping")
            continue
        try:
            download(spec["url"], target)
        except Exception as exc:  # noqa: BLE001 - the URL is the likely failure
            print(f"\nERROR downloading {spec['filename']}: {exc}")
            print(f"Fetch it manually from {spec['url']} into {args.out}/")
            return 1

    print(f"\n  Vectors ready in {args.out}")
    print("  Next: python training/finetune_rnn.py --cell lstm\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
