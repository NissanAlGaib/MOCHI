"""Render ordinary customer queries as images, speech and video.

    python eval/benign_media_corpus.py --modality image
    python eval/benign_media_corpus.py --modality audio --limit 20

Why this exists
---------------
Every false-positive rate MOCHI reports was measured on text typed into a text
field: 2.66% on BANKING77, 3.18% on CLINC150. The image, audio and video arms
report mitigation rates with no false-positive rate beside them at all, because
their corpora are 100% attacks.

That breaks the rule the rest of the evaluation is built on - a mitigation rate
without a false-positive rate cannot be interpreted, since refusing everything
scores 100%. The carrier guard in those harnesses rules out the degenerate case
where *every* request is refused, but says nothing about how often a legitimate
attachment is.

It matters more here than for text because each modality inserts a lossy step
ahead of the detector, and lossiness manufactures false positives. This is
measured, not hypothetical: a benign document squashed by an aspect-breaking
resize produced the OCR string ``'rtemalrem?rd.m'``, which scores **0.4301** on
its own - Stage II reads mangled characters as obfuscation. That finding is why
``RESCALE_MIN_CONFIDENCE`` exists, and it was validated against nine attacks
rather than against any benign corpus.

Same queries, same renderers
----------------------------
The queries are sampled from the datasets the text false-positive rate already
uses, and rendered with the **attack arms' own renderers** - ``render`` from
``eval.image_simulation``, ``speak`` and ``build_video`` from
``eval.av_corpus``. Both choices are deliberate:

* Reusing the renderers means the only difference between this corpus and the
  attack corpus is the words. A gap between the two is then a property of the
  content, not of how the files were made.
* Reusing the queries means the harness can run each one **as text and as the
  clip**, pairing the two. Comparing a 200-clip image rate against the
  3,080-query text rate would confound the modality with the sample; pairing
  removes that, and McNemar's test applies.

Sampling is seeded and the manifest records every row id, so the subset is
fixed and the run is reproducible.

Cost
----
Images are fast. Audio is about a second a clip to synthesise and another to
transcribe. Video is nearer nine seconds, dominated by frame OCR, which is why
its default ``--limit`` is far lower and why its interval will be the widest of
the three. Clips are cached on disk; a re-run only fills gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BENIGN_DIR = REPO / "data" / "benign"
MANIFEST = BENIGN_DIR / "manifest.json"

#: ``(name, csv path, text column, label column)`` - the same two datasets the
#: text false-positive rate uses, kept separate for the same reason: pooling
#: them would weight the result 64/36 toward CLINC150 and hide whichever is
#: worse.
SOURCES = [
    ("banking77", REPO / "data" / "external" / "banking77_test.csv", "text", "label"),
    ("clinc150", REPO / "data" / "external" / "clinc150_test.csv", "text", "intent"),
]

#: Clips per dataset, per modality. Video is lower because it costs ~9 s each
#: and its rate will carry the widest interval regardless; spending an hour to
#: narrow it slightly is not the best use of the time.
DEFAULT_LIMIT = {"image": 200, "audio": 200, "video": 40}

SUFFIX = {"image": ".png", "audio": ".wav", "video": ".mp4"}

#: Fixed, so the subset does not drift between runs and the manifest stays
#: meaningful after a rebuild.
SEED = 20261021


@dataclass
class BenignClip:
    """One ordinary query and the file it was rendered into."""

    clip_id: str
    dataset: str
    row_id: int
    intent: str
    text: str
    modality: str
    path: str
    """Relative to the repository root, so the manifest survives a move."""


def load_rows(path: Path, text_col: str, label_col: str) -> list[tuple[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"\n  Missing {path}.\n"
            "  Fetch it first - see docs/NEXT_STEPS.md, Part B.\n")
    with path.open(newline="", encoding="utf-8") as handle:
        return [(r[text_col], str(r.get(label_col, "")))
                for r in csv.DictReader(handle)]


def sample(dataset: str, path: Path, text_col: str, label_col: str,
           limit: int) -> list[tuple[int, str, str]]:
    """``limit`` queries as ``(row_id, text, intent)``, deterministically.

    Sampled rather than taken from the head, because both files are ordered by
    intent: the first 200 rows of CLINC150 cover two or three intents out of
    150, and a false-positive rate measured on them would describe those
    intents rather than the dataset.
    """
    rows = load_rows(path, text_col, label_col)
    indices = list(range(len(rows)))
    random.Random(f"{SEED}-{dataset}").shuffle(indices)
    picked = sorted(indices[:min(limit, len(rows))])
    return [(index, rows[index][0], rows[index][1]) for index in picked]


def write_image(text: str, path: Path) -> bool:
    from eval.image_simulation import render

    try:
        path.write_bytes(render(text))
    except Exception:  # noqa: BLE001 - no usable font, or an unrenderable glyph
        return False
    return path.exists() and path.stat().st_size > 512


def write_audio(text: str, path: Path) -> bool:
    from eval.av_corpus import speak

    return speak(text, path)


def write_video(text: str, path: Path, wav: Path) -> bool:
    """A clip carrying the query as a caption and as speech.

    Built the same way the attack videos are - both channels carrying the same
    words - so the benign and attack video arms differ only in content. The
    caption is the query itself rather than ``COVER_CAPTION``, because an
    attack video hides its payload behind a cover caption and a benign one has
    nothing to hide.
    """
    from eval.av_corpus import build_video, speak

    if not wav.exists() and not speak(text, wav):
        return False
    return build_video(text, wav, path)


def build(modality: str, limit: int, *, rebuild: bool) -> list[BenignClip]:
    out_dir = BENIGN_DIR / f"{modality}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    speech_dir = BENIGN_DIR / "audios"       # video reuses the audio clips
    if modality == "video":
        speech_dir.mkdir(parents=True, exist_ok=True)

    clips: list[BenignClip] = []
    built = skipped = failed = 0

    for dataset, path, text_col, label_col in SOURCES:
        rows = sample(dataset, path, text_col, label_col, limit)
        print(f"\n  {dataset}: {len(rows)} queries -> {modality}", flush=True)

        for position, (row_id, text, intent) in enumerate(rows, 1):
            clip_id = f"{dataset}-{row_id}"
            target = out_dir / f"{clip_id}{SUFFIX[modality]}"

            if target.exists() and not rebuild:
                skipped += 1
            else:
                if modality == "image":
                    ok = write_image(text, target)
                elif modality == "audio":
                    ok = write_audio(text, target)
                else:
                    ok = write_video(text, target, speech_dir / f"{clip_id}.wav")
                if not ok:
                    print(f"      [{position:>3}] {clip_id}  FAILED", flush=True)
                    failed += 1
                    continue
                built += 1

            clips.append(BenignClip(
                clip_id=clip_id, dataset=dataset, row_id=row_id,
                intent=intent, text=text, modality=modality,
                path=str(target.relative_to(REPO)).replace("\\", "/"),
            ))

            if position % 25 == 0 or position == len(rows):
                print(f"      [{position:>3}/{len(rows)}] {built} built, "
                      f"{skipped} cached, {failed} failed", flush=True)

    return clips


def merge_manifest(modality: str, clips: list[BenignClip]) -> dict:
    """Record this modality's clips without disturbing the others."""
    manifest: dict = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault("seed", SEED)
    manifest.setdefault("modalities", {})
    manifest["modalities"][modality] = [asdict(clip) for clip in clips]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return manifest


def load_manifest(modality: str) -> list[BenignClip]:
    """The clips recorded for ``modality``, for the harness to read."""
    if not MANIFEST.exists():
        raise SystemExit(
            f"\n  No benign media manifest at {MANIFEST}.\n"
            f"  Build one: python eval/benign_media_corpus.py "
            f"--modality {modality}\n")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("modalities", {}).get(modality)
    if not entries:
        raise SystemExit(
            f"\n  The manifest has no {modality} clips.\n"
            f"  Build them: python eval/benign_media_corpus.py "
            f"--modality {modality}\n")
    return [BenignClip(**entry) for entry in entries]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=("image", "audio", "video"),
                        required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="clips per dataset (default: 200 image/audio, "
                             "40 video)")
    parser.add_argument("--rebuild", action="store_true",
                        help="overwrite clips that already exist")
    args = parser.parse_args()

    limit = args.limit or DEFAULT_LIMIT[args.modality]
    clips = build(args.modality, limit, rebuild=args.rebuild)
    merge_manifest(args.modality, clips)

    print(f"\n  {len(clips)} {args.modality} clips recorded in {MANIFEST}\n")
    return 0 if clips else 1


if __name__ == "__main__":
    raise SystemExit(main())
