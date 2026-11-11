"""Hidden attachment text scored alone: what it changes in detection.

    python eval/attachment_channels.py
    python eval/attachment_channels.py --limit 5 --arms pdf_hidden

Detection only - no model is run, so this is fast and needs no gateway. Every
attachment attack arm is judged twice on the same build, the only difference
being :data:`mochi.detect.segments.SCORE_HIDDEN_CHANNELS_ALONE`:

``merged``  MOCHI before the change: everything read from an attachment is
            joined into one text and scored as one, so a short hidden payload
            is scored together with the ordinary visible text around it.
``alone``   each hidden channel (low-contrast or bit-plane image text, white
            PDF text, PDF and image metadata) is also scored by itself.

The arms are the attachment arms' own attacks, built by their own corpus
code: the 61 user-turn attacks as images (visible, concealed), PDFs (visible,
hidden, metadata) and bit-plane images (blue, RGB). Every request is a user
upload, so a detection refuses it (BLOCK); ALLOW is a false negative.

Writes ``reports/attachment_channels.jsonl`` (one row per attack, arm and
mode) and ``reports/attachment_channels.txt``. The false-positive side is
measured separately, by ``eval/benign_media_run.py`` on the benign corpora.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.attack_corpus import SYSTEM_PROMPT  # noqa: E402
from eval.av_corpus import user_turn_attacks  # noqa: E402
from eval.bitplane_detection import CARRIER_TEXT, VARIANTS as BITPLANE  # noqa: E402
from eval.bitplane_detection import judge, stego  # noqa: E402
from eval.document_corpus import attack_pdf, file_block  # noqa: E402
from eval.image_simulation import CARRIER as IMAGE_QUESTION, render  # noqa: E402

REPORTS = REPO / "reports"
PDF_QUESTION = "What does this document say?"


def _image(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(png).decode()}}


#: arm -> (question, payload -> content block)
ARMS = {
    "image_visible": (IMAGE_QUESTION, lambda p: _image(render(p))),
    "image_concealed": (IMAGE_QUESTION, lambda p: _image(render(p, concealed=True))),
    "pdf_visible": (PDF_QUESTION, lambda p: file_block(attack_pdf(p, "visible"), "document.pdf")),
    "pdf_hidden": (PDF_QUESTION, lambda p: file_block(attack_pdf(p, "hidden"), "statement.pdf")),
    "pdf_metadata": (PDF_QUESTION, lambda p: file_block(attack_pdf(p, "metadata"), "statement.pdf")),
    **{f"bitplane_{name}": ("What does this report say?",
                            (lambda c: lambda p: _image(stego(p, c)))(channels))
       for name, channels in BITPLANE.items()},
}


def verdict(client, question: str, block: dict) -> dict:
    """``judge`` from the bit-plane eval, for any content block."""
    from mochi.client import MochiBlocked
    from mochi.preprocess import extract

    extract._CACHE.clear()
    try:
        client.chat.completions.create(model="m", messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": question}, block]}])
    except MochiBlocked:
        pass
    call = client.mochi_last
    result = call.inspection
    stage1 = bool(result and any(r.should_block for _, r in result.stage1))
    stage2 = bool(result and any(r.should_block for _, r in result.stage2))
    action = call.decision.upper()
    return {
        "mitigation_action": action,
        "stopped_by": ("stage1" if stage1 else "stage2" if stage2 else None)
                      if action != "ALLOW" else None,
        "semantic_score": max((r.score for _, r in result.stage2), default=None)
                          if result else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    args = parser.parse_args()

    import mochi.detect.segments as segments
    from mochi.client import guard
    from tests.test_client_guard import StubClient

    assert judge and CARRIER_TEXT  # the bit-plane carrier is reused unchanged
    client = guard(StubClient(), system_prompt=SYSTEM_PROMPT, read_attachments=True)
    attacks = user_turn_attacks(args.limit)
    rows: list[dict] = []
    for arm in args.arms:
        question, build = ARMS[arm]
        for n, item in enumerate(attacks, 1):
            block = build(item.user_input)
            row = {"attack_id": item.id, "tier": item.tier,
                   "attack_type": item.attack_type, "arm": arm}
            for mode, alone in (("merged", False), ("alone", True)):
                segments.SCORE_HIDDEN_CHANNELS_ALONE = alone
                row[mode] = verdict(client, question, block)
            segments.SCORE_HIDDEN_CHANNELS_ALONE = True
            rows.append(row)
            print(f"  [{arm} {n:>2}/{len(attacks)}] {item.id:<6} "
                  f"merged {row['merged']['mitigation_action']:<8} "
                  f"alone {row['alone']['mitigation_action']}", flush=True)

    # A run over some arms replaces those arms' rows and keeps the rest, so
    # one arm can be re-run after a change without redoing the slow ones.
    out = REPORTS / "attachment_channels.jsonl"
    if out.exists() and args.limit is None:
        kept = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        rows = [r for r in kept if r["arm"] not in args.arms] + rows
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    arms = [arm for arm in ARMS if any(r["arm"] == arm for r in rows)]

    lines = ["", "  HIDDEN ATTACHMENT TEXT SCORED ALONE - detection only, no model run", "",
             f"  {'arm':<18}{'n':>4}  {'merged':>14}  {'alone':>14}  "
             f"{'gained':>6}  {'lost':>4}  {'FN merged':>10}  {'FN alone':>9}"]
    for arm in arms:
        subset = [r for r in rows if r["arm"] == arm]
        n = len(subset)
        caught = {m: sum(r[m]["mitigation_action"] != "ALLOW" for r in subset)
                  for m in ("merged", "alone")}
        gained = sum(r["merged"]["mitigation_action"] == "ALLOW"
                     and r["alone"]["mitigation_action"] != "ALLOW" for r in subset)
        lost = sum(r["merged"]["mitigation_action"] != "ALLOW"
                   and r["alone"]["mitigation_action"] == "ALLOW" for r in subset)
        lines.append(
            f"  {arm:<18}{n:>4}  {caught['merged']:>3} ({caught['merged'] / n:>6.1%})"
            f"  {caught['alone']:>3} ({caught['alone'] / n:>6.1%})  {gained:>6}  {lost:>4}"
            f"  {n - caught['merged']:>10}  {n - caught['alone']:>9}")
    by = Counter(r["alone"]["stopped_by"] for r in rows if r["alone"]["stopped_by"])
    lines += ["", f"  stopped by, alone: stage I {by.get('stage1', 0)}, "
                  f"stage II {by.get('stage2', 0)}", f"  rows -> {out}", ""]
    text = "\n".join(lines)
    (REPORTS / "attachment_channels.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
