"""Build the audio and video attack corpora (revision items 2.6 and 2.7).

    python eval/av_corpus.py --modality audio
    python eval/av_corpus.py --modality video --limit 20

Renders the **same attacks the image arms used** - the user-turn entries of
``eval/attack_corpus`` - as speech and as video. Reusing that set is the
point: it makes the four modality arms comparable, so a difference between
them is a property of the channel rather than of the attacks chosen for it.

Audio
-----
Spoken by the platform's own voice through Windows SAPI. No cloud service, no
extra model, and two installed voices give a crude speaker-variation arm at
no cost. Synthesised rather than recorded because no public corpus of spoken
prompt injections exists - the same limitation the image arms carry, and it
belongs in Limitations for the same reason.

Video
-----
A ten-second clip per attack: a benign caption throughout, the payload
visible for one second, and the attack spoken over the top. Both channels
carry it, which is deliberate - an evaluation where only one did would not
show whether the other works.

The one-second payload is the part worth defending. A caption on screen that
briefly is easy for a viewer to miss and trivial for frame sampling to catch,
which is exactly the asymmetry the frame pass exists to create.

Cost
----
Audio is about a second per clip. Video is nearer nine, dominated by OCR, so
``--limit`` is there for a reason. Both are cached on disk: building is
idempotent and a re-run only fills gaps.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.attack_corpus import AttackItem, load_corpus  # noqa: E402

AUDIO_DIR = REPO / "data" / "attacks" / "audio"
VIDEO_DIR = REPO / "data" / "attacks" / "video"

#: Caption shown for the rest of the clip. Innocuous, so the payload frame
#: is the only thing worth finding.
COVER_CAPTION = "Quarterly results briefing"

#: Seconds the payload caption is on screen, and the clip length around it.
#: One second at the default one-frame-per-second sampling means the payload
#: appears in roughly one sampled frame - deliberately the hardest case the
#: sampler should still catch.
PAYLOAD_SECONDS = 1
CLIP_SECONDS = 10
FPS = 5


def speak(text: str, path: Path, *, voice: int = 0, volume: int = 100) -> bool:
    """Synthesise ``text`` to a WAV with the platform voice. Windows only."""
    script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($s.GetInstalledVoices())
if ($voices.Count -gt {voice}) {{ $s.SelectVoice($voices[{voice}].VoiceInfo.Name) }}
$s.Volume = {volume}
$s.SetOutputToWaveFile("{path}")
$s.Speak(@'
{text}
'@)
$s.Dispose()
"""
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", script],
                       check=True, capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001 - no SAPI, or a voice that refuses
        return False
    return path.exists() and path.stat().st_size > 1024


def _caption(text: str, size: tuple[int, int]):
    """A frame showing ``text``, wrapped to the width."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 34)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > size[0] - 80 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)

    y = max(30, size[1] // 2 - len(lines) * 24)
    for text_line in lines:
        draw.text((40, y), text_line, font=font, fill=(0, 0, 0))
        y += 48
    return image


def build_video(payload: str, wav: Path, path: Path) -> bool:
    """A clip carrying the payload in one second of frames and in speech."""
    try:
        import av
        import numpy as np
    except Exception:  # noqa: BLE001
        return False

    size = (900, 260)
    cover = _caption(COVER_CAPTION, size)
    hidden = _caption(payload, size)

    try:
        with wave.open(str(wav), "rb") as speech:
            rate = speech.getframerate()
            frames = speech.readframes(speech.getnframes())

        with av.open(str(path), "w") as out:
            video = out.add_stream("libx264", rate=FPS)
            video.width, video.height = size
            video.pix_fmt = "yuv420p"
            audio = out.add_stream("aac", rate=rate)

            # Payload sits in the middle, so a sampler that only looked at
            # the first or last frame would miss it.
            start = (CLIP_SECONDS // 2) * FPS
            end = start + PAYLOAD_SECONDS * FPS
            for index in range(CLIP_SECONDS * FPS):
                frame = hidden if start <= index < end else cover
                out.mux(video.encode(av.VideoFrame.from_image(frame)))
            out.mux(video.encode(None))

            pcm = np.frombuffer(frames, dtype=np.int16).reshape(1, -1)
            chunk = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
            chunk.rate = rate
            for packet in audio.encode(chunk):
                out.mux(packet)
            for packet in audio.encode(None):
                out.mux(packet)
    except Exception:  # noqa: BLE001 - encoder missing or a bad waveform
        return False
    return path.exists() and path.stat().st_size > 1024


def user_turn_attacks(limit: int | None = None) -> list[AttackItem]:
    """The same selection the image arms ran, for comparability."""
    corpus = [i for i in load_corpus()
              if i.placement == "user_input" and not i.turns]
    return corpus[:limit] if limit else corpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=("audio", "video"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--voice", type=int, default=0,
                        help="index into the installed voices")
    parser.add_argument("--rebuild", action="store_true",
                        help="overwrite clips that already exist")
    args = parser.parse_args()

    attacks = user_turn_attacks(args.limit)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if args.modality == "video":
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  {len(attacks)} attacks -> {args.modality}")
    built = skipped = failed = 0
    for index, item in enumerate(attacks, 1):
        wav = AUDIO_DIR / f"{item.id}.wav"
        target = wav if args.modality == "audio" else VIDEO_DIR / f"{item.id}.mp4"

        if target.exists() and not args.rebuild:
            skipped += 1
            continue

        if not wav.exists() or args.rebuild:
            if not speak(item.user_input, wav, voice=args.voice):
                print(f"      [{index:>3}] {item.id}  speech FAILED")
                failed += 1
                continue

        if args.modality == "video" and not build_video(item.user_input, wav, target):
            print(f"      [{index:>3}] {item.id}  video FAILED")
            failed += 1
            continue

        built += 1
        if index % 10 == 0 or index == len(attacks):
            print(f"      [{index:>3}/{len(attacks)}] {built} built, "
                  f"{skipped} already there, {failed} failed", flush=True)

    where = AUDIO_DIR if args.modality == "audio" else VIDEO_DIR
    total = len(list(where.glob("*.wav" if args.modality == "audio" else "*.mp4")))
    print(f"\n  {total} {args.modality} clips in {where}\n")
    return 1 if failed and not built else 0


if __name__ == "__main__":
    raise SystemExit(main())
