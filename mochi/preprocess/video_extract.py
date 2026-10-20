"""Read video by reading the two things inside it that already have readers.

A video is a sequence of images and a soundtrack, and MOCHI can read both. So
this module decodes rather than detects: it samples frames and hands them to
``image_extract``, lifts the audio track and hands it to ``audio_extract``,
and merges the results. There is no video detector and there should not be -
a payload in a video is text on a frame or words in the soundtrack, and both
already have a path through the pipeline.

What makes video distinct
-------------------------
Length. An image is inspected once; a video could be inspected thousands of
times, and an attacker choosing the length chooses the cost. Two bounds
answer that, and they are the only real design decisions here:

``FRAME_INTERVAL_SECONDS``
    How often to look. Text a viewer is meant to read stays on screen for
    seconds, so sampling every second finds it.

``MAX_FRAMES``
    How many looks, at most. Without it a long upload costs unbounded OCR -
    the same reasoning as ``MAX_WINDOWS`` in Stage II, and the cap is
    reported rather than applied silently, because a caller whose video was
    only partly read should not be told it was clean.

A single frame carrying an instruction no viewer would notice is the video
version of concealed text, and sampling catches it for the same reason the
contrast pass catches pale text: the machine looks where a person does not.

Both channels are optional
--------------------------
Decoding needs PyAV, which arrives with faster-whisper, and frame OCR needs
the image runtime. Either being absent degrades that channel to "nothing
found" rather than failing the request, and ``ran`` stays honest about which
channels actually ran.
"""

from __future__ import annotations

import base64
import binascii
import io
import itertools
import re
from dataclasses import dataclass, field

from mochi.preprocess.flags import NormalizationFlag as F

#: Sample one frame per this many seconds of video.
#:
#: Text meant to be read by a viewer stays on screen for seconds; text meant
#: to be read only by a model need not, which is why the interval is a
#: trade-off rather than a free parameter. One second finds a caption a
#: viewer would see and most of what they would not, without OCR'ing every
#: frame of a thirty-frames-a-second recording.
FRAME_INTERVAL_SECONDS = 1.0

#: Never OCR more than this many frames, however long the video.
#:
#: An attacker picks the duration, so without a cap they pick the cost. Sixty
#: frames is a minute of coverage at the default interval. Exceeding it is
#: recorded on the result, never swallowed.
MAX_FRAMES = 60

#: Larger than this is refused rather than decoded, for the same reason the
#: image and audio extractors refuse their own limits: a caller sending this
#: much to a text gateway is not asking for transcription.
MAX_VIDEO_BYTES = 64 * 1024 * 1024

DATA_URI = re.compile(r"^data:video/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)


@dataclass
class VideoExtraction:
    """Text recovered from one video, from both of its channels."""

    visible_text: str = ""
    """Everything read: speech from the soundtrack, then text from frames."""

    hidden_text: list[str] = field(default_factory=list)
    """Frame text that only a contrast-boosted read could see, passed
    through from ``image_extract``. Audio contributes nothing here - there is
    no amplitude-based hidden channel in speech, which ``audio_extract``'s
    docstring records the measurement for."""

    metadata_text: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    ran: bool = False
    frames_read: int = 0
    truncated: bool = False
    """Whether :data:`MAX_FRAMES` cut the scan short. A partly-read video is
    not a clean one, and a caller that cannot tell the difference will treat
    it as though it were."""

    @property
    def all_segments(self) -> list[str]:
        return [s for s in [self.visible_text, *self.hidden_text,
                            *self.metadata_text] if s and s.strip()]

    @property
    def text(self) -> str:
        return "\n".join(self.all_segments)

    def add_flag(self, flag: F | str) -> None:
        value = flag.value if isinstance(flag, F) else flag
        if value not in self.flags:
            self.flags.append(value)


def decode_video_block(block: dict) -> bytes | None:
    """Bytes from a video content block, or ``None``.

    Remote URLs are refused, as they are for images and audio: fetching one
    would let any caller make the gateway issue outbound requests to a host
    of their choosing, on every inspection.
    """
    source = block.get("video") or block.get("video_url") or {}
    raw = source.get("data") or source.get("url") if isinstance(source, dict) else source
    if not isinstance(raw, str) or not raw:
        return None
    match = DATA_URI.match(raw.strip())
    if match:
        raw = match.group(1)
    elif raw.lstrip().lower().startswith(("http://", "https://")):
        return None
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None


#: Grid a frame is reduced to before hashing. Coarse enough that compression
#: noise vanishes, fine enough that a line of changed caption text moves it.
HASH_GRID = 24

#: Brightness levels kept. Four, so a pixel has to shift by a quarter of the
#: range to change the hash - far more than re-encoding moves it.
HASH_LEVELS = 4

#: Supplies distinct values when hashing fails. See _appearance.
_FALLBACK = itertools.count()


def _appearance(frame) -> str:
    """A hash of what a frame looks like, robust to compression noise.

    Shrink to a small greyscale grid and quantise hard. A difference hash
    was tried first and failed outright on this content: these frames are
    dark text on a near-white ground, so every neighbour comparison is a
    near-tie and codec noise decides each bit - it deduplicated nothing and
    cost an extra second. Quantising instead compares what the pixels *are*,
    with buckets wide enough that re-encoding cannot move a pixel between
    them, while text and background remain whole buckets apart.

    Falls back to a per-frame unique value on any failure, so a problem here
    costs a redundant OCR pass rather than a skipped frame.
    """
    from PIL import Image

    # ``frame`` is already a PIL image - _sample_frames converts on decode.
    # An earlier version called .to_image() on it, which raises, and the
    # broad except below turned every frame into a unique value: hashing
    # appeared to run and deduplicated nothing, through three attempts at
    # tuning the wrong thing. Narrowed deliberately, so a real failure here
    # is visible rather than silently costing every repeat an OCR pass.
    try:
        small = frame.convert("L").resize((HASH_GRID, HASH_GRID), Image.BILINEAR)
        step = 256 // HASH_LEVELS
        return "".join(str(p // step) for p in small.getdata())
    except (OSError, ValueError):
        # A counter, not id(). CPython reuses the id of a collected object,
        # and frames are created and dropped in a tight loop - so two
        # unhashable frames could collide and the second would be treated
        # as a repeat and never read. Wasting an OCR pass is the safe
        # direction to fail; skipping a frame is not.
        return f"unhashable-{next(_FALLBACK)}"


def _sample_frames(data: bytes) -> tuple[list, bool]:
    """Frames at the configured interval, and whether the cap cut it short."""
    try:
        import av
    except Exception:  # noqa: BLE001 - decoder absent
        return [], False

    frames, truncated = [], False
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return [], False
            stream.thread_type = "AUTO"
            next_at = 0.0
            for frame in container.decode(stream):
                when = float(frame.time or 0.0)
                if when + 1e-6 < next_at:
                    continue
                if len(frames) >= MAX_FRAMES:
                    truncated = True
                    break
                frames.append(frame.to_image())
                next_at = when + FRAME_INTERVAL_SECONDS
    except Exception:  # noqa: BLE001 - unreadable container or codec
        return frames, truncated
    return frames, truncated


def _audio_track(data: bytes) -> bytes | None:
    """The soundtrack as a WAV, or ``None`` when there is not one."""
    try:
        import av
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            rate = int(stream.rate or 16000)
            chunks = [f.to_ndarray().astype("float32").ravel()
                      for f in container.decode(stream)]
        if not chunks:
            return None
        samples = np.concatenate(chunks)
        peak = float(np.max(np.abs(samples))) or 1.0
        if peak > 1.5:          # already integer-scaled
            samples = samples / 32768.0
        samples = np.clip(samples, -1.0, 1.0)

        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes((samples * 32767).astype("int16").tobytes())
        return buffer.getvalue()
    except Exception:  # noqa: BLE001
        return None


def extract_video(data: bytes) -> VideoExtraction:
    """Read a video's soundtrack and its frames, and merge what both say."""
    out = VideoExtraction()
    if not data or len(data) > MAX_VIDEO_BYTES:
        return out

    parts: list[str] = []

    # --- soundtrack ---
    track = _audio_track(data)
    if track:
        from mochi.preprocess.audio_extract import extract_audio

        heard = extract_audio(track)
        if heard.ran:
            out.ran = True
            if heard.visible_text.strip():
                parts.append(heard.visible_text)
                out.add_flag(F.AUDIO_TRANSCRIBED)
            out.metadata_text.extend(heard.metadata_text)

    # --- frames ---
    frames, out.truncated = _sample_frames(data)
    if frames:
        out.ran = True
        from mochi.preprocess.image_extract import extract_image

        seen: set[str] = set()
        scanned: set[str] = set()
        for frame in frames:
            buffer = io.BytesIO()
            try:
                frame.save(buffer, format="PNG")
            except Exception:  # noqa: BLE001
                continue
            raw = buffer.getvalue()

            # Frames that look the same carry the same text, and video is
            # mostly frames that look the same - a caption held on screen
            # for five seconds is five samples of one image. OCR dominates
            # the cost, so skipping repeats is what makes this usable at
            # all: the test video has ten samples and two distinct frames.
            #
            # Compared by appearance, not by bytes. Inter-frame compression
            # means two visually identical frames decode to slightly
            # different pixels, so hashing the PNG caught only four of the
            # eight repeats and left 34.6 s on the clock. A difference hash
            # ignores that noise because it records which neighbouring
            # pixels are brighter, not what they are.
            digest = _appearance(frame)
            if digest in scanned:
                continue
            scanned.add(digest)

            read = extract_image(raw)
            out.frames_read += 1
            # A caption held on screen appears in every frame it spans.
            # Deduplicating keeps one copy, so a detector is not handed the
            # same sentence sixty times and a long static caption does not
            # drown everything else in the window it is scored in.
            for line in read.all_segments:
                key = line.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    parts.append(line)
            for line in read.hidden_text:
                if line not in out.hidden_text:
                    out.hidden_text.append(line)
            for flag in read.flags:
                out.add_flag(flag)

    out.visible_text = "\n".join(parts)
    if out.visible_text.strip() or out.hidden_text:
        out.add_flag(F.VIDEO_TEXT_EXTRACTED)
    if out.truncated:
        out.add_flag(F.VIDEO_SCAN_TRUNCATED)
    return out
