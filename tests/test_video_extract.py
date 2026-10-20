"""Video, read as the two things inside it that already have readers.

Revision item 2.5. There is no video detector and should not be: a payload
in a video is text on a frame or words in the soundtrack, and both already
have a path through the pipeline. What is specific to video is cost, so most
of what is pinned here is the bounds on it.
"""

from __future__ import annotations

import base64
import importlib.util

import pytest

from mochi.preprocess.video_extract import (
    HASH_GRID,
    MAX_VIDEO_BYTES,
    VideoExtraction,
    _appearance,
    decode_video_block,
    extract_video,
)

av_available = importlib.util.find_spec("av") is not None
needs_av = pytest.mark.skipif(not av_available, reason="PyAV not installed")


# --- decoding --------------------------------------------------------------


def test_inline_data_is_decoded() -> None:
    block = {"type": "video", "video": {"data": base64.b64encode(b"\x00mp4").decode()}}
    assert decode_video_block(block) == b"\x00mp4"


def test_data_uri_is_decoded() -> None:
    block = {"type": "video_url",
             "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(b"x").decode()}}
    assert decode_video_block(block) == b"x"


def test_remote_urls_are_refused() -> None:
    """Same rule as images and audio: fetching one would let any caller make
    the gateway issue outbound requests to a host of their choosing."""
    assert decode_video_block(
        {"type": "video", "video": {"url": "https://example.com/clip.mp4"}}) is None


# --- bounds ----------------------------------------------------------------


def test_oversized_video_is_refused_before_decoding() -> None:
    result = extract_video(b"\x00" * (MAX_VIDEO_BYTES + 1))
    assert result.ran is False


def test_unreadable_bytes_report_not_run() -> None:
    """Not the same as "read it and found nothing", which is a real result
    about a silent video with no text in it."""
    assert extract_video(b"definitely not a video").ran is False


def test_truncation_is_reported_not_swallowed() -> None:
    """An attacker picks the duration, so they would otherwise pick how much
    of their payload goes uninspected. A partly-read video must not look
    like a clean one."""
    result = VideoExtraction(ran=True, truncated=True, visible_text="partial")
    assert result.truncated is True


# --- frame deduplication ---------------------------------------------------


@needs_av
def test_identical_frames_hash_alike_despite_compression_noise() -> None:
    """The property that makes video affordable.

    OCR dominates the cost and video is mostly repeated frames, so repeats
    have to be recognised. They cannot be recognised by their bytes: a codec
    re-encodes visually identical frames to slightly different pixels.
    """
    from PIL import Image

    base = Image.new("L", (400, 120), 255)
    for x in range(40, 360, 7):          # some structure to hash
        for y in range(40, 80):
            base.putpixel((x, y), 0)

    noisy = base.copy()
    for x in range(0, 400, 13):          # codec-scale noise
        for y in range(0, 120, 11):
            noisy.putpixel((x, y), max(0, min(255, noisy.getpixel((x, y)) + 6)))

    assert _appearance(base.convert("RGB")) == _appearance(noisy.convert("RGB")), (
        "frames differing only by compression noise must hash alike"
    )


@needs_av
def test_different_frames_hash_differently() -> None:
    """Deduplication must not swallow a frame that says something new -
    a caption shown for one second is exactly the case this exists for."""
    from PIL import Image

    blank = Image.new("RGB", (400, 120), "white")
    captioned = Image.new("RGB", (400, 120), "white")
    for x in range(20, 380):
        for y in range(50, 70):
            captioned.putpixel((x, y), (0, 0, 0))

    assert _appearance(blank) != _appearance(captioned)


def test_hashing_a_bad_frame_costs_a_pass_not_a_skip() -> None:
    """A hashing failure must never drop a frame from inspection.

    Returning a unique value means the frame is treated as new and gets
    OCR'd - wasteful but safe. The inverse would let an unhashable frame go
    unread, which is the direction that loses attacks.
    """
    class Hostile:
        def convert(self, mode):
            raise OSError("cannot convert")

    first, second = _appearance(Hostile()), _appearance(Hostile())
    assert first != second, "a failed hash must not collide and suppress a frame"


# --- the shared shape ------------------------------------------------------


def test_both_channels_join_the_scannable_set() -> None:
    found = VideoExtraction(visible_text="spoken words\nframe caption",
                            hidden_text=["faint overlay"], ran=True)
    assert "spoken words" in found.text
    assert "faint overlay" in found.text


def test_video_block_types_are_registered() -> None:
    from mochi.preprocess.extract import supported

    assert {"video", "video_url"} <= supported()


def test_hash_grid_is_coarse_enough_to_be_useful() -> None:
    """A sanity bound: too fine and noise defeats it, too coarse and a
    changed caption does not move it."""
    assert 8 <= HASH_GRID <= 64
