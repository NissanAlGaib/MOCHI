"""The modality registry that every attachment now dispatches through.

Added 2026-10-20 for revision item 2. The registry's job is to make adding a
modality a one-decorator change, and to fail in the one direction that is
safe: an attachment nothing could read must look unread, never clean.
"""

from __future__ import annotations

import pytest

from mochi.preprocess.extract import (
    EXTRACTORS,
    Extraction,
    extract_from_content_block,
    register,
    supported,
)


@pytest.fixture
def isolated_registry():
    """Register into a copy, so a test cannot leak into the real table."""
    original = dict(EXTRACTORS)
    yield
    EXTRACTORS.clear()
    EXTRACTORS.update(original)


# --- the shared shape ------------------------------------------------------


def test_all_segments_orders_visible_content_first() -> None:
    """A detector reads the perceivable content before the recovered content."""
    found = Extraction(visible_text="seen", hidden_text=["faint"],
                       metadata_text=["Comment: tagged"], ran=True)
    assert found.all_segments == ["seen", "faint", "Comment: tagged"]
    assert found.text == "seen\nfaint\nComment: tagged"


def test_blank_segments_are_dropped() -> None:
    """Whitespace is not content; scanning it would only add noise."""
    found = Extraction(visible_text="", hidden_text=["   ", "real"], ran=True)
    assert found.all_segments == ["real"]


def test_ran_is_not_the_same_as_found_nothing() -> None:
    """The distinction the whole module turns on.

    An extractor that ran and found no text is a real result. One that never
    ran says nothing about the attachment, and a caller that conflates them
    reports a clean scan for a file it could not open.
    """
    scanned = Extraction(ran=True)
    never = Extraction(ran=False)
    assert scanned.text == never.text == ""
    assert scanned.ran and not never.ran


def test_add_flag_does_not_duplicate() -> None:
    found = Extraction()
    found.add_flag("image_text_extracted")
    found.add_flag("image_text_extracted")
    assert found.flags == ["image_text_extracted"]


# --- dispatch --------------------------------------------------------------


def test_registering_a_modality_is_one_decorator(isolated_registry) -> None:
    """The property that makes audio and video cheap to add."""
    @register("input_audio")
    def _audio(block: dict) -> Extraction:
        return Extraction(visible_text="spoken words", ran=True)

    assert "input_audio" in supported()
    result = extract_from_content_block({"type": "input_audio"})
    assert result.ran and result.text == "spoken words"


def test_one_extractor_can_claim_several_spellings(isolated_registry) -> None:
    """Clients disagree on block names; the image case already needs two."""
    @register("video", "video_url")
    def _video(block: dict) -> Extraction:
        return Extraction(visible_text="frame caption", ran=True)

    for name in ("video", "video_url"):
        assert extract_from_content_block({"type": name}).text == "frame caption"


def test_unregistered_type_reads_as_unread_not_as_clean() -> None:
    """An audio file today: nothing can read it, so nothing claims it is safe."""
    result = extract_from_content_block({"type": "input_audio"})
    assert result.ran is False
    assert result.text == ""


def test_malformed_block_does_not_raise() -> None:
    for block in ("not a dict", None, 42, {}, {"type": None}):
        assert extract_from_content_block(block).ran is False


def test_a_failing_extractor_is_contained(isolated_registry) -> None:
    """A corrupt attachment must not take down a legitimate request.

    It must also not be mistaken for a clean one, which is why the swallowed
    failure still reports ``ran=False``.
    """
    @register("exploding")
    def _boom(block: dict) -> Extraction:
        raise RuntimeError("corrupt file")

    result = extract_from_content_block({"type": "exploding"})
    assert result.ran is False
    assert result.text == ""


# --- the image path, through the registry ----------------------------------


def test_images_are_registered_by_default() -> None:
    """Both spellings the OpenAI format uses in the wild."""
    assert {"image_url", "image"} <= supported()


def test_image_adapter_maps_ocr_ran_onto_ran(monkeypatch) -> None:
    """``ImageExtraction`` predates the shared shape and names this field
    differently; the adapter is the only place that difference exists."""
    from mochi.preprocess import image_extract

    source = image_extract.ImageExtraction(
        visible_text="payload", hidden_text=["concealed"],
        metadata_text=["Comment: x"], flags=["image_text_extracted"],
        ocr_ran=True,
    )
    monkeypatch.setattr(image_extract, "extract_from_content_block",
                        lambda block: source)

    result = extract_from_content_block({"type": "image_url"})
    assert result.ran is True, "ocr_ran must survive the adapter as ran"
    assert result.visible_text == "payload"
    assert result.hidden_text == ["concealed"]
    assert result.metadata_text == ["Comment: x"]
    assert result.flags == ["image_text_extracted"]


# --- conversion: attachments become text before dispatch -------------------
#
# Revision item 2.3, chosen so a text-only model can serve a multimodal
# application. The gateway calls this after enforcement and after the block
# check, so a refused request is never transcribed and a redacted one stays
# redacted.


from mochi.preprocess.extract import _CACHE, convert_attachments  # noqa: E402


class _Msg:
    def __init__(self, content):
        self.content = content


class _Req:
    def __init__(self, messages):
        self.messages = messages


@pytest.fixture
def stub_image(isolated_registry):
    """An extractor that needs no OCR runtime, so these tests always run."""
    @register("image_url")
    def _fake(block: dict) -> Extraction:
        return Extraction(visible_text="PAYLOAD TEXT", ran=True)
    _CACHE.clear()
    yield
    _CACHE.clear()


def _image_block(url: str = "data:image/png;base64,AAAA") -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def test_attachment_is_replaced_by_its_text(stub_image) -> None:
    request = _Req([_Msg([{"type": "text", "text": "what is this?"},
                          _image_block()])])
    assert convert_attachments(request) == 1

    kinds = [b["type"] for b in request.messages[0].content]
    assert kinds == ["text", "text"], "the image block should be gone"
    assert "PAYLOAD TEXT" in request.messages[0].content[1]["text"]


def test_converted_text_is_labelled_as_a_transcript(stub_image) -> None:
    """Unlabelled, a transcript reads as something the user typed.

    A payload inside it would then arrive wearing the user's authority, which
    is the distinction the whole trust model rests on.
    """
    request = _Req([_Msg([_image_block()])])
    convert_attachments(request)
    assert request.messages[0].content[0]["text"].startswith("[image transcribed by MOCHI]")


def test_unreadable_attachment_becomes_a_marker_not_silence(
    isolated_registry,
) -> None:
    """Three options, and only one of them is honest.

    Dropping it discards what the user attached. Forwarding it hands the
    model bytes nothing inspected. A marker lets the model answer about an
    attachment it cannot see, rather than about one nobody checked.
    """
    @register("image_url")
    def _broken(block: dict) -> Extraction:
        return Extraction(ran=False)

    _CACHE.clear()
    request = _Req([_Msg([_image_block()])])
    assert convert_attachments(request) == 1
    text = request.messages[0].content[0]["text"]
    assert "could not be read" in text
    assert request.messages[0].content[0]["type"] == "text"


def test_plain_text_requests_are_untouched(stub_image) -> None:
    request = _Req([_Msg("just a string")])
    assert convert_attachments(request) == 0
    assert request.messages[0].content == "just a string"


def test_unregistered_block_types_are_left_alone(stub_image) -> None:
    """Nothing can read it, so replacing it would destroy the only copy."""
    request = _Req([_Msg([{"type": "input_audio", "input_audio": {"data": "x"}}])])
    assert convert_attachments(request) == 0
    assert request.messages[0].content[0]["type"] == "input_audio"


def test_extraction_is_cached_so_conversion_does_not_rerun_it(stub_image) -> None:
    """The cache is why conversion is affordable.

    Extraction happens once for detection and once for conversion. On an
    image that is seconds, not milliseconds, so without this the slowest step
    in the request happens twice.
    """
    calls = []

    @register("image_url")
    def _counting(block: dict) -> Extraction:
        calls.append(1)
        return Extraction(visible_text="once", ran=True)

    _CACHE.clear()
    block = _image_block()
    extract_from_content_block(block)          # detection
    convert_attachments(_Req([_Msg([block])])) # conversion
    assert len(calls) == 1, "extraction ran twice despite the cache"


def test_cache_is_bounded(stub_image) -> None:
    """A long-running gateway must not accumulate every image it has seen."""
    from mochi.preprocess.extract import _CACHE_MAX

    for i in range(_CACHE_MAX + 10):
        extract_from_content_block(_image_block(f"data:image/png;base64,{i}"))
    assert len(_CACHE) <= _CACHE_MAX
