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
