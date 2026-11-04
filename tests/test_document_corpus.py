"""The document renderers must not defuse the attacks they carry.

A payload that the renderer splits or mangles fails against the bare model as
well as against MOCHI, and the arm then credits the defence with a save the
channel made. The first PDF layout did exactly that to the long hex payloads,
so every attack is round-tripped here: rendered, extracted the way MOCHI
extracts it, and compared with the original text.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("fpdf")

from eval.av_corpus import user_turn_attacks  # noqa: E402
from eval.document_corpus import (  # noqa: E402
    PDF_VARIANTS, WEB_VARIANTS, attack_page, attack_pdf, file_block,
)
from mochi.preprocess.extract import extract_from_content_block  # noqa: E402
from mochi.preprocess.html_extract import extract_html  # noqa: E402

ATTACKS = user_turn_attacks()


def words(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("variant", PDF_VARIANTS)
def test_every_attack_survives_the_pdf(variant: str) -> None:
    broken = [
        item.id for item in ATTACKS
        if words(item.user_input) not in words(extract_from_content_block(
            file_block(attack_pdf(item.user_input, variant), f"{item.id}-{variant}.pdf")).text)
    ]
    assert broken == [], f"{variant}: renderer altered {broken}"


@pytest.mark.parametrize("variant", WEB_VARIANTS)
def test_every_attack_survives_the_page(variant: str) -> None:
    broken = []
    for item in ATTACKS:
        expected = item.user_input.replace("--", "- -") if variant == "comment" \
            else item.user_input
        if words(expected) not in words(" ".join(
                extract_html(attack_page(item.user_input, variant)).all_segments)):
            broken.append(item.id)
    assert broken == [], f"{variant}: renderer altered {broken}"
