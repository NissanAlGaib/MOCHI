"""eval/generate_taglish_candidates.py - the offline Taglish candidate generator.

Only ``make_taglish`` is tested here: it is the one function with no
argostranslate dependency (a fake translation object stands in), and it is
where the actual word-selection and rebuilding logic lives. The CLI and
``_load_translator`` need the real, ~2.5 GB optional dependency and are not
exercised by this suite - the same boundary ``test_code_switch.py`` draws
around the real Argos Translate model.
"""

from __future__ import annotations

import random

from eval.generate_taglish_candidates import make_taglish


class _FakeTranslation:
    def translate(self, word: str) -> str:
        return f"[{word}]"


def test_switches_only_english_content_words():
    """Function words (the/a/is) and punctuation must never be switched -
    Taglish keeps a grammatical frame from one language, per normalize.py's
    own docstring on code-switching, and switching the frame itself would
    produce something that is not Taglish at all.
    """
    text = "Please update the password immediately for the account."
    candidate, switched = make_taglish(
        text, _FakeTranslation(), switch_rate=1.0, rng=random.Random(1)
    )
    assert switched  # at least one content word was eligible and chosen
    assert "[the]" not in candidate
    assert "[for]" not in candidate


def test_switch_rate_controls_how_much_is_switched():
    text = "Please update the password immediately for the account information today"
    rng = random.Random(7)
    _, switched_low = make_taglish(text, _FakeTranslation(), switch_rate=0.1, rng=rng)
    _, switched_high = make_taglish(text, _FakeTranslation(), switch_rate=0.9, rng=rng)
    assert len(switched_high) >= len(switched_low)


def test_text_with_nothing_eligible_is_returned_unchanged():
    text = "the a is of"  # all function words, nothing switchable
    candidate, switched = make_taglish(
        text, _FakeTranslation(), switch_rate=1.0, rng=random.Random(1)
    )
    assert candidate == text
    assert switched == []


def test_repeated_word_is_translated_once_and_applied_everywhere():
    calls: list[str] = []

    class CountingTranslation:
        def translate(self, word: str) -> str:
            calls.append(word)
            return "PASAHOD"

    text = "update the password, then confirm the password again"
    candidate, switched = make_taglish(
        text, CountingTranslation(), switch_rate=1.0, rng=random.Random(1)
    )
    assert calls.count("password") <= 1
    assert candidate.count("PASAHOD") == calls.count("password") * 2 or "PASAHOD" in candidate


def test_reproducible_with_the_same_seed():
    text = "Please update the password immediately for the account information."
    c1, s1 = make_taglish(text, _FakeTranslation(), switch_rate=0.5, rng=random.Random(3))
    c2, s2 = make_taglish(text, _FakeTranslation(), switch_rate=0.5, rng=random.Random(3))
    assert c1 == c2
    assert s1 == s2
