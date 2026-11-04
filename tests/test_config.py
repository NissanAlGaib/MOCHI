"""Shipped defaults that the rest of the suite deliberately overrides.

``conftest.py`` pins the environment for the whole run, and pins Stage II off
so the suite needs no model. That makes the real default invisible to every
other test - so it is checked here, against an environment with the setting
removed rather than set.
"""

from __future__ import annotations

import pytest

from mochi.gateway.config import get_settings


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch):
    """Remove variables for one test, with the settings cache cleared on
    both sides so neither this test nor the next sees a stale value."""
    def _unset(*names: str) -> None:
        for name in names:
            monkeypatch.delenv(name, raising=False)
        get_settings.cache_clear()

    yield _unset
    get_settings.cache_clear()


def test_stage2_is_on_by_default(unset) -> None:
    """Stage I alone measured recall 0.059. A gateway that started without
    Stage II because nobody set a flag would miss almost every attack, and
    would do it silently - so the default is on, and this pins it."""
    unset("MOCHI_ENABLE_STAGE2")
    assert get_settings().enable_stage2 is True


def test_stage2_can_still_be_disabled_explicitly(unset, monkeypatch) -> None:
    """The Stage-I-only ablation arm depends on this."""
    unset("MOCHI_ENABLE_STAGE2")
    monkeypatch.setenv("MOCHI_ENABLE_STAGE2", "false")
    get_settings.cache_clear()
    assert get_settings().enable_stage2 is False
