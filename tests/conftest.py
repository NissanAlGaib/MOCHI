"""Isolate the test suite from whatever is in the developer's ``.env``.

``mochi.gateway.config`` calls ``load_dotenv()`` at import and caches the
result with ``lru_cache``, so every test inherits the machine's real
configuration. That is fine until the configuration changes, at which point
tests start failing for reasons that have nothing to do with the code under
test.

That is exactly what happened on 7 October 2026. Enabling Stage II in ``.env``
for the attack simulation broke two tests:

``test_unbuilt_stages_report_not_run``
    asserts Stage II reports ``not_run`` - true only while Stage II is off.

``test_mochi_only_fields_are_stripped_before_upstream``
    sends ``<p>scraped page</p>`` as web content and expects 200. With Stage II
    live that string scores 0.6733 and the request is refused. The test is
    about dropping MOCHI-only fields before the upstream call; whether its
    sample text happens to trip a detector is irrelevant to what it checks.

Neither was a real regression. Both were the suite reading a file it should
never have depended on.

The fixture below pins the configuration the suite assumes, so the tests pass
on any machine regardless of local ``.env`` contents, and a developer running
the gateway with Stage II enabled no longer sees phantom failures.

**Known gap:** with Stage II pinned off, no test exercises the gateway with the
real semantic detector loaded. Tests that need Stage II inject a deterministic
stub instead (see ``FixedScorer`` in ``test_mitigate.py``), which keeps the
suite fast and runnable without a 466 MB model. The cost is that defects only
visible with the real model loaded - as the system-prompt enforcement defect
was - cannot surface here. That class of problem needs the system-level
evaluation in ``eval/``, not unit tests.
"""

from __future__ import annotations

import os

import pytest

#: Configuration the suite is written against. Values chosen to match the
#: documented defaults in ``mochi/gateway/config.py``, not to make tests pass:
#: a fixture that silently diverges from the shipped defaults would hide real
#: regressions rather than isolate irrelevant ones.
TEST_ENVIRONMENT = {
    "MOCHI_ENABLE_STAGE1": "true",
    "MOCHI_ENABLE_STAGE2": "false",
    "MOCHI_ENABLE_TAGALOG_TRANSLATION": "false",
    "MOCHI_ENABLE_SESSION_RISK": "true",
    "MOCHI_ENABLE_OUTBOUND": "true",
    "MOCHI_LOG_PAYLOADS": "false",
    "MOCHI_SYSTEM_PROMPT_FILE": "",
    "MOCHI_ENFORCE_ON_TRUSTED": "false",
    "MOCHI_STAGE2_DEVICE": "",
    "OPENAI_API_KEY": "sk-test-key-not-real",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "TARGET_LLM_PROVIDER": "openai",
    "TARGET_LLM_MODEL": "gpt-4o-mini",
}


@pytest.fixture(autouse=True, scope="session")
def pinned_settings() -> None:
    """Replace the process environment for the whole session.

    Session-scoped and autouse: ``get_settings`` is ``lru_cache``d, so the
    first call anywhere freezes the configuration for the run. Patching
    per-test would be both slower and unreliable, because whichever test ran
    first would win.
    """
    from mochi.gateway.config import get_settings

    saved = {key: os.environ.get(key) for key in TEST_ENVIRONMENT}
    os.environ.update(TEST_ENVIRONMENT)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
