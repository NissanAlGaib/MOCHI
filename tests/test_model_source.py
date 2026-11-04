"""Where Stage II's weights come from, and that loading them is safe.

None of this loads a model. Resolution is tested against folders made in a
temporary directory and a stubbed download, so the suite still runs on a
machine with no weights and no network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from mochi.detect import stage2_semantic as s2
from mochi.detect.stage2_semantic import (
    HUB_PREFIX,
    MODEL_FILES,
    ModelUnavailable,
    resolve_model_dir,
)

REPO = Path(__file__).resolve().parents[1]


def complete_model(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        (folder / name).write_text("x", encoding="utf-8")
    return folder


@pytest.fixture(autouse=True)
def no_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution reads MOCHI_STAGE2_MODEL_DIR first, so a developer who has it
    set would otherwise see every test here resolve to their own folder."""
    monkeypatch.delenv("MOCHI_STAGE2_MODEL_DIR", raising=False)


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Stub ``snapshot_download``: records each request and serves a complete
    folder, so the Hub path is exercised without the network."""
    calls: list[dict] = []
    served = complete_model(tmp_path / "hub-cache")

    def fake(**kwargs):
        calls.append(kwargs)
        return str(served)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return calls


@pytest.fixture
def no_local_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """What an installed package sees: the clone's models/ does not exist."""
    monkeypatch.setattr(s2, "DEFAULT_MODEL_DIR", tmp_path / "absent")


# --- resolution order -------------------------------------------------------

def test_a_local_folder_is_used_as_given(tmp_path: Path) -> None:
    folder = complete_model(tmp_path / "weights")
    assert resolve_model_dir(folder) == folder


def test_a_missing_local_folder_fails_as_missing_and_is_not_downloaded(
        tmp_path: Path, downloads) -> None:
    """``models/e5-fine-tuned`` and ``owner/name`` are the same shape. A typo
    in a path must not fetch whatever a stranger published under that name."""
    with pytest.raises(ModelUnavailable, match="No Stage II model at"):
        resolve_model_dir(tmp_path / "models" / "e5-fine-tuned")
    assert downloads == []


def test_a_hub_spec_downloads_the_pinned_revision(downloads) -> None:
    path = resolve_model_dir(f"{HUB_PREFIX}acme/mochi-e5@abc123")

    assert downloads == [{"repo_id": "acme/mochi-e5", "revision": "abc123",
                          "allow_patterns": list(MODEL_FILES)}]
    assert (path / "model.safetensors").exists()


def test_an_unpinned_hub_spec_is_allowed_but_logged(downloads, caplog) -> None:
    resolve_model_dir(f"{HUB_PREFIX}acme/mochi-e5")
    assert downloads[0]["revision"] is None
    assert "not pinned" in caplog.text


def test_the_clone_folder_wins_when_present(monkeypatch, tmp_path, downloads) -> None:
    """Running from a clone - the evaluation, the tests - must keep loading the
    local weights, never a download."""
    local = complete_model(tmp_path / "clone-models")
    monkeypatch.setattr(s2, "DEFAULT_MODEL_DIR", local)
    monkeypatch.setattr(s2, "DEFAULT_MODEL_REPO", "acme/mochi-e5")
    assert resolve_model_dir(None) == local
    assert downloads == []


def test_an_installed_package_downloads_the_published_weights(
        monkeypatch, no_local_default, downloads) -> None:
    monkeypatch.setattr(s2, "DEFAULT_MODEL_REPO", "acme/mochi-e5")
    monkeypatch.setattr(s2, "DEFAULT_MODEL_REVISION", "abc123")
    resolve_model_dir(None)
    assert downloads[0]["repo_id"] == "acme/mochi-e5"
    assert downloads[0]["revision"] == "abc123"


def test_the_published_weights_are_pinned_to_a_commit() -> None:
    """A branch name would let a re-upload swap the detector under every
    installed copy; only a full commit hash names one exact set of weights."""
    assert s2.DEFAULT_MODEL_REPO
    assert re.fullmatch(r"[0-9a-f]{40}", s2.DEFAULT_MODEL_REVISION or "")


def test_with_nothing_local_and_nothing_published_the_error_says_what_to_do(
        monkeypatch, no_local_default) -> None:
    monkeypatch.setattr(s2, "DEFAULT_MODEL_REPO", None)
    with pytest.raises(ModelUnavailable) as raised:
        resolve_model_dir(None)
    assert "MOCHI_STAGE2_MODEL_DIR" in str(raised.value)
    assert "hf:owner/name@revision" in str(raised.value)


def test_a_partial_folder_names_what_is_missing(tmp_path: Path) -> None:
    folder = complete_model(tmp_path / "weights")
    (folder / "mochi_head.pt").unlink()
    with pytest.raises(ModelUnavailable, match="missing mochi_head.pt"):
        resolve_model_dir(folder)


def test_a_failed_download_explains_the_offline_route(monkeypatch) -> None:
    import huggingface_hub

    def offline(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    with pytest.raises(ModelUnavailable, match="without internet access"):
        resolve_model_dir(f"{HUB_PREFIX}acme/mochi-e5@abc123")


def test_constructing_a_scorer_does_not_resolve_or_download(downloads) -> None:
    """Resolution may mean a 468 MB download, so it waits for first use."""
    scorer = s2.E5Scorer(f"{HUB_PREFIX}acme/mochi-e5@abc123")
    assert downloads == []
    assert scorer.model_dir is None


# --- loading safety ---------------------------------------------------------

def _torch_load_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"):
            yield node


def test_every_torch_load_refuses_pickled_code() -> None:
    """``torch.load`` unpickles unless told not to, and unpickling can run
    code. The weights may now be downloaded, and the dependency floor allows
    torch 2.4 and 2.5, where ``weights_only`` is off by default. So every call
    says it explicitly - this fails if one is added that does not."""
    unsafe, checked = [], 0
    for folder in ("mochi", "training", "eval"):
        for path in sorted((REPO / folder).rglob("*.py")):
            for call in _torch_load_calls(path):
                checked += 1
                flag = next((kw.value for kw in call.keywords
                             if kw.arg == "weights_only"), None)
                if getattr(flag, "value", None) is not True:
                    unsafe.append(f"{path.relative_to(REPO).as_posix()}:{call.lineno}")
    assert checked, "found no torch.load calls - the scan itself is broken"
    assert not unsafe, f"torch.load without weights_only=True: {unsafe}"


def test_the_environment_setting_is_honoured_by_the_wrapper_too(
        monkeypatch, tmp_path, no_local_default) -> None:
    """The proxy reads MOCHI_STAGE2_MODEL_DIR through its settings; the
    wrapper passes no path at all. Resolution reads it itself, so one setting
    covers both surfaces and the error messages are true for both."""
    folder = complete_model(tmp_path / "weights")
    monkeypatch.setenv("MOCHI_STAGE2_MODEL_DIR", str(folder))
    assert resolve_model_dir(None) == folder


def test_an_explicit_path_beats_the_environment(monkeypatch, tmp_path) -> None:
    explicit = complete_model(tmp_path / "explicit")
    monkeypatch.setenv("MOCHI_STAGE2_MODEL_DIR", str(tmp_path / "elsewhere"))
    assert resolve_model_dir(explicit) == explicit
