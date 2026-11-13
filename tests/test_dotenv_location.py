"""The gateway reads the .env in the folder it is started from.

python-dotenv's default searches upward from the file that calls it, which
for an installed copy is inside site-packages, so ``mochi-gateway`` started
in a folder holding a ``.env`` ignored it and ran on defaults. ``python -c``
hid the bug, because dotenv treats it as interactive and searches the
current folder; a script file is what ``mochi-gateway`` is, so the test runs
one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_env_in_the_current_folder_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("MOCHI_PORT=8123\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("from mochi.gateway.config import get_settings\n"
                     "print(get_settings().port)\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "MOCHI_PORT"}
    env["PYTHONPATH"] = str(REPO)
    out = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "8123"


def test_a_real_environment_variable_still_wins(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("MOCHI_PORT=8123\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("from mochi.gateway.config import get_settings\n"
                     "print(get_settings().port)\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO), "MOCHI_PORT": "8456"}
    out = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "8456"
