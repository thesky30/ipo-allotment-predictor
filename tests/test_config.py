import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import config


def test_load_dotenv_parses_values(tmp_path, monkeypatch):
    for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        '# a comment line\n'
        'LLM_API_KEY="sk-abc123"\n'
        'LLM_MODEL=deepseek-chat   # inline comment\n'
        'LLM_BASE_URL=\n'          # empty value -> skipped
        '\n',
        encoding="utf-8",
    )
    config._load_dotenv(env)
    assert os.environ["LLM_API_KEY"] == "sk-abc123"      # quotes stripped
    assert os.environ["LLM_MODEL"] == "deepseek-chat"    # inline comment stripped
    assert "LLM_BASE_URL" not in os.environ              # empty placeholder skipped


def test_existing_env_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "real-from-shell")
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=from-file\n", encoding="utf-8")
    config._load_dotenv(env)
    assert os.environ["LLM_API_KEY"] == "real-from-shell"


def test_missing_dotenv_is_noop(tmp_path):
    config._load_dotenv(tmp_path / "nope.env")  # must not raise
