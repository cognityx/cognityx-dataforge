from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cognityx_dataforge.cli import main
from cognityx_dataforge.config import DataForgeConfig, resolve_dataforge_config


def _config(path: Path) -> Path:
    path.write_text(
        '[models.generator]\nmodel="model-a"\n'
        '[inference]\n'
        'base_url="https://user:password@example.test/v1?access_token=never-show"\n',
        encoding="utf-8",
    )
    return path


def test_static_config_show_uses_execution_resolver_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    path = _config(tmp_path / "dataforge.toml")
    monkeypatch.setattr(
        "cognityx_dataforge.cli._runtime",
        lambda _args: pytest.fail("Storage runtime was constructed"),
    )
    monkeypatch.setattr(
        "cognityx_dataforge.cli._jobs",
        lambda _args: pytest.fail("Jobs database was constructed"),
    )

    assert main(["config", "show", "--config", str(path)]) == 0
    shown = json.loads(capsys.readouterr().out)

    assert shown["configuration_kind"] == "scientific-workload"
    assert shown["master_config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    output = json.dumps(shown)
    assert "user:password" not in output
    assert "never-show" not in output
    assert shown["config_layers"][0]["changed_keys"] == [
        "inference.base_url",
        "models.generator.model",
    ]
    assert shown["field_sources"]["generator.model"] == str(path.resolve())
    assert shown["field_sources"]["generator.provider"] == "built-in"
    assert resolve_dataforge_config(path).configuration == DataForgeConfig.load(path)


def test_static_config_validate_is_same_selection(tmp_path: Path, capsys) -> None:
    path = _config(tmp_path / "dataforge.toml")
    assert main(["config", "show", "--config", str(path)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert main(["config", "validate", "--config", str(path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert shown["master_config"] == validated["master_config"]


def test_missing_config_is_nonzero_json(tmp_path: Path, capsys) -> None:
    assert main(["config", "validate", "--config", str(tmp_path / "missing.toml")]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
