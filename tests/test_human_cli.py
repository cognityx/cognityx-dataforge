import json

from cognityx_dataforge import cli
from cognityx_dataforge.human import render_human


def test_human_renderer_covers_empty_table_nested_and_full_values() -> None:
    assert render_human([]) == "No records."
    uri = "storage://local-main/datasets/example/manifest.json"
    output = render_human(
        [{"dataset_id": "dataset-complete-identifier", "manifest_uri": uri}]
    )
    assert "dataset-complete-identifier" in output
    assert uri in output
    assert "\x1b" not in output
    assert "Nested:\n  State: ready" in render_human({"nested": {"state": "ready"}})


def test_config_json_default_is_unchanged_and_human_calls_resolver_once(
    monkeypatch, capsys
) -> None:
    calls = 0
    payload = {
        "component": "dataforge",
        "valid": True,
        "master_config": {"path": "/tmp/dataforge.toml", "sha256": "a" * 64},
        "config_layers": [],
        "overrides": [],
        "effective": {"recipe": "paragraph-qa"},
        "warnings": [],
        "errors": [],
    }

    class Resolution:
        def to_dict(self):
            return payload

    def resolve(path):
        nonlocal calls
        calls += 1
        return Resolution()

    monkeypatch.setattr(cli, "resolve_dataforge_config", resolve)
    assert cli.main(["config", "show", "--config", "input.toml"]) == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert calls == 1

    assert cli.main(["config", "show", "--config", "input.toml", "--human"]) == 0
    output = capsys.readouterr().out
    assert calls == 2
    assert "Component: dataforge" in output
    assert "a" * 64 in output
    assert not output.lstrip().startswith("{")


def test_dataset_export_remains_silent_and_has_no_human_option(capsys) -> None:
    parser_exit = None
    try:
        cli.main(
            [
                "dataset",
                "export",
                "storage://local-main/datasets/example/manifest.json",
                "--output",
                "records.jsonl",
                "--human",
            ]
        )
    except SystemExit as exc:
        parser_exit = exc.code

    assert parser_exit == 2
    assert capsys.readouterr().out == ""


def test_job_cancel_default_preserves_existing_json_key_order(
    monkeypatch, capsys
) -> None:
    class Jobs:
        def request_cancel(self, job_id):
            assert job_id == "job-full-id"

        def get(self, job_id):
            assert job_id == "job-full-id"
            return type("Record", (), {"state": "cancel_requested"})()

    monkeypatch.setattr(cli, "_jobs", lambda args: Jobs())
    assert cli.main(["job", "cancel", "job-full-id"]) is None
    assert capsys.readouterr().out == (
        '{\n  "job_id": "job-full-id",\n  "state": "cancel_requested"\n}\n'
    )
