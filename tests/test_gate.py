import json

import pytest

from src.gate import main, passes


def test_gate_pass():
    assert passes({"accuracy": 0.97}, 0.95)


def test_gate_fail():
    assert not passes({"accuracy": 0.90}, 0.95)


def test_gate_cli_exits_nonzero_below_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))
    (tmp_path / "params.yaml").write_text("evaluate:\n  min_accuracy: 0.95\n")
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
