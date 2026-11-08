from __future__ import annotations

import json

import pytest

from mtmbot.cli import main
from mtmbot.data import write_candles


def test_validate_ok(capsys) -> None:
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "config OK" in out


def test_version_exits() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_no_command_prints_help(capsys) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_backtest_end_to_end(tmp_path, candle_factory) -> None:
    candles = candle_factory([100.0 * 1.002**index for index in range(900)])
    data = write_candles(tmp_path / "data.csv", candles)
    report_json = tmp_path / "report.json"
    report_html = tmp_path / "report.html"

    code = main(
        [
            "backtest",
            "--data",
            str(data),
            "--json",
            str(report_json),
            "--html",
            str(report_html),
            "--out-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert code == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["bars"] == 900
    assert "performance" in payload
    assert "<html" in report_html.read_text(encoding="utf-8")


def test_backtest_with_missing_file(tmp_path, capsys) -> None:
    code = main(["backtest", "--data", str(tmp_path / "nope.csv")])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_download_rejects_unsupported_timeframe(tmp_path) -> None:
    # 90s cannot be built from any Coinbase granularity, so this fails before
    # any network request is made.
    code = main(
        [
            "download",
            "--start",
            "2024-01-01",
            "--timeframe",
            "90s",
            "--out",
            str(tmp_path / "out.csv"),
        ]
    )
    assert code == 2


def test_live_requires_confirmation(tmp_path) -> None:
    code = main(["live", "--state", str(tmp_path / "s.json")])
    assert code == 2
