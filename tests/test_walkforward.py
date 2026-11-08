from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise

import pytest

from mtmbot.cli import main
from mtmbot.config import BotConfig, RiskConfig, StrategyConfig
from mtmbot.data import write_candles
from mtmbot.models import Candle
from mtmbot.walkforward import (
    WalkForwardError,
    WalkForwardSettings,
    expand_grid,
    run_walk_forward,
    select_strategy,
    split_folds,
)

GRID = {"atr_stop_mult": (1.5, 2.5)}


def _config() -> BotConfig:
    return BotConfig(
        symbol="BTC/USD",
        timeframe="1h",
        starting_cash=10_000.0,
        strategy=StrategyConfig(
            fast_ema=10,
            slow_ema=30,
            rsi_period=14,
            atr_period=14,
            htf_factor=4,
            atr_stop_mult=2.0,
            reward_risk=2.0,
            rsi_long_min=52.0,
            rsi_short_max=48.0,
        ),
        risk=RiskConfig(daily_loss_limit_pct=20.0, max_drawdown_pct=50.0),
    )


def _settings(**overrides: object) -> WalkForwardSettings:
    base = WalkForwardSettings(folds=2, train_ratio=0.6, objective="return", min_trades=0)
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_expand_grid_is_deterministic() -> None:
    grid = {"fast_ema": (13, 21), "slow_ema": (55, 89)}
    combos = expand_grid(grid)
    assert len(combos) == 4
    assert combos[0] == {"fast_ema": 13, "slow_ema": 55}
    assert combos == expand_grid(grid)


def test_expand_grid_rejects_unknown_parameter() -> None:
    with pytest.raises(WalkForwardError, match="unknown strategy parameter"):
        expand_grid({"magic": (1, 2)})


def test_expand_grid_rejects_wrong_types() -> None:
    with pytest.raises(WalkForwardError, match="integers"):
        expand_grid({"fast_ema": (13.5,)})
    with pytest.raises(WalkForwardError, match="numbers"):
        expand_grid({"atr_stop_mult": ("wide",)})


def test_split_folds_windows_are_contiguous() -> None:
    windows = split_folds(1_000, 0.6, 4, 100)
    assert windows[0] == (0, 600, 700)
    for previous, current in pairwise(windows):
        assert previous[2] == current[1]  # test windows never overlap or skip
        assert current[1] - current[0] == 600
        assert current[2] - current[1] == 100
    assert windows[-1][2] <= 1_000


def test_split_folds_refuses_short_history() -> None:
    with pytest.raises(WalkForwardError, match="not enough history"):
        split_folds(300, 0.6, 4, 200)


def test_settings_validation() -> None:
    with pytest.raises(WalkForwardError):
        WalkForwardSettings(folds=1)
    with pytest.raises(WalkForwardError):
        WalkForwardSettings(objective="vibes")
    with pytest.raises(WalkForwardError):
        WalkForwardSettings(train_ratio=0.95)
    with pytest.raises(WalkForwardError):
        WalkForwardSettings(min_trades=-1)


def test_walk_forward_raises_for_tiny_series(candle_factory) -> None:
    with pytest.raises(WalkForwardError):
        run_walk_forward(_config(), candle_factory([100.0, 101.0]), _settings())


def test_walk_forward_chains_capital_across_folds(candle_factory) -> None:
    from conftest import trending_closes

    candles = candle_factory(trending_closes(900))
    result = run_walk_forward(_config(), candles, _settings(), GRID)

    assert len(result.folds) == 2
    assert result.grid_size == 2
    assert result.oos_equity_curve[0].ts == result.folds[0].test_start
    for previous, current in pairwise(result.folds):
        assert current.equity_start == pytest.approx(previous.equity_end)
    timestamps = [point.ts for point in result.oos_equity_curve]
    assert all(later > earlier for earlier, later in pairwise(timestamps))


def test_walk_forward_reports_is_and_oos(candle_factory) -> None:
    from conftest import trending_closes

    candles = candle_factory(trending_closes(900))
    result = run_walk_forward(_config(), candles, _settings(), GRID)

    assert result.is_mean_return_pct is not None
    assert result.is_median_return_pct is not None
    assert result.oos_performance.trades >= 0
    if abs(result.is_mean_return_pct) > 1e-9:
        assert result.efficiency == pytest.approx(
            result.oos_return_pct / result.is_mean_return_pct
        )

    payload = result.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["summary"]["oos_return_pct"] == pytest.approx(result.oos_return_pct)
    assert len(payload["folds"]) == 2
    assert payload["folds"][0]["params"]


def test_selection_never_reads_the_test_window(candle_factory) -> None:
    """Mutating only fold 0's test bars must not change fold 0's parameters."""
    from conftest import trending_closes

    candles = candle_factory(trending_closes(900))
    settings = _settings()
    first = run_walk_forward(_config(), candles, settings, GRID)

    fold = first.folds[0]
    mutated = []
    for candle in candles:
        if fold.test_start <= candle.ts <= fold.test_end:
            mutated.append(
                Candle(
                    ts=candle.ts,
                    open=candle.open * 1.4,
                    high=candle.high * 1.4,
                    low=candle.low * 1.4,
                    close=candle.close * 1.4,
                    volume=candle.volume,
                )
            )
        else:
            mutated.append(candle)

    second = run_walk_forward(_config(), mutated, settings, GRID)
    assert second.folds[0].params == fold.params
    assert second.folds[0].train_start == fold.train_start


def test_select_strategy_ignores_unusable_combinations(candle_factory) -> None:
    from conftest import trending_closes

    candles = candle_factory(trending_closes(600))
    selection = select_strategy(
        _config(),
        candles,
        _settings(),
        {"fast_ema": (10, 60), "slow_ema": (30,)},  # 60/30 is invalid and must be skipped
    )
    assert selection.params["fast_ema"] == 10
    assert selection.candidates == 1


def test_warmup_uses_the_slowest_grid_candidate(candle_factory) -> None:
    from conftest import trending_closes

    candles = candle_factory(trending_closes(900))
    with pytest.raises(WalkForwardError, match="not enough history"):
        run_walk_forward(_config(), candles, _settings(), {"slow_ema": (200,)})


def test_flat_market_gives_flat_out_of_sample(candle_factory) -> None:
    candles = candle_factory([100.0] * 900, wick=0.0002)
    result = run_walk_forward(_config(), candles, _settings(min_trades=10), GRID)
    assert result.oos_return_pct == pytest.approx(0.0)
    assert any("relaxed" in note for note in result.notes)


def test_walkforward_cli_end_to_end(tmp_path, candle_factory) -> None:
    # The default grid includes slow_ema=89, which needs 445 warm-up bars, so
    # each test window must be at least that long.
    candles = candle_factory([100.0 * 1.0015**index for index in range(2_600)])
    data = write_candles(tmp_path / "data.csv", candles)
    report_json = tmp_path / "wf.json"
    report_html = tmp_path / "wf.html"

    code = main(
        [
            "walkforward",
            "--data",
            str(data),
            "--folds",
            "2",
            "--min-trades",
            "0",
            "--json",
            str(report_json),
            "--html",
            str(report_html),
        ]
    )
    assert code == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["settings"]["folds"] == 2
    assert "oos_performance" in payload
    assert "<html" in report_html.read_text(encoding="utf-8")


def test_walkforward_cli_short_history(tmp_path, candle_factory) -> None:
    candles = candle_factory([100.0 * 1.001**index for index in range(400)])
    data = write_candles(tmp_path / "data.csv", candles)
    assert main(["walkforward", "--data", str(data), "--folds", "4"]) == 2
