from __future__ import annotations

import pytest

from mtmbot.config import BotConfig, ConfigError, load_config


def test_defaults_are_valid() -> None:
    config = BotConfig()
    assert config.step_seconds == 3600
    assert config.htf_seconds == 4 * 3600
    assert config.risk.max_positions == 1


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        BotConfig.from_dict({"symbol": "BTC/USD", "starting_cash_typo": 100.0})


def test_unknown_nested_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        BotConfig.from_dict({"risk": {"risk_per_trade_pctt": 1.0}})


def test_fast_ema_must_be_below_slow() -> None:
    with pytest.raises(ConfigError):
        BotConfig.from_dict({"strategy": {"fast_ema": 60, "slow_ema": 20}})


def test_symbol_requires_quote() -> None:
    with pytest.raises(ConfigError):
        BotConfig(symbol="BTCUSD")


def test_invalid_timeframe_is_rejected() -> None:
    with pytest.raises(ConfigError):
        BotConfig(timeframe="7x")


def test_negative_cash_is_rejected() -> None:
    with pytest.raises(ConfigError):
        BotConfig(starting_cash=-5.0)


def test_daily_limit_must_be_sane_against_drawdown() -> None:
    with pytest.raises(ConfigError):
        BotConfig.from_dict(
            {"risk": {"daily_loss_limit_pct": 90.0, "max_drawdown_pct": 10.0}}
        )


def test_risk_pct_bounds() -> None:
    with pytest.raises(ConfigError):
        BotConfig.from_dict({"risk": {"risk_per_trade_pct": 0.0}})
    with pytest.raises(ConfigError):
        BotConfig.from_dict({"risk": {"risk_per_trade_pct": 50.0}})


def test_rsi_thresholds_must_not_overlap() -> None:
    with pytest.raises(ConfigError):
        BotConfig.from_dict({"strategy": {"rsi_long_min": 45.0, "rsi_short_max": 55.0}})


def test_toml_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'symbol = "ETH/USD"\ntimeframe = "15m"\nstarting_cash = 5000.0\n\n'
        "[risk]\nrisk_per_trade_pct = 1.5\n\n"
        "[costs]\ntaker_bps = 8.0\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.symbol == "ETH/USD"
    assert config.timeframe == "15m"
    assert config.step_seconds == 900
    assert config.risk.risk_per_trade_pct == 1.5
    assert config.costs.taker_bps == 8.0


def test_missing_config_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("/definitely/not/here.toml")


def test_invalid_toml(tmp_path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("symbol = [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(path)


def test_summary_shape() -> None:
    summary = BotConfig().summary()
    assert summary["symbol"] == "BTC/USD"
    assert set(summary) >= {"symbol", "timeframe", "strategy", "risk", "costs"}


def test_max_positions_must_stay_one() -> None:
    with pytest.raises(ConfigError, match="one position"):
        BotConfig.from_dict({"risk": {"max_positions": 3}})
