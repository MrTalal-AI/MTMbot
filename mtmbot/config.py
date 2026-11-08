"""Typed configuration with strict validation.

Unknown keys are rejected on purpose: a typo such as ``risk_per_trade_pctt``
must fail loudly instead of silently running with a default position size.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mtmbot.timeframe import TimeframeError, parse_timeframe

__all__ = [
    "BotConfig",
    "ConfigError",
    "CostsConfig",
    "ExchangeConfig",
    "RiskConfig",
    "StrategyConfig",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or unsafe."""


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true/false, got {type(value).__name__}")
    return value


def _as_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _as_table(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a table, got {type(value).__name__}")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {where}: {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Parameters of the Multi-Timeframe Momentum strategy."""

    fast_ema: int = 21
    slow_ema: int = 55
    rsi_period: int = 14
    rsi_long_min: float = 52.0
    rsi_short_max: float = 48.0
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    reward_risk: float = 2.0
    htf_factor: int = 4
    allow_shorts: bool = True
    exit_on_bias_flip: bool = True
    breakeven_at_r: float = 1.0
    trail_at_r: float = 2.0
    trail_atr_mult: float = 2.0

    def __post_init__(self) -> None:
        for name in ("fast_ema", "slow_ema", "rsi_period", "atr_period", "htf_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"strategy.{name} must be an integer")
            if value < 1:
                raise ConfigError(f"strategy.{name} must be >= 1, got {value}")
        if self.fast_ema >= self.slow_ema:
            raise ConfigError(
                f"strategy.fast_ema ({self.fast_ema}) must be < strategy.slow_ema "
                f"({self.slow_ema})"
            )
        if self.htf_factor < 2:
            raise ConfigError(f"strategy.htf_factor must be >= 2, got {self.htf_factor}")
        if not 0.0 <= self.rsi_long_min <= 100.0:
            raise ConfigError(f"strategy.rsi_long_min must be in [0, 100], got {self.rsi_long_min}")
        if not 0.0 <= self.rsi_short_max <= 100.0:
            raise ConfigError(
                f"strategy.rsi_short_max must be in [0, 100], got {self.rsi_short_max}"
            )
        if self.rsi_long_min <= self.rsi_short_max:
            raise ConfigError(
                "strategy.rsi_long_min must be above strategy.rsi_short_max "
                "(otherwise both directions fire at once)"
            )
        if not 0.0 < self.atr_stop_mult <= 20.0:
            raise ConfigError(
                f"strategy.atr_stop_mult must be in (0, 20], got {self.atr_stop_mult}"
            )
        if not 0.0 < self.reward_risk <= 50.0:
            raise ConfigError(f"strategy.reward_risk must be in (0, 50], got {self.reward_risk}")
        if not 0.0 <= self.breakeven_at_r <= 20.0:
            raise ConfigError(
                f"strategy.breakeven_at_r must be in [0, 20], got {self.breakeven_at_r}"
            )
        if not 0.0 <= self.trail_at_r <= 50.0:
            raise ConfigError(f"strategy.trail_at_r must be in [0, 50], got {self.trail_at_r}")
        if not 0.0 < self.trail_atr_mult <= 20.0:
            raise ConfigError(
                f"strategy.trail_atr_mult must be in (0, 20], got {self.trail_atr_mult}"
            )


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Hard limits. These are checked before every order, never optimistic."""

    risk_per_trade_pct: float = 0.5
    max_positions: int = 1
    max_notional_pct: float = 100.0
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 20.0
    min_notional: float = 10.0
    max_stop_distance_pct: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_per_trade_pct <= 10.0:
            raise ConfigError(
                f"risk.risk_per_trade_pct must be in (0, 10], got {self.risk_per_trade_pct}"
            )
        if isinstance(self.max_positions, bool) or not isinstance(self.max_positions, int):
            raise ConfigError("risk.max_positions must be an integer")
        if self.max_positions < 1:
            raise ConfigError(f"risk.max_positions must be >= 1, got {self.max_positions}")
        if not 0.0 < self.max_notional_pct <= 1000.0:
            raise ConfigError(
                f"risk.max_notional_pct must be in (0, 1000], got {self.max_notional_pct}"
            )
        if not 0.0 < self.daily_loss_limit_pct <= 100.0:
            raise ConfigError(
                f"risk.daily_loss_limit_pct must be in (0, 100], got {self.daily_loss_limit_pct}"
            )
        if not 0.0 < self.max_drawdown_pct <= 100.0:
            raise ConfigError(
                f"risk.max_drawdown_pct must be in (0, 100], got {self.max_drawdown_pct}"
            )
        if self.min_notional < 0:
            raise ConfigError(f"risk.min_notional must be >= 0, got {self.min_notional}")
        if not 0.0 < self.max_stop_distance_pct <= 50.0:
            raise ConfigError(
                "risk.max_stop_distance_pct must be in (0, 50], got "
                f"{self.max_stop_distance_pct}"
            )
        if self.daily_loss_limit_pct > self.max_drawdown_pct * 2:
            raise ConfigError(
                "risk.daily_loss_limit_pct is far above max_drawdown_pct; "
                "the daily limit would never matter"
            )


@dataclass(frozen=True, slots=True)
class CostsConfig:
    """Trading costs. Defaults are conservative spot taker assumptions."""

    taker_bps: float = 10.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.taker_bps <= 500.0:
            raise ConfigError(f"costs.taker_bps must be in [0, 500], got {self.taker_bps}")
        if not 0.0 <= self.slippage_bps <= 500.0:
            raise ConfigError(f"costs.slippage_bps must be in [0, 500], got {self.slippage_bps}")

    @property
    def taker_rate(self) -> float:
        """Taker fee as a fraction."""
        return self.taker_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        """Slippage as a fraction."""
        return self.slippage_bps / 10_000.0


@dataclass(frozen=True, slots=True)
class ExchangeConfig:
    """Live execution settings (ccxt). Only used by ``mtmbot live``."""

    id: str = "kraken"
    market_type: str = "spot"
    sandbox: bool = True
    api_key_env: str = "MTMBOT_API_KEY"
    api_secret_env: str = "MTMBOT_API_SECRET"

    def __post_init__(self) -> None:
        _as_str(self.id, "exchange.id")
        if self.market_type not in ("spot", "swap", "future"):
            raise ConfigError(
                f"exchange.market_type must be spot/swap/future, got {self.market_type!r}"
            )
        _as_str(self.api_key_env, "exchange.api_key_env")
        _as_str(self.api_secret_env, "exchange.api_secret_env")


def _strategy_from(data: Mapping[str, Any]) -> StrategyConfig:
    fields = {
        "fast_ema": ("fast_ema", _as_int),
        "slow_ema": ("slow_ema", _as_int),
        "rsi_period": ("rsi_period", _as_int),
        "rsi_long_min": ("rsi_long_min", _as_float),
        "rsi_short_max": ("rsi_short_max", _as_float),
        "atr_period": ("atr_period", _as_int),
        "atr_stop_mult": ("atr_stop_mult", _as_float),
        "reward_risk": ("reward_risk", _as_float),
        "htf_factor": ("htf_factor", _as_int),
        "allow_shorts": ("allow_shorts", _as_bool),
        "exit_on_bias_flip": ("exit_on_bias_flip", _as_bool),
        "breakeven_at_r": ("breakeven_at_r", _as_float),
        "trail_at_r": ("trail_at_r", _as_float),
        "trail_atr_mult": ("trail_atr_mult", _as_float),
    }
    _reject_unknown(data, set(fields), "[strategy]")
    kwargs: dict[str, Any] = {key: caster(data[key], f"strategy.{key}") for key, (_, caster) in fields.items()
              if key in data}
    return StrategyConfig(**kwargs)


def _risk_from(data: Mapping[str, Any]) -> RiskConfig:
    fields = {
        "risk_per_trade_pct": _as_float,
        "max_positions": _as_int,
        "max_notional_pct": _as_float,
        "daily_loss_limit_pct": _as_float,
        "max_drawdown_pct": _as_float,
        "min_notional": _as_float,
        "max_stop_distance_pct": _as_float,
    }
    _reject_unknown(data, set(fields), "[risk]")
    kwargs: dict[str, Any] = {key: caster(data[key], f"risk.{key}") for key, caster in fields.items()
              if key in data}
    return RiskConfig(**kwargs)


def _costs_from(data: Mapping[str, Any]) -> CostsConfig:
    fields = {"taker_bps": _as_float, "slippage_bps": _as_float}
    _reject_unknown(data, set(fields), "[costs]")
    kwargs: dict[str, Any] = {key: caster(data[key], f"costs.{key}") for key, caster in fields.items()
              if key in data}
    return CostsConfig(**kwargs)


def _exchange_from(data: Mapping[str, Any]) -> ExchangeConfig:
    fields = {
        "id": _as_str,
        "market_type": _as_str,
        "sandbox": _as_bool,
        "api_key_env": _as_str,
        "api_secret_env": _as_str,
    }
    _reject_unknown(data, set(fields), "[exchange]")
    kwargs: dict[str, Any] = {key: caster(data[key], f"exchange.{key}") for key, caster in fields.items()
              if key in data}
    return ExchangeConfig(**kwargs)


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Top level bot configuration."""

    symbol: str = "BTC/USD"
    timeframe: str = "1h"
    starting_cash: float = 10_000.0
    data: str = ""
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

    def __post_init__(self) -> None:
        symbol = _as_str(self.symbol, "symbol")
        if "/" not in symbol:
            raise ConfigError(f"symbol must look like BASE/QUOTE, got {symbol!r}")
        base, quote = symbol.split("/", 1)
        if not base or not quote:
            raise ConfigError(f"symbol must look like BASE/QUOTE, got {symbol!r}")
        if not isinstance(self.strategy, StrategyConfig):
            raise ConfigError("strategy must be a StrategyConfig")
        if not isinstance(self.risk, RiskConfig):
            raise ConfigError("risk must be a RiskConfig")
        if not isinstance(self.costs, CostsConfig):
            raise ConfigError("costs must be a CostsConfig")
        if not isinstance(self.exchange, ExchangeConfig):
            raise ConfigError("exchange must be an ExchangeConfig")
        cash = _as_float(self.starting_cash, "starting_cash")
        if cash <= 0:
            raise ConfigError(f"starting_cash must be > 0, got {cash}")
        try:
            parse_timeframe(self.timeframe)
        except TimeframeError as exc:
            raise ConfigError(f"invalid timeframe in config: {exc}") from exc
        # The strategy must be able to warm up on the higher timeframe too.
        htf_step = parse_timeframe(self.timeframe) * self.strategy.htf_factor
        if htf_step <= 0:
            raise ConfigError("higher timeframe is invalid")
        if self.risk.max_positions != 1:
            raise ConfigError(
                "this bot manages one position at a time; set risk.max_positions = 1"
            )

    @property
    def step_seconds(self) -> int:
        """Base timeframe length in seconds."""
        return parse_timeframe(self.timeframe)

    @property
    def htf_seconds(self) -> int:
        """Higher timeframe length in seconds."""
        return self.step_seconds * self.strategy.htf_factor

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BotConfig:
        """Build a config from a decoded TOML/JSON mapping."""
        if not isinstance(data, Mapping):
            raise ConfigError("config root must be a table")
        allowed = {"symbol", "timeframe", "starting_cash", "data", "strategy", "risk", "costs",
                   "exchange"}
        _reject_unknown(data, allowed, "config root")
        kwargs: dict[str, Any] = {}
        if "symbol" in data:
            kwargs["symbol"] = _as_str(data["symbol"], "symbol")
        if "timeframe" in data:
            kwargs["timeframe"] = _as_str(data["timeframe"], "timeframe")
        if "starting_cash" in data:
            kwargs["starting_cash"] = _as_float(data["starting_cash"], "starting_cash")
        if "data" in data:
            kwargs["data"] = _as_str(data["data"], "data")
        if "strategy" in data:
            kwargs["strategy"] = _strategy_from(_as_table(data["strategy"], "strategy"))
        if "risk" in data:
            kwargs["risk"] = _risk_from(_as_table(data["risk"], "risk"))
        if "costs" in data:
            kwargs["costs"] = _costs_from(_as_table(data["costs"], "costs"))
        if "exchange" in data:
            kwargs["exchange"] = _exchange_from(_as_table(data["exchange"], "exchange"))
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, path: str | Path) -> BotConfig:
        """Read and validate a TOML config file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigError(f"config file not found: {file_path}")
        try:
            payload = tomllib.loads(file_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file_path} is not valid TOML: {exc}") from exc
        return cls.from_dict(payload)

    def summary(self) -> dict[str, Any]:
        """Flat, JSON friendly view used in reports and logs."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "starting_cash": self.starting_cash,
            "strategy": {
                "fast_ema": self.strategy.fast_ema,
                "slow_ema": self.strategy.slow_ema,
                "rsi_period": self.strategy.rsi_period,
                "atr_period": self.strategy.atr_period,
                "atr_stop_mult": self.strategy.atr_stop_mult,
                "reward_risk": self.strategy.reward_risk,
                "htf_factor": self.strategy.htf_factor,
                "allow_shorts": self.strategy.allow_shorts,
            },
            "risk": {
                "risk_per_trade_pct": self.risk.risk_per_trade_pct,
                "daily_loss_limit_pct": self.risk.daily_loss_limit_pct,
                "max_drawdown_pct": self.risk.max_drawdown_pct,
                "max_notional_pct": self.risk.max_notional_pct,
            },
            "costs": {
                "taker_bps": self.costs.taker_bps,
                "slippage_bps": self.costs.slippage_bps,
            },
        }


def load_config(path: str | Path | None = None) -> BotConfig:
    """Load a config file, falling back to validated defaults."""
    if path is None:
        return BotConfig()
    return BotConfig.from_toml(path)
