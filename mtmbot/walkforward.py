"""Walk-forward validation.

The only honest way to test a parameter search:

1. Split the history into consecutive *test* windows and, for each one, use
   only the data that came **before** it to pick parameters.
2. Trade that test window with the parameters chosen from the past, carrying
   the capital forward fold by fold.
3. Compare the in-sample (train) result with the out-of-sample (test) result
   and report the degradation instead of hiding it.

No test-window bar ever influences the parameters used to trade it, so the
out-of-sample column is the only evidence. A large gap between the two columns
means the parameter search is fitting noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import product
from statistics import fmean, median
from typing import Any

from mtmbot.backtest import BacktestError, BacktestResult, EquityPoint, run_backtest
from mtmbot.config import BotConfig, ConfigError, StrategyConfig
from mtmbot.metrics import Performance, compute_performance
from mtmbot.models import Candle, Trade
from mtmbot.strategy.mtm import MTMStrategy

__all__ = [
    "DEFAULT_GRID",
    "FoldResult",
    "Selection",
    "WalkForwardError",
    "WalkForwardResult",
    "WalkForwardSettings",
    "expand_grid",
    "run_walk_forward",
    "select_strategy",
    "split_folds",
]

OBJECTIVES = ("sharpe", "sortino", "return", "expectancy", "calmar")

DEFAULT_GRID: dict[str, tuple[float | int, ...]] = {
    "fast_ema": (13, 21),
    "slow_ema": (55, 89),
    "atr_stop_mult": (1.5, 2.5),
    "reward_risk": (2.0, 3.0),
}

_PARAM_TYPES: dict[str, type] = {
    "fast_ema": int,
    "slow_ema": int,
    "rsi_period": int,
    "atr_period": int,
    "htf_factor": int,
    "rsi_long_min": float,
    "rsi_short_max": float,
    "atr_stop_mult": float,
    "reward_risk": float,
    "breakeven_at_r": float,
    "trail_at_r": float,
    "trail_atr_mult": float,
    "allow_shorts": bool,
    "exit_on_bias_flip": bool,
}

_PARAM_LABELS = {
    "fast_ema": "fast",
    "slow_ema": "slow",
    "rsi_period": "rsi",
    "atr_period": "atr",
    "htf_factor": "htf",
    "rsi_long_min": "rsiL",
    "rsi_short_max": "rsiS",
    "atr_stop_mult": "stop",
    "reward_risk": "rr",
    "breakeven_at_r": "be",
    "trail_at_r": "trail",
    "trail_atr_mult": "trailX",
    "allow_shorts": "shorts",
    "exit_on_bias_flip": "flip",
}


class WalkForwardError(RuntimeError):
    """Raised when a walk-forward run is impossible or meaningless."""


@dataclass(frozen=True, slots=True)
class WalkForwardSettings:
    """How to cut the history and how to score candidates."""

    folds: int = 4
    train_ratio: float = 0.6
    objective: str = "sharpe"
    min_trades: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.folds, bool) or not isinstance(self.folds, int):
            raise WalkForwardError("folds must be an integer")
        if self.folds < 2:
            raise WalkForwardError(f"folds must be >= 2, got {self.folds}")
        if not 0.2 <= self.train_ratio <= 0.9:
            raise WalkForwardError(
                f"train_ratio must be in [0.2, 0.9], got {self.train_ratio}"
            )
        if self.objective not in OBJECTIVES:
            raise WalkForwardError(
                f"objective must be one of {', '.join(OBJECTIVES)}; got {self.objective!r}"
            )
        if isinstance(self.min_trades, bool) or not isinstance(self.min_trades, int):
            raise WalkForwardError("min_trades must be an integer")
        if self.min_trades < 0:
            raise WalkForwardError(f"min_trades must be >= 0, got {self.min_trades}")


def expand_grid(grid: Mapping[str, Sequence[float | int | bool]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a deterministic list of candidates."""
    if not isinstance(grid, Mapping):
        raise WalkForwardError("grid must be a mapping of parameter -> values")
    if not grid:
        raise WalkForwardError("grid is empty")
    unknown = sorted(set(grid) - set(_PARAM_TYPES))
    if unknown:
        raise WalkForwardError(
            f"unknown strategy parameter(s) in grid: {', '.join(unknown)}"
        )

    keys = sorted(grid)
    values_per_key: list[Sequence[float | int | bool]] = []
    for key in keys:
        values = grid[key]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise WalkForwardError(f"grid[{key!r}] must be a list of values")
        if not values:
            raise WalkForwardError(f"grid[{key!r}] is empty")
        values_per_key.append(values)

    combos: list[dict[str, Any]] = []
    for combo in product(*values_per_key):
        params: dict[str, Any] = {}
        for key, value in zip(keys, combo, strict=True):
            expected = _PARAM_TYPES[key]
            if expected is bool:
                if not isinstance(value, bool):
                    raise WalkForwardError(f"grid[{key!r}] values must be true/false")
                params[key] = value
            elif expected is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise WalkForwardError(f"grid[{key!r}] values must be integers")
                params[key] = value
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise WalkForwardError(f"grid[{key!r}] values must be numbers")
                if not math.isfinite(float(value)):
                    raise WalkForwardError(f"grid[{key!r}] contains a non-finite value")
                params[key] = float(value)
        combos.append(params)
    return combos


def split_folds(
    total_bars: int, train_ratio: float, folds: int, warmup_bars: int
) -> list[tuple[int, int, int]]:
    """Return ``(train_start, test_start, test_end)`` index triples.

    Test windows are contiguous and never overlap; each training window is the
    fixed block that ends exactly when its test window starts (rolling, not
    anchored), so the parameters adapt to the recent past.
    """
    if total_bars <= 0:
        raise WalkForwardError("no candles to split")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 1:
        raise WalkForwardError(f"folds must be a positive integer, got {folds!r}")
    if not 0.0 < train_ratio < 1.0:
        raise WalkForwardError(f"train_ratio must be in (0, 1), got {train_ratio!r}")

    train_size = int(total_bars * train_ratio)
    test_size = (total_bars - train_size) // folds
    minimum = max(warmup_bars, 2)
    if train_size < minimum or test_size < minimum:
        raise WalkForwardError(
            f"not enough history for walk-forward: {total_bars} bars gives a "
            f"{train_size}-bar train window and a {test_size}-bar test window, but the "
            f"strategy needs at least {minimum} bars for each. Download more data or "
            "use fewer folds."
        )
    windows: list[tuple[int, int, int]] = []
    for index in range(folds):
        test_start = train_size + index * test_size
        windows.append((test_start - train_size, test_start, test_start + test_size))
    return windows


@dataclass(frozen=True, slots=True)
class Selection:
    """The parameter choice made on one training window."""

    params: dict[str, Any]
    strategy: StrategyConfig
    train_result: BacktestResult
    candidates: int
    relaxed: bool = False


def _params_label(params: Mapping[str, Any]) -> str:
    return " ".join(
        f"{_PARAM_LABELS.get(key, key)}={value}" for key, value in sorted(params.items())
    )


def _score(performance: Performance, objective: str) -> float:
    values = {
        "sharpe": performance.sharpe,
        "sortino": performance.sortino,
        "return": performance.total_return_pct,
        "expectancy": performance.expectancy_r,
        "calmar": performance.calmar,
    }
    score = values[objective]
    return score if math.isfinite(score) else float("-inf")


def select_strategy(
    config: BotConfig,
    train_candles: Sequence[Candle],
    settings: WalkForwardSettings,
    grid: Mapping[str, Sequence[float | int | bool]] | None = None,
    combos: Sequence[dict[str, Any]] | None = None,
) -> Selection:
    """Pick the best parameters using *only* ``train_candles``."""
    candidates = list(combos) if combos is not None else expand_grid(grid or DEFAULT_GRID)
    if not candidates:
        raise WalkForwardError("grid produced no parameter combinations")

    evaluated: list[tuple[dict[str, Any], StrategyConfig, BacktestResult]] = []
    for params in candidates:
        try:
            strategy = replace(config.strategy, **params)
        except (ConfigError, TypeError, ValueError):
            continue
        trial = replace(config, strategy=strategy)
        try:
            result = run_backtest(trial, train_candles)
        except BacktestError:
            continue
        evaluated.append((dict(params), strategy, result))

    if not evaluated:
        raise WalkForwardError(
            "no grid combination produced a usable backtest on the training window"
        )

    eligible = [
        item for item in evaluated if item[2].performance.trades >= settings.min_trades
    ]
    relaxed = not eligible
    pool = eligible or evaluated
    best = pool[0]
    best_score = _score(best[2].performance, settings.objective)
    for item in pool[1:]:
        score = _score(item[2].performance, settings.objective)
        if score > best_score + 1e-12:
            best = item
            best_score = score

    return Selection(
        params=best[0],
        strategy=best[1],
        train_result=best[2],
        candidates=len(evaluated),
        relaxed=relaxed,
    )


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One train/test split and what it produced."""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    params: dict[str, Any]
    is_performance: Performance
    oos_performance: Performance
    candidates: int
    equity_start: float
    equity_end: float

    @property
    def params_label(self) -> str:
        return _params_label(self.params)


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregated walk-forward outcome: the out-of-sample column is the result."""

    symbol: str
    timeframe_seconds: int
    settings: WalkForwardSettings
    folds: tuple[FoldResult, ...]
    oos_equity_curve: tuple[EquityPoint, ...]
    oos_trades: tuple[Trade, ...]
    oos_performance: Performance
    is_mean_return_pct: float
    is_median_return_pct: float
    efficiency: float
    stability_pct: float
    grid_size: int
    notes: tuple[str, ...] = ()
    config_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def oos_return_pct(self) -> float:
        return self.oos_performance.total_return_pct

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the whole run."""
        return {
            "symbol": self.symbol,
            "timeframe_seconds": self.timeframe_seconds,
            "settings": {
                "folds": self.settings.folds,
                "train_ratio": self.settings.train_ratio,
                "objective": self.settings.objective,
                "min_trades": self.settings.min_trades,
            },
            "grid_size": self.grid_size,
            "summary": {
                "oos_return_pct": self.oos_return_pct,
                "is_mean_return_pct": self.is_mean_return_pct,
                "is_median_return_pct": self.is_median_return_pct,
                "efficiency": self.efficiency,
                "stability_pct": self.stability_pct,
                "oos_trades": self.oos_performance.trades,
            },
            "oos_performance": self.oos_performance.to_dict(),
            "folds": [
                {
                    "index": fold.index,
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    "params": fold.params,
                    "candidates": fold.candidates,
                    "equity_start": fold.equity_start,
                    "equity_end": fold.equity_end,
                    "is_performance": fold.is_performance.to_dict(),
                    "oos_performance": fold.oos_performance.to_dict(),
                }
                for fold in self.folds
            ],
            "notes": list(self.notes),
            "config": self.config_summary,
        }


def _max_warmup(config: BotConfig, combos: Sequence[dict[str, Any]]) -> int:
    """Warm-up needed by the slowest candidate in the grid.

    Using only the base config would let a fold test window be long enough for
    the default parameters but too short for a slower grid candidate.
    """
    warmup = MTMStrategy(config.strategy, config.step_seconds).warmup_bars
    for params in combos:
        try:
            candidate = replace(config.strategy, **params)
        except (ConfigError, TypeError, ValueError):
            continue
        warmup = max(warmup, MTMStrategy(candidate, config.step_seconds).warmup_bars)
    return warmup


def run_walk_forward(
    config: BotConfig,
    candles: Sequence[Candle],
    settings: WalkForwardSettings | None = None,
    grid: Mapping[str, Sequence[float | int | bool]] | None = None,
) -> WalkForwardResult:
    """Run a rolling walk-forward and return in-sample *and* out-of-sample stats."""
    if not isinstance(config, BotConfig):
        raise TypeError(f"config must be a BotConfig, got {type(config).__name__}")
    active = settings or WalkForwardSettings()
    series = list(candles)
    if len(series) < 3:
        raise WalkForwardError("need more than 2 candles for walk-forward")

    combos = expand_grid(grid or DEFAULT_GRID)
    warmup = _max_warmup(config, combos)
    windows = split_folds(len(series), active.train_ratio, active.folds, warmup)

    equity = config.starting_cash
    curve: list[EquityPoint] = []
    exposures: list[float] = []
    all_trades: list[Trade] = []
    fees = 0.0
    notes: list[str] = []
    folds: list[FoldResult] = []
    chosen: list[str] = []

    for index, (train_start, test_start, test_end) in enumerate(windows, start=1):
        if equity <= 0:
            notes.append(f"fold {index}: capital exhausted; remaining folds skipped")
            break
        train = series[train_start:test_start]
        test = series[test_start:test_end]
        selection = select_strategy(config, train, active, combos=combos)
        if selection.relaxed:
            notes.append(
                f"fold {index}: no candidate reached {active.min_trades} in-sample trades; "
                "the minimum was relaxed for this fold"
            )

        trial_config = replace(config, strategy=selection.strategy, starting_cash=equity)
        oos = run_backtest(trial_config, test)
        folds.append(
            FoldResult(
                index=index,
                train_start=train[0].ts,
                train_end=train[-1].ts,
                test_start=test[0].ts,
                test_end=test[-1].ts,
                params=selection.params,
                is_performance=selection.train_result.performance,
                oos_performance=oos.performance,
                candidates=selection.candidates,
                equity_start=equity,
                equity_end=oos.ending_equity,
            )
        )
        chosen.append(_params_label(selection.params))
        curve.extend(oos.equity_curve)
        exposures.extend(
            1.0 if point.exposure > 0 else 0.0 for point in oos.equity_curve
        )
        all_trades.extend(oos.trades)
        fees += oos.performance.fees_paid
        if oos.open_position_at_end:
            notes.append(
                f"fold {index}: a position was still open at the end; equity is marked to market"
            )
        equity = oos.ending_equity

    if not folds or not curve:
        raise WalkForwardError("walk-forward produced no folds")

    oos_performance = compute_performance(
        equity_curve=[(point.ts, point.equity) for point in curve],
        exposure=exposures,
        trades=all_trades,
        step_seconds=config.step_seconds,
        starting_equity=config.starting_cash,
        fees_paid=fees,
    )

    is_returns = [fold.is_performance.total_return_pct for fold in folds]
    is_mean = fmean(is_returns)
    is_median = median(is_returns)
    efficiency = oos_performance.total_return_pct / is_mean if abs(is_mean) > 1e-9 else 0.0
    mode_count = max((chosen.count(label) for label in set(chosen)), default=0)
    stability = 100.0 * mode_count / len(chosen) if chosen else 0.0

    return WalkForwardResult(
        symbol=config.symbol,
        timeframe_seconds=config.step_seconds,
        settings=active,
        folds=tuple(folds),
        oos_equity_curve=tuple(curve),
        oos_trades=tuple(all_trades),
        oos_performance=oos_performance,
        is_mean_return_pct=is_mean,
        is_median_return_pct=is_median,
        efficiency=efficiency,
        stability_pct=stability,
        grid_size=len(combos),
        notes=tuple(notes),
        config_summary=config.summary(),
    )
