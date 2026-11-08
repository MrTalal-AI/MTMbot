"""Live / paper trading loop.

Design notes:

* The loop is bar-driven: it wakes shortly after each candle closes, which is
  exactly what the strategy was designed for.
* Every iteration is idempotent per bar: a bar already processed is skipped,
  so a restart never double-trades.
* Failures are logged and retried with backoff; after too many consecutive
  failures the engine stops instead of trading blind.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mtmbot.config import BotConfig
from mtmbot.data.base import CandleSource, DataError, sanitize
from mtmbot.execution import Broker, Portfolio
from mtmbot.models import Signal
from mtmbot.risk import RiskManager
from mtmbot.session import TradingSession
from mtmbot.state import StateError, restore_state, save_state
from mtmbot.strategy.mtm import MTMStrategy
from mtmbot.timeframe import next_bar_close

__all__ = ["EngineError", "LiveEngine", "StepResult", "build_session"]

_LOG = logging.getLogger("mtmbot.engine")


class EngineError(RuntimeError):
    """Raised when the engine cannot continue safely."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """What one iteration did."""

    new_bar: bool
    bar_ts: datetime | None = None
    signal: Signal | None = None
    equity: float = 0.0
    halt_reason: str | None = None


def build_session(config: BotConfig, broker: Broker) -> TradingSession:
    """Create a session from configuration (no state restore)."""
    strategy = MTMStrategy(config.strategy, config.step_seconds)
    portfolio = Portfolio.create(config.starting_cash)
    risk = RiskManager(config.risk, config.starting_cash)
    return TradingSession(
        symbol=config.symbol,
        portfolio=portfolio,
        risk=risk,
        broker=broker,
        strategy=strategy,
    )


@dataclass(slots=True)
class LiveEngine:
    """Bar-driven engine for live (real broker) or paper (simulated) trading."""

    config: BotConfig
    feed: CandleSource
    broker: Broker
    state_path: Path | None = None
    limit: int = 900
    max_consecutive_errors: int = 10
    session: TradingSession = field(init=False)
    last_bar_ts: datetime | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise EngineError(f"limit must be >= 1, got {self.limit!r}")
        self.session = build_session(self.config, self.broker)
        if self.state_path is not None:
            try:
                self.last_bar_ts = restore_state(self.state_path, self.session)
            except StateError:
                # Never trade on an untrusted state file.
                raise
            if self.last_bar_ts is not None:
                _LOG.info("restored state from %s (last bar %s)",
                          self.state_path, self.last_bar_ts.isoformat())

    # ------------------------------------------------------------------ core
    def step(self, now: datetime | None = None) -> StepResult:
        """Fetch data, process one new bar if there is one."""
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        candles = sanitize(
            self.feed.fetch_candles(self.config.symbol, self.config.step_seconds, self.limit),
            self.config.step_seconds,
            moment,
        )
        warmup = self.session.strategy.warmup_bars
        if len(candles) < warmup:
            raise EngineError(
                f"feed returned {len(candles)} closed bars, need {warmup} for warm-up"
            )
        prepared = self.session.strategy.prepare_from_base(candles)
        last = candles[-1]

        if self.last_bar_ts is not None and last.ts <= self.last_bar_ts:
            return StepResult(
                new_bar=False,
                bar_ts=last.ts,
                equity=self.session.last_equity,
                halt_reason=self.session.halt_reason,
            )

        self.session.begin_bar(last.close, moment)
        signal = self.session.end_bar(last, prepared, len(candles) - 1)
        self.last_bar_ts = last.ts
        self._persist()
        _LOG.info(
            "bar %s processed: equity %.2f, position=%s, signal=%s",
            last.ts.isoformat(),
            self.session.last_equity,
            "yes" if self.session.portfolio.positions else "no",
            signal.action.value,
        )
        return StepResult(
            new_bar=True,
            bar_ts=last.ts,
            signal=signal,
            equity=self.session.last_equity,
            halt_reason=self.session.halt_reason,
        )

    def _persist(self) -> None:
        if self.state_path is None:
            return
        try:
            save_state(self.state_path, self.session, self.last_bar_ts)
        except OSError as exc:
            _LOG.error("could not save state to %s: %s", self.state_path, exc)

    def seconds_until_next_bar(self, now: datetime | None = None) -> float:
        """Seconds to sleep before the next bar is expected to be final."""
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        target = next_bar_close(moment, self.config.step_seconds)
        return max(1.0, min(3600.0, (target - moment).total_seconds()))

    def run(
        self,
        *,
        once: bool = False,
        max_iterations: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Run the bar loop until stopped by the caller (Ctrl+C / SIGTERM)."""
        if max_iterations is not None and max_iterations < 1:
            raise EngineError(f"max_iterations must be >= 1, got {max_iterations!r}")
        errors = 0
        iterations = 0
        _LOG.info(
            "engine starting: %s %s (%ss bars), state=%s",
            self.config.symbol,
            self.config.timeframe,
            self.config.step_seconds,
            self.state_path or "memory",
        )
        try:
            while True:
                if max_iterations is not None and iterations >= max_iterations:
                    break
                iterations += 1
                try:
                    result = self.step()
                    errors = 0
                    if result.new_bar:
                        _LOG.info("equity %.2f", result.equity)
                except (DataError, EngineError) as exc:
                    errors += 1
                    _LOG.error("iteration failed (%d/%d): %s", errors,
                               self.max_consecutive_errors, exc)
                    if errors >= self.max_consecutive_errors:
                        raise EngineError(
                            f"stopping after {errors} consecutive failures: {exc}"
                        ) from exc
                except Exception as exc:
                    errors += 1
                    _LOG.exception("unexpected failure (%d/%d)", errors,
                                   self.max_consecutive_errors)
                    if errors >= self.max_consecutive_errors:
                        raise EngineError(
                            f"stopping after {errors} consecutive failures: {exc}"
                        ) from exc
                if once:
                    break
                sleep(self.seconds_until_next_bar())
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            _LOG.warning("interrupted by user; state saved at %s", self.state_path or "memory")
        finally:
            self._persist()
