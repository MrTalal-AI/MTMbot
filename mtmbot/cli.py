"""Command line interface: ``python -m mtmbot <command>``."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mtmbot import __version__
from mtmbot.backtest import BacktestError, BacktestResult, run_backtest
from mtmbot.ccxt_broker import CcxtBroker
from mtmbot.config import BotConfig, ConfigError, load_config
from mtmbot.data import (
    CoinbaseSource,
    DataError,
    KrakenSource,
    download_history,
    read_candles,
    write_candles,
)
from mtmbot.data.base import CandleSource
from mtmbot.engine import EngineError, LiveEngine
from mtmbot.execution import BrokerError, PaperBroker
from mtmbot.logs import setup_logging
from mtmbot.report import write_html, write_json, write_walkforward_html
from mtmbot.state import StateError
from mtmbot.timeframe import TimeframeError, parse_timeframe
from mtmbot.walkforward import (
    WalkForwardError,
    WalkForwardResult,
    WalkForwardSettings,
    run_walk_forward,
)

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_USAGE = 2


def _parse_moment(text: str, name: str) -> datetime:
    raw = text.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(f"--{name} is not an ISO date/datetime: {text!r}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _slug(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '-')}_{timeframe}"


def _load_config(args: argparse.Namespace) -> BotConfig:
    config = load_config(args.config)
    overrides: dict[str, Any] = {}
    if getattr(args, "symbol", None):
        overrides["symbol"] = args.symbol
    if getattr(args, "timeframe", None):
        overrides["timeframe"] = args.timeframe
    if getattr(args, "cash", None) is not None:
        overrides["starting_cash"] = args.cash
    return replace(config, **overrides) if overrides else config


def _make_feed(name: str) -> CandleSource:
    if name == "kraken":
        return KrakenSource()
    if name == "coinbase":
        return CoinbaseSource()
    raise ConfigError(f"unknown data feed {name!r}; choose kraken or coinbase")


def _print_backtest(result: BacktestResult) -> None:
    perf = result.performance
    first = result.equity_curve[0].ts.date()
    last = result.equity_curve[-1].ts.date()
    factor = "inf" if math.isinf(perf.profit_factor) else f"{perf.profit_factor:.2f}"
    lines = [
        "",
        f"MTMbot backtest  {result.symbol}  {first} -> {last}",
        f"  bars            {result.bars:,} ({result.htf_bars:,} higher-timeframe bars, "
        f"{result.data_gaps} gaps)",
        f"  starting equity {result.starting_equity:,.2f}",
        f"  ending equity   {result.ending_equity:,.2f}"
        f"{'  (position still open)' if result.open_position_at_end else ''}",
        f"  net return      {perf.total_return_pct:+.2f}%   "
        f"(buy & hold {result.buy_hold_return_pct:+.2f}%)",
        f"  CAGR            {perf.cagr_pct:+.2f}%",
        f"  max drawdown    -{perf.max_drawdown_pct:.2f}% "
        f"({perf.max_drawdown_days:.1f} days underwater)",
        f"  sharpe/sortino  {perf.sharpe:.2f} / {perf.sortino:.2f}",
        f"  trades          {perf.trades} (win rate {perf.win_rate_pct:.1f}%, "
        f"profit factor {factor})",
        f"  expectancy      {perf.expectancy_r:+.2f}R  "
        f"(best {perf.best_r:+.2f}R / worst {perf.worst_r:+.2f}R)",
        f"  exposure        {perf.exposure_pct:.1f}% of the time",
        f"  fees paid       {perf.fees_paid:,.2f}",
        "",
    ]
    print("\n".join(lines))


def _cmd_backtest(args: argparse.Namespace) -> int:
    config = _load_config(args)
    data_path = args.data or config.data
    if not data_path:
        raise ConfigError("no data: pass --data file.csv (or set data = \"...\" in the config)")
    candles = read_candles(data_path)
    result = run_backtest(config, candles)
    _print_backtest(result)

    stem = _slug(config.symbol, config.timeframe)
    default_dir = Path(args.out_dir)
    json_path = Path(args.json) if args.json else default_dir / f"{stem}_backtest.json"
    html_path = Path(args.html) if args.html else default_dir / f"{stem}_backtest.html"
    write_json(json_path, result)
    write_html(html_path, result, title=f"MTMbot backtest - {config.symbol} {config.timeframe}")
    print(f"reports written: {json_path}  and  {html_path}")
    return EXIT_OK


def _print_walkforward(result: WalkForwardResult) -> None:
    perf = result.oos_performance
    lines = [
        "",
        f"MTMbot walk-forward  {result.symbol}  ({len(result.folds)} folds, "
        f"objective={result.settings.objective}, grid={result.grid_size})",
        f"  {'fold':<5}{'train window':<25}{'test window':<25}{'params':<36}"
        f"{'IS%':>8}{'OOS%':>9}",
    ]
    for fold in result.folds:
        train = f"{fold.train_start.date()}..{fold.train_end.date()}"
        test = f"{fold.test_start.date()}..{fold.test_end.date()}"
        lines.append(
            f"  {fold.index:<5}{train:<25}{test:<25}{fold.params_label:<36}"
            f"{fold.is_performance.total_return_pct:>+8.2f}"
            f"{fold.oos_performance.total_return_pct:>+9.2f}"
        )
    lines.extend(
        [
            "",
            f"  out-of-sample (stitched)  return {perf.total_return_pct:+.2f}%  "
            f"max drawdown -{perf.max_drawdown_pct:.2f}%",
            f"                            sharpe {perf.sharpe:.2f}  trades {perf.trades}  "
            f"win rate {perf.win_rate_pct:.1f}%  expectancy {perf.expectancy_r:+.2f}R",
            f"  in-sample mean return     {result.is_mean_return_pct:+.2f}%  "
            f"(median {result.is_median_return_pct:+.2f}%, overlapping windows)",
            f"  OOS / IS efficiency       {result.efficiency:+.2f}",
            f"  parameter stability       {result.stability_pct:.0f}% of folds chose the "
            "same parameters",
        ]
    )
    for note in result.notes:
        lines.append(f"  note: {note}")
    lines.append("  only the out-of-sample column is evidence.")
    lines.append("")
    print("\n".join(lines))


def _load_walkforward_grid(path: str | None) -> dict[str, list[Any]] | None:
    if not path:
        return None
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"grid file not found: {config_path}")
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid TOML: {exc}") from exc
    table = payload.get("grid", payload)
    if not isinstance(table, dict):
        raise ConfigError("the grid file must contain a [grid] table")
    grid: dict[str, list[Any]] = {}
    for key, value in table.items():
        if not isinstance(value, list):
            raise ConfigError(f"grid[{key!r}] must be a list of values")
        grid[str(key)] = value
    return grid


def _cmd_walkforward(args: argparse.Namespace) -> int:
    config = _load_config(args)
    data_path = args.data or config.data
    if not data_path:
        raise ConfigError("no data: pass --data file.csv (or set data in the config)")
    candles = read_candles(data_path)
    settings = WalkForwardSettings(
        folds=args.folds,
        train_ratio=args.train_ratio,
        objective=args.objective,
        min_trades=args.min_trades,
    )
    result = run_walk_forward(config, candles, settings, grid=_load_walkforward_grid(args.grid))
    _print_walkforward(result)

    stem = _slug(config.symbol, config.timeframe)
    default_dir = Path(args.out_dir)
    json_path = Path(args.json) if args.json else default_dir / f"{stem}_walkforward.json"
    html_path = Path(args.html) if args.html else default_dir / f"{stem}_walkforward.html"
    write_json(json_path, result)
    write_walkforward_html(
        html_path, result, title=f"MTMbot walk-forward - {config.symbol} {config.timeframe}"
    )
    print(f"reports written: {json_path}  and  {html_path}")
    return EXIT_OK


def _cmd_download(args: argparse.Namespace) -> int:
    step = parse_timeframe(args.timeframe)
    start = _parse_moment(args.start, "start")
    end = _parse_moment(args.end, "end") if args.end else datetime.now(UTC)
    if end <= start:
        raise ConfigError("--end must be after --start")
    candles = download_history(args.symbol, step, start, end, source=args.source)
    out = Path(args.out)
    write_candles(out, candles)
    print(
        f"downloaded {len(candles):,} candles for {args.symbol} {args.timeframe} "
        f"({candles[0].ts.date()} -> {candles[-1].ts.date()}) into {out}"
    )
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print("config OK")
    print(json.dumps(config.summary(), indent=2))
    return EXIT_OK


def _cmd_paper(args: argparse.Namespace) -> int:
    config = _load_config(args)
    feed = _make_feed(args.feed)
    broker = PaperBroker(costs=config.costs)
    engine = LiveEngine(
        config=config,
        feed=feed,
        broker=broker,
        state_path=Path(args.state) if args.state else None,
        limit=args.limit,
    )
    print(
        f"paper trading {config.symbol} {config.timeframe} on {feed.name} data; "
        f"risk {config.risk.risk_per_trade_pct}% per trade"
    )
    engine.run(once=args.once, max_iterations=args.iterations)
    session = engine.session
    print(
        f"paper session: equity {session.last_equity:,.2f}, "
        f"{len(session.portfolio.trades)} closed trade(s), "
        f"halt={session.halt_reason or 'none'}"
    )
    return EXIT_OK


def _cmd_live(args: argparse.Namespace) -> int:
    if not args.confirm_live:
        raise ConfigError(
            "live trading moves real money. Re-run with --confirm-live after checking the "
            "config, and set MTMBOT_LIVE_OK=1 plus your API keys in the environment."
        )
    config = _load_config(args)
    feed = _make_feed(args.feed)
    broker = CcxtBroker(config.exchange, config.symbol)
    engine = LiveEngine(
        config=config,
        feed=feed,
        broker=broker,
        state_path=Path(args.state) if args.state else None,
        limit=args.limit,
    )
    engine.run(once=args.once, max_iterations=args.iterations)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser (also used by tests)."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    common.add_argument("--log-file", default=None, help="also append logs to this file")

    parser = argparse.ArgumentParser(
        prog="mtmbot",
        description="Multi-Timeframe Momentum trading bot (backtest, paper, live).",
    )
    parser.add_argument("--version", action="version", version=f"mtmbot {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    download = subparsers.add_parser(
        "download", parents=[common], help="download historical candles to CSV"
    )
    download.add_argument("--symbol", default="BTC/USD")
    download.add_argument("--timeframe", default="1h")
    download.add_argument("--start", required=True, help="ISO date, e.g. 2023-01-01")
    download.add_argument("--end", default=None, help="ISO date, defaults to now")
    download.add_argument("--source", default="coinbase", choices=["coinbase"])
    download.add_argument("--out", required=True)
    download.set_defaults(handler=_cmd_download)

    backtest = subparsers.add_parser(
        "backtest", parents=[common], help="run a backtest on a CSV file"
    )
    backtest.add_argument("--config", default=None, help="path to config.toml")
    backtest.add_argument("--data", default=None, help="path to candles CSV")
    backtest.add_argument("--symbol", default=None, help="override config symbol")
    backtest.add_argument("--timeframe", default=None, help="override config timeframe")
    backtest.add_argument("--cash", type=float, default=None, help="override starting cash")
    backtest.add_argument("--json", default=None, help="JSON report path")
    backtest.add_argument("--html", default=None, help="HTML report path")
    backtest.add_argument("--out-dir", default="reports", help="default report directory")
    backtest.set_defaults(handler=_cmd_backtest)

    walkforward = subparsers.add_parser(
        "walkforward",
        parents=[common],
        help="validate parameter choices out of sample (rolling walk-forward)",
    )
    walkforward.add_argument("--config", default=None, help="path to config.toml")
    walkforward.add_argument("--data", default=None, help="path to candles CSV")
    walkforward.add_argument("--symbol", default=None)
    walkforward.add_argument("--timeframe", default=None)
    walkforward.add_argument("--cash", type=float, default=None)
    walkforward.add_argument("--folds", type=int, default=4)
    walkforward.add_argument("--train-ratio", type=float, default=0.6)
    walkforward.add_argument(
        "--objective", default="sharpe", choices=["sharpe", "sortino", "return", "expectancy", "calmar"]
    )
    walkforward.add_argument("--min-trades", type=int, default=10)
    walkforward.add_argument("--grid", default=None, help="TOML file with a [grid] table")
    walkforward.add_argument("--json", default=None, help="JSON report path")
    walkforward.add_argument("--html", default=None, help="HTML report path")
    walkforward.add_argument("--out-dir", default="reports", help="default report directory")
    walkforward.set_defaults(handler=_cmd_walkforward)

    validate = subparsers.add_parser(
        "validate", parents=[common], help="validate a config file"
    )
    validate.add_argument("--config", default=None)
    validate.add_argument("--symbol", default=None)
    validate.add_argument("--timeframe", default=None)
    validate.add_argument("--cash", type=float, default=None)
    validate.set_defaults(handler=_cmd_validate)

    for name, handler, help_text in (
        ("paper", _cmd_paper, "paper trade against live public market data"),
        ("live", _cmd_live, "live trading through ccxt (real money)"),
    ):
        command = subparsers.add_parser(name, parents=[common], help=help_text)
        command.add_argument("--config", default=None)
        command.add_argument("--symbol", default=None)
        command.add_argument("--timeframe", default=None)
        command.add_argument("--cash", type=float, default=None)
        command.add_argument("--feed", default="kraken", choices=["kraken", "coinbase"])
        command.add_argument("--state", default=None, help="state file for restarts")
        command.add_argument("--limit", type=int, default=900, help="history bars to request")
        command.add_argument("--once", action="store_true", help="process one bar and exit")
        command.add_argument(
            "--iterations", type=int, default=None, help="stop after N loop iterations"
        )
        if name == "live":
            command.add_argument(
                "--confirm-live",
                action="store_true",
                help="required acknowledgement that real funds are at risk",
            )
        command.set_defaults(handler=handler)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    setup_logging(getattr(args, "log_level", "INFO"), getattr(args, "log_file", None))
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (
        ConfigError,
        DataError,
        BacktestError,
        EngineError,
        WalkForwardError,
        StateError,
        BrokerError,
        TimeframeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
