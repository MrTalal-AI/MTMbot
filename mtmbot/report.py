"""Self-contained HTML/JSON reports with inline SVG charts (no dependencies)."""

from __future__ import annotations

import html
import json
import math
import os
import tempfile
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol

from mtmbot.backtest import BacktestResult
from mtmbot.walkforward import WalkForwardResult

__all__ = [
    "render_html",
    "render_walkforward_html",
    "write_html",
    "write_json",
    "write_walkforward_html",
]


class _Serialisable(Protocol):
    def to_dict(self) -> dict[str, Any]:
        ...


_CSS = """
  :root {
    --bg:#020617; --panel:#0f172a; --line:#1e293b; --text:#e2e8f0;
    --muted:#94a3b8; --pos:#34d399; --neg:#f87171; --accent:#22d3ee;
  }
  * { box-sizing:border-box }
  body { margin:0; padding:32px; background:var(--bg); color:var(--text);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif }
  h1 { font-size:26px; margin:0 0 4px }
  h2 { font-size:17px; margin:32px 0 12px; color:var(--muted); font-weight:600;
       text-transform:uppercase; letter-spacing:.08em }
  .meta { color:var(--muted); margin-bottom:24px }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:14px 16px; display:flex; flex-direction:column; gap:6px }
  .card span { color:var(--muted); font-size:12px; text-transform:uppercase;
               letter-spacing:.06em }
  .card strong { font-size:20px; font-variant-numeric:tabular-nums }
  .pos { color:var(--pos) } .neg { color:var(--neg) }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums }
  th,td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
          font-size:13px }
  th { color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px;
       letter-spacing:.06em }
  .note { margin-top:28px; color:var(--muted); font-size:12px; border-top:1px solid var(--line);
          padding-top:16px }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
          border:1px solid var(--line); color:var(--muted) }
  ul.notes { color:var(--muted); font-size:13px; padding-left:18px }
"""


def _atomic_write(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".report-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(tmp_name).replace(target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def _fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    if not math.isfinite(value):
        return "&#8734;"
    return f"{value:,.{digits}f}{suffix}"


def _polyline(values: list[float], width: float, height: float, pad: float = 6.0) -> str:
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        span = 1.0
    step = (width - 2 * pad) / (len(values) - 1)
    points = []
    for index, value in enumerate(values):
        x = pad + index * step
        y = pad + (high - value) / span * (height - 2 * pad)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _drawdown_series(values: list[float]) -> list[float]:
    peak = values[0]
    out: list[float] = []
    for value in values:
        peak = max(peak, value)
        out.append(0.0 if peak <= 0 else 100.0 * (peak - value) / peak)
    return out


def _svg(equity: list[float], width: int = 900, height: int = 240) -> str:
    drawdown = _drawdown_series(equity)
    worst = max(drawdown) if drawdown else 0.0
    eq_points = _polyline(equity, width, height)
    dd_points = _polyline(
        [height - (value / worst * height if worst > 0 else 0.0) for value in drawdown],
        width,
        height,
    )
    return f"""
    <svg viewBox="0 0 {width} {height + 90}" role="img" aria-label="Equity curve and drawdown"
         style="width:100%;height:auto;background:#0f172a;border-radius:12px">
      <text x="12" y="20" fill="#94a3b8" font-size="12">Equity</text>
      <polyline points="{eq_points}" fill="none" stroke="#22d3ee" stroke-width="2"/>
      <g transform="translate(0,{height + 40})">
        <text x="12" y="0" fill="#94a3b8" font-size="12">Drawdown (worst {worst:.2f}%)</text>
        <polyline points="{dd_points}" fill="none" stroke="#f87171" stroke-width="2"/>
      </g>
    </svg>
    """


def _cards(items: list[tuple[str, str, bool]]) -> str:
    return "".join(
        f'<div class="card"><span>{name}</span>'
        f'<strong class="{"pos" if good else "neg"}">{value}</strong></div>'
        for name, value, good in items
    )


def _page(title: str, heading: str, meta: str, body: str, footer: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body>
  <h1>{html.escape(heading)}</h1>
  <div class="meta">{meta}</div>
  {body}
  <div class="note">{footer}</div>
</body></html>
"""


def render_html(result: BacktestResult, *, title: str | None = None) -> str:
    """Build a standalone backtest tearsheet."""
    perf = result.performance
    heading = title or f"MTMbot backtest - {result.symbol}"
    equity = [point.equity for point in result.equity_curve]
    first = result.equity_curve[0].ts.astimezone(UTC).date()
    last = result.equity_curve[-1].ts.astimezone(UTC).date()
    factor = "&#8734;" if not math.isfinite(perf.profit_factor) else _fmt(perf.profit_factor)
    cards = _cards(
        [
            ("Net return", _fmt(perf.total_return_pct, 2, "%"), perf.total_return_pct >= 0),
            (
                "Buy &amp; hold",
                _fmt(result.buy_hold_return_pct, 2, "%"),
                result.buy_hold_return_pct >= 0,
            ),
            ("CAGR", _fmt(perf.cagr_pct, 2, "%"), perf.cagr_pct >= 0),
            ("Max drawdown", _fmt(-perf.max_drawdown_pct, 2, "%"), False),
            ("Sharpe", _fmt(perf.sharpe, 2), perf.sharpe >= 0),
            ("Sortino", _fmt(perf.sortino, 2), perf.sortino >= 0),
            ("Profit factor", factor, perf.profit_factor >= 1),
            ("Win rate", _fmt(perf.win_rate_pct, 1, "%"), perf.win_rate_pct >= 50),
            ("Trades", str(perf.trades), True),
            ("Expectancy", _fmt(perf.expectancy_r, 2, "R"), perf.expectancy_r >= 0),
            ("Exposure", _fmt(perf.exposure_pct, 1, "%"), True),
            ("Fees paid", _fmt(perf.fees_paid, 2), False),
        ]
    )
    rows = "".join(
        f"<tr><td>{trade.entry_ts.astimezone(UTC).strftime('%Y-%m-%d %H:%M')}</td>"
        f"<td>{trade.exit_ts.astimezone(UTC).strftime('%Y-%m-%d %H:%M')}</td>"
        f"<td>{html.escape(trade.side.value)}</td>"
        f"<td>{trade.entry_price:,.2f}</td><td>{trade.exit_price:,.2f}</td>"
        f"<td class=\"{'pos' if trade.pnl >= 0 else 'neg'}\">{trade.pnl:,.2f}</td>"
        f"<td>{trade.r_multiple:+.2f}R</td>"
        f"<td>{html.escape(trade.exit_reason)}</td></tr>"
        for trade in result.trades[-50:]
    )
    config_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for section, values in result.config_summary.items()
        for key, value in (values.items() if isinstance(values, dict) else [(section, values)])
    )
    meta = (
        f"{first} &rarr; {last} &middot; {result.bars:,} bars &middot; "
        f'<span class="pill">{html.escape(str(result.config_summary.get("timeframe")))}</span>'
        f" &middot; started with {result.starting_equity:,.2f} &middot; ended at "
        f"{result.ending_equity:,.2f}"
    )
    body = f"""
  <div class="grid">{cards}</div>
  <h2>Equity &amp; drawdown</h2>
  {_svg(equity)}
  <h2>Trades (last 50)</h2>
  <table><thead><tr><th>Entry</th><th>Exit</th><th>Side</th><th>Entry px</th>
  <th>Exit px</th><th>PnL</th><th>R</th><th>Reason</th></tr></thead>
  <tbody>{rows or '<tr><td colspan="8">No trades</td></tr>'}</tbody></table>
  <h2>Parameters</h2>
  <table><tbody>{config_rows}</tbody></table>
"""
    footer = (
        f"Simulated with {result.config_summary.get('costs', {}).get('taker_bps', 0)} bps fees "
        f"and {result.config_summary.get('costs', {}).get('slippage_bps', 0)} bps slippage per "
        "fill. Backtests are a hypothesis test, not a promise: past performance does not "
        "guarantee future results. Verify on a paper account before risking real money."
    )
    return _page(heading, heading, meta, body, footer)


def render_walkforward_html(result: WalkForwardResult, *, title: str | None = None) -> str:
    """Build a walk-forward tearsheet that puts in-sample and OOS side by side."""
    oos = result.oos_performance
    heading = title or f"MTMbot walk-forward - {result.symbol}"
    cards = _cards(
        [
            ("Out-of-sample return", _fmt(oos.total_return_pct, 2, "%"), oos.total_return_pct >= 0),
            (
                "In-sample mean",
                _fmt(result.is_mean_return_pct, 2, "%"),
                result.is_mean_return_pct >= 0,
            ),
            ("OOS / IS efficiency", _fmt(result.efficiency, 2), result.efficiency >= 0),
            ("OOS max drawdown", _fmt(-oos.max_drawdown_pct, 2, "%"), False),
            ("OOS Sharpe", _fmt(oos.sharpe, 2), oos.sharpe >= 0),
            ("OOS Sortino", _fmt(oos.sortino, 2), oos.sortino >= 0),
            ("OOS trades", str(oos.trades), True),
            ("OOS win rate", _fmt(oos.win_rate_pct, 1, "%"), oos.win_rate_pct >= 50),
            ("OOS expectancy", _fmt(oos.expectancy_r, 2, "R"), oos.expectancy_r >= 0),
            ("Param stability", _fmt(result.stability_pct, 0, "%"), result.stability_pct >= 50),
            ("Folds", str(len(result.folds)), True),
            ("Grid size", str(result.grid_size), True),
        ]
    )
    rows = "".join(
        f"<tr><td>{fold.index}</td>"
        f"<td>{fold.train_start.date()} &rarr; {fold.train_end.date()}</td>"
        f"<td>{fold.test_start.date()} &rarr; {fold.test_end.date()}</td>"
        f"<td>{html.escape(fold.params_label)}</td>"
        f"<td class=\"{'pos' if fold.is_performance.total_return_pct >= 0 else 'neg'}\">"
        f"{fold.is_performance.total_return_pct:+.2f}%</td>"
        f"<td class=\"{'pos' if fold.oos_performance.total_return_pct >= 0 else 'neg'}\">"
        f"{fold.oos_performance.total_return_pct:+.2f}%</td>"
        f"<td>{fold.is_performance.trades}</td>"
        f"<td>{fold.oos_performance.trades}</td>"
        f"<td>{fold.equity_end:,.2f}</td></tr>"
        for fold in result.folds
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in result.notes)
    equity = [point.equity for point in result.oos_equity_curve]
    first = result.oos_equity_curve[0].ts.astimezone(UTC).date()
    last = result.oos_equity_curve[-1].ts.astimezone(UTC).date()
    meta = (
        f"{first} &rarr; {last} &middot; {len(result.folds)} folds &middot; "
        f'<span class="pill">objective {html.escape(result.settings.objective)}</span>'
        f" &middot; train {result.settings.train_ratio:.0%} &middot; grid "
        f"{result.grid_size} combos &middot; min trades {result.settings.min_trades}"
    )
    body = f"""
  <div class="grid">{cards}</div>
  <h2>Out-of-sample equity (capital carried fold to fold)</h2>
  {_svg(equity)}
  <h2>In-sample vs out-of-sample by fold</h2>
  <table><thead><tr><th>Fold</th><th>Train window</th><th>Test window</th><th>Chosen params</th>
  <th>IS return</th><th>OOS return</th><th>IS trades</th><th>OOS trades</th><th>OOS equity</th>
  </tr></thead><tbody>{rows}</tbody></table>
  <h2>Notes</h2>
  <ul class="notes">{notes or "<li>No caveats recorded.</li>"}</ul>
"""
    footer = (
        "Only the out-of-sample column is evidence: parameters were chosen using data "
        "before each test window and applied forward. In-sample returns come from "
        "overlapping rolling windows, so their mean is a diagnostic of how much the search "
        "was fitting, not an achievable result. A large OOS/IS gap means the parameters "
        "are fitting noise. Backtests are not a promise of future returns."
    )
    return _page(heading, heading, meta, body, footer)


def write_html(path: str | Path, result: BacktestResult, *, title: str | None = None) -> Path:
    """Write the backtest HTML tearsheet atomically."""
    return _atomic_write(path, render_html(result, title=title))


def write_walkforward_html(
    path: str | Path, result: WalkForwardResult, *, title: str | None = None
) -> Path:
    """Write the walk-forward HTML tearsheet atomically."""
    return _atomic_write(path, render_walkforward_html(result, title=title))


def write_json(path: str | Path, payload: dict[str, Any] | _Serialisable) -> Path:
    """Write a JSON report atomically (strict: no NaN/Infinity)."""
    data = payload if isinstance(payload, dict) else payload.to_dict()
    text = json.dumps(data, indent=2, allow_nan=False, sort_keys=False)
    return _atomic_write(path, text + "\n")
