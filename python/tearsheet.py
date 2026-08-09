"""
tearsheet.py — Full Performance Tearsheet
==========================================
Computes and prints/exports all 8 performance metrics:
  ① Annualised return
  ② Sharpe ratio
  ③ Sortino ratio
  ④ Max drawdown
  ⑤ Calmar ratio
  ⑥ Win rate
  ⑦ Profit factor
  ⑧ 95% VaR (daily)

Also computes:
  - Monthly P&L heatmap
  - Drawdown periods analysis
  - Rolling Sharpe chart data
  - In-sample vs OOS comparison
"""

import csv
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path


# ─── Data structures ─────────────────────────────────────────────────────────
@dataclass
class TradeRecord:
    ticker_a:     str
    ticker_b:     str
    direction:    str       # LONG_A or SHORT_A
    entry_bar:    int
    exit_bar:     int
    hold_bars:    int
    entry_spread: float
    exit_spread:  float
    entry_z:      float
    exit_z:       float
    hedge_ratio:  float
    position_size:float
    gross_pnl:    float
    cost:         float
    net_pnl:      float
    stopped:      bool


@dataclass
class TearsheetMetrics:
    # Trade stats
    total_trades:      int   = 0
    winning_trades:    int   = 0
    losing_trades:     int   = 0
    stopped_trades:    int   = 0
    win_rate:          float = 0.0
    avg_hold_bars:     float = 0.0
    avg_win_pnl:       float = 0.0
    avg_loss_pnl:      float = 0.0
    profit_factor:     float = 0.0

    # P&L
    total_gross_pnl:   float = 0.0
    total_costs:       float = 0.0
    total_net_pnl:     float = 0.0

    # 8 risk-adjusted metrics
    ann_return:        float = 0.0   # ①
    sharpe:            float = 0.0   # ②
    sortino:           float = 0.0   # ③
    max_drawdown:      float = 0.0   # ④
    calmar:            float = 0.0   # ⑤
    # win_rate already above              ⑥
    # profit_factor already above         ⑦
    var_95:            float = 0.0   # ⑧

    # Equity curve stats
    initial_nav:       float = 1_000_000.0
    final_nav:         float = 1_000_000.0
    equity_curve:      List[float] = field(default_factory=list)
    daily_pnl:         List[float] = field(default_factory=list)


# ─── Statistics helpers ───────────────────────────────────────────────────────
SQRT_252 = math.sqrt(252)

def _mean(v: List[float]) -> float:
    return sum(v) / len(v) if v else 0.0

def _std(v: List[float], ddof: int = 1) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    s = sum((x - m) ** 2 for x in v)
    return math.sqrt(s / (len(v) - ddof))

def _percentile(v: List[float], p: float) -> float:
    """Linear interpolation percentile."""
    if not v:
        return 0.0
    sv   = sorted(v)
    idx  = p * (len(sv) - 1)
    lo   = int(idx)
    frac = idx - lo
    if lo + 1 >= len(sv):
        return sv[-1]
    return sv[lo] + frac * (sv[lo + 1] - sv[lo])

def _max_drawdown(equity: List[float]) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    mdd  = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd

def _sharpe(daily_rets: List[float], rf_daily: float = 0.0) -> float:
    exc = [r - rf_daily for r in daily_rets]
    s   = _std(exc)
    return _mean(exc) / s * SQRT_252 if s > 1e-10 else 0.0

def _sortino(daily_rets: List[float], rf_daily: float = 0.0) -> float:
    m = _mean(daily_rets) - rf_daily
    down = [r - rf_daily for r in daily_rets if r < rf_daily]
    if not down:
        return math.inf   # no downside observed
    dstd = math.sqrt(sum(x**2 for x in down) / len(down)) * SQRT_252
    return m * SQRT_252 / dstd if dstd > 1e-10 else 0.0

def _calmar(equity: List[float], ann_return: float) -> float:
    mdd = _max_drawdown(equity)
    return ann_return / mdd if mdd > 1e-10 else math.inf   # no drawdown observed

def _rolling(v: List[float], window: int, fn) -> List[float]:
    out = [0.0] * len(v)
    for i in range(window - 1, len(v)):
        out[i] = fn(v[i - window + 1: i + 1])
    return out


# ─── Tearsheet class ──────────────────────────────────────────────────────────
class Tearsheet:
    """
    Computes the full performance tearsheet from a list of trades.
    Can load directly from CSV (output of the C++ system) or from
    a list of TradeRecord objects.
    """

    def __init__(self, initial_nav: float = 1_000_000.0):
        self.initial_nav = initial_nav
        self.trades:      List[TradeRecord]  = []
        self.metrics:     TearsheetMetrics   = TearsheetMetrics()

    # ── Load from CSV ─────────────────────────────────────────────────────────
    def load_csv(self, path: str) -> "Tearsheet":
        """Load trades from the CSV produced by C++ TradeLog::exportTradesCSV."""
        self.trades.clear()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = TradeRecord(
                        ticker_a      = row.get("ticker_a", "A"),
                        ticker_b      = row.get("ticker_b", "B"),
                        direction     = row.get("direction", "LONG_A"),
                        entry_bar     = int(row.get("entry_bar", 0)),
                        exit_bar      = int(row.get("exit_bar", 0)),
                        hold_bars     = int(row.get("hold_bars", 0)),
                        entry_spread  = float(row.get("entry_spread", 0)),
                        exit_spread   = float(row.get("exit_spread", 0)),
                        entry_z       = float(row.get("entry_z", 0)),
                        exit_z        = float(row.get("exit_z", 0)),
                        hedge_ratio   = float(row.get("hedge_ratio", 1)),
                        position_size = float(row.get("position_size", 0)),
                        gross_pnl     = float(row.get("gross_pnl", 0)),
                        cost          = float(row.get("cost", 0)),
                        net_pnl       = float(row.get("net_pnl", 0)),
                        stopped       = row.get("stopped", "0") == "1",
                    )
                    self.trades.append(t)
                except (ValueError, KeyError):
                    continue
        return self

    def add_trade(self, t: TradeRecord) -> "Tearsheet":
        self.trades.append(t)
        return self

    # ── Compute all metrics ───────────────────────────────────────────────────
    def compute(self) -> TearsheetMetrics:
        m = TearsheetMetrics()
        m.initial_nav = self.initial_nav

        if not self.trades:
            self.metrics = m
            return m

        # Trade-level stats
        m.total_trades   = len(self.trades)
        wins  = [t for t in self.trades if t.net_pnl > 0]
        loses = [t for t in self.trades if t.net_pnl <= 0]
        stops = [t for t in self.trades if t.stopped]

        m.winning_trades = len(wins)
        m.losing_trades  = len(loses)
        m.stopped_trades = len(stops)
        m.win_rate       = m.winning_trades / m.total_trades if m.total_trades else 0.0
        m.avg_hold_bars  = _mean([t.hold_bars for t in self.trades])
        m.avg_win_pnl    = _mean([t.net_pnl for t in wins])   if wins  else 0.0
        m.avg_loss_pnl   = _mean([abs(t.net_pnl) for t in loses]) if loses else 0.0

        sum_win  = sum(t.net_pnl       for t in wins)
        sum_loss = sum(abs(t.net_pnl)  for t in loses)
        if sum_loss > 1e-6:
            m.profit_factor = sum_win / sum_loss
        else:
            # no losses at all: infinite profit factor if there were wins,
            # otherwise undefined (no trades / no wins either) — not 0.0,
            # which would misleadingly read as a bad strategy
            m.profit_factor = math.inf if sum_win > 1e-6 else 0.0

        # P&L
        m.total_gross_pnl = sum(t.gross_pnl for t in self.trades)
        m.total_costs     = sum(t.cost       for t in self.trades)
        m.total_net_pnl   = sum(t.net_pnl    for t in self.trades)

        # Build equity curve: one point per trade exit
        equity  = [self.initial_nav]
        pnl_ser = [t.net_pnl for t in self.trades]
        for p in pnl_ser:
            equity.append(equity[-1] + p)

        m.equity_curve = equity
        m.daily_pnl    = pnl_ser
        m.final_nav    = equity[-1]

        # Daily return series
        daily_rets = []
        for i, p in enumerate(pnl_ser):
            base = equity[i]
            daily_rets.append(p / base if base > 0 else 0.0)

        # ── 8 metrics ──
        m.ann_return  = _mean(daily_rets) * 252.0                    # ①
        m.sharpe      = _sharpe(daily_rets)                           # ②
        m.sortino     = _sortino(daily_rets)                          # ③
        m.max_drawdown= _max_drawdown(equity)                         # ④
        m.calmar      = _calmar(equity, m.ann_return)                 # ⑤
        # ⑥ = win_rate (already set)
        # ⑦ = profit_factor (already set)
        m.var_95      = -_percentile(pnl_ser, 0.05)                  # ⑧

        self.metrics = m
        return m

    # ── Print tearsheet ───────────────────────────────────────────────────────
    def print(self, title: str = "STRATEGY TEARSHEET"):
        m = self.metrics
        LINE = "═" * 58

        def row(label, val, unit="", bar_val=None, bar_max=1.0):
            bar = ""
            if bar_val is not None:
                filled = int(min(max(bar_val / bar_max, 0), 1) * 18)
                bar = "█" * filled + "░" * (18 - filled)
            print(f"  {label:<28} {val:<12} {unit:<6} {bar}")

        print(f"\n╔{LINE}╗")
        print(f"║  {title:<56}║")
        print(f"╠{LINE}╣")

        print(f"║  {'─'*56}║")
        print(f"║  TRADE STATISTICS{' '*40}║")
        print(f"║  {'─'*56}║")
        row("Total trades",     f"{m.total_trades}")
        row("Win / Loss / Stop", f"{m.winning_trades}/{m.losing_trades}/{m.stopped_trades}")
        row("Win rate",          f"{m.win_rate*100:.1f}",    "%",
            m.win_rate * 100, 100)
        row("Profit factor",     f"{m.profit_factor:.3f}",   "",
            min(m.profit_factor, 3.0), 3.0)
        row("Avg win P&L",       f"{m.avg_win_pnl:+.2f}")
        row("Avg loss P&L",      f"{m.avg_loss_pnl:.2f}")
        row("Avg hold duration", f"{m.avg_hold_bars:.1f}",   "bars")

        print(f"║  {'─'*56}║")
        print(f"║  P&L BREAKDOWN{' '*43}║")
        print(f"║  {'─'*56}║")
        row("Gross P&L",         f"{m.total_gross_pnl:+.2f}")
        row("Transaction costs", f"{-m.total_costs:+.2f}")
        row("Net P&L",           f"{m.total_net_pnl:+.2f}")
        row("Final NAV",         f"{m.final_nav:.2f}")

        print(f"║  {'─'*56}║")
        print(f"║  RISK-ADJUSTED PERFORMANCE (8 METRICS){' '*19}║")
        print(f"║  {'─'*56}║")
        row("① Ann. return",     f"{m.ann_return*100:.2f}", "%",
            max(m.ann_return * 100, 0), 30)
        row("② Sharpe ratio",    f"{m.sharpe:.3f}",  "",
            max(m.sharpe, 0), 3.0)
        row("③ Sortino ratio",   f"{min(m.sortino, 99):.3f}", "",
            max(min(m.sortino, 3), 0), 3.0)
        row("④ Max drawdown",    f"{m.max_drawdown*100:.1f}", "%",
            m.max_drawdown * 100, 50)
        row("⑤ Calmar ratio",    f"{min(m.calmar, 99):.3f}",  "",
            max(min(m.calmar, 3), 0), 3.0)
        row("⑥ Win rate",        f"{m.win_rate*100:.1f}",  "%",
            m.win_rate * 100, 100)
        row("⑦ Profit factor",   f"{m.profit_factor:.3f}", "",
            min(m.profit_factor, 3.0), 3.0)
        row("⑧ 95% VaR (1-day)", f"{m.var_95:.2f}")
        print(f"╚{LINE}╝\n")

    # ── Monthly P&L heatmap (text) ────────────────────────────────────────────
    def monthly_heatmap(self) -> Dict[str, float]:
        """Group net P&L by month index (bar // 21)."""
        monthly: Dict[int, float] = {}
        for t in self.trades:
            month_idx = t.entry_bar // 21
            monthly[month_idx] = monthly.get(month_idx, 0.0) + t.net_pnl

        print("\n── Monthly P&L (approx 21 bars/month) ─────────────")
        all_keys = sorted(monthly)
        for k in all_keys:
            v    = monthly[k]
            sign = "+" if v >= 0 else ""
            bar  = "█" * min(int(abs(v) / 500), 30)
            print(f"  Month {k:3d}: {sign}{v:>10.2f}  {bar}")
        print()
        return monthly

    # ── Rolling Sharpe (window=20 trades) ────────────────────────────────────
    def rolling_sharpe(self, window: int = 20) -> List[float]:
        pnl = [t.net_pnl for t in self.trades]
        if len(pnl) < window:
            return []
        out = []
        for i in range(window - 1, len(pnl)):
            w = pnl[i - window + 1: i + 1]
            s = _std(w)
            out.append(_mean(w) / s * math.sqrt(252) if s > 1e-10 else 0.0)
        return out

    # ── Drawdown periods ──────────────────────────────────────────────────────
    def drawdown_periods(self) -> List[Dict]:
        eq  = self.metrics.equity_curve
        if len(eq) < 2:
            return []
        periods, peak_val, peak_idx, in_dd = [], eq[0], 0, False
        for i, v in enumerate(eq):
            if v > peak_val:
                if in_dd:
                    periods[-1]["end_idx"] = i
                    periods[-1]["recovery"] = i - periods[-1]["start_idx"]
                peak_val = v; peak_idx = i; in_dd = False
            else:
                dd = (peak_val - v) / peak_val if peak_val > 0 else 0
                if dd > 0.01 and not in_dd:
                    periods.append({"start_idx": peak_idx,
                                    "trough_dd": dd, "end_idx": None,
                                    "recovery": None})
                    in_dd = True
                elif in_dd and dd > periods[-1]["trough_dd"]:
                    periods[-1]["trough_dd"] = dd
        return periods

    # ── Export metrics to CSV ─────────────────────────────────────────────────
    def export_metrics_csv(self, path: str):
        m = self.metrics
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("metric,value\n")
            for name, val in [
                ("total_trades",   m.total_trades),
                ("win_rate",       m.win_rate),
                ("profit_factor",  m.profit_factor),
                ("total_net_pnl",  m.total_net_pnl),
                ("ann_return",     m.ann_return),
                ("sharpe",         m.sharpe),
                ("sortino",        m.sortino),
                ("max_drawdown",   m.max_drawdown),
                ("calmar",         m.calmar),
                ("var_95",         m.var_95),
                ("avg_hold_bars",  m.avg_hold_bars),
                ("final_nav",      m.final_nav),
            ]:
                f.write(f"{name},{val:.6f}\n")
        print(f"[INFO] Wrote {path}")


# ── Compare in-sample vs OOS tearsheets ──────────────────────────────────────
def compare_is_oos(is_metrics: TearsheetMetrics,
                    oos_metrics: TearsheetMetrics):
    """Print IS vs OOS comparison — key overfitting check."""
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON           ║")
    print("║     (large gap = overfitting)                        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"  {'Metric':<20} {'In-Sample':>12} {'OOS':>12} {'Degradation':>14}")
    print(f"  {'─'*60}")

    def pct(a, b):
        if abs(a) < 1e-9:
            return "N/A"
        return f"{(b - a) / abs(a) * 100:+.1f}%"

    # higher_is_better=True metrics degrade when OOS falls well below IS;
    # Max DD is the opposite (higher drawdown is worse), so it degrades
    # when OOS rises well above IS. The original code only ever flagged
    # the Sharpe row — every other metric's degradation went unflagged
    # even though the table displays all of them.
    rows = [
        ("Sharpe",         is_metrics.sharpe,          oos_metrics.sharpe,          True),
        ("Sortino",        is_metrics.sortino,         oos_metrics.sortino,         True),
        ("Ann. Return",    is_metrics.ann_return*100,  oos_metrics.ann_return*100,  True),
        ("Max DD",         is_metrics.max_drawdown*100,oos_metrics.max_drawdown*100,False),
        ("Win Rate",       is_metrics.win_rate*100,    oos_metrics.win_rate*100,    True),
        ("Profit Factor",  is_metrics.profit_factor,   oos_metrics.profit_factor,   True),
    ]
    for label, iv, ov, higher_is_better in rows:
        if higher_is_better:
            degraded = ov < iv * 0.5
        else:
            degraded = iv > 1e-9 and ov > iv * 1.5
        flag = " ⚠" if degraded else ""
        print(f"  {label:<20} {iv:>12.3f} {ov:>12.3f} {pct(iv, ov):>14}{flag}")

    print(f"╚══════════════════════════════════════════════════════╝\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    trades_csv = sys.argv[1] if len(sys.argv) > 1 else "data/results/trades.csv"
    out_csv    = sys.argv[2] if len(sys.argv) > 2 else "data/results/metrics.csv"

    if not os.path.exists(trades_csv):
        # Demo with synthetic trades
        print("[INFO] No trades CSV found — generating demo data\n")
        ts = Tearsheet(initial_nav=1_000_000)
        import random; random.seed(42)
        for i in range(50):
            pnl = random.gauss(300, 1200)
            ts.add_trade(TradeRecord(
                ticker_a="HDFC", ticker_b="ICICI",
                direction="LONG_A" if pnl > 0 else "SHORT_A",
                entry_bar=i*5, exit_bar=i*5+4, hold_bars=4,
                entry_spread=random.gauss(0, 2),
                exit_spread=random.gauss(0, 1),
                entry_z=random.uniform(-3, -2),
                exit_z=random.uniform(-0.5, 0.5),
                hedge_ratio=1.5, position_size=100_000,
                gross_pnl=pnl + 100, cost=100, net_pnl=pnl,
                stopped=pnl < -2000
            ))
    else:
        ts = Tearsheet().load_csv(trades_csv)

    ts.compute()
    ts.print("STATISTICAL ARBITRAGE TEARSHEET")
    ts.monthly_heatmap()
    ts.export_metrics_csv(out_csv)