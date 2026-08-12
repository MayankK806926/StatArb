"""
visualise.py - Spread Visualisation Dashboard
===============================================
Produces a 4-panel plot:
  Panel 1: Normalised price series of A and B (rebased to 100)
  Panel 2: Spread with ±1σ, ±2σ, ±3σ bands + trade entry/exit markers
  Panel 3: Z-score time series with threshold lines + regime shading
  Panel 4: Cumulative P&L equity curve with drawdown shading

Usage:
    python visualise.py                          # demo with synthetic data
    python visualise.py --prices data/prices/    \\
                        --trades data/results/trades.csv \\
                        --equity data/results/equity.csv \\
                        --tickerA HDFC --tickerB ICICI
    python visualise.py --no-show --out dashboard.png
"""

import os
import csv
import math
import random
import argparse
from typing import List, Optional, Tuple

# ─── Pure-Python helpers (same as Utils.h, no numpy needed for core logic) ────
def _mean(v): return sum(v) / len(v) if v else 0.0
def _std(v):
    if len(v) < 2: return 0.0
    m = _mean(v); return math.sqrt(sum((x-m)**2 for x in v) / (len(v)-1))
def _rolling_mean(v, w):
    out = [0.0] * len(v); s = 0.0
    for i, x in enumerate(v):
        s += x
        if i >= w: s -= v[i-w]
        if i >= w-1: out[i] = s / w
    return out
def _rolling_std(v, w):
    out = [0.0] * len(v)
    for i in range(w-1, len(v)):
        seg = v[i-w+1:i+1]; m = _mean(seg)
        out[i] = math.sqrt(sum((x-m)**2 for x in seg) / (w-1)) if w > 1 else 0.0
    return out
def _zscore(v, w):
    rm = _rolling_mean(v, w); rs = _rolling_std(v, w)
    out = [0.0] * len(v)
    for i in range(len(v)):
        if rs[i] > 1e-10: out[i] = (v[i] - rm[i]) / rs[i]
    return out
def _max_drawdown_series(equity):
    """Return drawdown series (fraction below peak) at each bar."""
    peak = equity[0]; dd = []
    for v in equity:
        if v > peak: peak = v
        dd.append((peak - v) / peak if peak > 0 else 0.0)
    return dd
def _rebase(prices, base=100.0):
    if not prices or prices[0] == 0: return prices
    r = base / prices[0]
    return [p * r for p in prices]


# ─── CSV readers ──────────────────────────────────────────────────────────────
def read_price_csv(path: str) -> Tuple[List[str], List[float]]:
    dates, prices = [], []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dates.append(row.get("Date", row.get("date", "")))
                    key = next((k for k in row if "close" in k.lower()), None)
                    if key:
                        prices.append(float(row[key]))
                except (ValueError, StopIteration):
                    continue
    except FileNotFoundError:
        pass
    return dates, prices

def read_trades_csv(path: str) -> List[dict]:
    trades = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    trades.append({
                        "entry_bar":  int(row.get("entry_bar", 0)),
                        "exit_bar":   int(row.get("exit_bar",  0)),
                        "direction":  row.get("direction", "LONG_A"),
                        "entry_z":    float(row.get("entry_z", 0)),
                        "exit_z":     float(row.get("exit_z",  0)),
                        "net_pnl":    float(row.get("net_pnl", 0)),
                        "stopped":    row.get("stopped", "0") == "1",
                        "hedge_ratio":float(row.get("hedge_ratio", 1.0)),
                    })
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return trades

def read_equity_csv(path: str) -> List[float]:
    equity = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    equity.append(float(row.get("equity", 0)))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return equity


# ─── Dashboard class ──────────────────────────────────────────────────────────
class SpreadDashboard:
    """
    4-panel spread visualisation dashboard.
    Works with matplotlib if available; falls back to ASCII art.
    """

    def __init__(self, ticker_a: str = "A", ticker_b: str = "B",
                  z_window: int = 30, entry_z: float = 2.0):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.z_window = z_window
        self.entry_z  = entry_z

    def plot(self, prices_a: List[float], prices_b: List[float],
              spread: List[float], equity: List[float],
              trades: Optional[List[dict]] = None,
              regimes: Optional[List[int]] = None,
              save_path: Optional[str] = None,
              show: bool = True):
        """
        Main entry point. Tries matplotlib first; falls back to ASCII.
        """
        try:
            self._plot_matplotlib(prices_a, prices_b, spread, equity,
                                   trades, regimes, save_path, show)
        except ImportError:
            print("[WARN] matplotlib not found. Using ASCII dashboard.\n")
            self._plot_ascii(prices_a, prices_b, spread, equity, trades)

    def _plot_matplotlib(self, prices_a, prices_b, spread, equity,
                          trades, regimes, save_path, show):
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec

        n   = len(spread)
        xs  = list(range(n))
        z   = _zscore(spread, self.z_window)
        rm  = _rolling_mean(spread, self.z_window)
        rs  = _rolling_std(spread,  self.z_window)

        fig = plt.figure(figsize=(16, 11), facecolor="#0f0f0f")
        fig.suptitle(
            f"StatArb Dashboard  ·  {self.ticker_a} / {self.ticker_b}",
            color="white", fontsize=14, fontweight="bold", y=0.98)

        gs  = gridspec.GridSpec(4, 1, hspace=0.45, top=0.94, bottom=0.05)
        axs = [fig.add_subplot(gs[i]) for i in range(4)]

        COLORS = {
            "a": "#4FC3F7", "b": "#F48FB1",
            "spread": "#B0BEC5",
            "entry_long": "#66BB6A", "entry_short": "#EF5350",
            "exit": "#FFA726", "stop": "#AB47BC",
            "equity": "#80CBC4",
            "drawdown": "#EF5350",
            "regime_sit": "#FF6F0028",
            "grid": "#1e1e1e", "text": "#9e9e9e",
        }
        for ax in axs:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#777", labelsize=8)
            ax.grid(color=COLORS["grid"], linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color("#333")

        # ── Panel 1: Normalised Prices ──
        ax = axs[0]
        rA = _rebase(prices_a[:n]); rB = _rebase(prices_b[:n])
        ax.plot(xs, rA, color=COLORS["a"],  lw=1.0, label=self.ticker_a)
        ax.plot(xs, rB, color=COLORS["b"],  lw=1.0, label=self.ticker_b, alpha=0.8)
        ax.set_title("Normalised Prices (rebased to 100)",
                      color=COLORS["text"], fontsize=9, loc="left")
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
                   framealpha=0.5)
        ax.set_ylabel("Price", color=COLORS["text"], fontsize=8)

        # ── Panel 2: Spread + σ bands + trade markers ──
        ax = axs[1]
        ax.plot(xs, spread, color=COLORS["spread"], lw=0.8, label="Spread")
        ax.plot(xs, rm,     color="#888", lw=0.6, linestyle="--", alpha=0.6)

        # Band fills
        b1u = [rm[i] + rs[i]       for i in range(n)]
        b1d = [rm[i] - rs[i]       for i in range(n)]
        b2u = [rm[i] + 2*rs[i]     for i in range(n)]
        b2d = [rm[i] - 2*rs[i]     for i in range(n)]
        b3u = [rm[i] + 3*rs[i]     for i in range(n)]
        b3d = [rm[i] - 3*rs[i]     for i in range(n)]

        ax.fill_between(xs, b1d, b1u, alpha=0.06, color="#4FC3F7")
        ax.fill_between(xs, b2d, b1d, alpha=0.10, color="#4FC3F7")
        ax.fill_between(xs, b1u, b2u, alpha=0.10, color="#4FC3F7")
        ax.fill_between(xs, b3d, b2d, alpha=0.12, color="#EF5350")
        ax.fill_between(xs, b2u, b3u, alpha=0.12, color="#EF5350")

        # Trade markers
        if trades:
            for t in trades:
                eb, xb = t["entry_bar"], t["exit_bar"]
                if eb >= n: continue
                col = COLORS["entry_long"] if t["direction"] == "LONG_A" \
                      else COLORS["entry_short"]
                ax.axvline(eb, color=col, alpha=0.5, lw=0.8)
                scol = COLORS["stop"] if t["stopped"] else COLORS["exit"]
                if xb < n:
                    ax.axvline(xb, color=scol, alpha=0.35, lw=0.6,
                                linestyle=":")

        ax.set_title("Spread  ·  ±1σ / ±2σ / ±3σ bands",
                      color=COLORS["text"], fontsize=9, loc="left")
        ax.set_ylabel("Spread", color=COLORS["text"], fontsize=8)

        # ── Panel 3: Z-score ──
        ax = axs[2]
        # Colour fill: long zone green, short zone red, flat grey
        for i in range(1, n):
            col = ("#66BB6A22" if z[i] < -self.entry_z
                   else "#EF535022" if z[i] > self.entry_z
                   else "#88888811")
            ax.axvspan(i-1, i, color=col, linewidth=0)

        ax.plot(xs, z, color="#CFD8DC", lw=0.7)
        for level, col in [(self.entry_z,  "#66BB6A80"),
                            (-self.entry_z, "#EF535080"),
                            (3.5,           "#AB47BC60"),
                            (-3.5,          "#AB47BC60"),
                            (0.0,           "#55555580")]:
            ax.axhline(level, color=col, lw=0.6, linestyle="--")

        # Regime shading (sit-out = red)
        if regimes and len(regimes) == n:
            for i in range(1, n):
                if regimes[i] == 1:
                    ax.axvspan(i-1, i, color="#FF6F0018", linewidth=0)

        ax.set_title("Z-score  ·  dashed = ±entry / ±stop thresholds"
                      + ("  ·  orange shading = HMM sit-out" if regimes else ""),
                      color=COLORS["text"], fontsize=9, loc="left")
        ax.set_ylabel("Z-score", color=COLORS["text"], fontsize=8)
        ax.set_ylim(-5, 5)

        # ── Panel 4: Equity + Drawdown ──
        ax = axs[3]
        eq = equity if equity else [1e6] * n
        dd = _max_drawdown_series(eq)
        xs_eq = list(range(len(eq)))

        ax.fill_between(xs_eq, eq, eq[0],
                         where=[v >= eq[0] for v in eq],
                         alpha=0.25, color=COLORS["equity"])
        ax.fill_between(xs_eq, eq, eq[0],
                         where=[v < eq[0] for v in eq],
                         alpha=0.20, color=COLORS["drawdown"])
        ax.plot(xs_eq, eq, color=COLORS["equity"], lw=1.0, label="Equity")

        ax2 = ax.twinx()
        ax2.fill_between(xs_eq, [d*100 for d in dd], 0,
                          alpha=0.30, color=COLORS["drawdown"])
        ax2.set_ylabel("Drawdown %", color=COLORS["drawdown"], fontsize=8)
        ax2.tick_params(colors=COLORS["drawdown"], labelsize=8)
        ax2.set_facecolor("#1a1a1a")
        ax2.invert_yaxis()

        ax.set_title("Equity Curve  ·  Drawdown (right axis)",
                      color=COLORS["text"], fontsize=9, loc="left")
        ax.set_ylabel("NAV", color=COLORS["text"], fontsize=8)
        ax.set_xlabel("Bar", color=COLORS["text"], fontsize=8)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                         facecolor="#0f0f0f")
            print(f"[INFO] Dashboard saved to {save_path}")

        if show:
            plt.show()
        plt.close()

    def _plot_ascii(self, prices_a, prices_b, spread, equity, trades):
        """Fallback ASCII dashboard when matplotlib is unavailable."""
        WIDTH = 72
        n     = len(spread)
        z     = _zscore(spread, self.z_window)

        def ascii_line(values, lo, hi, width, title):
            print(f"\n  {title}")
            print("  " + "─" * width)
            rows = 8
            for row in range(rows, -1, -1):
                threshold = lo + (hi - lo) * row / rows
                line = ""
                for i in range(0, n, max(1, n // width)):
                    v = values[min(i, n-1)]
                    if abs(v - threshold) < (hi - lo) / rows / 2:
                        line += "·"
                    elif v >= threshold:
                        line += "█"
                    else:
                        line += " "
                label = f"{threshold:>7.2f} │"
                print(f"  {label}{line}")
            print("  " + " " * 9 + "└" + "─" * width)

        print("\n╔" + "═" * WIDTH + "╗")
        print(f"║  StatArb ASCII Dashboard  ·  {self.ticker_a}/{self.ticker_b}"
              + " " * (WIDTH - 36 - len(self.ticker_a) - len(self.ticker_b)) + "║")
        print("╚" + "═" * WIDTH + "╝")

        if spread:
            mn, mx = min(spread), max(spread)
            ascii_line(spread, mn, mx, WIDTH - 10,
                        f"SPREAD  (min={mn:.2f}, max={mx:.2f})")

        if z:
            ascii_line(z, -4, 4, WIDTH - 10,
                        f"Z-SCORE  (entry=±{self.entry_z})")

        if equity:
            mn, mx = min(equity), max(equity)
            ascii_line(equity, mn, mx, WIDTH - 10,
                        f"EQUITY  (start={equity[0]:.0f}, end={equity[-1]:.0f})")

        if trades:
            total_pnl = sum(t.get("net_pnl", 0) for t in trades)
            wins = sum(1 for t in trades if t.get("net_pnl", 0) > 0)
            print(f"\n  Trades: {len(trades)}  "
                  f"Wins: {wins}  "
                  f"Net P&L: {total_pnl:+.2f}\n")


# ── Synthetic demo data generator ─────────────────────────────────────────────
def _make_demo_data(n: int = 400, seed: int = 42):
    random.seed(seed)

    def rw(start, vol, drift=0.0002):
        p = [start]
        for _ in range(n - 1):
            z = random.gauss(0, 1)
            p.append(p[-1] * math.exp(drift + vol * z))
        return p

    def ou(theta=0.05, sigma=1.0):
        x = [0.0]
        for _ in range(n - 1):
            z = random.gauss(0, 1)
            x.append(x[-1] * (1 - theta) + sigma * z)
        return x

    pB     = rw(500, 0.012)
    noise  = ou(theta=0.05, sigma=3.0)
    pA     = [1.5 * pB[i] + 20 + noise[i] for i in range(n)]
    spread = noise  # true spread = noise

    # Fake equity
    equity = [1_000_000.0]
    for i in range(1, n):
        equity.append(equity[-1] + random.gauss(200, 800))

    # Fake trades
    trades = []
    i = 30
    while i < n - 20:
        skip = random.randint(10, 30)
        if i + skip >= n:
            break
        pnl = random.gauss(300, 1200)
        trades.append({
            "entry_bar": i,
            "exit_bar":  i + skip,
            "direction": random.choice(["LONG_A", "SHORT_A"]),
            "entry_z":   random.uniform(-3, -2),
            "exit_z":    random.uniform(-0.5, 0.5),
            "net_pnl":   pnl,
            "stopped":   pnl < -2000,
        })
        i += skip + random.randint(5, 15)

    return pA, pB, spread, equity, trades


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StatArb Spread Dashboard")
    parser.add_argument("--prices",   default="data/prices",
                        help="Directory with <TICKER>.csv files")
    parser.add_argument("--trades",   default="data/results/trades.csv")
    parser.add_argument("--equity",   default="data/results/equity.csv")
    parser.add_argument("--tickerA",  default="HDFC")
    parser.add_argument("--tickerB",  default="ICICI")
    parser.add_argument("--zwindow",  type=int, default=30)
    parser.add_argument("--out",      default="data/results/dashboard.png")
    parser.add_argument("--no-show",  action="store_true")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║     StatArb - Spread Visualisation Dashboard     ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Try loading real data
    path_a  = os.path.join(args.prices, f"{args.tickerA}.csv")
    path_b  = os.path.join(args.prices, f"{args.tickerB}.csv")
    _, pA   = read_price_csv(path_a)
    _, pB   = read_price_csv(path_b)
    trades  = read_trades_csv(args.trades)
    equity  = read_equity_csv(args.equity)

    if not pA or not pB:
        print("[INFO] Price data not found - using synthetic demo prices\n")
        demo_pA, demo_pB, demo_spread, demo_equity, demo_trades = _make_demo_data(400)
        pA, pB, spread = demo_pA, demo_pB, demo_spread
        # Only fall back to demo trades/equity if real ones weren't loaded -
        # missing price CSVs shouldn't discard a real trades.csv/equity.csv
        # that DID load successfully.
        if not trades:
            trades = demo_trades
        if not equity:
            equity = demo_equity
    else:
        # Align lengths
        n      = min(len(pA), len(pB))
        pA, pB = pA[:n], pB[:n]
        # Use the actual fitted hedge ratio from the trades CSV rather than
        # a hardcoded guess - the real β varies per pair (the C++ engine's
        # Kalman filter tracks it dynamically), so a fixed constant here
        # would show a spread the backtest never actually traded.
        if trades and "hedge_ratio" in trades[0]:
            beta = trades[0]["hedge_ratio"]
        else:
            beta = 1.0
            print("[WARN] No trades.csv hedge ratio available - using β=1.0 "
                  "for the spread plot; pass a real trades.csv for an accurate spread.\n")
        spread = [pA[i] - beta * pB[i] for i in range(n)]

    db = SpreadDashboard(ticker_a=args.tickerA, ticker_b=args.tickerB,
                          z_window=args.zwindow)
    db.plot(pA, pB, spread, equity, trades,
             save_path=args.out, show=not args.no_show)