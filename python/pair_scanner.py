"""
pair_scanner.py — NIFTY-50 Cointegration Pair Scanner
======================================================
Scans all C(n,2) pairs from a universe of NSE stocks.
Can use either:
  1. The compiled C++ statarb binary (fastest, ~3s for 1225 pairs)
  2. Pure-Python fallback (slower, ~45s for 1225 pairs)

Reports top cointegrated pairs ranked by ADF p-value + Hurst exponent.

Usage:
    python pair_scanner.py                        # scan default NIFTY-50 subset
    python pair_scanner.py --tickers HDFC ICICI TCS INFY WIPRO HCLTECH
    python pair_scanner.py --binary ../cpp/build/statarb
    python pair_scanner.py --top 10
"""

import os
import csv
import math
import random
import subprocess
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ─── NIFTY-50 subset (NSE tickers without .NS suffix) ────────────────────────
NIFTY50_PAIRS_UNIVERSE = [
    "HDFC", "ICICI", "TCS", "INFY", "WIPRO",
    "HCLTECH", "RELIANCE", "ONGC", "AXISBANK", "KOTAKBANK",
    "SBIN", "BAJFINANCE",
]

# Known classic pairs (for validation)
CLASSIC_PAIRS = [
    ("HDFC",  "ICICI",    "Banking — large cap"),
    ("TCS",   "INFY",     "IT services — large cap"),
    ("WIPRO", "HCLTECH",  "IT services — mid cap"),
    ("RELIANCE", "ONGC",  "Energy — different profiles"),
]


# ─── Data structures ──────────────────────────────────────────────────────────
@dataclass
class PairResult:
    ticker_a:       str
    ticker_b:       str
    beta:           float    # OLS hedge ratio
    adf_stat:       float    # ADF test statistic
    p_value:        float    # MacKinnon p-value
    hurst:          float    # Hurst exponent
    half_life:      float    # bars
    spread_std:     float    # spread std dev
    cointegrated:   bool
    score:          float    # combined ranking score (lower = better)
    sector_match:   bool = False
    note:           str  = ""


# ─── Pure-Python stats (no numpy) ─────────────────────────────────────────────
def _mean(v): return sum(v) / len(v) if v else 0.0

def _std(v, ddof=1):
    if len(v) < 2: return 0.0
    m = _mean(v)
    return math.sqrt(sum((x-m)**2 for x in v) / (len(v)-ddof))

def _ols(x, y):
    """Returns (alpha, beta, residuals)."""
    n  = min(len(x), len(y))
    mx = _mean(x[:n]); my = _mean(y[:n])
    sxy = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    sxx = sum((x[i]-mx)**2 for i in range(n))
    if sxx < 1e-14: return 0.0, 0.0, y[:n]
    beta  = sxy / sxx
    alpha = my - beta * mx
    resid = [y[i] - (alpha + beta*x[i]) for i in range(n)]
    return alpha, beta, resid

def _adf_approx(series) -> Tuple[float, float]:
    """
    Simplified ADF test (no lags, constant only).
    Returns (statistic, approx_p_value).
    Accurate enough for screening; use C++ engine for final decisions.
    """
    n = len(series)
    if n < 20: return 0.0, 1.0

    delta = [series[i+1]-series[i] for i in range(n-1)]
    level = series[:-1]

    # OLS: delta = a + b*level
    alpha, beta, resid = _ols(level, delta)

    if abs(beta) < 1e-12: return 0.0, 1.0

    # SE of beta
    ss_res = sum(r**2 for r in resid)
    s2     = ss_res / (n - 3)
    mx     = _mean(level)
    sxx    = sum((x-mx)**2 for x in level)
    if sxx < 1e-14 or s2 < 1e-14: return 0.0, 1.0
    se     = math.sqrt(s2 / sxx)
    t_stat = beta / se

    # Approximate p-value (MacKinnon table approximation).
    # Kept identical to CointegrationEngine::mackinnon_pvalue() in the C++
    # engine (same anchor points/breakpoints) so the Python fallback
    # screener and the C++ screener rank pairs the same way.
    if t_stat < -5.0: return t_stat, 0.0001
    if t_stat > 0.0:  return t_stat, 0.9999
    if t_stat <= -4.0:    p = 0.0001 + (t_stat + 5.0) / 1.0  * 0.0009
    elif t_stat <= -3.45: p = 0.001  + (t_stat + 4.0) / 0.55 * 0.009
    elif t_stat <= -2.87: p = 0.01   + (t_stat + 3.45) / 0.58 * 0.04
    elif t_stat <= -2.57: p = 0.05   + (t_stat + 2.87) / 0.30 * 0.05
    elif t_stat <= -1.94: p = 0.10   + (t_stat + 2.57) / 0.63 * 0.15
    else: p = min(0.25 + (t_stat + 1.94) / 1.94 * 0.50, 0.99)
    return t_stat, max(0.0001, min(0.9999, p))

def _half_life(spread) -> float:
    n = len(spread)
    if n < 10: return 999.0
    delta  = [spread[i+1]-spread[i] for i in range(n-1)]
    level  = spread[:-1]
    _, b, _ = _ols(level, delta)
    if b >= 0: return 999.0
    return math.log(2) / (-b)

def _hurst(series, min_size=20) -> float:
    n = len(series)
    if n < min_size: return 0.5
    lags  = [l for l in [4,8,16,32,64] if l < n]
    logN, logRS = [], []
    for lag in lags:
        rs_vals = []
        for start in range(0, n-lag, lag):
            sub = series[start:start+lag]
            m   = _mean(sub)
            dev = [x-m for x in sub]
            cum = []
            s   = 0.0
            for d in dev: s += d; cum.append(s)
            R   = max(cum) - min(cum)
            S   = _std(sub)
            if S > 1e-10: rs_vals.append(R/S)
        if rs_vals:
            logN.append(math.log(lag))
            logRS.append(math.log(_mean(rs_vals)))
    if len(logN) < 2: return 0.5
    _, h, _ = _ols(logN, logRS)
    return max(0.0, min(1.0, h))


# ─── CSV price reader ─────────────────────────────────────────────────────────
def load_prices(data_dir: str, tickers: List[str]) -> Dict[str, List[float]]:
    prices = {}
    for t in tickers:
        path = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(path):
            continue
        vals = []
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = next((k for k in row if "close" in k.lower()), None)
                    if key:
                        try: vals.append(float(row[key]))
                        except ValueError: pass
        except Exception:
            pass
        if len(vals) >= 50:
            prices[t] = vals
    return prices


# ─── Python screener ──────────────────────────────────────────────────────────
def screen_python(prices: Dict[str, List[float]],
                   p_threshold: float = 0.05,
                   hurst_threshold: float = 0.5,
                   min_hl: float = 5.0,
                   max_hl: float = 120.0,
                   verbose: bool = True) -> List[PairResult]:
    tickers = sorted(prices.keys())
    total   = len(tickers) * (len(tickers)-1) // 2
    results = []
    done    = 0

    if verbose:
        print(f"[Python] Screening {total} pairs from {len(tickers)} tickers...\n")

    for i, a in enumerate(tickers):
        for j in range(i+1, len(tickers)):
            b    = tickers[j]
            pA   = prices[a]; pB = prices[b]
            n    = min(len(pA), len(pB))
            pA, pB = pA[:n], pB[:n]
            done += 1

            # OLS regression
            alpha, beta, resid = _ols(pB, pA)
            if beta <= 0.05: continue

            # ADF on residuals
            t_stat, p_val = _adf_approx(resid)

            if p_val >= p_threshold: continue

            # Hurst + half-life
            H  = _hurst(resid)
            hl = _half_life(resid)

            if H >= hurst_threshold: continue
            if not (min_hl <= hl <= max_hl): continue

            pr = PairResult(
                ticker_a=a, ticker_b=b,
                beta=beta, adf_stat=t_stat, p_value=p_val,
                hurst=H, half_life=hl,
                spread_std=_std(resid),
                cointegrated=True,
                score=p_val * 0.6 + H * 0.4,
            )
            results.append(pr)

            if verbose:
                print(f"  [PASS] {a}/{b}  β={beta:.3f}  "
                      f"ADF={t_stat:.3f}  p={p_val:.3f}  "
                      f"H={H:.3f}  HL={hl:.0f}d")

    results.sort(key=lambda x: x.score)
    if verbose:
        print(f"\n[Python] {len(results)}/{done} pairs cointegrated.\n")
    return results


# ─── C++ binary bridge ────────────────────────────────────────────────────────
def screen_cpp(binary: str, data_dir: str,
                tickers: List[str]) -> Optional[List[PairResult]]:
    """
    Call the compiled C++ statarb binary in --mode screen and parse output.
    Typical speedup: 10–15× over Python.
    """
    if not os.path.exists(binary):
        print(f"[WARN] C++ binary not found at {binary}. Using Python screener.")
        return None

    ticker_str = ",".join(tickers)
    cmd = [binary, "--mode", "screen",
           "--data",    data_dir,
           "--tickers", ticker_str,
           "--quiet"]

    print(f"[C++] Running: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[WARN] C++ binary error: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("[WARN] C++ binary timed out.")
        return None
    except Exception as e:
        print(f"[WARN] Could not run C++ binary: {e}")
        return None

    # Parse stdout for [PASS] lines
    results = []
    parse_failures = 0
    pass_lines = 0
    for line in result.stdout.splitlines():
        if "[PASS]" not in line: continue
        pass_lines += 1
        try:
            # C++ line: "  [PASS] HDFC/ICICI  β=1.5000  ADF=-4.12  p=0.0012  H=0.42  HL=15d"
            # .split() → ["[PASS]", "HDFC/ICICI", "β=...", "ADF=...", "p=...", "H=...", "HL=...']
            # (previously indexed one-off from parts[2..7], which always
            # silently produced zero results — nothing ever surfaced this
            # because the failure was swallowed with a bare `continue`.)
            parts  = line.split()
            pair   = parts[1]                          # e.g. HDFC/ICICI
            a, b   = pair.split("/")
            beta   = float(parts[2].split("=")[1])
            adf    = float(parts[3].split("=")[1])
            pval   = float(parts[4].split("=")[1])
            hurst  = float(parts[5].split("=")[1])
            hl_str = parts[6].split("=")[1].rstrip("d")
            hl     = float(hl_str)
            results.append(PairResult(
                ticker_a=a, ticker_b=b, beta=beta,
                adf_stat=adf, p_value=pval, hurst=hurst,
                half_life=hl, spread_std=0.0,
                cointegrated=True,
                score=pval*0.6 + hurst*0.4,
            ))
        except (IndexError, ValueError):
            parse_failures += 1
            continue

    if parse_failures:
        print(f"[WARN] Failed to parse {parse_failures}/{pass_lines} [PASS] line(s) "
              f"from the C++ binary's output — its log format may have changed. "
              f"Results below may be incomplete.")

    results.sort(key=lambda x: x.score)
    return results


# ─── Print results table ──────────────────────────────────────────────────────
def print_results(results: List[PairResult], top: int = 10):
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  TOP {top} COINTEGRATED PAIRS                                      ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"  {'Rank':<5} {'Pair':<16} {'β':>7} {'ADF':>8} {'p-val':>8} "
          f"{'Hurst':>7} {'HL(d)':>7} {'Score':>7}")
    print("  " + "─" * 65)
    for rank, pr in enumerate(results[:top], 1):
        pair  = f"{pr.ticker_a}/{pr.ticker_b}"
        flag  = " ★" if any(pr.ticker_a == c[0] and pr.ticker_b == c[1]
                              for c in CLASSIC_PAIRS) else ""
        print(f"  {rank:<5} {pair:<16} {pr.beta:>7.3f} {pr.adf_stat:>8.3f} "
              f"{pr.p_value:>8.4f} {pr.hurst:>7.3f} {int(pr.half_life):>7} "
              f"{pr.score:>7.4f}{flag}")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("\n  ★ = known classic NSE pair\n")


# ─── Export to CSV ────────────────────────────────────────────────────────────
def export_pairs_csv(results: List[PairResult], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank","ticker_a","ticker_b","beta",
                          "adf_stat","p_value","hurst","half_life_bars",
                          "spread_std","score"])
        for i, pr in enumerate(results, 1):
            writer.writerow([i, pr.ticker_a, pr.ticker_b,
                              f"{pr.beta:.4f}", f"{pr.adf_stat:.4f}",
                              f"{pr.p_value:.4f}", f"{pr.hurst:.4f}",
                              f"{pr.half_life:.1f}", f"{pr.spread_std:.4f}",
                              f"{pr.score:.4f}"])
    print(f"[INFO] Wrote {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NIFTY-50 Pairs Scanner")
    parser.add_argument("--tickers", nargs="+", default=NIFTY50_PAIRS_UNIVERSE)
    parser.add_argument("--data",    default="data/prices",
                        help="Directory with <TICKER>.csv files")
    parser.add_argument("--binary",  default="../cpp/build/statarb",
                        help="Path to compiled C++ statarb binary")
    parser.add_argument("--out",     default="data/results/pairs.csv")
    parser.add_argument("--top",     type=int, default=10)
    parser.add_argument("--pval",    type=float, default=0.05)
    parser.add_argument("--force-python", action="store_true",
                        help="Skip C++ binary, use Python screener")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║     StatArb — NIFTY Pair Scanner                 ║")
    print("╚══════════════════════════════════════════════════╝\n")
    print(f"Universe : {len(args.tickers)} tickers")
    print(f"Pairs    : {len(args.tickers)*(len(args.tickers)-1)//2}")
    print(f"Data dir : {args.data}\n")

    # Try C++ binary first (much faster)
    results = None
    if not args.force_python:
        results = screen_cpp(args.binary, args.data, args.tickers)

    # Fall back to Python
    if results is None:
        prices = load_prices(args.data, args.tickers)
        if not prices:
            print("[WARN] No price data found in", args.data)
            print("       Run: python data_fetch.py --synthetic 600")
            print("       Or:  python data_fetch.py  (downloads from yfinance)")
            # Demo with synthetic data
            print("\n[INFO] Generating demo synthetic pairs for illustration...\n")
            random.seed(42)
            prices = {}
            for t in args.tickers[:6]:
                p = [500.0]
                for _ in range(499):
                    p.append(p[-1] * math.exp(0.0002 + 0.012*random.gauss(0,1)))
                prices[t] = p
            # Make HDFC and ICICI cointegrated
            if "HDFC" in prices and "ICICI" in prices:
                ou = [0.0]
                for _ in range(499):
                    ou.append(ou[-1]*0.95 + 0.5*random.gauss(0,1))
                prices["HDFC"] = [1.5*prices["ICICI"][i]+20+ou[i]
                                   for i in range(500)]
        results = screen_python(prices, p_threshold=args.pval)

    if not results:
        print("[INFO] No cointegrated pairs found with current settings.")
        print("       Try relaxing --pval threshold or generating more data.")
        return

    print_results(results, top=args.top)
    export_pairs_csv(results, args.out)

    # Print suggested next command
    if results:
        best = results[0]
        print(f"\nSuggested next step (deep analysis on best pair):")
        print(f"  cd cpp")
        print(f"  ./build/statarb --mode pair "
              f"--tickerA {best.ticker_a} --tickerB {best.ticker_b} --debug")
        print(f"  python tearsheet.py data/results/trades.csv")
        print(f"  python visualise.py --tickerA {best.ticker_a} "
              f"--tickerB {best.ticker_b}\n")


if __name__ == "__main__":
    main()