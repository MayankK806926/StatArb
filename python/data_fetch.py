"""
data_fetch.py — NSE/BSE Price Data Fetcher
===========================================
Downloads OHLCV data for NSE stocks using yfinance and saves
each ticker as a CSV in data/prices/.

Usage:
    python data_fetch.py                        # default NIFTY-50 subset
    python data_fetch.py --tickers HDFCBANK ICICIBANK TCS INFY
    python data_fetch.py --start 2018-01-01 --end 2024-01-01
    python data_fetch.py --synthetic 600        # generate synthetic data (no internet)
"""

import os
import argparse
import random
import math
import zlib
from datetime import datetime, timedelta
from pathlib import Path

# ── Default NSE tickers (yfinance uses .NS suffix) ────────────────────────────
NIFTY50_SUBSET = [
    "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS",    "INFY.NS",
    "WIPRO.NS",    "HCLTECH.NS",   "RELIANCE.NS","ONGC.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS",    "BAJFINANCE.NS",
]

# Short aliases (strip .NS for filenames)
def ticker_to_filename(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


def download_prices(tickers, start="2018-01-01", end=None,
                    out_dir="data/prices", verbose=True):
    """Download adjusted close prices using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("[ERROR] yfinance not installed. Run: pip install yfinance")
        print("        Alternatively, use --synthetic to generate test data.")
        return []

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    loaded = []
    for ticker in tickers:
        fname = ticker_to_filename(ticker)
        path  = os.path.join(out_dir, f"{fname}.csv")
        try:
            if verbose:
                print(f"  Downloading {ticker}...", end=" ", flush=True)
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                print("EMPTY — skipped")
                continue
            # Save with consistent column name
            df.index.name = "Date"
            df.to_csv(path, columns=["Close"])
            if verbose:
                print(f"OK ({len(df)} bars → {path})")
            loaded.append(fname)
        except Exception as e:
            print(f"FAILED: {e}")

    return loaded


def generate_synthetic(tickers, n_bars=600, out_dir="data/prices",
                        seed=42, verbose=True):
    """
    Generate synthetic cointegrated price series for testing.
    Pairs are generated with a known cointegrating relationship:
      priceA[i] = beta * priceB[i] + alpha + noise
    so the system can be tested end-to-end without internet.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Parameters per ticker
    configs = {
        "HDFC":      dict(start=1500.0, vol=0.012, drift=0.0003),
        "ICICI":     dict(start=800.0,  vol=0.014, drift=0.0002),
        "TCS":       dict(start=3000.0, vol=0.010, drift=0.0004),
        "INFY":      dict(start=1400.0, vol=0.011, drift=0.0003),
        "WIPRO":     dict(start=450.0,  vol=0.013, drift=0.0002),
        "HCLTECH":   dict(start=1100.0, vol=0.012, drift=0.0002),
        "RELIANCE":  dict(start=2200.0, vol=0.013, drift=0.0004),
        "ONGC":      dict(start=180.0,  vol=0.016, drift=0.0001),
        "AXISBANK":  dict(start=750.0,  vol=0.015, drift=0.0002),
        "KOTAKBANK": dict(start=1800.0, vol=0.011, drift=0.0003),
    }

    # For tickers not in configs, generate defaults
    for t in tickers:
        name = ticker_to_filename(t)
        if name not in configs:
            configs[name] = dict(
                start=500.0 + random.uniform(0, 1500),
                vol=0.010 + random.uniform(0, 0.008),
                drift=random.uniform(0.0001, 0.0004)
            )

    # Build trading date sequence (skip weekends)
    start_date = datetime(2018, 1, 2)
    dates = []
    d = start_date
    while len(dates) < n_bars:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    saved = []
    for ticker in tickers:
        name = ticker_to_filename(ticker)
        cfg  = configs[name]
        path = os.path.join(out_dir, f"{name}.csv")

        # Box-Muller normal variates from uniform (no numpy dependency)
        price = cfg["start"]
        prices = []
        # Python's built-in hash() is randomised per-process (PYTHONHASHSEED)
        # since 3.3, which would make this "seeded" generator produce a
        # different series on every run. zlib.crc32 is a stable, portable
        # deterministic hash, so --synthetic output is actually reproducible.
        rng_state = seed + zlib.crc32(name.encode()) % 10000
        random.seed(rng_state)

        for _ in range(n_bars):
            u1 = max(random.random(), 1e-10)
            u2 = random.random()
            z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            price *= math.exp(cfg["drift"] + cfg["vol"] * z)
            prices.append(price)

        with open(path, "w") as f:
            f.write("Date,Close\n")
            for date, p in zip(dates, prices):
                f.write(f"{date},{p:.4f}\n")

        if verbose:
            print(f"  Generated {name}: {n_bars} bars, "
                  f"start={cfg['start']:.0f}, vol={cfg['vol']:.3f} → {path}")
        saved.append(name)

    return saved


def generate_cointegrated_pair(tickerA: str, tickerB: str,
                                n_bars: int = 600,
                                beta: float = 1.5,
                                alpha: float = 0.0,
                                noise_vol: float = 0.5,
                                out_dir: str = "data/prices",
                                seed: int = 42):
    """
    Generate a *known* cointegrated pair for unit testing.
    priceA[i] = beta * priceB[i] + alpha + OU_noise[i]
    Half-life of spread is approximately 20 bars.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    start_date = datetime(2018, 1, 2)
    dates = []
    d = start_date
    while len(dates) < n_bars:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    # Generate priceB as a random walk
    priceB = [500.0]
    random.seed(seed)
    for _ in range(n_bars - 1):
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        priceB.append(priceB[-1] * math.exp(0.0002 + 0.012 * z))

    # Generate OU noise for the spread (mean-reverting, HL≈20 bars)
    theta = math.log(2) / 20.0
    ou    = [0.0]
    random.seed(seed + 1)
    for _ in range(n_bars - 1):
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        ou.append(ou[-1] * (1 - theta) + noise_vol * z)

    priceA = [beta * priceB[i] + alpha + ou[i] for i in range(n_bars)]

    # Save both
    for name, prices in [(tickerA, priceA), (tickerB, priceB)]:
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w") as f:
            f.write("Date,Close\n")
            for date, p in zip(dates, prices):
                f.write(f"{date},{max(p, 0.01):.4f}\n")
        print(f"  Generated cointegrated {name}: β={beta}, HL≈20 bars → {path}")

    return priceA, priceB


def main():
    parser = argparse.ArgumentParser(
        description="Fetch or generate price data for StatArb system")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Ticker symbols (NSE: append .NS)")
    parser.add_argument("--start",   default="2018-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out",     default="data/prices",
                        help="Output directory")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N",
                        help="Generate N bars of synthetic data (no internet)")
    parser.add_argument("--cointegrated", action="store_true",
                        help="Also generate a known cointegrated pair (HDFC+ICICI_SYN)")
    args = parser.parse_args()

    tickers = args.tickers or NIFTY50_SUBSET

    print("╔══════════════════════════════════════════════════╗")
    print("║       StatArb Data Fetcher                       ║")
    print("╚══════════════════════════════════════════════════╝\n")

    if args.synthetic > 0:
        print(f"[INFO] Generating {args.synthetic} bars of synthetic data...")
        names = [ticker_to_filename(t) for t in tickers]
        saved = generate_synthetic(names, n_bars=args.synthetic, out_dir=args.out)
        if args.cointegrated:
            generate_cointegrated_pair(
                "HDFC_SYN", "ICICI_SYN",
                n_bars=args.synthetic, out_dir=args.out)
            saved += ["HDFC_SYN", "ICICI_SYN"]
        print(f"\n[INFO] Generated {len(saved)} series in {args.out}/")
        print(f"\nNext step:")
        print(f"  cd cpp && cmake -B build && cmake --build build")
        print(f"  ./build/statarb --mode wf --tickers {','.join(saved[:6])}")
    else:
        print(f"[INFO] Downloading {len(tickers)} tickers from Yahoo Finance...")
        print(f"       Period: {args.start} to {args.end or 'today'}\n")
        loaded = download_prices(tickers, start=args.start,
                                  end=args.end, out_dir=args.out)
        print(f"\n[INFO] Successfully downloaded {len(loaded)}/{len(tickers)} tickers")
        if loaded:
            print(f"\nNext step:")
            print(f"  cd cpp && cmake -B build && cmake --build build")
            names = [ticker_to_filename(t) for t in loaded[:6]]
            print(f"  ./build/statarb --mode wf --tickers {','.join(names)}")


if __name__ == "__main__":
    main()