#!/usr/bin/env python3
"""
Fetch ETH/USDT 1m candles from Binance.
Uses official monthly ZIPs when available, and falls back to live REST API
for any months not yet published.

Example:
    uv run get_data.py --start 2024-01 --end 2025-11 --out ethusdt_1m.csv --parquet
"""

import argparse, os, io, zipfile, requests, time
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE_ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m"
API_URL = "https://api.binance.com/api/v3/klines"

# ---------------------------------------------------------------------


def download_archive_month(year, month):
    fname = f"ETHUSDT-1m-{year}-{month:02d}.zip"
    url = f"{BASE_ARCHIVE}/{fname}"
    print(f"📦 Archive: {url}")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"⚠️  No archive for {year}-{month:02d} (HTTP {r.status_code})")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None)
    except Exception as e:
        print(f"⚠️  Failed to parse archive {year}-{month:02d}: {e}")
        return None

    df.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ][: df.shape[1]]
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    df = df.dropna(subset=["open_time"])
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


# ---------------------------------------------------------------------


def fetch_from_api_month(year, month, symbol="ETHUSDT"):
    print(f"🌐 API: fetching {year}-{month:02d}")
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    all_rows = []
    limit = 1000
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": limit,
            "startTime": start_ms,
        }
        r = requests.get(API_URL, params=params, timeout=15)
        if r.status_code != 200:
            print(f"⚠️  API error {r.status_code} — stopping month early")
            break
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        last_close = data[-1][6]  # close_time
        start_ms = last_close + 1
        time.sleep(0.25)  # respect rate limits

    if not all_rows:
        print(f"⚠️  No API data for {year}-{month:02d}")
        return None

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


# ---------------------------------------------------------------------


def ewma_vol(df, lam=0.94):
    df["ΔP"] = df["close"].pct_change()
    df["σ2"] = df["ΔP"].ewm(alpha=(1 - lam)).mean() ** 2
    return df


# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--out", default="ethusdt_1m.csv")
    ap.add_argument("--parquet", action="store_true")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m")
    end = datetime.strptime(args.end, "%Y-%m")

    frames = []
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        df = download_archive_month(y, m)
        if df is None or df.empty:
            df = fetch_from_api_month(y, m)
        if df is not None and not df.empty:
            frames.append(df)
        m += 1
        if m > 12:
            m, y = 1, y + 1

    if not frames:
        print("❌ No data gathered.")
        return

    df = pd.concat(frames).sort_values("open_time").reset_index(drop=True)
    df = ewma_vol(df)
    df.to_csv(args.out, index=False)
    print(f"✅ Saved {len(df):,} rows → {args.out}")

    if args.parquet:
        pq_path = args.out.replace(".csv", ".parquet")
        df.to_parquet(pq_path, index=False)
        print(f"✅ Saved Parquet → {pq_path}")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
