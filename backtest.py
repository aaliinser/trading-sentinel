#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI + Bollinger Bands REVERSAL — Binary Options Backtest
M30 logic tested on 5m bars / 15m expiry (3 candles) — clean engine, no look-ahead.
Filters added vs H2: expanding bandwidth (momentum) + min_bb_width (skip dead markets).
RSI thresholds 70/30 (per the new strategy) instead of H2's 75/25.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───
ASSETS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
PERIOD = "60d"
SIGNAL_INTERVAL = "5m"
EXPIRY_CANDLES = 3          # 3 x 5m = 15 minutes

RSI_PERIOD = 14
BB_PERIOD = 20
BB_MULT = 2.0
RSI_OVERSOLD = 30.0         # strategy uses 30 (H2 used 25)
RSI_OVERBOUGHT = 70.0       # strategy uses 70 (H2 used 75)
MIN_BB_WIDTH = 0.005        # skip dead/low-volatility markets

STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN_WR = 1.0 / (1.0 + PAYOUT) * 100.0   # 52.6316%

# ─── DATA LAYER (stable, no MultiIndex issues) ───
def flatten_columns(df):
    OHLC = ("Open", "High", "Low", "Close")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c) for c in df.columns]
        return df
    df = df.copy()
    price_level = None
    for lvl in range(df.columns.nlevels):
        values = {str(v) for v in df.columns.get_level_values(lvl)}
        if values & set(OHLC):
            price_level = lvl
            break
    if price_level is None:
        price_level = df.columns.nlevels - 1
    df.columns = [str(c) for c in df.columns.get_level_values(price_level)]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df

def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period=PERIOD, interval=SIGNAL_INTERVAL)
        if df.empty:
            return pd.DataFrame()
        df = flatten_columns(df)
        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [{ticker}] missing columns {missing} - skipped")
            return pd.DataFrame()
        df = df[required].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
        return df
    except Exception as e:
        print(f"  [{ticker}] Fetch Error: {e}")
        return pd.DataFrame()

# ─── INDICATORS (TradingView-exact math) ───
def wilder_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    denom = avg_gain + avg_loss
    rsi = 100.0 * avg_gain / denom
    return rsi.where(denom > 0, 50.0)

def bollinger(close, period=BB_PERIOD, mult=BB_MULT):
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)   # population std
    upper = ma + mult * sd
    lower = ma - mult * sd
    width = (upper - lower) / ma                                  # normalised bandwidth
    return upper, lower, width

# ─── BACKTEST PER ASSET ───
def backtest_asset(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 60:
        print(f"  [{ticker}] skipped (insufficient data)")
        return []

    close = df["Close"].astype(float)
    df["RSI"] = wilder_rsi(close, RSI_PERIOD)
    df["BB_Upper"], df["BB_Lower"], df["BB_Width"] = bollinger(close, BB_PERIOD, BB_MULT)
    df = df.dropna(subset=["RSI", "BB_Upper", "BB_Lower", "BB_Width", "Close"])

    trades = []
    # need current bar + previous bar (for expanding) + 3 future bars (for expiry)
    for i in range(1, len(df) - EXPIRY_CANDLES):
        row = df.iloc[i]
        prev_width = float(df["BB_Width"].iloc[i - 1])
        curr_width = float(row["BB_Width"])

        # Filter 1: skip dead / low-volatility markets
        if curr_width < MIN_BB_WIDTH:
            continue
        # Filter 2: bandwidth must be EXPANDING (momentum present)
        if curr_width <= prev_width:
            continue

        r = float(row["RSI"])
        cl = float(row["Close"])
        bu = float(row["BB_Upper"])
        bl = float(row["BB_Lower"])

        direction = None
        if r <= RSI_OVERSOLD and cl <= bl:
            direction = "CALL"
        elif r >= RSI_OVERBOUGHT and cl >= bu:
            direction = "PUT"
        if direction is None:
            continue

        entry_price = cl
        exit_price = float(df["Close"].iloc[i + EXPIRY_CANDLES])
        if exit_price == entry_price:
            continue   # exact tie -> void

        win = (exit_price > entry_price) if direction == "CALL" else (exit_price < entry_price)
        trades.append({
            "Asset": ticker,
            "Direction": direction,
            "Entry": entry_price,
            "Exit": exit_price,
            "RSI": round(r, 2),
            "Width": round(curr_width, 5),
            "Won": win,
        })

    print(f"  [{ticker}] trades: {len(trades)}")
    return trades

# ─── METRICS & REPORT ───
def win_rate(wins, total):
    return (wins / total * 100.0) if total else 0.0

def report(all_trades):
    print("\n" + "=" * 68)
    print("RSI + BOLLINGER REVERSAL | RSI(14) 70/30 + BB(20,2.0)")
    print("Filters: expanding bandwidth + min width 0.005 | 5m signal / 15m expiry")
    print("=" * 68)

    if not all_trades:
        print("No trades generated. Nothing to evaluate.")
        return

    n = len(all_trades)
    wins = sum(1 for t in all_trades if t["Won"])
    losses = n - wins
    wr = win_rate(wins, n)
    net = wins * (STAKE * PAYOUT) - losses * STAKE

    print(f"Total trades      : {n}")
    print(f"Wins / Losses     : {wins} / {losses}")
    print(f"Win rate          : {wr:.2f}%")
    print(f"Break-even needed : {BREAKEVEN_WR:.2f}%   (payout {PAYOUT:.0%})")
    print(f"Edge vs break-even: {wr - BREAKEVEN_WR:+.2f} pp")
    print(f"Net P&L           : ${net:+.2f}   (stake ${STAKE:.2f})")

    # Statistical significance vs break-even
    p_hat = wr / 100.0
    p_star = 1.0 / (1.0 + PAYOUT)
    se = (p_star * (1.0 - p_star) / n) ** 0.5
    z = (p_hat - p_star) / se if se > 0 else 0.0
    print(f"\nz vs break-even   : {z:+.2f}", end="  ")
    print("-> SIGNIFICANT EDGE" if z > 1.96 else "-> NOT statistically distinguishable from noise")

    # Robustness: chronological 50/50 split
    mid = n // 2
    h1, h2 = all_trades[:mid], all_trades[mid:]
    w1 = sum(1 for t in h1 if t["Won"])
    w2 = sum(1 for t in h2 if t["Won"])
    wr1, wr2 = win_rate(w1, len(h1)), win_rate(w2, len(h2))
    print("\n" + "-" * 68)
    print("ROBUSTNESS CHECK (chronological 50/50 split)")
    print("-" * 68)
    print(f"First half  : {len(h1):>4} trades | WR {wr1:.2f}% {'OK' if wr1 >= BREAKEVEN_WR else 'FAIL'}")
    print(f"Second half : {len(h2):>4} trades | WR {wr2:.2f}% {'OK' if wr2 >= BREAKEVEN_WR else 'FAIL'}")
    robust = (wr1 >= BREAKEVEN_WR) and (wr2 >= BREAKEVEN_WR)
    print(f"\nVERDICT     : {'ROBUST' if robust else 'NOT ROBUST'}")

    if n < 300:
        print(f"\nWARNING     : n={n} < 300. Treat as preliminary only.")

    # Per-asset breakdown
    print("\n" + "-" * 68)
    print("BREAKDOWN BY ASSET")
    print("-" * 68)
    assets = sorted({t["Asset"] for t in all_trades})
    for a in assets:
        sub = [t for t in all_trades if t["Asset"] == a]
        sn = len(sub)
        sw = sum(1 for t in sub if t["Won"])
        print(f"  {a:<10} {sn:>4} trades | WR {win_rate(sw, sn):.2f}%")

    # Per-direction breakdown
    print("\n" + "-" * 68)
    print("BREAKDOWN BY DIRECTION")
    print("-" * 68)
    for dr in ["CALL", "PUT"]:
        sub = [t for t in all_trades if t["Direction"] == dr]
        sn = len(sub)
        sw = sum(1 for t in sub if t["Won"])
        print(f"  {dr:<6} {sn:>4} trades | WR {win_rate(sw, sn):.2f}%")

# ─── MAIN ───
def main():
    print("=" * 68)
    print("RSI + BB REVERSAL BACKTEST | last 60d | 5 assets")
    print("=" * 68)
    all_trades = []
    for ticker in ASSETS:
        all_trades.extend(backtest_asset(ticker))
    report(all_trades)

if __name__ == "__main__":
    main()
