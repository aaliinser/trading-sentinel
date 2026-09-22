#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 Popular Binary Options Strategies — Clean Backtest Engine
All on 5m timeframe, 15m expiry (3 candles), same assets, same data.
Signal on CLOSED candle i, entry = close[i], exit = close[i+3].
Ties skipped. No look-ahead. Standard indicator math.

S1: Support/Resistance Reversal (20-bar swing levels)
S2: Pin Bar / Rejection Candle (long wick, small body)
S3: Stochastic (14,3,3) crossover in oversold/overbought
S4: MACD (12,26,9) crossover + EMA50 trend filter
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

ASSETS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
PAYOUT = 0.90
BREAKEVEN = 1.0 / (1.0 + PAYOUT) * 100.0  # 52.63%

INTERVAL = "5m"
PERIOD = "60d"
EXPIRY_CANDLES = 3

# ─── Data helpers ───
def flatten_columns(df):
    OHLC = ("Open", "High", "Low", "Close")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c) for c in df.columns]
        return df
    df = df.copy()
    lvl = None
    for L in range(df.columns.nlevels):
        if {str(v) for v in df.columns.get_level_values(L)} & set(OHLC):
            lvl = L
            break
    if lvl is None:
        lvl = df.columns.nlevels - 1
    df.columns = [str(c) for c in df.columns.get_level_values(lvl)]
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

def get_data():
    out = {}
    for t in ASSETS:
        try:
            df = yf.Ticker(t).history(period=PERIOD, interval=INTERVAL)
            if df is None or df.empty:
                continue
            df = flatten_columns(df)
            if not all(c in df.columns for c in ["Open","High","Low","Close"]):
                continue
            df = df[["Open","High","Low","Close"]].astype(float).dropna()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if len(df) > 200:
                out[t] = df
        except Exception as e:
            print(f"  [{t}] fetch error: {e}")
    return out

# ─── S1: Support/Resistance Reversal ───
def strategy_S1(df):
    """Buy when price touches 20-bar support with bullish close.
       Sell when price touches 20-bar resistance with bearish close."""
    trades = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    o = df["Open"].values
    LOOKBACK = 20
    for i in range(LOOKBACK, len(df) - EXPIRY_CANDLES):
        support = min(l[i-LOOKBACK:i])
        resistance = max(h[i-LOOKBACK:i])
        # CALL: low touches support (within 0.1%) AND closes bullish
        if l[i] <= support * 1.001 and c[i] > o[i]:
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("CALL", entry, exit_px, exit_px > entry))
        # PUT: high touches resistance (within 0.1%) AND closes bearish
        elif h[i] >= resistance * 0.999 and c[i] < o[i]:
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("PUT", entry, exit_px, exit_px < entry))
    return trades

# ─── S2: Pin Bar / Rejection Candle ───
def strategy_S2(df):
    """Pin bar: small body, long rejection wick.
       Bullish pin: lower wick >= 2x body AND body < 33% of range.
       Bearish pin: upper wick >= 2x body AND body < 33% of range."""
    trades = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    o = df["Open"].values
    for i in range(1, len(df) - EXPIRY_CANDLES):
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        body = abs(c[i] - o[i])
        upper_wick = h[i] - max(c[i], o[i])
        lower_wick = min(c[i], o[i]) - l[i]
        if body / rng >= 0.33:
            continue  # body too big
        # Bullish pin: long lower wick
        if lower_wick >= 2 * body and lower_wick > upper_wick:
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("CALL", entry, exit_px, exit_px > entry))
        # Bearish pin: long upper wick
        elif upper_wick >= 2 * body and upper_wick > lower_wick:
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("PUT", entry, exit_px, exit_px < entry))
    return trades

# ─── S3: Stochastic (14,3,3) ───
def strategy_S3(df):
    """Buy: %K crosses above %D while %K < 20.
       Sell: %K crosses below %D while %K > 80."""
    trades = []
    c = df["Close"]
    high = df["High"]
    low = df["Low"]
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    denom = high14 - low14
    raw_k = 100 * (c - low14) / denom.replace(0, np.nan)
    K = raw_k.rolling(3).mean()
    D = K.rolling(3).mean()
    K = K.fillna(50).values
    D = D.fillna(50).values
    c_vals = c.values
    for i in range(20, len(df) - EXPIRY_CANDLES):
        # CALL: K crosses above D from below 20
        if K[i-1] <= D[i-1] and K[i] > D[i] and K[i] < 20:
            entry = c_vals[i]
            exit_px = c_vals[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("CALL", entry, exit_px, exit_px > entry))
        # PUT: K crosses below D from above 80
        elif K[i-1] >= D[i-1] and K[i] < D[i] and K[i] > 80:
            entry = c_vals[i]
            exit_px = c_vals[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("PUT", entry, exit_px, exit_px < entry))
    return trades

# ─── S4: MACD (12,26,9) + EMA50 trend filter ───
def strategy_S4(df):
    """Buy: MACD line crosses above signal line + close > EMA50.
       Sell: MACD line crosses below signal line + close < EMA50."""
    trades = []
    c = df["Close"].values
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    for i in range(60, len(df) - EXPIRY_CANDLES):
        # CALL: MACD crosses above signal + bullish trend
        if macd[i-1] <= signal[i-1] and macd[i] > signal[i] and c[i] > ema50[i]:
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("CALL", entry, exit_px, exit_px > entry))
        # PUT: MACD crosses below signal + bearish trend
        elif macd[i-1] >= signal[i-1] and macd[i] < signal[i] and c[i] < ema50[i]:
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px != entry:
                trades.append(("PUT", entry, exit_px, exit_px < entry))
    return trades

STRATEGIES = [
    ("S1: Support/Resistance Reversal", strategy_S1),
    ("S2: Pin Bar / Rejection Candle",  strategy_S2),
    ("S3: Stochastic (14,3,3)",         strategy_S3),
    ("S4: MACD (12,26,9) + EMA50",      strategy_S4),
]

def stats(trades):
    n = len(trades)
    if n == 0:
        return n, 0, 0.0, 0.0
    w = sum(1 for t in trades if t[3])
    wr = 100.0 * w / n
    p_star = 1.0 / (1.0 + PAYOUT)
    se = (p_star * (1 - p_star) / n) ** 0.5
    z = (wr / 100.0 - p_star) / se if se > 0 else 0.0
    return n, w, wr, z

def main():
    data = get_data()
    print(f"\n{'='*72}")
    print(f"4 POPULAR BINARY OPTIONS STRATEGIES — 5m signal / 15m expiry")
    print(f"Assets: {len(data)} | Break-even: {BREAKEVEN:.2f}%")
    print(f"{'='*72}")

    for name, fn in STRATEGIES:
        all_trades = []
        for sym, df in data.items():
            t = fn(df)
            all_trades.extend([(sym, dr, e, w) for (dr, _, _, w) in t for e in [0]])
            # rebuild with proper structure
        # rebuild with symbol tracking
        trades = []
        for sym, df in data.items():
            for (dr, entry, exit_px, win) in fn(df):
                trades.append((sym, dr, win))

        n, w, wr, z = stats(trades)
        print(f"\n--- {name} ---")
        if n == 0:
            print("NO SIGNALS GENERATED")
            continue
        print(f"trades: {n} | wins: {w} | WR: {wr:.2f}% | z: {z:+.2f}", end="  ")
        print("SIGNIFICANT" if z > 1.96 else "not significant")
        print(f"edge vs break-even ({BREAKEVEN:.2f}%): {wr - BREAKEVEN:+.2f} pp")
        mid = n // 2
        if mid > 0:
            h1 = trades[:mid]
            h2 = trades[mid:]
            _, _, wr1, _ = stats(h1)
            _, _, wr2, _ = stats(h2)
            ok1 = "OK" if wr1 >= BREAKEVEN else "FAIL"
            ok2 = "OK" if wr2 >= BREAKEVEN else "FAIL"
            print(f"robustness: 1st half {wr1:.2f}% {ok1} | 2nd half {wr2:.2f}% {ok2}")
        by_sym = {}
        for sym, dr, win in trades:
            by_sym.setdefault(sym, []).append(win)
        print("per asset: " + " | ".join(
            f"{s.split('=')[0]} {100*sum(v)/len(v):.0f}%({len(v)})"
            for s, v in sorted(by_sym.items())))

    print(f"\n{'='*72}")
    print(f"SUMMARY TABLE")
    print(f"{'='*72}")
    for name, fn in STRATEGIES:
        trades = []
        for sym, df in data.items():
            for (dr, entry, exit_px, win) in fn(df):
                trades.append((sym, dr, win))
        n, w, wr, z = stats(trades)
        status = "WINNER" if wr >= BREAKEVEN and z > 1.96 else \
                 "MARGINAL" if wr >= BREAKEVEN else "LOSER"
        print(f"{name[:35]:35} | n={n:5d} | WR={wr:5.2f}% | z={z:+.2f} | {status}")

if __name__ == "__main__":
    main()
