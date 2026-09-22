#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMA 9/21/50 Crossover — Binary Options Backtest (clean engine, no look-ahead)
Signal on CLOSED candle i: EMA9 x EMA21 cross + close vs EMA50 filter.
Entry = close[i], Exit = close[i+k]. Ties skipped. Wilder-free (standard EMA).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

ASSETS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
STAKE, PAYOUT = 6.0, 0.90
BREAKEVEN = 1.0 / (1.0 + PAYOUT) * 100.0

VARIANTS = [
    ("V1: 5m signal / 15m expiry", "60d", "5m", 3),
    ("V2: 5m signal / 5m expiry",  "60d", "5m", 1),
    ("V3: 1h signal / 60m expiry", "60d", "60m", 1),
]

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

def get_data(period, interval):
    out = {}
    for t in ASSETS:
        try:
            df = yf.Ticker(t).history(period=period, interval=interval)
            if df is None or df.empty:
                continue
            df = flatten_columns(df)
            if "Close" not in df.columns:
                continue
            c = df["Close"].astype(float).dropna()
            c.index = pd.to_datetime(c.index, utc=True)
            c = c[~c.index.duplicated(keep="last")].sort_index()
            if len(c) > 100:
                out[t] = c
        except Exception as e:
            print(f"  [{t}] fetch error: {e}")
    return out

def emas(c):
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    return e9, e21, e50

def run_variant(name, period, interval, k):
    data = get_data(period, interval)
    trades = []
    for sym, c in data.items():
        e9, e21, e50 = emas(c)
        cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
        cross_dn = (e9 < e21) & (e9.shift(1) >= e21.shift(1))
        for i in range(60, len(c) - k):
            cu, cd = bool(cross_up.iloc[i]), bool(cross_dn.iloc[i])
            if not (cu or cd):
                continue
            cl = float(c.iloc[i])
            if cu and cl > float(e50.iloc[i]):
                dr = "CALL"
            elif cd and cl < float(e50.iloc[i]):
                dr = "PUT"
            else:
                continue
            ex = float(c.iloc[i + k])
            if ex == cl:
                continue
            win = (ex > cl) if dr == "CALL" else (ex < cl)
            trades.append((sym, dr, win))
    return trades

def stats(trades):
    n = len(trades)
    if n == 0:
        return n, 0, 0.0, 0.0
    w = sum(1 for t in trades if t[2])
    wr = 100.0 * w / n
    p_star = 1.0 / (1.0 + PAYOUT)
    se = (p_star * (1 - p_star) / n) ** 0.5
    z = (wr / 100.0 - p_star) / se if se > 0 else 0.0
    return n, w, wr, z

def main():
    print("=" * 70)
    print("EMA 9/21/50 CROSSOVER | binary-adapted | clean engine")
    print("=" * 70)
    for name, period, interval, k in VARIANTS:
        trades = run_variant(name, period, interval, k)
        n, w, wr, z = stats(trades)
        print(f"\n--- {name} ---")
        print(f"trades: {n} | wins: {w} | WR: {wr:.2f}% | z: {z:+.2f}", end="  ")
        if n == 0:
            print("(no data)")
            continue
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
        if by_sym:
            print("per asset: " + " | ".join(
                f"{s.split('=')[0]} {100*sum(v)/len(v):.0f}%({len(v)})"
                for s, v in sorted(by_sym.items())))
        if n < 300:
            print(f"WARNING: n={n} < 300 -> preliminary only")

if __name__ == "__main__":
    main()
