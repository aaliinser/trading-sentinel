#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2 Strategy — 6-Month Backtest (EXACT v5.7.1 logic)
RSI Wilder(14) + Bollinger Bands(20, 2.0, ddof=0)
Entry: RSI <= 25 AND close <= lower band (CALL)
       RSI >= 75 AND close >= upper band (PUT)
5m timeframe, 15m expiry (3 candles)
No cooldown, no blackout — pure strategy performance
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

ASSETS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
RSI_P = 14
BB_P = 20
BB_K = 2.0
RSI_HI = 75.0
RSI_LO = 25.0
PAYOUT = 0.90
BREAKEVEN = 1.0 / (1.0 + PAYOUT) * 100.0

INTERVAL = "5m"
PERIOD = "180d"  # 6 months
EXPIRY_CANDLES = 3  # 15 minutes

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
                print(f"  [{t}] no data")
                continue
            df = flatten_columns(df)
            if not all(c in df.columns for c in ["Open","High","Low","Close"]):
                print(f"  [{t}] missing columns")
                continue
            df = df[["Open","High","Low","Close"]].astype(float).dropna()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if len(df) > 100:
                out[t] = df
                print(f"  [{t}] loaded {len(df)} candles")
            else:
                print(f"  [{t}] insufficient data")
        except Exception as e:
            print(f"  [{t}] fetch error: {e}")
    return out

def compute_indicators(df):
    """EXACT same RSI Wilder + BB as v5.7.1"""
    c = df["Close"]
    # RSI Wilder
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / RSI_P
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=RSI_P).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=RSI_P).mean()
    denom = avg_gain + avg_loss
    rsi = 100.0 * avg_gain / denom
    rsi = rsi.where(denom > 0, 50.0)
    # Bollinger Bands
    mid = c.rolling(BB_P, min_periods=BB_P).mean()
    sd = c.rolling(BB_P, min_periods=BB_P).std(ddof=0)
    bu = mid + BB_K * sd
    bl = mid - BB_K * sd
    return rsi, bu, bl, sd

def run_strategy(data):
    """Run H2 strategy on all assets — returns list of (symbol, direction, win, timestamp, v4)"""
    trades = []
    for sym, df in data.items():
        rsi, bu, bl, sd = compute_indicators(df)
        c = df["Close"].values
        r = rsi.values
        upper = bu.values
        lower = bl.values
        sdv = sd.values
        
        for i in range(max(RSI_P, BB_P), len(df) - EXPIRY_CANDLES):
            if np.isnan(r[i]) or np.isnan(sdv[i]) or sdv[i] <= 0:
                continue
            
            entry = c[i]
            exit_px = c[i + EXPIRY_CANDLES]
            if exit_px == entry:
                continue
            
            # CALL: RSI <= 25 AND close <= lower band
            if r[i] <= RSI_LO and c[i] <= lower[i]:
                win = exit_px > entry
                over = (lower[i] - c[i]) / sdv[i]
                v4 = over >= 0.5
                trades.append((sym, "CALL", win, df.index[i], v4))
            
            # PUT: RSI >= 75 AND close >= upper band
            elif r[i] >= RSI_HI and c[i] >= upper[i]:
                win = exit_px < entry
                over = (c[i] - upper[i]) / sdv[i]
                v4 = over >= 0.5
                trades.append((sym, "PUT", win, df.index[i], v4))
    
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

def monthly_breakdown(trades):
    """Group trades by month"""
    by_month = {}
    for sym, dr, win, ts, v4 in trades:
        month_key = ts.strftime("%Y-%m")
        if month_key not in by_month:
            by_month[month_key] = []
        by_month[month_key].append((sym, dr, win, ts, v4))
    return by_month

def main():
    print("=" * 72)
    print("H2 STRATEGY — 6-MONTH BACKTEST (EXACT v5.7.1 LOGIC)")
    print("RSI Wilder(14) 25/75 + BB(20,2.0,ddof=0) | 5m / 15m expiry")
    print("=" * 72)
    
    print("\nLoading data...")
    data = get_data()
    
    if len(data) == 0:
        print("ERROR: No data loaded")
        return
    
    print(f"\nRunning strategy on {len(data)} assets...")
    trades = run_strategy(data)
    
    if len(trades) == 0:
        print("ERROR: No trades generated")
        return
    
    # Overall stats
    n, w, wr, z = stats(trades)
    print(f"\n{'='*72}")
    print(f"OVERALL RESULTS (6 MONTHS)")
    print(f"{'='*72}")
    print(f"Total trades: {n}")
    print(f"Wins: {w} | Losses: {n - w}")
    print(f"Win rate: {wr:.2f}%")
    print(f"Z-score: {z:+.2f}", end="  ")
    print("SIGNIFICANT" if z > 1.96 else "not significant")
    print(f"Edge vs break-even ({BREAKEVEN:.2f}%): {wr - BREAKEVEN:+.2f} pp")
    
    # Robustness
    mid = n // 2
    if mid > 0:
        _, _, wr1, _ = stats(trades[:mid])
        _, _, wr2, _ = stats(trades[mid:])
        ok1 = "OK" if wr1 >= BREAKEVEN else "FAIL"
        ok2 = "OK" if wr2 >= BREAKEVEN else "FAIL"
        print(f"Robustness: 1st half {wr1:.2f}% {ok1} | 2nd half {wr2:.2f}% {ok2}")
    
    # V4 breakdown
    v4_trades = [t for t in trades if t[4]]
    non_v4 = [t for t in trades if not t[4]]
    if v4_trades:
        _, _, wr_v4, _ = stats(v4_trades)
        _, _, wr_nv4, _ = stats(non_v4)
        print(f"\nV4 signals: {len(v4_trades)} trades | {wr_v4:.2f}% WR")
        print(f"Non-V4: {len(non_v4)} trades | {wr_nv4:.2f}% WR")
    
    # Per asset
    print(f"\n{'='*72}")
    print(f"PER ASSET BREAKDOWN")
    print(f"{'='*72}")
    by_sym = {}
    for t in trades:
        sym = t[0]
        if sym not in by_sym:
            by_sym[sym] = []
        by_sym[sym].append(t)
    
    for sym in sorted(by_sym.keys()):
        sym_trades = by_sym[sym]
        sn, sw, swr, sz = stats(sym_trades)
        print(f"{sym.split('=')[0]:10} | n={sn:5d} | WR={swr:5.2f}% | z={sz:+.2f}")
    
    # Monthly breakdown
    print(f"\n{'='*72}")
    print(f"MONTHLY BREAKDOWN")
    print(f"{'='*72}")
    by_month = monthly_breakdown(trades)
    for month in sorted(by_month.keys()):
        month_trades = by_month[month]
        mn, mw, mwr, mz = stats(month_trades)
        status = "WINNER" if mwr >= BREAKEVEN and mz > 1.96 else \
                 "MARGINAL" if mwr >= BREAKEVEN else "LOSER"
        print(f"{month} | n={mn:5d} | WR={mwr:5.2f}% | z={mz:+.2f} | {status}")
    
    # Direction breakdown
    print(f"\n{'='*72}")
    print(f"DIRECTION BREAKDOWN")
    print(f"{'='*72}")
    calls = [t for t in trades if t[1] == "CALL"]
    puts = [t for t in trades if t[1] == "PUT"]
    if calls:
        cn, cw, cwr, cz = stats(calls)
        print(f"CALL | n={cn:5d} | WR={cwr:5.2f}% | z={cz:+.2f}")
    if puts:
        pn, pw, pwr, pz = stats(puts)
        print(f"PUT  | n={pn:5d} | WR={pwr:5.2f}% | z={pz:+.2f}")
    
    # Final verdict
    print(f"\n{'='*72}")
    print(f"FINAL VERDICT")
    print(f"{'='*72}")
    if wr >= 56 and z > 1.96:
        print("EXCELLENT: Strategy is profitable with statistical significance")
        print("Ready for live trading with confidence")
    elif wr >= BREAKEVEN:
        print("MARGINAL: Strategy is profitable but needs more data")
        print("Continue monitoring live performance")
    else:
        print("FAILING: Strategy is below break-even")
        print("Do not trade live — investigate parameters")

if __name__ == "__main__":
    main()
