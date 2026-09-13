#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2 Mean Reversion Backtest - FINAL BULLETPROOF VERSION
Logic: RSI(14) + BB(20, 2.0) on 5m, Expiry = exactly 3 candles later (15m)
Fixes: Removes yfinance 1m limits and pandas datetime merge errors entirely.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───
ASSETS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
PERIOD = "60d"
SIGNAL_INTERVAL = "5m"
EXPIRY_CANDLES = 3  # 3 candles * 5m = 15 minutes

RSI_PERIOD = 14
BB_PERIOD = 20
BB_MULT = 2.0
RSI_OVERSOLD = 25.0
RSI_OVERBOUGHT = 75.0

STAKE = 6.0
PAYOUT = 0.90
BREAK_EVEN_WR = 1.0 / (1.0 + PAYOUT) * 100.0  # 52.6316%

# ─── DATA LAYER ───
def get_data(ticker: str) -> pd.DataFrame:
    try:
        # yf.Ticker.history is the most stable method, avoids MultiIndex issues
        df = yf.Ticker(ticker).history(period=PERIOD, interval=SIGNAL_INTERVAL)
        if df.empty:
            return pd.DataFrame()
        
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return df
    except Exception as e:
        print(f"  [{ticker}] Fetch Error: {e}")
        return pd.DataFrame()

# ─── INDICATORS (Exact TradingView Math) ───
def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    denom = avg_gain + avg_loss
    rsi = 100.0 * avg_gain / denom
    return rsi.where(denom > 0, 50.0)

def bollinger(close: pd.Series, period: int = BB_PERIOD, mult: float = BB_MULT):
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0) # ddof=0 matches TradingView
    return ma, ma + mult * sd, ma - mult * sd

# ─── BACKTEST ENGINE ───
def backtest_asset(ticker: str) -> pd.DataFrame:
    df = get_data(ticker)
    if df.empty or len(df) < 30:
        print(f"  [{ticker}] Skipped (insufficient data)")
        return pd.DataFrame()

    close = df['Close'].astype(float)
    df['RSI'] = wilder_rsi(close, RSI_PERIOD)
    df['BB_MA'], df['BB_Upper'], df['BB_Lower'] = bollinger(close, BB_PERIOD, BB_MULT)
    
    # Drop rows where indicators are NaN (first 20 periods)
    df = df.dropna(subset=['RSI', 'BB_Upper', 'BB_Lower', 'Close'])
    
    # Signal Logic
    call_cond = (df['RSI'] <= RSI_OVERSOLD) & (df['Close'] <= df['BB_Lower'])
    put_cond = (df['RSI'] >= RSI_OVERBOUGHT) & (df['Close'] >= df['BB_Upper'])
    
    signals = df[call_cond | put_cond].copy()
    if signals.empty:
        print(f"  [{ticker}] 0 signals")
        return pd.DataFrame()

    signals['Direction'] = np.where(call_cond.reindex(signals.index, fill_value=False), 'CALL', 'PUT')
    
    trades_list = []
    for idx, row in signals.iterrows():
        # Find the index position of this signal in the main dataframe
        pos = df.index.get_loc(idx)
        exit_pos = pos + EXPIRY_CANDLES
        
        if exit_pos >= len(df):
            continue # Not enough data to resolve expiry
            
        entry_price = row['Close']
        exit_price = df.iloc[exit_pos]['Close']
        
        if row['Direction'] == 'CALL':
            won = exit_price > entry_price
        else:
            won = exit_price < entry_price
            
        if exit_price == entry_price:
            continue # Exact tie (rare, treated as void)
            
        pnl = (STAKE * PAYOUT) if won else -STAKE
        
        trades_list.append({
            'Asset': ticker,
            'SignalTime': idx,
            'Direction': row['Direction'],
            'Entry': entry_price,
            'Exit': exit_price,
            'RSI': row['RSI'],
            'Won': won,
            'PnL': pnl
        })
        
    print(f"  [{ticker}] Trades: {len(trades_list)}")
    return pd.DataFrame(trades_list)

# ─── METRICS & REPORTING ───
def win_rate(df: pd.DataFrame) -> float:
    return float(df["Won"].mean() * 100.0) if len(df) else 0.0

def report(trades: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("H2 MEAN REVERSION - RESULTS")
    print("=" * 66)

    if trades.empty:
        print("No trades generated. Nothing to evaluate.")
        return

    trades = trades.sort_values("SignalTime").reset_index(drop=True)
    n = len(trades)
    wins = int(trades["Won"].sum())
    losses = n - wins
    wr = win_rate(trades)
    net = float(trades["PnL"].sum())

    print(f"Window            : {trades['SignalTime'].min().strftime('%Y-%m-%d')} to {trades['SignalTime'].max().strftime('%Y-%m-%d')}")
    print(f"Total trades      : {n}")
    print(f"Wins / Losses     : {wins} / {losses}")
    print(f"Win rate          : {wr:.2f}%")
    print(f"Break-even needed : {BREAK_EVEN_WR:.2f}%   (payout {PAYOUT:.0%})")
    print(f"Edge vs break-even: {wr - BREAK_EVEN_WR:+.2f} pp")
    print(f"Net P&L           : ${net:+.2f}   (turnover ${n * STAKE:,.2f})")

    # Statistical significance (Z-Score)
    p_hat = wr / 100.0
    p_star = 1.0 / (1.0 + PAYOUT)
    se_h0 = np.sqrt(p_star * (1.0 - p_star) / n)
    z = (p_hat - p_star) / se_h0 if se_h0 > 0 else 0
    
    print(f"\nz vs break-even   : {z:+.2f}", end="  ")
    print("-> SIGNIFICANT EDGE" if z > 1.96 else "-> NOT statistically distinguishable from noise")

    # Robustness Check
    mid = n // 2
    h1, h2 = trades.iloc[:mid], trades.iloc[mid:]
    wr1, wr2 = win_rate(h1), win_rate(h2)

    print("\n" + "-" * 66)
    print("ROBUSTNESS CHECK (chronological 50/50 split)")
    print("-" * 66)
    print(f"First half  : {len(h1):>4} trades | WR {wr1:.2f}% {'✅' if wr1 >= BREAK_EVEN_WR else '❌'}")
    print(f"Second half : {len(h2):>4} trades | WR {wr2:.2f}% {'✅' if wr2 >= BREAK_EVEN_WR else '❌'}")
    
    robust = (wr1 >= BREAK_EVEN_WR) and (wr2 >= BREAK_EVEN_WR)
    print(f"\nVERDICT     : {'✅ ROBUST' if robust else '❌ NOT ROBUST'}")
    
    if n < 300:
        print(f"\n⚠️ WARNING: n={n} < 300. Treat as preliminary.")

    # Breakdown
    print("\n" + "-" * 66)
    print("BREAKDOWN BY ASSET")
    print("-" * 66)
    by_asset = trades.groupby("Asset").agg(Trades=("Won", "size"), Wins=("Won", "sum"), PnL=("PnL", "sum"))
    by_asset["WR%"] = (by_asset["Wins"] / by_asset["Trades"] * 100).round(2)
    print(by_asset.to_string())

# ─── MAIN ───
def main() -> None:
    print("=" * 66)
    print("H2 MEAN REVERSION | RSI(14) + BB(20, 2.0) | 5m signal / 15m expiry")
    print(f"Period: last {PERIOD} | Break-even WR: {BREAK_EVEN_WR:.2f}%")
    print("=" * 66)

    frames = []
    for ticker in ASSETS:
        try:
            frames.append(backtest_asset(ticker))
        except Exception as exc:
            print(f"  [{ticker}] ERROR: {type(exc).__name__}: {exc}")

    frames = [f for f in frames if not f.empty]
    all_trades = pd.concat(frames, ignore_index=True).sort_values("SignalTime").reset_index(drop=True) if frames else pd.DataFrame()

    report(all_trades)
    
    if not all_trades.empty:
        all_trades.to_csv("h2_trades.csv", index=False)
        print(f"\n✅ Trade log written to h2_trades.csv ({len(all_trades)} rows)")

if __name__ == "__main__":
    main()
