#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2 Mean Reversion Backtest - Fixed & Optimized for yfinance
Logic: RSI(14) + BB(20, 2.0) on 5m, Expiry = 3 candles later (15m)
"""
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── Constants ───
SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN_WR = 52.6316

def get_data(symbol):
    try:
        # Fetch ONLY 5m data to avoid yfinance 1m limitations
        df = yf.download(symbol, period=f"{DAYS}d", interval="5m", progress=False)
        if df.empty:
            return None
        
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
            
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        print(f"  [{symbol}] Fetch Error: {e}")
        return None

def calculate_indicators(df):
    c = df['Close']
    
    # 1. RSI (Wilder's Smoothing)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # 2. Bollinger Bands (ddof=0 for population std, matching TradingView)
    df['BB_Mid'] = c.rolling(window=20).mean()
    df['BB_Std'] = c.rolling(window=20).std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + (2.0 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2.0 * df['BB_Std'])
    
    return df

def run_backtest():
    print("="*70)
    print("H2 MEAN REVERSION | RSI(14) + BB(20, 2.0) | 5m signal / 15m expiry")
    print(f"Period: last {DAYS}d | Break-even WR: {BREAKEVEN_WR}%")
    print("="*70)
    
    all_trades = []
    
    for sym in SYMBOLS:
        print(f"  Processing {sym}...")
        df = get_data(sym)
        if df is None or len(df) < 30:
            continue
            
        df = calculate_indicators(df)
        
        # Start after indicators are ready (20 periods), end 3 candles before EOF for expiry
        for i in range(20, len(df) - 3):
            row = df.iloc[i]
            
            if pd.isna(row['RSI']) or pd.isna(row['BB_Lower']):
                continue
                
            signal = None
            if row['RSI'] <= 25.0 and row['Close'] <= row['BB_Lower']:
                signal = 'CALL'
            elif row['RSI'] >= 75.0 and row['Close'] >= row['BB_Upper']:
                signal = 'PUT'
                
            if signal:
                entry_price = row['Close']
                # Exit is exactly 3 candles later (15 minutes)
                exit_row = df.iloc[i + 3]
                exit_price = exit_row['Close']
                
                if signal == 'CALL':
                    win = exit_price > entry_price
                else:
                    win = exit_price < entry_price
                    
                # Ignore exact ties (rare, but mathematically correct)
                if exit_price == entry_price:
                    continue
                    
                pnl = (STAKE * PAYOUT) if win else -STAKE
                
                all_trades.append({
                    'Symbol': sym,
                    'Entry_Time': row.name,
                    'Signal': signal,
                    'Win': win,
                    'PnL': pnl
                })
                
    return pd.DataFrame(all_trades)

def evaluate_results(trades_df):
    if trades_df.empty:
        print("\nNo trades generated. Check data or strategy logic.")
        return
        
    n = len(trades_df)
    wins = trades_df['Win'].sum()
    wr = (wins / n) * 100
    total_pnl = trades_df['PnL'].sum()
    
    # Robustness: Split in half
    mid = n // 2
    wr_1 = (trades_df.iloc[:mid]['Win'].sum() / mid) * 100 if mid > 0 else 0
    wr_2 = (trades_df.iloc[mid:]['Win'].sum() / (n - mid)) * 100 if (n - mid) > 0 else 0
    
    # Z-Score Test against Breakeven (p* = 1/1.9 ≈ 0.5263)
    p_star = 1 / (1 + 1/PAYOUT) # ~0.5263 for 90% payout
    se = np.sqrt(p_star * (1 - p_star) / n)
    z_score = ( (wr/100) - p_star ) / se if se > 0 else 0
    
    is_robust = (wr_1 >= BREAKEVEN_WR) and (wr_2 >= BREAKEVEN_WR) and (z_score >= 1.96)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Total Signals Found : {n}")
    print(f"Wins / Losses       : {int(wins)} / {int(n - wins)}")
    print(f"Win Rate (WR)       : {wr:.2f}%")
    print(f"Net PnL             : ${total_pnl:.2f}")
    print("-" * 70)
    print(f"Robustness Check (Split 50/50):")
    print(f"  First Half WR     : {wr_1:.2f}% {'✅' if wr_1 >= BREAKEVEN_WR else '❌'}")
    print(f"  Second Half WR    : {wr_2:.2f}% {'✅' if wr_2 >= BREAKEVEN_WR else '❌'}")
    print(f"  Z-Score (vs 52.63%): {z_score:.2f} {'✅' if z_score >= 1.96 else '❌'} (Needs >= 1.96)")
    print("-" * 70)
    if is_robust:
        print("FINAL VERDICT: ✅ STRATEGY IS STATISTICALLY ROBUST")
    else:
        print("FINAL VERDICT: ❌ STRATEGY IS NOT ROBUST (Failed split or Z-test)")
    print("="*70)

if __name__ == "__main__":
    df_trades = run_backtest()
    evaluate_results(df_trades)
