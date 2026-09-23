import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
TIMEFRAME = "1h"
PERIOD_LIMIT = "2y"
STAKE = 6.0
PAYOUT_RATE = 0.90
BREAK_EVEN_WR = 0.5263

# The Top 3 Winning Strategies from previous run
WINNING_STRATEGIES = [
    {'name': 'Stoch_K30_D8_OB90_OS20', 'type': 'Stoch_Extreme', 'params': {'k': 30, 'd': 8, 'ob': 90, 'os': 20}},
    {'name': 'Stoch_K14_D5_OB90_OS10', 'type': 'Stoch_Extreme', 'params': {'k': 14, 'd': 5, 'ob': 90, 'os': 10}},
    {'name': 'Stoch_K9_D5_OB90_OS10',   'type': 'Stoch_Extreme', 'params': {'k': 9,  'd': 5, 'ob': 90, 'os': 10}}
]

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def get_data(symbol, period=PERIOD_LIMIT, interval=TIMEFRAME):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty: return None
        df = flatten_columns(df)
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols): return None
        df = df.dropna(subset=['Close'])
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        return df[['Open', 'High', 'Low', 'Close']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ==========================================
# INDICATORS & LOGIC
# ==========================================

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_stoch(high, low, close, k_period, d_period):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k.fillna(50), stoch_d.fillna(50)

def validate_strategy(strategy_cfg, symbol_df):
    """
    Runs the strategy on a single symbol and returns trades split by time.
    Returns: list of dicts {'half': 1 or 2, 'win': bool}
    """
    trades = []
    n = len(symbol_df)
    if n < 200: return trades
    
    c_series = symbol_df['Close']
    h_series = symbol_df['High']
    l_series = symbol_df['Low']
    
    # Pre-calculate Trend Filter (EMA200)
    ema200 = calc_ema(c_series, 200).values
    
    # Calculate Stochastic
    params = strategy_cfg['params']
    sk, sd = calc_stoch(h_series, l_series, c_series, params['k'], params['d'])
    sk_v, sd_v = sk.values, sd.values
    
    min_idx = max(params['k'], params['d']) + 1
    
    # Determine midpoint for Split-Sample
    mid_point = n // 2
    
    for i in range(min_idx, n-1):
        curr_close = c_series.iloc[i]
        next_close = c_series.iloc[i+1]
        
        # Skip ties
        if curr_close == next_close: continue
        
        # Trend Filter
        trend_up = curr_close > ema200[i]
        trend_down = curr_close < ema200[i]
        
        dir_sig = 0
        k_val, d_val = sk_v[i], sd_v[i]
        pk, pd = sk_v[i-1], sd_v[i-1]
        
        if np.isnan(k_val) or np.isnan(d_val): continue
        
        # Bullish Crossover in OS Zone
        if pk <= pd and k_val > d_val and k_val < params['os']:
            dir_sig = 1
        # Bearish Crossover in OB Zone
        elif pk >= pd and k_val < d_val and k_val > params['ob']:
            dir_sig = -1
            
        # Apply Trend Filter
        if dir_sig == 1 and not trend_up: dir_sig = 0
        if dir_sig == -1 and not trend_down: dir_sig = 0
        
        if dir_sig != 0:
            win = (next_close > curr_close) if dir_sig == 1 else (next_close < curr_close)
            
            # Assign to Half 1 or Half 2 based on index
            half = 1 if i < mid_point else 2
            
            trades.append({'half': half, 'win': win})
            
    return trades

def main():
    print("="*60)
    print("FINAL VALIDATION: SPLIT-SAMPLE ROBUSTNESS TEST")
    print(f"Strategies Tested: {len(WINNING_STRATEGIES)}")
    print("="*60)
    
    # Fetch Data Once
    symbol_data = {}
    for sym in SYMBOLS:
        print(f"Fetching {sym}...")
        df = get_data(sym)
        if df is not None and len(df) > 200:
            symbol_data[sym] = df
    
    if not symbol_data:
        print("No data found."); return

    final_report = []

    for strat in WINNING_STRATEGIES:
        print(f"\n>>> Validating: {strat['name']}")
        all_trades_h1 = [] # First half of timeline
        all_trades_h2 = [] # Second half of timeline
        
        for sym, df in symbol_data.items():
            trades = validate_strategy(strat, df)
            for t in trades:
                if t['half'] == 1:
                    all_trades_h1.append(t['win'])
                else:
                    all_trades_h2.append(t['win'])
        
        # Calculate Stats for each half
        def get_stats(trade_list):
            if not trade_list: return 0, 0
            total = len(trade_list)
            wins = sum(trade_list)
            wr = wins / total
            return wr, total
            
        wr_h1, count_h1 = get_stats(all_trades_h1)
        wr_h2, count_h2 = get_stats(all_trades_h2)
        total_count = count_h1 + count_h2
        overall_wr = (sum(all_trades_h1) + sum(all_trades_h2)) / total_count if total_count > 0 else 0
        
        # Z-Score Calculation for Overall
        p_null = BREAK_EVEN_WR
        std_dev = np.sqrt(total_count * p_null * (1-p_null))
        z_score = ((sum(all_trades_h1) + sum(all_trades_h2)) - (total_count * p_null)) / std_dev if std_dev > 0 else 0
        
        # Robustness Check
        is_robust = False
        verdict = "FAILING"
        
        if wr_h1 >= BREAK_EVEN_WR and wr_h2 >= BREAK_EVEN_WR:
            is_robust = True
            verdict = "ROBUST ✅"
        else:
            verdict = "FRAGILE ❌"
            
        # Final Verdict Logic
        final_verdict = "INVALID"
        if is_robust and z_score > 1.96:
            final_verdict = "EXCELLENT 🏆"
        elif is_robust and z_score > 0:
            final_verdict = "VIABLE ⚠️"
        else:
            final_verdict = "REJECTED 🛑"

        report_entry = {
            'name': strat['name'],
            'total_trades': total_count,
            'overall_wr': overall_wr,
            'z_score': z_score,
            'wr_half1': wr_h1,
            'count_half1': count_h1,
            'wr_half2': wr_h2,
            'count_half2': count_h2,
            'robust': is_robust,
            'verdict': final_verdict
        }
        
        final_report.append(report_entry)
        
        print(f"   Total Trades: {total_count}")
        print(f"   Overall WR:   {overall_wr*100:.2f}% | Z-Score: {z_score:.2f}")
        print(f"   Half 1 (Old): WR {wr_h1*100:.2f}% ({count_h1} trades)")
        print(f"   Half 2 (New): WR {wr_h2*100:.2f}% ({count_h2} trades)")
        print(f"   Status:       {verdict}")
        print(f"   FINAL VERDICT:{final_verdict}")

    # Summary Table
    print("\n" + "="*60)
    print("SUMMARY OF VALIDATION")
    print("="*60)
    print(f"{'Strategy':<25} | {'Trades':>6} | {'WR%':>6} | {'Z':>5} | {'H1 WR':>6} | {'H2 WR':>6} | {'Verdict':>10}")
    print("-"*85)
    
    winners_found = False
    for res in final_report:
        marker = ""
        if res['verdict'] == "EXCELLENT 🏆": 
            marker = "🟢 PASS"; winners_found = True
        elif res['verdict'] == "VIABLE ⚠️": 
            marker = "🟡 MAYBE"
        else: 
            marker = "🔴 FAIL"
            
        print(f"{res['name']:<25} | {res['total_trades']:>6} | {res['overall_wr']*100:>6.2f} | {res['z_score']:>5.2f} | {res['wr_half1']*100:>6.2f} | {res['wr_half2']*100:>6.2f} | {marker}")

    if winners_found:
        print("\n🎉 CONGRATULATIONS! You have statistically verified strategies.")
    else:
        print("\n⚠️ WARNING: No strategies passed the strict robustness test.")
        print("The initial high Win Rates were likely due to specific market conditions (Overfitting).")

if __name__ == "__main__":
    main()
