import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION FOR MASSIVE SAMPLE SIZE TEST
# ==========================================
# Expanded Symbol List to reach >3000 trades
SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", # Majors
    "USDCAD=X", "USDCHF=X",                                     # Majors
    "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X", # Euro Crosses
    "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",             # Pound Crosses
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",                         # Aussie Crosses
    "CADCHF=X", "CADJPY=X", "CHFJPY=X"                          # Minor Crosses
]

TIMEFRAME = "15m"
PERIOD_LIMIT = "60d" # Max available for intraday in free tier
BREAK_EVEN_WR = 0.5263
TARGET_TRADES = 3000 # Goal

# The Top 4 Winning Strategies from previous validation
WINNING_STRATEGIES = [
    {'name': 'BB_15_Std2.3', 'type': 'BB_Bounce', 'params': {'period': 15, 'std': 2.3}},
    {'name': 'BB_14_Std2.4', 'type': 'BB_Bounce', 'params': {'period': 14, 'std': 2.4}},
    {'name': 'BB_13_Std2.2', 'type': 'BB_Bounce', 'params': {'period': 13, 'std': 2.2}},
    {'name': 'BB_12_Std2.4', 'type': 'BB_Bounce', 'params': {'period': 12, 'std': 2.4}}
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

def calc_bb(close, period, std_mult):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = sma + (std * std_mult)
    lower = sma - (std * std_mult)
    return sma, upper, lower

def validate_strategy(strategy_cfg, symbol_df):
    trades = []
    n = len(symbol_df)
    if n < 50: return trades
    
    c_series = symbol_df['Close']
    
    # Trend Filter (EMA200)
    ema200 = calc_ema(c_series, 200).values
    
    stype = strategy_cfg['type']
    params = strategy_cfg['params']
    
    if stype == 'BB_Bounce':
        _, ub, lb = calc_bb(c_series, params['period'], params['std'])
        ub_v, lb_v = ub.values, lb.values
        min_idx = params['period'] + 1
        
        mid_point = n // 2
        
        for i in range(min_idx, n-1):
            curr_close = c_series.iloc[i]
            next_close = c_series.iloc[i+1]
            
            if curr_close == next_close: continue
            
            trend_up = curr_close > ema200[i]
            trend_down = curr_close < ema200[i]
            
            dir_sig = 0
            u_val, l_val = ub_v[i], lb_v[i]
            if np.isnan(u_val): continue
            
            # Buy: Touch Lower Band AND Uptrend
            if curr_close <= l_val and trend_up: dir_sig = 1
            # Sell: Touch Upper Band AND Downtrend
            elif curr_close >= u_val and trend_down: dir_sig = -1
                
            if dir_sig != 0:
                win = (next_close > curr_close) if dir_sig == 1 else (next_close < curr_close)
                half = 1 if i < mid_point else 2
                trades.append({'half': half, 'win': win})
                
    return trades

def main():
    print("="*60)
    print("MASSIVE SAMPLE SIZE VALIDATION (>3000 Trades Target)")
    print(f"Timeframe: {TIMEFRAME} | Symbols Count: {len(SYMBOLS)}")
    print("="*60)
    
    # Fetch Data for ALL symbols
    symbol_data = {}
    successful_fetches = 0
    for sym in SYMBOLS:
        print(f"Fetching {sym}...", end=" ")
        df = get_data(sym)
        if df is not None and len(df) > 100:
            symbol_data[sym] = df
            successful_fetches += 1
            print("OK")
        else:
            print("SKIP/EMPTY")
    
    print(f"\nSuccessfully loaded data for {successful_fetches} out of {len(SYMBOLS)} symbols.")
    
    if successful_fetches < 10:
        print("⚠️ Warning: Too few symbols loaded. Sample size might be insufficient.")

    final_report = []

    for strat in WINNING_STRATEGIES:
        print(f"\n>>> Validating: {strat['name']} across all symbols...")
        all_trades_h1 = [] 
        all_trades_h2 = [] 
        
        total_symbols_processed = 0
        
        for sym, df in symbol_data.items():
            trades = validate_strategy(strat, df)
            total_symbols_processed += 1
            for t in trades:
                if t['half'] == 1:
                    all_trades_h1.append(t['win'])
                else:
                    all_trades_h2.append(t['win'])
        
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
        
        p_null = BREAK_EVEN_WR
        std_dev = np.sqrt(total_count * p_null * (1-p_null))
        z_score = ((sum(all_trades_h1) + sum(all_trades_h2)) - (total_count * p_null)) / std_dev if std_dev > 0 else 0
        
        is_robust = False
        verdict = "FAILING"
        
        if wr_h1 >= BREAK_EVEN_WR and wr_h2 >= BREAK_EVEN_WR:
            is_robust = True
            verdict = "ROBUST ✅"
        else:
            verdict = "FRAGILE ❌"
            
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

    print("\n" + "="*60)
    print("SUMMARY OF MASSIVE SAMPLE VALIDATION")
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
        print("\n🎉 SUCCESS! Verified strategies exist even on M15 with Trend Filtering.")
    else:
        print("\n⚠️ WARNING: High Z-scores were likely due to specific market regime in last 60 days.")

if __name__ == "__main__":
    main()
