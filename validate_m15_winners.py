import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION FOR M15 VALIDATION
# ==========================================
SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
TIMEFRAME = "15m"
PERIOD_LIMIT = "60d"
BREAK_EVEN_WR = 0.5263

# The Top Winners from Random Search (Focusing on BB strategies)
WINNING_STRATEGIES = [
    {'name': 'BB_15_Std2.3', 'type': 'BB_Bounce', 'params': {'period': 15, 'std': 2.3}},
    {'name': 'BB_14_Std2.4', 'type': 'BB_Bounce', 'params': {'period': 14, 'std': 2.4}},
    {'name': 'BB_13_Std2.2', 'type': 'BB_Bounce', 'params': {'period': 13, 'std': 2.2}},
    {'name': 'BB_12_Std2.4', 'type': 'BB_Bounce', 'params': {'period': 12, 'std': 2.4}},
    {'name': 'RSI_17_OB79_OS19', 'type': 'RSI_MeanRev', 'params': {'period': 17, 'ob': 79, 'os': 19}} # The lone survivor
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

def calc_rsi(close, period):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def validate_strategy(strategy_cfg, symbol_df):
    trades = []
    n = len(symbol_df)
    if n < 50: return trades
    
    c_series = symbol_df['Close']
    
    # Trend Filter (EMA200) - Critical for stability
    ema200 = calc_ema(c_series, 200).values
    
    stype = strategy_cfg['type']
    params = strategy_cfg['params']
    
    # Calculate Indicators
    if stype == 'BB_Bounce':
        _, ub, lb = calc_bb(c_series, params['period'], params['std'])
        ub_v, lb_v = ub.values, lb.values
        min_idx = params['period'] + 1
        
    elif stype == 'RSI_MeanRev':
        ind_vals = calc_rsi(c_series, params['period']).values
        min_idx = params['period'] + 1
    else:
        return trades

    mid_point = n // 2
    
    for i in range(min_idx, n-1):
        curr_close = c_series.iloc[i]
        next_close = c_series.iloc[i+1]
        
        if curr_close == next_close: continue
        
        # Trend Filter Logic
        trend_up = curr_close > ema200[i]
        trend_down = curr_close < ema200[i]
        
        dir_sig = 0
        
        if stype == 'BB_Bounce':
            u_val, l_val = ub_v[i], lb_v[i]
            if np.isnan(u_val): continue
            # Buy when touching Lower Band AND Price is ABOVE EMA200 (Uptrend Pullback)
            # Sell when touching Upper Band AND Price is BELOW EMA200 (Downtrend Rally)
            if curr_close <= l_val and trend_up: dir_sig = 1
            elif curr_close >= u_val and trend_down: dir_sig = -1
            
        elif stype == 'RSI_MeanRev':
            val = ind_vals[i]
            if np.isnan(val): continue
            # Similar logic: Only buy oversold in uptrend, sell overbought in downtrend
            if val < params['os'] and trend_up: dir_sig = 1
            elif val > params['ob'] and trend_down: dir_sig = -1
            
        if dir_sig != 0:
            win = (next_close > curr_close) if dir_sig == 1 else (next_close < curr_close)
            half = 1 if i < mid_point else 2
            trades.append({'half': half, 'win': win})
            
    return trades

def main():
    print("="*60)
    print("FINAL VALIDATION: SPLIT-SAMPLE ROBUSTNESS TEST (M15)")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT}")
    print("="*60)
    
    symbol_data = {}
    for sym in SYMBOLS:
        print(f"Fetching {sym}...")
        df = get_data(sym)
        if df is not None and len(df) > 100:
            symbol_data[sym] = df
    
    if not symbol_data:
        print("No data found."); return

    final_report = []

    for strat in WINNING_STRATEGIES:
        print(f"\n>>> Validating: {strat['name']}")
        all_trades_h1 = [] 
        all_trades_h2 = [] 
        
        for sym, df in symbol_data.items():
            trades = validate_strategy(strat, df)
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
        
        # Strict Rule: Must be profitable in BOTH halves
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
    print("SUMMARY OF M15 VALIDATION")
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
