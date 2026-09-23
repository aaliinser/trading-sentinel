import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"] # FIXED TYPO HERE
TIMEFRAME = "1h"
PERIOD_LIMIT = "2y"
STAKE = 6.0
PAYOUT_RATE = 0.90
BREAK_EVEN_WR = 0.5263

# Expanded Parameter Grids for Deeper Search
PARAM_GRIDS = {
    'RSI': {'period': [7, 14, 21, 28], 'overbought': [60, 70, 80], 'oversold': [20, 30, 40]},
    'BB': {'period': [10, 20, 30], 'std_dev': [1.5, 1.8, 2.0, 2.5]}, # Added tighter bands
    'EMA_Cross': {'fast': [5, 9, 12, 20], 'slow': [21, 26, 50, 100]}, # Wider range
    'Stoch': {'k_period': [9, 14, 21, 30], 'd_period': [3, 5, 8], 'ob_level': [70, 80, 90], 'os_level': [10, 20, 30]}
}

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
# FAST INDICATOR CALCULATORS
# ==========================================

def calc_rsi(close, period):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_bb(close, period, std_mult):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = sma + (std * std_mult)
    lower = sma - (std * std_mult)
    return sma, upper, lower

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_stoch(high, low, close, k_period, d_period):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k.fillna(50), stoch_d.fillna(50)

# ==========================================
# STRATEGY GENERATION ENGINE WITH TREND FILTER
# ==========================================

class StrategyGeneratorV2:
    def __init__(self):
        self.strategies = []
        self._generate_all_combinations()

    def _generate_all_combinations(self):
        # Generate combos... (Same logic as before but expanded grids)
        for p, ob, os in product(PARAM_GRIDS['RSI']['period'], PARAM_GRIDS['RSI']['overbought'], PARAM_GRIDS['RSI']['oversold']):
            self.strategies.append({'type': 'RSI_MeanRev', 'params': {'period': p, 'ob': ob, 'os': os}, 'name': f"RSI_{p}_OB{ob}_OS{os}"})

        for p, s in product(PARAM_GRIDS['BB']['period'], PARAM_GRIDS['BB']['std_dev']):
            self.strategies.append({'type': 'BB_Bounce', 'params': {'period': p, 'std': s}, 'name': f"BB_{p}_Std{s}"})

        for f, sl in product(PARAM_GRIDS['EMA_Cross']['fast'], PARAM_GRIDS['EMA_Cross']['slow']):
            if f >= sl: continue
            self.strategies.append({'type': 'EMA_Cross', 'params': {'fast': f, 'slow': sl}, 'name': f"EMACross_F{f}_S{sl}"})

        for kp, dp, ob, os in product(PARAM_GRIDS['Stoch']['k_period'], PARAM_GRIDS['Stoch']['d_period'], 
                                      PARAM_GRIDS['Stoch']['ob_level'], PARAM_GRIDS['Stoch']['os_level']):
            self.strategies.append({'type': 'Stoch_Extreme', 'params': {'k': kp, 'd': dp, 'ob': ob, 'os': os}, 'name': f"Stoch_K{kp}_D{dp}_OB{ob}_OS{os}"})

        print(f"Generated {len(self.strategies)} unique strategy configurations.")

    def run_backtest_for_symbol(self, symbol_df, strategy_config):
        trades = []
        n = len(symbol_df)
        if n < 200: return trades # Need enough data for EMA200 filter
        
        c_series = symbol_df['Close']
        h_series = symbol_df['High']
        l_series = symbol_df['Low']
        
        # Pre-calculate Global Trend Filter (EMA200)
        ema200 = calc_ema(c_series, 200).values
        
        stype = strategy_config['type']
        params = strategy_config['params']
        
        # Calculate specific indicators
        if stype == 'RSI_MeanRev':
            ind_vals = calc_rsi(c_series, params['period']).values
            min_idx = params['period'] + 1
            
        elif stype == 'BB_Bounce':
            _, ub, lb = calc_bb(c_series, params['period'], params['std'])
            ub_v, lb_v = ub.values, lb.values
            ind_vals = None # Handled differently
            min_idx = params['period'] + 1
            
        elif stype == 'EMA_Cross':
            ema_f = calc_ema(c_series, params['fast']).values
            ema_s = calc_ema(c_series, params['slow']).values
            min_idx = max(params['fast'], params['slow']) + 1
            
        elif stype == 'Stoch_Extreme':
            sk, sd = calc_stoch(h_series, l_series, c_series, params['k'], params['d'])
            sk_v, sd_v = sk.values, sd.values
            min_idx = max(params['k'], params['d']) + 1
        else:
            return trades

        # Main Loop
        for i in range(min_idx, n-1):
            curr_close = c_series.iloc[i]
            next_close = c_series.iloc[i+1]
            
            # SKIP if tie
            if curr_close == next_close: continue
            
            # --- TREND FILTER LOGIC ---
            # Only take CALL if Price > EMA200
            # Only take PUT if Price < EMA200
            trend_up = curr_close > ema200[i]
            trend_down = curr_close < ema200[i]
            
            dir_sig = 0
            
            if stype == 'RSI_MeanRev':
                val = ind_vals[i]
                if np.isnan(val): continue
                if val < params['os']: dir_sig = 1 # Buy Signal
                elif val > params['ob']: dir_sig = -1 # Sell Signal
                
            elif stype == 'BB_Bounce':
                u_val, l_val = ub_v[i], lb_v[i]
                if np.isnan(u_val): continue
                if curr_close <= l_val: dir_sig = 1
                elif curr_close >= u_val: dir_sig = -1
                
            elif stype == 'EMA_Cross':
                pf, ps = ema_f[i-1], ema_s[i-1]
                cf, cs = ema_f[i], ema_s[i]
                if np.isnan(pf): continue
                if pf <= ps and cf > cs: dir_sig = 1
                elif pf >= ps and cf < cs: dir_sig = -1
                
            elif stype == 'Stoch_Extreme':
                k_val, d_val = sk_v[i], sd_v[i]
                pk, pd = sk_v[i-1], sd_v[i-1]
                if np.isnan(k_val): continue
                if pk <= pd and k_val > d_val and k_val < params['os']: dir_sig = 1
                elif pk >= pd and k_val < d_val and k_val > params['ob']: dir_sig = -1
            
            # APPLY TREND FILTER
            if dir_sig == 1 and not trend_up: dir_sig = 0 # Reject Counter-Trend Buy
            if dir_sig == -1 and not trend_down: dir_sig = 0 # Reject Counter-Trend Sell
            
            if dir_sig != 0:
                win = (next_close > curr_close) if dir_sig == 1 else (next_close < curr_close)
                trades.append(win)
                    
        return trades

def main():
    print("="*60)
    print("AUTO-STRATEGY GENERATOR V2 (With Trend Filter)")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT}")
    print("="*60)
    
    gen = StrategyGeneratorV2()
    total_strats = len(gen.strategies)
    print(f"Total Configurations to Test: {total_strats}")
    
    symbol_data = {}
    for sym in SYMBOLS:
        print(f"Fetching {sym}...")
        df = get_data(sym)
        if df is not None and len(df) > 200:
            symbol_data[sym] = df
    
    if not symbol_data:
        print("No data found."); return

    results_pool = []
    count_processed = 0
    
    for strat_cfg in gen.strategies:
        all_trades_combined = []
        for sym, df in symbol_data.items():
            trades = gen.run_backtest_for_symbol(df, strat_cfg)
            all_trades_combined.extend(trades)
            
        if len(all_trades_combined) < 50: continue
            
        wins = sum(all_trades_combined)
        total = len(all_trades_combined)
        wr = wins / total
        
        p_null = BREAK_EVEN_WR
        std_dev = np.sqrt(total * p_null * (1-p_null))
        z_score = (wins - (total * p_null)) / std_dev if std_dev > 0 else 0
        
        results_pool.append({
            'name': strat_cfg['name'],
            'trades': total,
            'wr': wr,
            'z': z_score
        })
        
        count_processed += 1
        if count_processed % 50 == 0:
            print(f"Processed {count_processed}/{total_strats}...")

    results_pool.sort(key=lambda x: x['z'], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 20 PERFORMING STRATEGIES (Filtered by Trend)")
    print("="*60)
    print(f"{'Rank':<5} | {'Strategy Name':<25} | {'Trades':>6} | {'WR%':>6} | {'Z-Score':>7}")
    print("-"*70)
    
    top_n = min(20, len(results_pool))
    for i in range(top_n):
        res = results_pool[i]
        marker = ""
        if res['z'] > 1.96: marker = "  SIGNIFICANT!"
        elif res['z'] > 0: marker = " ✅ Positive Edge"
        else: marker = " ❌ Negative Edge"
        
        print(f"{i+1:<5} | {res['name']:<25} | {res['trades']:>6} | {res['wr']*100:>6.2f} | {res['z']:>7.2f}{marker}")

if __name__ == "__main__":
    main()
