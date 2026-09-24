import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import random
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION FOR M15 RANDOM SEARCH
# ==========================================
SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
TIMEFRAME = "15m"
PERIOD_LIMIT = "60d"    # Max data available for intraday
STAKE = 6.0
PAYOUT_RATE = 0.90
BREAK_EVEN_WR = 0.5263
MIN_TRADES_REQUIRED = 100 # Strict filter to avoid noise
TARGET_STRATEGIES_COUNT = 1000 # Number of random configs to test

# Seed for reproducibility if needed
RANDOM_SEED = 42

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
# INDICATOR CALCULATORS (Vectorized)
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

def calc_macd(close, fast, slow, signal):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.values, signal_line.values, hist.values

# ==========================================
# RANDOM STRATEGY GENERATOR
# ==========================================

class RandomStrategyGenerator:
    def __init__(self, count):
        self.strategies = []
        self._generate_random_configs(count)

    def _generate_random_configs(self, count):
        """Generates N random strategy configurations."""
        types = ['RSI_MeanRev', 'BB_Bounce', 'EMA_Cross', 'Stoch_Extreme', 'MACD_Hist']
        
        for i in range(count):
            strat_type = random.choice(types)
            
            if strat_type == 'RSI_MeanRev':
                p = random.randint(5, 30)
                ob = random.randint(60, 85)
                os = random.randint(15, 40)
                self.strategies.append({
                    'id': f"R{i}", 'type': strat_type, 
                    'params': {'period': p, 'ob': ob, 'os': os},
                    'name': f"RSI_{p}_OB{ob}_OS{os}"
                })
                
            elif strat_type == 'BB_Bounce':
                p = random.randint(10, 40)
                s = round(random.uniform(1.5, 3.0), 1)
                self.strategies.append({
                    'id': f"B{i}", 'type': strat_type, 
                    'params': {'period': p, 'std': s},
                    'name': f"BB_{p}_Std{s}"
                })
                
            elif strat_type == 'EMA_Cross':
                f = random.randint(3, 15)
                sl = random.randint(20, 60)
                if f >= sl: continue # Invalid
                self.strategies.append({
                    'id': f"E{i}", 'type': strat_type, 
                    'params': {'fast': f, 'slow': sl},
                    'name': f"EMACross_F{f}_S{sl}"
                })
                
            elif strat_type == 'Stoch_Extreme':
                kp = random.randint(5, 30)
                dp = random.randint(3, 10)
                ob = random.randint(75, 95)
                os = random.randint(5, 25)
                self.strategies.append({
                    'id': f"S{i}", 'type': strat_type, 
                    'params': {'k': kp, 'd': dp, 'ob': ob, 'os': os},
                    'name': f"Stoch_K{kp}_D{dp}_OB{ob}_OS{os}"
                })
                
            elif strat_type == 'MACD_Hist':
                f = random.randint(8, 15)
                sl = random.randint(20, 30)
                sig = random.randint(7, 12)
                thresh = round(random.uniform(0.0001, 0.001), 4) # Small threshold for Hist crossover
                self.strategies.append({
                    'id': f"M{i}", 'type': strat_type, 
                    'params': {'fast': f, 'slow': sl, 'signal': sig, 'thresh': thresh},
                    'name': f"MACD_F{f}_S{sl}_Sig{sig}"
                })

        print(f"Generated {len(self.strategies)} unique random strategy configurations.")

    def run_backtest_for_symbol(self, symbol_df, strategy_config):
        trades = []
        n = len(symbol_df)
        if n < 50: return trades
        
        c_series = symbol_df['Close']
        h_series = symbol_df['High']
        l_series = symbol_df['Low']
        
        # Pre-calculate Trend Filter (EMA200) - Optional but recommended
        ema200 = calc_ema(c_series, 200).values
        
        stype = strategy_config['type']
        params = strategy_config['params']
        
        # Calculate Indicators
        if stype == 'RSI_MeanRev':
            ind_vals = calc_rsi(c_series, params['period']).values
            min_idx = params['period'] + 1
            
        elif stype == 'BB_Bounce':
            _, ub, lb = calc_bb(c_series, params['period'], params['std'])
            ub_v, lb_v = ub.values, lb.values
            ind_vals = None
            min_idx = params['period'] + 1
            
        elif stype == 'EMA_Cross':
            ema_f = calc_ema(c_series, params['fast']).values
            ema_s = calc_ema(c_series, params['slow']).values
            min_idx = max(params['fast'], params['slow']) + 1
            
        elif stype == 'Stoch_Extreme':
            sk, sd = calc_stoch(h_series, l_series, c_series, params['k'], params['d'])
            sk_v, sd_v = sk.values, sd.values
            min_idx = max(params['k'], params['d']) + 1
            
        elif stype == 'MACD_Hist':
            m, s, h = calc_macd(c_series, params['fast'], params['slow'], params['signal'])
            min_idx = max(params['fast'], params['slow'], params['signal']) + 1
        else:
            return trades

        # Main Loop
        for i in range(min_idx, n-1):
            curr_close = c_series.iloc[i]
            next_close = c_series.iloc[i+1]
            
            if curr_close == next_close: continue
            
            # Trend Filter Logic (Only trade with trend)
            trend_up = curr_close > ema200[i]
            trend_down = curr_close < ema200[i]
            
            dir_sig = 0
            
            if stype == 'RSI_MeanRev':
                val = ind_vals[i]
                if np.isnan(val): continue
                if val < params['os']: dir_sig = 1
                elif val > params['ob']: dir_sig = -1
                
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
                
            elif stype == 'MACD_Hist':
                prev_h = h[i-1]
                curr_h = h[i]
                if np.isnan(prev_h): continue
                # Simple Histogram Crossover Zero Line
                if prev_h <= 0 and curr_h > params['thresh']: dir_sig = 1
                elif prev_h >= 0 and curr_h < -params['thresh']: dir_sig = -1
            
            # Apply Trend Filter
            if dir_sig == 1 and not trend_up: dir_sig = 0
            if dir_sig == -1 and not trend_down: dir_sig = 0
            
            if dir_sig != 0:
                win = (next_close > curr_close) if dir_sig == 1 else (next_close < curr_close)
                trades.append(win)
                    
        return trades

def main():
    print("="*60)
    print("RANDOM SEARCH ENGINE: 1000 STRATEGIES ON M15")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT}")
    print("="*60)
    
    gen = RandomStrategyGenerator(TARGET_STRATEGIES_COUNT)
    total_strats = len(gen.strategies)
    
    # Fetch Data Once
    symbol_data = {}
    for sym in SYMBOLS:
        print(f"Fetching {sym}...")
        df = get_data(sym)
        if df is not None and len(df) > 200:
            symbol_data[sym] = df
    
    if not symbol_data:
        print("No data found."); return

    results_pool = []
    processed_count = 0
    
    # Iterate through EVERY generated strategy
    for strat_cfg in gen.strategies:
        all_trades_combined = []
        for sym, df in symbol_data.items():
            trades = gen.run_backtest_for_symbol(df, strat_cfg)
            all_trades_combined.extend(trades)
            
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{total_strats} strategies... Found {len(results_pool)} viable candidates so far.")
            
        # STRICT FILTERING
        if len(all_trades_combined) < MIN_TRADES_REQUIRED: 
            continue
            
        wins = sum(all_trades_combined)
        total = len(all_trades_combined)
        wr = wins / total
        
        # Only keep strategies with WR > 55% initially to reduce load for Z-score check later if needed
        # But let's calculate Z anyway for accuracy
        p_null = BREAK_EVEN_WR
        std_dev = np.sqrt(total * p_null * (1-p_null))
        z_score = (wins - (total * p_null)) / std_dev if std_dev > 0 else 0
        
        # Store result ONLY if it looks promising (WR > 54%) to save memory/screen space
        if wr > 0.54: 
            results_pool.append({
                'name': strat_cfg['name'],
                'trades': total,
                'wr': wr,
                'z': z_score
            })

    # SORT RESULTS BY Z-SCORE DESCENDING
    results_pool.sort(key=lambda x: x['z'], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 20 PERFORMING STRATEGIES FROM RANDOM SEARCH (M15)")
    print("(Filtered by Min Trades > 100 AND WR > 54%)")
    print("="*60)
    
    if not results_pool:
        print("❌ NO STRATEGIES PASSED THE STRICT FILTERS.")
        print("Conclusion: On M15/60d, there is no consistent statistical edge using these standard indicators.")
    else:
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
