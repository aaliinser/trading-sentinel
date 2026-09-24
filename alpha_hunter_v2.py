import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION FOR MASSIVE SEARCH
# ==========================================
SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "EURGBP=X", "EURAUD=X", "EURCAD=X", 
    "EURCHF=X", "EURNZD=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", 
    "GBPNZD=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "CADCHF=X", 
    "CADJPY=X", "CHFJPY=X"
] # 22 Symbols for maximum diversification

TIMEFRAME = "15m"
PERIOD_LIMIT = "60d" # Max available data
BREAK_EVEN_WR = 0.5263
MIN_TRADES_THRESHOLD = 300 # Strict filter to avoid noise
TARGET_WIN_RATE = 0.58     # Only show strategies with >58% WR

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
# INDICATOR LIBRARY (Vectorized & Fast)
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

def calc_williams_r(high, low, close, period):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.fillna(-50)

def calc_cci(high, low, close, period):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.fillna(0)

def calc_roc(close, period):
    return close.pct_change(periods=period) * 100

# ==========================================
# STRATEGY GENERATOR ENGINE
# ==========================================

class AlphaHunterEngine:
    def __init__(self):
        self.strategies = []
        self._generate_combinations()

    def _generate_combinations(self):
        """Generates ~2000 unique strategy configurations."""
        
        # 1. RSI Variations (Periods: 7-25, OB/OS: Wide Range)
        for p in range(7, 26, 2):
            for ob in [65, 70, 75, 80]:
                for os in [20, 25, 30, 35]:
                    self.strategies.append({'type': 'RSI', 'params': {'p': p, 'ob': ob, 'os': os}, 'name': f"RSI_{p}_{ob}_{os}"})

        # 2. Bollinger Bands Variations (Periods: 10-30, Std: 1.5-3.0)
        for p in range(10, 31, 2):
            for s in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8]:
                self.strategies.append({'type': 'BB', 'params': {'p': p, 's': s}, 'name': f"BB_{p}_{s}"})

        # 3. Stochastic Variations (K: 5-25, D: 3-10, Levels: Extremes)
        for k in range(5, 26, 2):
            for d in [3, 5, 8]:
                for ob in [75, 80, 85, 90]:
                    for os in [10, 15, 20, 25]:
                        self.strategies.append({'type': 'Stoch', 'params': {'k': k, 'd': d, 'ob': ob, 'os': os}, 'name': f"Stoch_{k}_{d}_{ob}_{os}"})

        # 4. Williams %R Variations
        for p in range(10, 21, 2):
            for level in [-10, -20, -80, -90]: # Overbought/Oversold thresholds
                self.strategies.append({'type': 'WR', 'params': {'p': p, 'level': level}, 'name': f"WR_{p}_{abs(level)}"})

        # 5. CCI Variations
        for p in range(14, 25, 2):
            for level in [100, 150, 200]:
                self.strategies.append({'type': 'CCI', 'params': {'p': p, 'level': level}, 'name': f"CCI_{p}_{level}"})

        # 6. MACD Histogram Crossings
        for f in [8, 12, 16]:
            for s in [20, 26, 30]:
                for sig in [7, 9, 12]:
                    if f >= s: continue
                    self.strategies.append({'type': 'MACD_Hist', 'params': {'f': f, 's': s, 'sig': sig}, 'name': f"MACDH_{f}_{s}_{sig}"})

        # 7. ROC Momentum Reversals
        for p in [10, 12, 14, 20]:
            for thresh in [1.0, 2.0, 3.0]:
                self.strategies.append({'type': 'ROC', 'params': {'p': p, 'thresh': thresh}, 'name': f"ROC_{p}_{thresh}"})

        print(f"Generated {len(self.strategies)} unique strategy configurations.")

    def evaluate_strategy_on_split(self, strat_cfg, df_full):
        """
        Evaluates strategy on First Half (Training) and Second Half (Validation).
        Returns: (is_valid, overall_wr, z_score, total_trades)
        """
        n = len(df_full)
        mid = n // 2
        
        # Split Data
        df_train = df_full.iloc[:mid].copy()
        df_val = df_full.iloc[mid:].copy()
        
        if len(df_train) < 50 or len(df_val) < 50:
            return False, 0, 0, 0

        # Calculate Indicators for Train
        trades_train = self._run_logic(strat_cfg, df_train)
        trades_val = self._run_logic(strat_cfg, df_val)
        
        total_trades = len(trades_train) + len(trades_val)
        
        if total_trades < MIN_TRADES_THRESHOLD:
            return False, 0, 0, 0
            
        wins_total = sum(trades_train) + sum(trades_val)
        wr_total = wins_total / total_trades
        
        # STRICT FILTER: Must be profitable in BOTH halves to pass validation
        wr_train = sum(trades_train)/len(trades_train) if trades_train else 0
        wr_val = sum(trades_val)/len(trades_val) if trades_val else 0
        
        is_robust = (wr_train >= BREAK_EVEN_WR) and (wr_val >= BREAK_EVEN_WR)
        
        if not is_robust:
            return False, wr_total, 0, total_trades # Failed robustness check

        # Calculate Z-Score for the whole set
        p_null = BREAK_EVEN_WR
        std_dev = np.sqrt(total_trades * p_null * (1-p_null))
        z_score = (wins_total - (total_trades * p_null)) / std_dev if std_dev > 0 else 0
        
        return True, wr_total, z_score, total_trades

    def _run_logic(self, cfg, df):
        """Executes the specific strategy logic on a dataframe slice."""
        trades = []
        n = len(df)
        if n < 50: return trades
        
        c = df['Close']
        h = df['High']
        l = df['Low']
        
        # Trend Filter (EMA200) - Applied universally for stability
        ema200 = calc_ema(c, 200).values
        
        stype = cfg['type']
        params = cfg['params']
        
        # Pre-calculate indicators based on type
        if stype == 'RSI':
            ind = calc_rsi(c, params['p']).values
            min_idx = params['p'] + 1
        elif stype == 'BB':
            _, ub, lb = calc_bb(c, params['p'], params['s'])
            ub_v, lb_v = ub.values, lb.values
            min_idx = params['p'] + 1
        elif stype == 'Stoch':
            sk, sd = calc_stoch(h, l, c, params['k'], params['d'])
            sk_v, sd_v = sk.values, sd.values
            min_idx = max(params['k'], params['d']) + 1
        elif stype == 'WR':
            wr = calc_williams_r(h, l, c, params['p']).values
            min_idx = params['p'] + 1
        elif stype == 'CCI':
            cci = calc_cci(h, l, c, params['p']).values
            min_idx = params['p'] + 1
        elif stype == 'MACD_Hist':
            _, _, hist = calc_macd(c, params['f'], params['s'], params['sig'])
            min_idx = max(params['f'], params['s'], params['sig']) + 1
        elif stype == 'ROC':
            roc = calc_roc(c, params['p']).values
            min_idx = params['p'] + 1
        else:
            return trades

        # Main Loop
        for i in range(min_idx, n-1):
            curr_close = c.iloc[i]
            next_close = c.iloc[i+1]
            
            if curr_close == next_close: continue
            
            trend_up = curr_close > ema200[i]
            trend_down = curr_close < ema200[i]
            
            dir_sig = 0
            
            if stype == 'RSI':
                val = ind[i]
                if np.isnan(val): continue
                # Mean Reversion Logic
                if val < params['os'] and trend_up: dir_sig = 1
                elif val > params['ob'] and trend_down: dir_sig = -1
                
            elif stype == 'BB':
                u_val, l_val = ub_v[i], lb_v[i]
                if np.isnan(u_val): continue
                if curr_close <= l_val and trend_up: dir_sig = 1
                elif curr_close >= u_val and trend_down: dir_sig = -1
                
            elif stype == 'Stoch':
                k_val, d_val = sk_v[i], sd_v[i]
                pk, pd = sk_v[i-1], sd_v[i-1]
                if np.isnan(k_val): continue
                if pk <= pd and k_val > d_val and k_val < params['os'] and trend_up: dir_sig = 1
                elif pk >= pd and k_val < d_val and k_val > params['ob'] and trend_down: dir_sig = -1
                
            elif stype == 'WR':
                val = wr[i]
                if np.isnan(val): continue
                if val < params['level'] and trend_up: dir_sig = 1 # Oversold
                elif val > params['level'] and trend_down: dir_sig = -1 # Overbought (Note: WR is negative)
                
            elif stype == 'CCI':
                val = cci[i]
                if np.isnan(val): continue
                if val < -params['level'] and trend_up: dir_sig = 1
                elif val > params['level'] and trend_down: dir_sig = -1
                
            elif stype == 'MACD_Hist':
                prev_h = hist[i-1]
                curr_h = hist[i]
                if np.isnan(prev_h): continue
                if prev_h <= 0 and curr_h > 0 and trend_up: dir_sig = 1
                elif prev_h >= 0 and curr_h < 0 and trend_down: dir_sig = -1
                
            elif stype == 'ROC':
                val = roc[i]
                if np.isnan(val): continue
                if val < -params['thresh'] and trend_up: dir_sig = 1
                elif val > params['thresh'] and trend_down: dir_sig = -1
            
            if dir_sig != 0:
                win = (next_close > curr_close) if dir_sig == 1 else (next_close < curr_close)
                trades.append(win)
                
        return trades

def main():
    print("="*60)
    print("ALPHA HUNTER V2.0: 2000 STRATEGIES SCAN")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT}")
    print("="*60)
    
    engine = AlphaHunterEngine()
    total_strats = len(engine.strategies)
    
    # Fetch Data Once
    symbol_data = {}
    for sym in SYMBOLS:
        print(f"Fetching {sym}...", end=" ")
        df = get_data(sym)
        if df is not None and len(df) > 200:
            symbol_data[sym] = df
            print("OK")
        else:
            print("SKIP")
    
    if not symbol_data:
        print("No data found."); return

    winners = []
    processed_count = 0
    
    # Iterate through EVERY generated strategy
    for strat_cfg in engine.strategies:
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"Processed {processed_count}/{total_strats}... Found {len(winners)} gems so far.")
            
        all_trades_combined = []
        is_all_robust = True
        
        # Run on all symbols to aggregate stats
        for sym, df in symbol_data.items():
            # For speed, we only do the split-validation on the aggregated data? 
            # No, strict validation requires checking each symbol's consistency OR aggregating first.
            # To save time in this script, we will Aggregate Trades from All Symbols first,
            # THEN apply the Split-Sample check on the TOTAL pool.
            
            # Optimization: Instead of running split per symbol, run full backtest per symbol, collect trades, then split the BIG list.
            pass 

        # RE-EFFICIENT APPROACH:
        # 1. Collect ALL trades from ALL symbols for this strategy config.
        global_trades_list = []
        for sym, df in symbol_data.items():
            t_list = engine._run_logic(strat_cfg, df)
            global_trades_list.extend(t_list)
            
        if len(global_trades_list) < MIN_TRADES_THRESHOLD:
            continue
            
        # 2. Perform Split-Sample Check on the GLOBAL list
        n_total = len(global_trades_list)
        mid_point = n_total // 2
        
        train_part = global_trades_list[:mid_point]
        val_part = global_trades_list[mid_point:]
        
        wr_train = sum(train_part) / len(train_part)
        wr_val = sum(val_part) / len(val_part)
        
        # Robustness Gate
        if wr_train < BREAK_EVEN_WR or wr_val < BREAK_EVEN_WR:
            continue
            
        # Success! Calculate Final Stats
        total_wins = sum(global_trades_list)
        overall_wr = total_wins / n_total
        z_score = (total_wins - (n_total * BREAK_EVEN_WR)) / np.sqrt(n_total * BREAK_EVEN_WR * (1-BREAK_EVEN_WR))
        
        # Quality Gate
        if overall_wr > TARGET_WIN_RATE and z_score > 2.0:
            winners.append({
                'name': strat_cfg['name'],
                'trades': n_total,
                'wr': overall_wr,
                'z': z_score,
                'wr_train': wr_train,
                'wr_val': wr_val
            })

    # SORT RESULTS BY Z-SCORE DESCENDING
    winners.sort(key=lambda x: x['z'], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 20 ELITE STRATEGIES (Validated & Robust)")
    print("="*60)
    
    if not winners:
        print("❌ NO STRATEGIES PASSED THE ULTRA-STRICT FILTERS.")
        print("Conclusion: The market regime in the last 60 days does not support simple indicator reversals with high certainty.")
    else:
        print(f"{'Rank':<5} | {'Strategy Name':<25} | {'Trades':>6} | {'WR%':>6} | {'Z':>5} | {'Train%':>6} | {'Val%':>6}")
        print("-"*80)
        
        top_n = min(20, len(winners))
        for i in range(top_n):
            res = winners[i]
            print(f"{i+1:<5} | {res['name']:<25} | {res['trades']:>6} | {res['wr']*100:>6.2f} | {res['z']:>5.2f} | {res['wr_train']*100:>6.2f} | {res['wr_val']*100:>6.2f}")

if __name__ == "__main__":
    main()
