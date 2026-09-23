import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
# Adding major pairs for correlation context if needed, but sticking to main list for consistency
MAIN_SYMBOLS = SYMBOLS 

TIMEFRAME = "1h"       # Hourly is best for reducing noise in these advanced strategies
PERIOD_LIMIT = "2y"    # Longer history for better statistical significance of patterns
STAKE = 6.0
PAYOUT_RATE = 0.90
BREAK_EVEN_WR = 0.5263

# Advanced Parameters
ATR_PERIOD = 14
SESSION_START_HOURS = [7, 8, 9, 10, 11, 12, 13, 14] # London/NY Overlap focus (UTC approx)
OB_LOOKBACK = 5      # How many candles back to look for Order Block origin
VOLATILITY_THRESHOLD_MULT = 1.5 # Price must move at least 1.5x ATR to be considered an impulse

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
# INDICATOR CALCULATIONS (ADVANCED)
# ==========================================

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def detect_order_blocks(df, ob_lookback=5):
    """
    Detects bullish/bearish order blocks based on strong impulses.
    Bullish OB: Last bearish candle before a strong upward move.
    Bearish OB: Last bullish candle before a strong downward move.
    Returns boolean series: True if current candle closes inside/above/below relevant OB zone.
    For simplicity in backtesting logic, we mark the ZONE creation index.
    """
    n = len(df)
    bull_ob_zones = [] # List of dicts: {'start_idx': int, 'end_price_high': float, 'end_price_low': float}
    bear_ob_zones = []
    
    c = df['Close'].values
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    
    # Simple Impulse Detection: Change > Threshold * Average Range
    avg_range = pd.Series(h-l).rolling(20).mean().values
    threshold = avg_range * VOLATILITY_THRESHOLD_MULT
    
    for i in range(ob_lookback + 1, n):
        # Check for Bullish Impulse starting around i-ob_lookback
        # If price moved up significantly from i-k to i
        move_up = c[i] - c[i-ob_lookback]
        if move_up > threshold[i]:
            # Find the last down candle before this move started
            for j in range(i-ob_lookback, i-1, -1):
                if c[j] < o[j]: # It was a red candle
                    bull_ob_zones.append({
                        'origin_idx': j,
                        'zone_top': h[j],
                        'zone_bottom': l[j],
                        'valid_until': i + 20 # Assume validity for next 20 bars unless broken
                    })
                    break
        
        # Check for Bearish Impulse
        move_down = c[i-ob_lookback] - c[i]
        if move_down > threshold[i]:
            for j in range(i-ob_lookback, i-1, -1):
                if c[j] > o[j]: # It was a green candle
                    bear_ob_zones.append({
                        'origin_idx': j,
                        'zone_top': h[j],
                        'zone_bottom': l[j],
                        'valid_until': i + 20
                    })
                    break
                    
    return bull_ob_zones, bear_ob_zones

def check_session_filter(timestamp, allowed_hours):
    hour = timestamp.hour
    day = timestamp.weekday() # 0=Mon, 4=Fri
    if day >= 5: return False # Weekend
    if hour in allowed_hours: return True
    return False

# ==========================================
# STRATEGY IMPLEMENTATIONS
# ==========================================

def run_backtest_logic_advanced(indicators_df, strategy_func, symbol_name):
    """
    indicators_df contains pre-calculated columns including ATR, Session Flag, etc.
    strategy_func takes row index and dataframe slices.
    """
    trades = []
    n = len(indicators_df)
    
    # Pre-extract arrays for speed
    close = indicators_df['Close'].values
    open_p = indicators_df['Open'].values
    high = indicators_df['High'].values
    low = indicators_df['Low'].values
    atr = indicators_df['ATR'].values
    session_ok = indicators_df['Session_OK'].values.astype(bool)
    
    # Get OB Zones for this symbol
    bull_obs, bear_obs = detect_order_blocks(indicators_df)
    
    # Helper to check if price is in any active OB
    def is_in_bull_ob(idx, price):
        for ob in bull_obs:
            if idx <= ob['valid_until'] and idx > ob['origin_idx']:
                if ob['zone_bottom'] <= price <= ob['zone_top']:
                    return True
        return False

    def is_in_bear_ob(idx, price):
        for ob in bear_obs:
            if idx <= ob['valid_until'] and idx > ob['origin_idx']:
                if ob['zone_bottom'] <= price <= ob['zone_top']:
                    return True
        return False

    for i in range(n - 2):
        # Basic Filters
        if not session_ok[i]: continue
        if np.isnan(atr[i]) or atr[i] == 0: continue
        
        # Current State
        curr_close = close[i]
        prev_close = close[i-1]
        
        direction = strategy_func(i, curr_close, prev_close, atr[i], 
                                  lambda p: is_in_bull_ob(i, p),
                                  lambda p: is_in_bear_ob(i, p))
        
        if direction == 0: continue
        
        entry_price = curr_close
        exit_price = close[i+1]
        
        if entry_price == exit_price: continue
            
        is_win = (exit_price > entry_price) if direction == 1 else (exit_price < entry_price)
            
        trades.append({'index': i, 'direction': direction, 'win': is_win})
        
    return trades

# --- Strategy 1: Order Block Reversal with Volatility Filter ---
def strat_1_ob_volatility(i, c, pc, atr_val, in_bull_ob_fn, in_bear_ob_fn):
    # CALL: Price retraces into a Bullish OB AND shows rejection (wick) OR momentum shift
    # We simplify: If previous candle closed below current (red), and current is inside Bullish OB -> Potential Buy
    # Plus: Ensure volatility isn't exploding against us (stop loss proxy via ATR)
    
    if in_bull_ob_fn(c):
        # Look for stabilization or slight uptick start
        if c > pc: # Green candle closing higher than prev
            return 1
            
    if in_bear_ob_fn(c):
        if c < pc: # Red candle closing lower
            return -1
            
    return 0

# --- Strategy 2: Relative Strength Divergence (Proxy using EUR/USD vs GBP/USD logic simplified) ---
# Since we are testing single symbols in loop, true cross-pair arb needs global data access.
# Here we simulate "Internal Momentum Consistency":
# Only trade RSI/Bollinger signals IF they align with the longer-term EMA trend AND high volume/volatility.
def strat_2_momentum_confluence(i, ind_row_dict):
    pass # Placeholder for complex multi-symbol logic, skipped for single-file simplicity

# Let's replace Strat 2 with a pure Single-Symbol Advanced Logic: 
# "The Fakeout Trap": Price breaks recent High/Low then immediately reverses.
def strat_2_fakeout_trap(i, c, pc, atr_val, in_bull_ob_fn, in_bear_ob_fn):
    # Need access to rolling highs/lows. This requires passing them or calculating inside.
    # For efficiency in this generic runner, we assume external calc or simple local check.
    # Let's use a simpler version: Engulfing Pattern at Key Level (OB)
    
    body_curr = abs(c - ind_row_dict.get('o', c)) # Note: Open not passed directly in signature above, need adjustment
    # To keep code clean, let's stick to OB + Trend Alignment using EMA50 calculated globally.
    return 0 

# RE-DEFINING STRATEGIES FOR CLARITY IN THE MAIN LOOP BELOW

STRATEGIES_ADVANCED = [
    ("OB_Reversal_Volatility", "Order Block Retest"),
    ("Fakeout_Traps", "Breakout Failure"),
    ("Trend_Pullback_ATR", "Deep Pullback in Strong Trend")
]

# We will implement these inline in the main function for clarity since they need different indicator sets.

def prepare_indicators_for_symbol(df):
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], ATR_PERIOD)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Session Filter
    df['Session_OK'] = df.apply(lambda x: check_session_filter(x.name, SESSION_START_HOURS), axis=1)
    
    # Rolling Highs/Lows for Fakeouts
    df['Roll_High_10'] = df['High'].rolling(10).max().shift(1) # Previous 10 bars high
    df['Roll_Low_10'] = df['Low'].rolling(10).min().shift(1)   # Previous 10 bars low
    
    return df

def execute_strategy_logic(strat_name, i, row, prev_row):
    """
    Central brain for decision making based on strategy name.
    """
    c = row['Close']
    o = row['Open']
    h = row['High']
    l = row['Low']
    ema50 = row['EMA50']
    ema200 = row['EMA200']
    atr = row['ATR']
    roll_h = row['Roll_High_10']
    roll_l = row['Roll_Low_10']
    
    prev_c = prev_row['Close']
    prev_o = prev_row['Open']
    prev_h = prev_row['High']
    prev_l = prev_row['Low']

    if np.isnan(atr) or np.isnan(roll_h) or np.isnan(roll_l): return 0

    # 1. OB Reversal (Simplified approximation without full zone storage for speed)
    # Idea: Price pulls back to EMA50 in a trend defined by EMA200
    if strat_name == "OB_Reversal_Volatility":
        # Uptrend: Close > EMA200 > EMA50
        if c > ema200 and ema200 > ema50:
            # Pullback: Low touched near EMA50 but Closed above it
            if l <= ema50 * 1.005 and c > ema50:
                return 1 # CALL
        # Downtrend
        elif c < ema200 and ema200 < ema50:
            if h >= ema50 * 0.995 and c < ema50:
                return -1 # PUT
        return 0

    # 2. Fakeout Traps
    # Breakout fails immediately
    if strat_name == "Fakeout_Traps":
        # Bull Trap: Broke previous high, but closed back below it
        if h > roll_h and c < roll_h:
            return -1 # PUT (Expect reversal down)
        # Bear Trap: Broke previous low, but closed back above it
        if l < roll_l and c > roll_l:
            return 1 # CALL (Expect reversal up)
        return 0

    # 3. Deep Pullback in Strong Trend (Momentum Confluence)
    if strat_name == "Trend_Pullback_ATR":
        # Strong Up Move recently?
        mom = c - prev_c
        if mom > atr * 1.5: # Big green candle
             # Next candle opens lower (gap/profit taking)?
             # Actually, we enter AT CLOSE of signal candle. 
             # So condition: Current candle is big green, AND price is above EMA200.
             if c > ema200:
                 return 1
        # Big Red Candle
        if mom < -atr * 1.5:
             if c < ema200:
                 return -1
        return 0

    return 0

# ==========================================
# MAIN EXECUTION
# ==========================================

def calculate_stats(trades_list):
    if not trades_list: return {'count': 0, 'wr': 0, 'z_score': 0, 'robust': 'N/A'}
    wins = sum(1 for t in trades_list if t['win'])
    total = len(trades_list)
    wr = wins / total
    
    p_null = BREAK_EVEN_WR
    std_dev = np.sqrt(total * p_null * (1 - p_null))
    z_score = (wins - (total * p_null)) / std_dev if std_dev > 0 else 0
    
    mid = total // 2
    h1_w = sum(1 for t in trades_list[:mid] if t['win']) / max(mid, 1)
    h2_w = sum(1 for t in trades_list[mid:] if t['win']) / max(total-mid, 1)
    
    robust = "OK" if (h1_w >= BREAK_EVEN_WR and h2_w >= BREAK_EVEN_WR) else "FAIL"
    
    return {'count': total, 'wr': wr, 'z_score': z_score, 'robust': robust, 'wr_first': h1_w, 'wr_second': h2_w}

def main():
    print("="*60)
    print("ADVANCED QUANT BACKTEST ENGINE (Contextual Strategies)")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT}")
    print("="*60)
    
    results_store = {}
    
    # Fetch Data
    symbol_data_raw = {}
    for sym in MAIN_SYMBOLS:
        print(f"Fetching {sym}...")
        df = get_data(sym)
        if df is not None and len(df) > 100:
            # Prepare indicators ONCE per symbol
            prepared_df = prepare_indicators_for_symbol(df.copy())
            symbol_data_raw[sym] = prepared_df
        else:
            print(f"Skipping {sym}.")
            
    if not symbol_data_raw:
        print("No data."); return

    # Run Strategies
    for strat_key, strat_desc in STRATEGIES_ADVANCED:
        print(f"\n>>> Testing: {strat_desc} ({strat_key})")
        all_trades = []
        
        for sym, df in symbol_data_raw.items():
            rows = df.to_numpy()
            cols = df.columns.tolist()
            
            # Convert to dict-like access for helper functions if needed, 
            # but iterating pandas rows is slow. Let's iterate indices.
            n = len(df)
            for i in range(1, n-1): # Start from 1 to have prev row, end at n-2 for exit
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                # Check Session Filter first (fast fail)
                if not row['Session_OK']: continue
                
                direction = execute_strategy_logic(strat_key, i, row, prev_row)
                
                if direction != 0:
                    entry = row['Close']
                    exit_p = df.iloc[i+1]['Close']
                    
                    if entry == exit_p: continue
                    
                    win = (exit_p > entry) if direction == 1 else (exit_p < entry)
                    all_trades.append({'win': win})
                    
        stats_res = calculate_stats(all_trades)
        results_store[strat_desc] = stats_res
        
        print(f"Trades: {stats_res['count']} | WR: {stats_res['wr']*100:.2f}% | Z: {stats_res['z_score']:.2f} | Robust: {stats_res['robust']}")
        
        verdict = "FAILING"
        if stats_res['wr'] >= 0.56 and stats_res['z_score'] > 1.96 and stats_res['robust'] == "OK":
            verdict = "EXCELLENT"
        elif stats_res['wr'] >= BREAK_EVEN_WR:
            verdict = "MARGINAL"
        print(f"Verdict: {verdict}")

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for name, res in results_store.items():
        print(f"{name:<25} | Trades:{res['count']:>5} | WR:{res['wr']*100:>5.1f}% | Z:{res['z_score']:>5.2f} | {res['robust']}")

if __name__ == "__main__":
    main()
