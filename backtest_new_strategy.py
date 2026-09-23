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
TIMEFRAME = "1h"          # Changed to 1 Hour
PERIOD_LIMIT = "1y"       # Request 1 year of hourly data for better sample size
STAKE = 6.0
PAYOUT_RATE = 0.90
BREAK_EVEN_WR = 0.5263    # ~52.63% needed for 90% payout

# Indicator Parameters (Standard settings work well for H1 too)
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
EMA_TREND_PERIOD = 50
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD_DEV = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20
WILLIAMS_R_PERIOD = 14
ROC_PERIOD = 12
SMA_ROC_PERIOD = 20

def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def get_data(symbol, period=PERIOD_LIMIT, interval=TIMEFRAME):
    """Fetch and clean data."""
    try:
        ticker = yf.Ticker(symbol)
        # Note: For 1h data, 'max' might fetch more than we want or fail if too old. 
        # '1y' is a safe bet for recent reliable data.
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data returned for {symbol} with period={period}, interval={interval}")
            return None
        
        df = flatten_columns(df)
        
        # Basic cleaning
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None
            
        df = df.dropna(subset=['Close'])
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ==========================================
# STRATEGY IMPLEMENTATIONS
# Same logic as before, but applied to H1 candles.
# Entry at Close[i], Exit at Close[i+1].
# ==========================================

def calculate_indicators(df):
    """Calculate common indicators efficiently."""
    c = df['Close'].values
    
    # EMAs
    ema_fast = pd.Series(c).ewm(span=EMA_FAST_PERIOD, adjust=False).mean().values
    ema_slow = pd.Series(c).ewm(span=EMA_SLOW_PERIOD, adjust=False).mean().values
    ema_trend = pd.Series(c).ewm(span=EMA_TREND_PERIOD, adjust=False).mean().values
    
    # RSI (Wilder's Smoothing)
    delta = pd.Series(c).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Initial SMA seed
    avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    
    rs_series = pd.Series(index=df.index, dtype=float)
    rs_series[:RSI_PERIOD-1] = np.nan
    
    # Wilder's smoothing loop
    current_avg_gain = avg_gain.iloc[RSI_PERIOD-1]
    current_avg_loss = avg_loss.iloc[RSI_PERIOD-1]
    
    for i in range(RSI_PERIOD, len(c)):
        g = max(0, delta.iloc[i])
        l = max(0, -delta.iloc[i])
        current_avg_gain = ((current_avg_gain * (RSI_PERIOD - 1)) + g) / RSI_PERIOD
        current_avg_loss = ((current_avg_loss * (RSI_PERIOD - 1)) + l) / RSI_PERIOD
        
        if current_avg_loss == 0:
            rs = 100
        else:
            rs = 100 - (100 / (1 + current_avg_gain / current_avg_loss))
        rs_series.iloc[i] = rs
        
    rsi = rs_series.values

    # Bollinger Bands (Population Std Dev ddof=0)
    sma_bb = pd.Series(c).rolling(window=BB_PERIOD).mean().values
    std_bb = pd.Series(c).rolling(window=BB_PERIOD).std(ddof=0).values
    bb_upper = sma_bb + BB_STD_DEV * std_bb
    bb_lower = sma_bb - BB_STD_DEV * std_bb
    
    # MACD
    macd_line = pd.Series(c).ewm(span=MACD_FAST, adjust=False).mean() - \
                pd.Series(c).ewm(span=MACD_SLOW, adjust=False).mean()
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line
    
    # Stochastic
    low_min = df['Low'].rolling(window=STOCH_K_PERIOD).min()
    high_max = df['High'].rolling(window=STOCH_K_PERIOD).max()
    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=STOCH_D_PERIOD).mean()
    
    # ADX
    def calc_adx(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

    adx, plus_di, minus_di = calc_adx(df['High'], df['Low'], df['Close'], ADX_PERIOD)
    
    # Donchian Channel
    don_high = df['High'].rolling(DONCHIAN_PERIOD).max()
    don_low = df['Low'].rolling(DONCHIAN_PERIOD).min()
    
    # Williams %R
    wr = -100 * (high_max - df['Close']) / (high_max - low_min)
    
    # ROC & SMA
    roc = df['Close'].pct_change(periods=ROC_PERIOD) * 100
    sma_roc = df['Close'].rolling(SMA_ROC_PERIOD).mean()

    return {
        'ema_fast': ema_fast, 'ema_slow': ema_slow, 'ema_trend': ema_trend,
        'rsi': rsi, 'bb_upper': bb_upper, 'bb_lower': bb_lower, 'sma_bb': sma_bb,
        'macd': macd_line.values, 'signal': signal_line.values, 'hist': hist.values,
        'stoch_k': stoch_k.values, 'stoch_d': stoch_d.values,
        'adx': adx.values, 'plus_di': plus_di.values, 'minus_di': minus_di.values,
        'don_high': don_high.values, 'don_low': don_low.values,
        'wr': wr.values,
        'roc': roc.values, 'sma_roc': sma_roc.values,
        'close': c, 'open': df['Open'].values, 'high': df['High'].values, 'low': df['Low'].values
    }

def run_backtest_logic(indicators, strategy_func):
    """Generic runner to apply strategy rules and generate trades."""
    trades = []
    n = len(indicators['close'])
    
    for i in range(n - 2):
        direction = strategy_func(i, indicators)
        if direction == 0: continue
        
        entry_price = indicators['close'][i]
        exit_price = indicators['close'][i+1]
        
        # Tie handling
        if entry_price == exit_price:
            continue
            
        is_win = False
        if direction == 1: # CALL
            is_win = exit_price > entry_price
        elif direction == -1: # PUT
            is_win = exit_price < entry_price
            
        trades.append({
            'index': i,
            'direction': direction,
            'win': is_win
        })
    return trades

# --- Strategy Definitions (Same as before) ---

def strat_1_ema_cross(i, ind):
    ef, es, et = ind['ema_fast'][i], ind['ema_slow'][i], ind['ema_trend'][i]
    prev_ef, prev_es = ind['ema_fast'][i-1], ind['ema_slow'][i-1]
    
    if prev_ef <= prev_es and ef > es and ind['close'][i] > et:
        return 1
    if prev_ef >= prev_es and ef < es and ind['close'][i] < et:
        return -1
    return 0

def strat_2_rsi_reversion(i, ind):
    rsi_val = ind['rsi'][i]
    if np.isnan(rsi_val): return 0
    if rsi_val < 30: return 1
    if rsi_val > 70: return -1
    return 0

def strat_3_bollinger_bounce(i, ind):
    c, lb, ub = ind['close'][i], ind['bb_lower'][i], ind['bb_upper'][i]
    if np.isnan(lb) or np.isnan(ub): return 0
    if c <= lb: return 1
    if c >= ub: return -1
    return 0

def strat_4_macd_cross(i, ind):
    m, s = ind['macd'][i], ind['signal'][i]
    pm, ps = ind['macd'][i-1], ind['signal'][i-1]
    
    if pm <= ps and m > s: return 1
    if pm >= ps and m < s: return -1
    return 0

def strat_5_stoch_extremes(i, ind):
    k, d = ind['stoch_k'][i], ind['stoch_d'][i]
    pk, pd = ind['stoch_k'][i-1], ind['stoch_d'][i-1]
    
    if np.isnan(k) or np.isnan(d): return 0
    if pk <= pd and k > d and k < 20: return 1
    if pk >= pd and k < d and k > 80: return -1
    return 0

def strat_6_price_action_hammer(i, ind):
    o, h, l, c = ind['open'][i], ind['high'][i], ind['low'][i], ind['close'][i]
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    if body == 0: return 0 
    
    if lower_wick > 2 * body and upper_wick < body and c > o:
        if c < ind['sma_bb'][i]: 
             return 1
             
    if upper_wick > 2 * body and lower_wick < body and c < o:
        if c > ind['sma_bb'][i]: 
             return -1
             
    return 0

def strat_7_adx_trend(i, ind):
    adx_val = ind['adx'][i]
    pdi, mdi = ind['plus_di'][i], ind['minus_di'][i]
    
    if np.isnan(adx_val) or adx_val < 25: return 0
    
    if pdi > mdi: return 1
    if mdi > pdi: return -1
    return 0

def strat_8_donchian_breakout(i, ind):
    c = ind['close'][i]
    dh, dl = ind['don_high'][i-1], ind['don_low'][i-1] 
    
    if np.isnan(dh) or np.isnan(dl): return 0
    
    if c > dh: return 1
    if c < dl: return -1
    return 0

def strat_9_williams_r(i, ind):
    wr_val = ind['wr'][i]
    if np.isnan(wr_val): return 0
    if wr_val < -80: return 1
    if wr_val > -20: return -1
    return 0

def strat_10_momentum_sma(i, ind):
    roc_val = ind['roc'][i]
    c, sma = ind['close'][i], ind['sma_roc'][i]
    
    if np.isnan(roc_val) or np.isnan(sma): return 0
    if roc_val > 0 and c > sma: return 1
    if roc_val < 0 and c < sma: return -1
    return 0

STRATEGIES = [
    ("EMA_Cross_Trend", strat_1_ema_cross),
    ("RSI_Reversion", strat_2_rsi_reversion),
    ("Bollinger_Bounce", strat_3_bollinger_bounce),
    ("MACD_Cross", strat_4_macd_cross),
    ("Stoch_Extremes", strat_5_stoch_extremes),
    ("PA_Hammer_Shoot", strat_6_price_action_hammer),
    ("ADX_Trend_Filter", strat_7_adx_trend),
    ("Donchian_Breakout", strat_8_donchian_breakout),
    ("Williams_R_Reversal", strat_9_williams_r),
    ("Momentum_SMA_Combo", strat_10_momentum_sma)
]

# ==========================================
# STATISTICS & REPORTING
# ==========================================

def calculate_stats(trades_list):
    if not trades_list:
        return {'count': 0, 'wr': 0, 'z_score': 0, 'robust': 'N/A'}
    
    wins = sum(1 for t in trades_list if t['win'])
    total = len(trades_list)
    wr = wins / total
    
    p_null = BREAK_EVEN_WR
    q_null = 1 - p_null
    expected_wins = total * p_null
    std_dev = np.sqrt(total * p_null * q_null)
    
    z_score = 0
    if std_dev > 0:
        z_score = (wins - expected_wins) / std_dev
        
    mid = total // 2
    first_half = trades_list[:mid]
    second_half = trades_list[mid:]
    
    def get_wr(lst):
        if not lst: return 0
        return sum(1 for x in lst if x['win']) / len(lst)
        
    wr_1 = get_wr(first_half)
    wr_2 = get_wr(second_half)
    
    robust_status = "FAIL"
    if wr_1 >= BREAK_EVEN_WR and wr_2 >= BREAK_EVEN_WR:
        robust_status = "OK"
    elif total < 10:
        robust_status = "INSUFFICIENT_DATA"
        
    return {
        'count': total,
        'wr': wr,
        'z_score': z_score,
        'robust': robust_status,
        'wr_first': wr_1,
        'wr_second': wr_2
    }

def main():
    print("="*60)
    print("QUANTITATIVE BACKTEST ENGINE - BINARY OPTIONS (H1)")
    print(f"Timeframe: {TIMEFRAME} | Duration: 1 Candle ({TIMEFRAME})")
    print(f"Payout: {PAYOUT_RATE*100}% | Stake: ${STAKE}")
    print("="*60)
    
    all_results = {}
    
    symbol_data = {}
    for sym in SYMBOLS:
        print(f"Fetching data for {sym}...")
        df = get_data(sym)
        if df is not None and len(df) > 50: 
            symbol_data[sym] = df
        else:
            print(f"Skipping {sym} due to insufficient data.")
            
    if not symbol_data:
        print("No valid data found. Exiting.")
        return

    for strat_name, strat_func in STRATEGIES:
        print(f"\nTesting Strategy: {strat_name}")
        global_trades = []
        asset_breakdown = {}
        dir_analysis = {'CALL': [], 'PUT': []}
        
        for sym, df in symbol_data.items():
            ind = calculate_indicators(df)
            trades = run_backtest_logic(ind, strat_func)
            
            for t in trades:
                t['symbol'] = sym
                global_trades.append(t)
                
                if sym not in asset_breakdown:
                    asset_breakdown[sym] = []
                asset_breakdown[sym].append(t)
                
                if t['direction'] == 1:
                    dir_analysis['CALL'].append(t)
                else:
                    dir_analysis['PUT'].append(t)
                    
        overall_stats = calculate_stats(global_trades)
        
        all_results[strat_name] = {
            'overall': overall_stats,
            'assets': asset_breakdown,
            'directions': dir_analysis
        }
        
        print("-"*40)
        print(f"Total Trades: {overall_stats['count']}")
        if overall_stats['count'] > 0:
            print(f"Win Rate:     {overall_stats['wr']*100:.2f}%")
            print(f"Z-Score:      {overall_stats['z_score']:.2f}")
            sig = "SIGNIFICANT" if overall_stats['z_score'] > 1.96 else "NOT SIGNIFICANT"
            print(f"Significance: {sig}")
            print(f"Robustness:   {overall_stats['robust']} (H1:{overall_stats.get('wr_first',0)*100:.1f}% H2:{overall_stats.get('wr_second',0)*100:.1f}%)")
            
            verdict = "FAILING"
            if overall_stats['wr'] >= 0.56 and overall_stats['z_score'] > 1.96 and overall_stats['robust'] == "OK":
                verdict = "EXCELLENT"
            elif overall_stats['wr'] >= BREAK_EVEN_WR and overall_stats['wr'] < 0.56:
                verdict = "MARGINAL"
            elif overall_stats['wr'] < BREAK_EVEN_WR:
                verdict = "FAILING"
            else:
                verdict = "NEUTRAL/INCONCLUSIVE"
                
            print(f"Verdict:      {verdict}")
        else:
            print("No trades generated.")
            
    print("\n\n" + "="*60)
    print("FINAL CONSOLIDATED REPORT (H1)")
    print("="*60)
    
    header = f"{'Strategy':<20} | {'Trades':>6} | {'WR%':>6} | {'Z-Score':>7} | {'Robust':>8} | {'Verdict':>10}"
    print(header)
    print("-"*len(header))
    
    best_strats = []
    
    for name, res in all_results.items():
        stats = res['overall']
        if stats['count'] == 0:
            continue
            
        wr_pct = stats['wr'] * 100
        z = stats['z_score']
        rob = stats['robust']
        
        v = "FAILING"
        if stats['wr'] >= 0.56 and z > 1.96 and rob == "OK":
            v = "EXCELLENT"
            best_strats.append(name)
        elif stats['wr'] >= BREAK_EVEN_WR and stats['wr'] < 0.56:
            v = "MARGINAL"
            
        print(f"{name:<20} | {stats['count']:>6} | {wr_pct:>6.2f} | {z:>7.2f} | {rob:>8} | {v:>10}")
        
    print("-"*len(header))
    
    if best_strats:
        print(f"\n🏆 TOP PERFORMERS (Excellent): {', '.join(best_strats)}")
    else:
        print("\n⚠️ No strategies achieved 'EXCELLENT' status based on strict criteria.")
        
    print("\nNote: Results depend on market conditions during the fetched period.")

if __name__ == "__main__":
    main()
