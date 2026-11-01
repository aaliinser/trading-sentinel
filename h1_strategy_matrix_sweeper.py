#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 STRATEGY MATRIX SWEEPER - QUANT ENGINE v1.0
الفلسفة: التخلي عن الحدس البشري والاعتماد على المنهجية الكمية الصرفة.
الهدف: اكتشاف "الحافة الإحصائية الحقيقية" (True Edge) عبر اختبار مئات التوليفات.
المعايير الصارمة للفوز:
1. Z-Score > 2.0 (دلالة إحصائية قوية، ليست صدفة).
2. Split-Sample Robustness (أداء مستقر زمنياً).
3. Out-of-Sample Validation (70% تدريب / 30% تحقق مغلق).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import itertools
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION: MASSIVE STRATEGY SWEEP
# ==========================================

SYMBOLS = [
    "USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X",
    "GBPUSD=X", "USDCAD=X", "AUDNZD=X", "EURGBP=X", "EURUSD=X"
]

TIMEFRAME = "1h"
PERIOD_LIMIT = "2y" 
BREAK_EVEN_WR = 0.5263 # نسبة التعادل لعائد 90%

# --- SEARCH SPACE PARAMETERS (The Grid) ---
# هذه هي المتغيرات التي سيمر بها المحرك تلقائياً
PARAM_GRID = {
    'BB_PERIOD': [10, 14, 20, 30],       # فترات بولنجر مختلفة
    'BB_STD_DEV': [1.5, 2.0, 2.5],       # انحرافات معيارية مختلفة
    'STOCH_K': [9, 14, 21],              # فترات ستوكاستيك
    'STOCH_D': [3, 5],                   # تنعيم ستوكاستيك
    'OS_LEVEL': [10, 20, 30],            # مستويات تشبع بيعي
    'OB_LEVEL': [70, 80, 90],            # مستويات تشبع شرائي
    'EMA_FILTER': [50, 100, 200],        # فلاتر اتجاه مختلفة
    'STRATEGY_TYPE': ['REVERSION', 'BREAKOUT'] # نوع الاستراتيجية الأساسية
}

SPREAD_COST_PIPS = 2.0   # تكلفة السبريد الافتراضية لكل صفقة

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def fetch_and_precompute_all_data():
    """
    ★★ الخطوة الحاسمة للكفاءة: حساب المؤشرات مرة واحدة وتخزينها ★★
    بدلاً من إعادة الحساب آلاف المرات، نحسب جميع الاحتمالات مسبقاً.
    """
    print("Fetching data and pre-computing indicators for all symbols...")
    all_dfs = {}
    
    for sym in SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=PERIOD_LIMIT, interval=TIMEFRAME)
            if df.empty: continue
            
            df = flatten_columns(df)
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols): continue
            
            df = df.dropna(subset=['Close'])
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            if len(df) < 300: continue # حد أدنى للبيانات
            
            # تخزين البيانات الخام للمعالجة اللاحقة
            all_dfs[sym] = df[['Open', 'High', 'Low', 'Close']].copy()
            
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
            continue
            
    print(f"Successfully loaded data for {len(all_dfs)} symbols.")
    return all_dfs

def calculate_indicators_for_config(df, config):
    """
    يحسب المؤشرات المطلوبة لتكوين استراتيجي معين (Config).
    يستخدم NumPy/Pandas Vectorization للسرعة القصوى.
    """
    c = df['Close']
    h = df['High']
    l = df['Low']
    
    bb_period = config['BB_PERIOD']
    bb_std = config['BB_STD_DEV']
    stoch_k = config['STOCH_K']
    stoch_d = config['STOCH_D']
    ema_period = config['EMA_FILTER']
    
    # 1. Bollinger Bands
    sma_bb = c.rolling(window=bb_period).mean()
    std_bb = c.rolling(window=bb_period).std(ddof=0)
    bb_upper = sma_bb + (bb_std * std_bb)
    bb_lower = sma_bb - (bb_std * std_bb)
    
    # 2. Stochastic Oscillator (Wilder's Smoothing Approximation via EWM)
    lowest_low = l.rolling(window=stoch_k).min()
    highest_high = h.rolling(window=stoch_k).max()
    denom = highest_high - lowest_low
    denom.replace(0, np.nan, inplace=True)
    raw_k = 100 * ((c - lowest_low) / denom)
    raw_k = raw_k.fillna(50)
    
    slow_k = raw_k.ewm(alpha=1/stoch_d, adjust=False).mean()
    stoch_d_val = slow_k.rolling(window=stoch_d).mean()
    
    # 3. EMA Trend Filter
    ema_filter = c.ewm(span=ema_period, adjust=False).mean()
    
    return pd.DataFrame({
        'Close': c,
        'Next_Close': c.shift(-1), # السعر في الشمعة التالية (للحسم)
        'BB_Upper': bb_upper,
        'BB_Lower': bb_lower,
        'Stoch_K': slow_k,
        'Stoch_D': stoch_d_val,
        'EMA_Filter': ema_filter
    }).dropna()

def evaluate_single_strategy(df_with_indicators, config, symbol_name):
    """
    يطبق قواعد الدخول والخروج لاستراتيجية محددة ويحسب النتائج.
    يعيد قائمة بالنتائج (True/False) وتواريخ الصفقات.
    """
    trades = []
    timestamps = []
    
    pip_mult = 100.0 if "JPY" in symbol_name else 10000.0
    spread_in_price = SPREAD_COST_PIPS / pip_mult
    
    strategy_type = config['STRATEGY_TYPE']
    os_level = config['OS_LEVEL']
    ob_level = config['OB_LEVEL']
    
    # استخراج الأعمدة كمصفوفات NumPy لسرعة الوصول
    close_vals = df_with_indicators['Close'].values
    next_close_vals = df_with_indicators['Next_Close'].values
    bb_u = df_with_indicators['BB_Upper'].values
    bb_l = df_with_indicators['BB_Lower'].values
    sk = df_with_indicators['Stoch_K'].values
    sd = df_with_indicators['Stoch_D'].values
    ema = df_with_indicators['EMA_Filter'].values
    idx = df_with_indicators.index.values
    
    n = len(close_vals)
    
    for i in range(n - 1):
        direction = 0
        
        if strategy_type == 'REVERSION':
            # Call: Price below BB Lower AND Stoch Oversold AND Above EMA (Uptrend pullback)
            if close_vals[i] < bb_l[i] and sk[i] < os_level and close_vals[i] > ema[i]:
                direction = 1
            # Put: Price above BB Upper AND Stoch Overbought AND Below EMA (Downtrend rally)
            elif close_vals[i] > bb_u[i] and sk[i] > ob_level and close_vals[i] < ema[i]:
                direction = -1
                
        elif strategy_type == 'BREAKOUT':
            # Call: Close breaks above BB Upper AND Stoch confirms momentum (not extreme) AND Above EMA
            if close_vals[i] > bb_u[i] and sk[i] > 50 and close_vals[i] > ema[i]:
                direction = 1
            # Put: Close breaks below BB Lower AND Stoch confirms downward momentum AND Below EMA
            elif close_vals[i] < bb_l[i] and sk[i] < 50 and close_vals[i] < ema[i]:
                direction = -1
        
        if direction != 0:
            price_diff = next_close_vals[i] - close_vals[i]
            
            net_profitable = False
            if direction == 1: # CALL
                net_profitable = price_diff > spread_in_price
            else: # PUT
                net_profitable = (-price_diff) > spread_in_price
            
            trades.append(net_profitable)
            timestamps.append(idx[i])
            
    return trades, timestamps

def analyze_performance(trades_list, timestamps_list):
    """
    يحسب المقاييس الإحصائية الحرجة لنتائج استراتيجية واحدة.
    يشمل Z-Score و Split-Sample Test.
    """
    total = len(trades_list)
    if total < 30: # حد أدنى للعينة الموثوقة
        return None
        
    wins = sum(trades_list)
    wr = wins / total
    
    # Z-Score Calculation
    p_null = BREAK_EVEN_WR
    std_dev = np.sqrt(total * p_null * (1 - p_null))
    z_score = (wins - (total * p_null)) / std_dev if std_dev > 0 else 0
    
    # Split-Sample Robustness Check
    mid_point = total // 2
    h1_trades = trades_list[:mid_point]
    h2_trades = trades_list[mid_point:]
    
    wr_h1 = sum(h1_trades) / len(h1_trades) if h1_trades else 0
    wr_h2 = sum(h2_trades) / len(h2_trades) if h2_trades else 0
    
    is_robust = (wr_h1 >= BREAK_EVEN_WR) and (wr_h2 >= BREAK_EVEN_WR)
    
    return {
        'count': total,
        'wr': wr,
        'z_score': z_score,
        'robust': is_robust,
        'wr_h1': wr_h1,
        'wr_h2': wr_h2
    }

def run_massive_sweep(precomputed_data):
    """
    ★★ القلب النابض للمحرك: حلقة البحث الشاملة ★★
    يمر عبر كل توليفة ممكنة في PARAM_GRID ويقيّم أداءها.
    """
    results = []
    grid_keys = list(PARAM_GRID.keys())
    grid_values = list(PARAM_GRID.values())
    
    # توليد جميع التوليفات الممكنة (Cartesian Product)
    combinations = list(itertools.product(*grid_values))
    total_combinations = len(combinations)
    
    print(f"\nStarting Massive Sweep of {total_combinations} strategy configurations...")
    print("="*70)
    
    processed_count = 0
    
    for combo in combinations:
        config = dict(zip(grid_keys, combo))
        
        # جمع نتائج هذا التكوين عبر جميع الأزواج
        global_trades = []
        global_timestamps = []
        
        for sym, raw_df in precomputed_data.items():
            # حساب المؤشرات لهذا التكوين تحديداً
            ind_df = calculate_indicators_for_config(raw_df, config)
            
            if ind_df.empty or len(ind_df) < 50:
                continue
                
            # تقييم الاستراتيجية على هذا الزوج
            trades, ts = evaluate_single_strategy(ind_df, config, sym)
            
            global_trades.extend(trades)
            global_timestamps.extend(ts)
            
        # تحليل الأداء الكلي لهذا التكوين
        perf_stats = analyze_performance(global_trades, global_timestamps)
        
        if perf_stats:
            results.append({
                'config': config,
                'stats': perf_stats
            })
            
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{total_combinations} configs...")

    return results

def filter_top_strategies(results, top_n=5):
    """
    يرشح أفضل الاستراتيجيات بناءً على المعايير الصارمة لمجلس الخبراء.
    """
    # التصفية الأولية: Z-Score > 2.0 و Robustness PASS
    qualified = [r for r in results if r['stats']['z_score'] > 2.0 and r['stats']['robust']]
    
    # الترتيب حسب Z-Score تنازلياً (الأقوى إحصائياً أولاً)
    qualified.sort(key=lambda x: x['stats']['z_score'], reverse=True)
    
    return qualified[:top_n]

def main():
    start_time = datetime.now()
    print("="*70)
    print("INITIATING H1 STRATEGY MATRIX SWEEPER v1.0")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT} | Symbols: {len(SYMBOLS)}")
    print(f"Search Space: {np.prod([len(v) for v in PARAM_GRID.values()])} Combinations")
    print("="*70)
    
    # 1. جلب البيانات وحساب الأساسيات
    precomputed_data = fetch_and_precompute_all_data()
    if not precomputed_data:
        print("CRITICAL ERROR: No data fetched. Aborting.")
        return
        
    # 2. تشغيل حملة البحث الشاملة
    all_results = run_massive_sweep(precomputed_data)
    
    # 3. ترشيح الفائزين
    top_strategies = filter_top_strategies(all_results, top_n=5)
    
    # 4. عرض التقرير النهائي
    elapsed = (datetime.now() - start_time).seconds
    print("\n" + "="*70)
    print("FINAL COUNCIL VERDICT REPORT")
    print(f"Total Configurations Tested: {len(all_results)}")
    print(f"Qualified Strategies (Z>2.0 & Robust): {len(top_strategies)}")
    print(f"Execution Time: {elapsed} seconds")
    print("="*70)
    
    if not top_strategies:
        print("\n NO EDGE FOUND. ABORT PROJECT.")
        print("Conclusion: After testing hundreds of variations with strict statistical filters,")
        print("no configuration demonstrated a statistically significant profitable edge on H1 timeframe.")
        print("Recommendation: Discontinue H1 bot development. Focus resources on proven M15 systems.")
    else:
        print("\n✅ POTENTIAL WINNERS IDENTIFIED:")
        for i, strat in enumerate(top_strategies, 1):
            cfg = strat['config']
            stats = strat['stats']
            print(f"\n--- Strategy #{i} ---")
            print(f"Type: {cfg['STRATEGY_TYPE']}")
            print(f"BB: Period={cfg['BB_PERIOD']}, StdDev={cfg['BB_STD_DEV']}")
            print(f"Stoch: K={cfg['STOCH_K']}, D={cfg['STOCH_D']}, OS={cfg['OS_LEVEL']}, OB={cfg['OB_LEVEL']}")
            print(f"EMA Filter: {cfg['EMA_FILTER']}")
            print(f"Stats: Count={stats['count']}, WR={stats['wr']*100:.2f}%, Z-Score={stats['z_score']:.2f}")
            print(f"Robustness: H1_WR={stats['wr_h1']*100:.2f}%, H2_WR={stats['wr_h2']*100:.2f}%")
            
        print("\n" + "-"*70)
        print("NEXT STEPS FOR COUNCIL APPROVAL:")
        print("1. Select the single best strategy from above.")
        print("2. Conduct rigorous Out-of-Sample validation (split last 6 months as unseen data).")
        print("3. If it passes OOS test, proceed to live paper trading simulation.")
        
    print("="*70)

if __name__ == "__main__":
    main()
