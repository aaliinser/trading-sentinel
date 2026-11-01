#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 BOLLINGER SQUEEZE SNIPER - BACKTEST ENGINE v1.0 (Council Approved Strategy)
الفلسفة: ركوب الموجة عند انفجار الطاقة المختزنة (Breakout)، وليس تخمين القاع/القمة (Reversion).
الإطار الزمني: ساعة واحدة (H1) | مدة العقد: شمعة واحدة (60 دقيقة).
المنطق: 
1. كشف الانضغاط (Squeeze): عرض البولنجر الحالي < 80% من متوسط العرض لآخر 20 شمعة.
2. فلتر الاتجاه (Trend Filter): السعر يجب أن يكون فوق EMA50 للشراء، وتحتها للبيع.
3. الزناد (Trigger): إغلاق الشمعة خارج حدود البولنجر المنضغط.
4. إدارة التكلفة: خصم سبريد واقعي (2 نقطة) من هامش الربح قبل احتساب الفوز.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION: H1 BOLLINGER SQUEEZE SNIPER
# ==========================================

SYMBOLS = [
    "USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X",
    "GBPUSD=X", "USDCAD=X", "AUDNZD=X", "EURGBP=X", "EURUSD=X"
]

TIMEFRAME = "1h"
PERIOD_LIMIT = "2y" 
BREAK_EVEN_WR = 0.5263 # نسبة التعادل لعائد 90%

# --- STRATEGY PARAMETERS ---
BB_PERIOD = 20           # فترة حساب المتوسط والانحراف المعياري للبولنجر
BB_STD_DEV = 2.0         # عدد الانحرافات المعيارية لتحديد الحدود
SQUEEZE_THRESHOLD = 0.80 # عتبة الانضغاط: العرض الحالي يجب أن يكون أقل من 80% من المتوسط التاريخي
EMA_TREND_PERIOD = 50    # فلتر الاتجاه السريع (أنسب للاختراقات من EMA200)
SPREAD_COST_PIPS = 2.0   # تكلفة السبريد الافتراضية لكل صفقة

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def get_data(symbol, period=PERIOD_LIMIT, interval=TIMEFRAME):
    """Fetches OHLCV data with strict gap validation."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty: 
            return None
        
        df = flatten_columns(df)
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols): 
            return None
            
        df = df.dropna(subset=['Close'])
        df.sort_index(inplace=True)
        
        # إزالة التكرارات لضمان استمرارية البيانات
        df = df[~df.index.duplicated(keep='first')]
        
        # التأكد من وجود بيانات كافية لحساب المؤشرات الطويلة
        min_required_len = max(BB_PERIOD * 2, EMA_TREND_PERIOD + 10)
        if len(df) < min_required_len:
            return None
            
        return df[['Open', 'High', 'Low', 'Close']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_squeeze_indicators(df):
    """
    يحسب مؤشرات البولنجر باندز، متوسط العرض (Bandwidth)، وخط الاتجاه EMA50.
    يضيف عموداً منطقياً (Boolean) لكشف حالة "الانضغاط" (Squeeze Active).
    """
    c = df['Close']
    
    # 1. حساب خطوط البولنجر الأساسية
    sma_bb = c.rolling(window=BB_PERIOD).mean()
    std_bb = c.rolling(window=BB_PERIOD).std(ddof=0)
    bb_upper = sma_bb + (BB_STD_DEV * std_bb)
    bb_lower = sma_bb - (BB_STD_DEV * std_bb)
    
    # 2. حساب عرض النطاق الحالي (Current Bandwidth)
    current_width = bb_upper - bb_lower
    
    # 3. حساب متوسط العرض التاريخي لآخر BB_PERIOD شمعة
    avg_width = current_width.rolling(window=BB_PERIOD).mean()
    
    # 4. تحديد حالة الانضغاط (Squeeze Condition)
    # إذا كان العرض الحالي أقل من العتبة المحددة من المتوسط، فنحن في وضع استعداد للانفجار
    squeeze_active = current_width < (avg_width * SQUEEZE_THRESHOLD)
    
    # 5. فلتر الاتجاه (EMA50)
    ema_50 = c.ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    
    return df.assign(
        BB_UPPER=bb_upper,
        BB_LOWER=bb_lower,
        CURRENT_WIDTH=current_width,
        AVG_WIDTH=avg_width,
        IS_SQUEEZED=squeeze_active,
        EMA_50=ema_50
    )

def run_backtest_v1(df, symbol_name): 
    """
    ينفذ قواعد الدخول الصارمة لاستراتيجية الاختراق من الانضغاط.
    """
    trades = []
    timestamps = []
    n = len(df)
    
    # نبدأ الفحص بعد استقرار جميع المؤشرات
    start_idx = max(BB_PERIOD * 2, EMA_TREND_PERIOD) + 1
    
    # تحديد معامل النقاط بناءً على نوع الزوج (JPY أو غيره)
    pip_mult = 100.0 if "JPY" in symbol_name else 10000.0
    
    for i in range(start_idx, n - 1):
        curr_close = df['Close'].iloc[i]
        next_close = df['Close'].iloc[i+1]
        timestamp = df.index[i]
        
        # جلب قيم المؤشرات الحالية
        is_squeezed = df['IS_SQUEEZED'].iloc[i]
        bb_upper = df['BB_UPPER'].iloc[i]
        bb_lower = df['BB_LOWER'].iloc[i]
        ema_50 = df['EMA_50'].iloc[i]
        
        direction = 0 # 0=None, 1=CALL, -1=PUT
        
        # الشرط الأساسي: يجب أن تكون الشمعة الحالية ضمن حالة انضغاط
        if is_squeezed:
            # --- BUY RULE (CALL) ---
            # 1. إغلاق الشمعة فوق الحد العلوي للبولنجر (اختراق صاعد)
            # 2. السعر فوق خط الاتجاه EMA50 (تأكيد الاتجاه العام)
            if curr_close > bb_upper and curr_close > ema_50:
                direction = 1
                
            # --- SELL RULE (PUT) ---
            # 1. إغلاق الشمعة تحت الحد السفلي للبولنجر (اختراق هابط)
            # 2. السعر تحت خط الاتجاه EMA50 (تأكيد الاتجاه العام)
            elif curr_close < bb_lower and curr_close < ema_50:
                direction = -1
        
        # إذا تحقق شرط الدخول، نحسب النتيجة مع مراعاة التكاليف
        if direction != 0:
            price_diff = next_close - curr_close
            
            # تحويل تكلفة السبريد إلى وحدات سعرية
            spread_in_price = SPREAD_COST_PIPS / pip_mult
            
            net_profitable = False
            
            if direction == 1: # CALL
                # هل الحركة الصافية للشمعة التالية غطت تكلفة السبريد؟
                net_profitable = price_diff > spread_in_price
                
            else: # PUT
                # هل الحركة الهابطة الصافية غطت تكلفة السبريد؟
                net_profitable = (-price_diff) > spread_in_price
            
            trades.append(net_profitable)
            timestamps.append(timestamp)
            
    return trades, timestamps

def analyze_results(all_trades, all_timestamps):
    total = len(all_trades)
    if total == 0:
        return {'count': 0, 'wr': 0, 'z_score': 0, 'robust': 'N/A', 'wr_h1': 0, 'wr_h2': 0}
        
    wins = sum(all_trades)
    wr = wins / total
    
    p_null = BREAK_EVEN_WR
    std_dev = np.sqrt(total * p_null * (1 - p_null))
    z_score = (wins - (total * p_null)) / std_dev if std_dev > 0 else 0
    
    mid_point = total // 2
    h1_trades = all_trades[:mid_point]
    h2_trades = all_trades[mid_point:]
    
    wr_h1 = sum(h1_trades) / len(h1_trades) if h1_trades else 0
    wr_h2 = sum(h2_trades) / len(h2_trades) if h2_trades else 0
    
    is_robust = (wr_h1 >= BREAK_EVEN_WR) and (wr_h2 >= BREAK_EVEN_WR)
    robust_status = "PASS ✅" if is_robust else "FAIL ❌"
    
    return {
        'count': total, 
        'wr': wr, 
        'z_score': z_score, 
        'robust': robust_status,
        'wr_h1': wr_h1,
        'wr_h2': wr_h2
    }

def main():
    print("="*70)
    print("VALIDATING: H1 BOLLINGER SQUEEZE SNIPER v1.0")
    print(f"Timeframe: {TIMEFRAME} | Period: {PERIOD_LIMIT} | Symbols: {len(SYMBOLS)}")
    print(f"Squeeze Threshold: {SQUEEZE_THRESHOLD*100}% of Avg Width")
    print(f"Trend Filter: EMA{EMA_TREND_PERIOD}")
    print(f"Cost Model: Spread={SPREAD_COST_PIPS} pips deducted.")
    print("="*70)
    
    global_trades = []
    global_timestamps = []
    
    for sym in SYMBOLS:
        print(f"Processing {sym}...", end=" ")
        df = get_data(sym)
        if df is not None:
            enriched_df = calculate_squeeze_indicators(df)
            trades, ts = run_backtest_v1(enriched_df, sym)
            global_trades.extend(trades)
            global_timestamps.extend(ts)
            print(f"Done ({len(trades)} signals)")
        else:
            print("Skipped (Insufficient Data/Error)")
            
    stats = analyze_results(global_trades, global_timestamps)
    
    print("\n" + "-"*70)
    print("FINAL STATISTICAL REPORT (v1.0 SQUEEZE SIMULATION)")
    print("-"*70)
    print(f"Total Signals Generated: {stats['count']}")
    print(f"Overall Win Rate:        {stats['wr']*100:.2f}%")
    print(f"Z-Score (Significance):  {stats['z_score']:.2f}")
    print(f"Robustness (Split-Samp): {stats['robust']}")
    print(f"  - First Half WR:       {stats['wr_h1']*100:.2f}%")
    print(f"  - Second Half WR:      {stats['wr_h2']*100:.2f}%")
    
    verdict = "FAILING"
    # معايير الحكم لمجلس الخبراء: Z-Score > 1.96 هو الحد الأدنى للدلالة الإحصائية الحقيقية
    if stats['count'] >= 50:
        if stats['wr'] >= 0.55 and stats['z_score'] > 1.96 and stats['robust'] == "PASS ✅":
            verdict = "EXCELLENT 🏆 (Ready for Production)"
        elif stats['wr'] >= BREAK_EVEN_WR and stats['z_score'] > 1.0:
            verdict = "VIABLE ⚠️ (Monitor Closely)"
        else:
            verdict = "FRAGILE ❌ (Do Not Trade Real Money)"
            
    print(f"\nVerdict:                 {verdict}")
    print("="*70)

if __name__ == "__main__":
    main()
