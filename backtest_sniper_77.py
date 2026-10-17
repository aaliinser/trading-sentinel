import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# --- إعدادات الاختبار التاريخي (Backtest Configuration) ---
SYMBOLS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 
    'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X', 'EURGBP=X', 'EURJPY=X'
]
INTERVAL = "1h"   # فريم الساعة (الأفضل للاختبار السنوي السريع والمستقر)
PERIOD = "1y"     # مدة الاختبار: سنة كاملة

# معاملات المؤشرات (نفس منطق القناص)
EMA_TREND_PERIOD = 200
EMA_VALUE_PERIOD = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN_STRENGTH = 20

# إعدادات إدارة المخاطر للاختبار
TAKE_PROFIT_PIPS = 30  # هدف الربح (مثلا 30 نقطة/بيپ)
STOP_LOSS_PIPS = 15    # وقف الخسارة (نسبة 1:2 عائد للمخاطرة)
PIP_SIZE = 0.0001      # حجم البيپ (يتغير لليين تلقائياً داخل الكود)

# ==========================================
# دوال الحساب الرياضي
# ==========================================

def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_adx(high, low, close, period):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(period).mean()

# ==========================================
# محرك الاختبار التاريخي (The Backtester)
# ==========================================

def run_backtest_for_symbol(symbol):
    print(f"\n🔄 Starting Backtest for {symbol} ({PERIOD}, {INTERVAL})...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=INTERVAL, period=PERIOD)
        
        if df.empty or len(df) < 300:
            print(f"⚠️ Insufficient data for {symbol}. Skipping.")
            return None
            
        df.columns = [col.lower() for col in df.columns]
        
        # حساب المؤشرات لجميع الشموع التاريخية دفعة واحدة
        df['EMA_200'] = get_ema(df['close'], EMA_TREND_PERIOD)
        df['EMA_50'] = get_ema(df['close'], EMA_VALUE_PERIOD)
        df['RSI'] = get_rsi(df['close'], RSI_PERIOD)
        df['ADX'] = get_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
        
        # تحديد حجم البيپ (اليين الياباني يحتاج 0.01 بدل 0.0001)
        pip_size = 0.01 if 'JPY' in symbol else PIP_SIZE
        tp_distance = TAKE_PROFIT_PIPS * pip_size
        sl_distance = STOP_LOSS_PIPS * pip_size
        
        wins = 0
        losses = 0
        total_trades = 0
        
        # المرور على كل شمعة سابقة لمعرفة هل كانت هناك صفقة؟
        # نبدأ من الشمعة رقم 200 لضمان وجود قيم EMAs صحيحة
        start_index = 200 
        
        for i in range(start_index, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # التحقق من شروط الدخول (نفس منطق القناص بالضبط)
            signal_type = None
            
            if curr['ADX'] >= ADX_MIN_STRENGTH:
                # شروط الشراء
                if curr['close'] > curr['EMA_200'] and \
                   prev['low'] <= curr['EMA_50'] * 1.001 and \
                   prev['low'] >= curr['EMA_50'] * 0.999 and \
                   40 < curr['RSI'] < 60 and \
                   curr['close'] > prev['open']:
                    signal_type = "BUY"
                    
                # شروط البيع
                elif curr['close'] < curr['EMA_200'] and \
                     prev['high'] >= curr['EMA_50'] * 0.999 and \
                     prev['high'] <= curr['EMA_50'] * 1.001 and \
                     40 < curr['RSI'] < 60 and \
                     curr['close'] < prev['open']:
                     signal_type = "SELL"
            
            if signal_type:
                total_trades += 1
                entry_price = curr['close']
                
                # محاكاة نتيجة الصفقة بناءً على الشموع التالية حتى تحقيق TP أو SL
                outcome = check_trade_outcome(df, i, signal_type, entry_price, tp_distance, sl_distance)
                
                if outcome == "WIN":
                    wins += 1
                elif outcome == "LOSS":
                    losses += 1
                # إذا كان DRAW (لم يتحقق أي منهما خلال نافذة زمنية معينة)، نتجاهله أو نحسبه حسب سياستك
                # هنا سنعتبره خسارة بسيطة للحفاظ على الصرامة، أو نتجاهله. لنجعله غير محسوب.
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        result_dict = {
            'Symbol': symbol,
            'Total Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Win Rate (%)': round(win_rate, 2),
            'Status': "✅ PASS (>77%)" if win_rate > 77 else ("⏳ NEUTRAL" if win_rate > 50 else "❌ FAIL")
        }
        
        return result_dict

    except Exception as e:
        print(f"❌ Critical Error in {symbol}: {e}")
        return None

def check_trade_outcome(df, entry_idx, direction, entry_price, tp_dist, sl_dist):
    """يفحص الشموع التالية لمعرفة هل حققت الصفقة ربحاً أم خسارة"""
    max_lookahead = 20 # نفترض أن الصفقة يجب أن تنتهي خلال 20 شمعة (20 ساعة لفريم H1)
    
    for j in range(entry_idx + 1, min(entry_idx + 1 + max_lookahead, len(df))):
        candle = df.iloc[j]
        
        if direction == "BUY":
            take_profit_level = entry_price + tp_dist
            stop_loss_level = entry_price - sl_dist
            
            # تحقق من لمس الوقف أولاً (الأكثر صرامة)
            if candle['low'] <= stop_loss_level:
                return "LOSS"
            # ثم تحقق من الهدف
            if candle['high'] >= take_profit_level:
                return "WIN"
                
        elif direction == "SELL":
            take_profit_level = entry_price - tp_dist
            stop_loss_level = entry_price + sl_dist
            
            if candle['high'] >= stop_loss_level:
                return "LOSS"
            if candle['low'] <= take_profit_level:
                return "WIN"
                
    return "DRAW" # لم يتحقق شيء خلال النافذة الزمنية

def main():
    print("\n" + "="*80)
    print("🔬 SNIPER STRATEGY - FULL YEAR HISTORICAL BACKTEST ENGINE")
    print(f"📅 Period: Last 1 Year | Timeframe: {INTERVAL.upper()}")
    print("="*80)
    
    all_results = []
    
    for symbol in SYMBOLS:
        res = run_backtest_for_symbol(symbol)
        if res:
            all_results.append(res)
            # طباعة النتيجة فوراً لكل زوج
            print(f"{res['Status']} [{res['Symbol']}] Win Rate: {res['Win Rate (%)']}% | Trades: {res['Total Trades']} (W:{res['Wins']}/L:{res['Losses']})")
    
    print("\n" + "="*80)
    print("📊 FINAL COMPREHENSIVE REPORT")
    print("="*80)
    
    if not all_results:
        print("No valid results generated.")
        return
        
    avg_win_rate = sum(r['Win Rate (%)'] for r in all_results) / len(all_results)
    total_all_trades = sum(r['Total Trades'] for r in all_results)
    total_wins = sum(r['Wins'] for r in all_results)
    
    print(f"• Average Win Rate Across All Pairs : {avg_win_rate:.2f}%")
    print(f"• Total Simulated Trades (1 Year)   : {total_all_trades}")
    print(f"• Total Winning Trades              : {total_wins}")
    print("-"*80)
    
    # ترتيب النتائج من الأعلى نجاحاً
    sorted_results = sorted(all_results, key=lambda x: x['Win Rate (%)'], reverse=True)
    
    print(f"{'PAIR':<12} | {'TRADES':<8} | {'WINS':<6} | {'LOSSES':<7} | {'RATE %':<8} | STATUS")
    print("-"*80)
    
    for r in sorted_results:
        status_icon = "🟢" if r['Win Rate (%)'] > 77 else ("🟡" if r['Win Rate (%)'] > 50 else "🔴")
        print(f"{r['Symbol']:<12} | {r['Total Trades']:<8} | {r['Wins']:<6} | {r['Losses']:<7} | {r['Win Rate (%)']:<8} | {status_icon} {r['Status']}")
        
    print("="*80)
    
    best_pair = sorted_results[0]['Symbol']
    best_rate = sorted_results[0]['Win Rate (%)']
    
    if best_rate > 77:
        print(f"🎯 CONCLUSION: Strategy VALIDATED! Best performer is {best_pair} with {best_rate}% success rate.")
        print("💡 Recommendation: Deploy this pair specifically for live trading monitoring.")
    else:
        print(f"⚠️ CONCLUSION: Overall average ({avg_win_rate:.2f}%) did not meet the strict >77% threshold globally.")
        print("💡 Insight: The strategy works better on specific pairs. Focus ONLY on pairs marked 🟢 above.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
