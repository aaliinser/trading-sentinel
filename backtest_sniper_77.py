import ccxt
import pandas as pd
import numpy as np
import datetime

# --- إعدادات الاستراتيجية ---
SYMBOLS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD', 'EUR/GBP', 'EUR/JPY']
TIMEFRAME = '5m'
EXCHANGE_ID = 'binance'

# معاملات المؤشرات
EMA_TREND_PERIOD = 200
EMA_VALUE_PERIOD = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN_STRENGTH = 20

exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})

# ==========================================
# دوال الحساب الرياضي (No External Libs)
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
# محرك التحليل الرئيسي
# ==========================================

def analyze_market():
    print("\n" + "="*60)
    print(f"🔬 SNIPER MOMENTUM TRAP - ANALYSIS REPORT")
    print(f"📅 Time: {datetime.datetime.now(datetime.timezone.utc)}")
    print("="*60)
    
    signals_found = 0
    
    for symbol in SYMBOLS:
        try:
            # جلب البيانات
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=300)
            if not ohlcv or len(ohlcv) < 250:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # حساب المؤشرات
            df['EMA_200'] = get_ema(df['close'], EMA_TREND_PERIOD)
            df['EMA_50'] = get_ema(df['close'], EMA_VALUE_PERIOD)
            df['RSI'] = get_rsi(df['close'], RSI_PERIOD)
            df['ADX'] = get_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
            
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # استخراج القيم الحالية
            price = last_row['close']
            ema_trend = last_row['EMA_200']
            ema_value = last_row['EMA_50']
            rsi_val = last_row['RSI']
            adx_val = last_row['ADX']
            
            # منطق الصفقة (Logic Check)
            signal_type = None
            
            # فلتر قوة الاتجاه
            if adx_val >= ADX_MIN_STRENGTH:
                
                # شروط الشراء (Buy Setup)
                # السعر فوق الترند + ارتداد من منطقة القيمة + زخم صحي
                if price > ema_trend and \
                   prev_row['low'] <= ema_value * 1.001 and \
                   prev_row['low'] >= ema_value * 0.999 and \
                   40 < rsi_val < 60 and \
                   price > prev_row['open']:
                    signal_type = "BUY 🟢 STRONG"
                    
                # شروط البيع (Sell Setup)
                # السعر تحت الترند + رفض من منطقة القيمة + زخم صحي
                elif price < ema_trend and \
                     prev_row['high'] >= ema_value * 0.999 and \
                     prev_row['high'] <= ema_value * 1.001 and \
                     40 < rsi_val < 60 and \
                     price < prev_row['open']:
                     signal_type = "SELL 🔴 STRONG"
            
            # عرض النتيجة في اللوج
            status_icon = "✅" if signal_type else "⏳"
            result_text = signal_type if signal_type else "WAITING FOR SETUP..."
            
            print(f"{status_icon} [{symbol}] Price: {price:.5f} | RSI: {rsi_val:.2f} | ADX: {adx_val:.2f}")
            print(f"   -> Signal: {result_text}")
            print("-" * 40)
            
            if signal_type:
                signals_found += 1
                
        except Exception as e:
            print(f"❌ Error scanning {symbol}: {str(e)[:50]}...")
            continue

    print("\n" + "="*60)
    if signals_found == 0:
        print("📊 SUMMARY: No valid sniper entries detected in this cycle.")
        print("💡 The market is either ranging or waiting for a pullback confirmation.")
    else:
        print(f" SUMMARY: {signals_found} HIGH PROBABILITY SIGNAL(S) DETECTED!")
    print("="*60 + "\n")

if __name__ == "__main__":
    analyze_market()
