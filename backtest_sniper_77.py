import pandas as pd
import numpy as np
import datetime
import os
import json
import urllib.request

# --- إعدادات الاستراتيجية (Forex Real Pairs) ---
# صيغة الرموز في OANDA تكون هكذا: EUR_USD وليس EUR/USD
SYMBOLS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 
    'AUD_USD', 'NZD_USD', 'USD_CAD', 'EUR_GBP', 'EUR_JPY'
]
TIMEFRAME_GRANULARITY = "M5" # فريم 5 دقائق
CANDLES_LIMIT = 300 # عدد الشموع المطلوبة للحساب

# معاملات المؤشرات
EMA_TREND_PERIOD = 200
EMA_VALUE_PERIOD = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN_STRENGTH = 20

# تهيئة الاتصال بـ OANDA (نستخدم الرابط العام للبيانات المجانية)
BASE_URL = "https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"

def get_oanda_data(instrument):
    """جلب بيانات الشموع من OANDA مجاناً"""
    url = BASE_URL.format(instrument=instrument) + f"?granularity={TIMEFRAME_GRANULARITY}&count={CANDLES_LIMIT}"
    
    try:
        req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode())
        
        candles = data.get('candles', [])
        if not candles:
            return None
            
        df = pd.DataFrame(candles)
        # تحويل الحقول إلى أرقام
        df['mid'] = df['mid'].astype(float)
        df['high'] = df['bid'].apply(lambda x: float(x)).astype(float) # تقريب بسيط
        df['low'] = df['ask'].apply(lambda x: float(x)).astype(float)  # تقريب بسيط
        df['close'] = df['mid'] 
        
        # تنظيف الأعمدة غير الضرورية
        cols_to_keep = ['time', 'open', 'high', 'low', 'close', 'volume']
        # OANDA يعطي open/high/low/close داخل كائنات فرعية أحياناً حسب النسخة، لذا نتأكد من وجودها
        if 'o' in df.columns: df.rename(columns={'o':'open'}, inplace=True)
        if 'h' in df.columns: df.rename(columns={'h':'high'}, inplace=True)
        if 'l' in df.columns: df.rename(columns={'l':'low'}, inplace=True)
        if 'c' in df.columns: df.rename(columns={'c':'close'}, inplace=True)
        if 'v' in df.columns: df.rename(columns={'v':'volume'}, inplace=True)
        
        final_cols = [col for col in cols_to_keep if col in df.columns]
        return df[final_cols].copy()

    except Exception as e:
        print(f"API Error {instrument}: {e}")
        return None

# ==========================================
# دوال الحساب الرياضي (Same Logic as before)
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
    print(f"🔬 SNIPER MOMENTUM TRAP - FOREX REAL EDITION")
    print(f"📅 Time: {datetime.datetime.now(datetime.timezone.utc)}")
    print("="*60)
    
    signals_found = 0
    
    for symbol in SYMBOLS:
        try:
            # جلب البيانات من OANDA
            df = get_oanda_data(symbol)
            
            if df is None or len(df) < 250:
                continue
                
            # حساب المؤشرات
            df['EMA_200'] = get_ema(df['close'], EMA_TREND_PERIOD)
            df['EMA_50'] = get_ema(df['close'], EMA_VALUE_PERIOD)
            df['RSI'] = get_rsi(df['close'], RSI_PERIOD)
            df['ADX'] = get_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
            
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            price = last_row['close']
            ema_trend = last_row['EMA_200']
            ema_value = last_row['EMA_50']
            rsi_val = last_row['RSI']
            adx_val = last_row['ADX']
            
            signal_type = None
            
            # فلتر قوة الاتجاه
            if adx_val >= ADX_MIN_STRENGTH:
                
                # شروط الشراء
                if price > ema_trend and \
                   prev_row['low'] <= ema_value * 1.001 and \
                   prev_row['low'] >= ema_value * 0.999 and \
                   40 < rsi_val < 60 and \
                   price > prev_row['open']:
                    signal_type = "BUY 🟢 STRONG"
                    
                # شروط البيع
                elif price < ema_trend and \
                     prev_row['high'] >= ema_value * 0.999 and \
                     prev_row['high'] <= ema_value * 1.001 and \
                     40 < rsi_val < 60 and \
                     price < prev_row['open']:
                     signal_type = "SELL 🔴 STRONG"
            
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
        print("💡 Market is consolidating or trending strongly without pullback yet.")
    else:
        print(f" SUMMARY: {signals_found} HIGH PROBABILITY SIGNAL(S) DETECTED!")
    print("="*60 + "\n")

if __name__ == "__main__":
    analyze_market()
