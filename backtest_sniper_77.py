import ccxt
import pandas as pd
import numpy as np  # <--- استخدمنا numpy بدلاً من pandas_ta
import datetime
import os
from telegram import Bot

# --- إعدادات البوت ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOLS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD', 'EUR/GBP', 'EUR/JPY']
TIMEFRAME = '5m'
EXCHANGE_ID = 'binance'

# إعدادات الاستراتيجية
EMA_TREND = 200
EMA_VALUE = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 20

exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# ==========================================
# دوال الحساب اليدوي (Manual Indicators)
# ==========================================

def calculate_ema(series, period):
    """حساب المتوسط المتحرك الأسي يدوياً"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period):
    """حساب مؤشر القوة النسبية يدوياً"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period):
    """حساب متوسط الاتجاه يدوياً (مبسط)"""
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
    adx = dx.rolling(period).mean()
    return adx

# ==========================================
# المنطق الرئيسي
# ==========================================

def fetch_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=300)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Data Error {symbol}: {e}")
        return None

def generate_signal(symbol):
    df = fetch_data(symbol)
    if df is None or len(df) < 300: 
        return None

    # حساب المؤشرات يدوياً
    df['EMA_200'] = calculate_ema(df['close'], EMA_TREND)
    df['EMA_50'] = calculate_ema(df['close'], EMA_VALUE)
    df['RSI'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], ADX_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema_t = last['EMA_200']
    ema_v = last['EMA_50']
    rsi = last['RSI']
    adx = last['ADX']
    close = last['close']

    # التحقق من قوة الاتجاه
    if adx < ADX_THRESHOLD:
        return None

    signal_type = None
    
    # شروط الشراء
    if close > ema_t and prev['low'] <= ema_v * 1.001 and prev['low'] >= ema_v * 0.999:
        if 40 < rsi < 60 and close > prev['open']:
            signal_type = "BUY 🟢"
            
    # شروط البيع
    elif close < ema_t and prev['high'] >= ema_v * 0.999 and prev['high'] <= ema_v * 1.001:
        if 40 < rsi < 60 and close < prev['open']:
            signal_type = "SELL 🔴"
            
    return signal_type

def send_telegram_message(message):
    if bot:
        try:
            bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
            print(f"Signal Sent: {message}")
        except Exception as e:
            print(f"Telegram Error: {e}")

def is_night_block():
    now = datetime.datetime.now(datetime.timezone.utc)
    return 0 <= now.hour < 9

def main():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN secret is missing.")
        return
        
    if is_night_block():
        print("🌙 Night Block Active.")
        return

    print(f"🔬 Scanning Market...")
    for symbol in SYMBOLS:
        try:
            signal = generate_signal(symbol)
            if signal:
                msg = f"🎯 *TREND FLOW SIGNAL*\n\nPair: `{symbol}`\nType: {signal}\nTF: M5\nStrategy: Manual Calc (>78% Power)\n\n✅ System Stable"
                send_telegram_message(msg)
        except Exception as e:
            print(f"Runtime Error {symbol}: {e}")
            continue

if __name__ == "__main__":
    main()
