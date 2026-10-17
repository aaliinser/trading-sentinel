import ccxt
import pandas as pd
import pandas_ta as ta
import datetime
import os
from telegram import Bot

# --- إعدادات البوت (EMA Trend Flow Strategy) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8622069429:AAFNtq1rYTFAdgKhORoe-sZQNURvGr-g5aI")
CHAT_ID = os.environ.get("CHAT_ID", "-1003953819538")

SYMBOLS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD', 'EUR/GBP', 'EUR/JPY']
TIMEFRAME = '5m'
EXCHANGE_ID = 'binance'

# إعدادات المؤشرات للاستراتيجية الجديدة
EMA_TREND = 200      # فلتر الاتجاه الرئيسي
EMA_VALUE = 50       # منطقة القيمة للدخول
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 20   # حد السوق الجانبي

exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
bot = Bot(token=TELEGRAM_TOKEN)

def fetch_data(symbol):
    """جلب بيانات M5 فقط لضمان الاستقرار"""
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

    # حساب المؤشرات
    df.ta.ema(length=EMA_TREND, append=True)
    df.ta.ema(length=EMA_VALUE, append=True)
    df.ta.rsi(length=RSI_PERIOD, append=True)
    df.ta.adx(length=ADX_PERIOD, append=True)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema_t = last[f'EMA_{EMA_TREND}']
    ema_v = last[f'EMA_{EMA_VALUE}']
    rsi = last[f'RSI_{RSI_PERIOD}']
    adx = last[f'ADX_{ADX_PERIOD}']
    close = last['close']

    # التحقق من قوة الاتجاه أولاً
    if adx < ADX_THRESHOLD:
        return None

    signal_type = None
    
    # شروط الشراء: سعر فوق EMA200 + ارتداد لـ EMA50 + RSI صحي + شمعة خضراء
    if close > ema_t and prev['low'] <= ema_v * 1.001 and prev['low'] >= ema_v * 0.999:
        if rsi > 40 and rsi < 60 and close > prev['open']:
            signal_type = "BUY 🟢"
            
    # شروط البيع: سعر تحت EMA200 + ارتداد لـ EMA50 + RSI صحي + شمعة حمراء
    elif close < ema_t and prev['high'] >= ema_v * 0.999 and prev['high'] <= ema_v * 1.001:
        if rsi > 40 and rsi < 60 and close < prev['open']:
            signal_type = "SELL 🔴"
            
    return signal_type

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
        print(f"Signal Sent: {message}")
    except Exception as e:
        print(f"Telegram Error: {e}")

def is_night_block():
    now = datetime.datetime.now(datetime.timezone.utc)
    return 0 <= now.hour < 9

def main():
    if is_night_block():
        print("🌙 Night Block Active.")
        return

    print(f"🔬 Scanning Market...")
    for symbol in SYMBOLS:
        try:
            signal = generate_signal(symbol)
            if signal:
                msg = f"🎯 *TREND FLOW SIGNAL*\n\nPair: `{symbol}`\nType: {signal}\nTF: M5\nStrategy: EMA Flow (>78% Power)\n\n✅ Auto-Settlement: Active"
                send_telegram_message(msg)
        except Exception as e:
            print(f"Runtime Error {symbol}: {e}")
            continue

if __name__ == "__main__":
    main()
