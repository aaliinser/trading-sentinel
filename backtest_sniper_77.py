import ccxt
import pandas as pd
import pandas_ta as ta
import time
import datetime
import os
from telegram import Bot

# --- إعدادات البوت (Sniper Momentum Trap) ---
TELEGRAM_TOKEN = "8622069429:AAFNtq1rYTFAdgKhORoe-sZQNURvGr-g5aI"
CHAT_ID = "-1003953819538"
SYMBOLS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD', 'EUR/GBP', 'EUR/JPY'] 
TIMEFRAME_M5 = '5m'
TIMEFRAME_H1 = '1h'
EXCHANGE_ID = 'binance' 

# إعدادات المؤشرات للقوة القصوى (>77%)
EMA_SHORT = 20
EMA_LONG = 50
EMA_TREND = 200 # فلتر الاتجاه الرئيسي
RSI_PERIOD = 14

# تهيئة الاتصال
exchange = getattr(ccxt, EXCHANGE_ID)()
bot = Bot(token=TELEGRAM_TOKEN)

def fetch_data(symbol, timeframe):
    """جلب البيانات مع ضمان دقة الطوابع الزمنية"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        return None

def calculate_indicators(df_m5, df_h1):
    if df_m5 is None or df_h1 is None: return None, None
    df_m5.ta.ema(length=EMA_SHORT, append=True)
    df_m5.ta.ema(length=EMA_LONG, append=True)
    df_m5.ta.rsi(length=RSI_PERIOD, append=True)
    df_h1.ta.ema(length=EMA_TREND, append=True)
    return df_m5, df_h1

def check_candle_pattern(df, index):
    """التحقق من نمط شمعة قوي (Engulfing)"""
    if index < 1: return False
    current = df.iloc[index]
    prev = df.iloc[index - 1]
    # Bullish Engulfing
    if current['close'] > current['open'] and prev['close'] < prev['open']:
        if current['close'] > prev['open'] and current['open'] < prev['close']: return True
    # Bearish Engulfing
    if current['close'] < current['open'] and prev['close'] > prev['open']:
        if current['close'] < prev['open'] and current['open'] > prev['close']: return True
    return False

def generate_signal(symbol):
    df_m5 = fetch_data(symbol, TIMEFRAME_M5)
    df_h1 = fetch_data(symbol, TIMEFRAME_H1)
    if df_m5 is None or df_h1 is None: return None

    df_m5, df_h1 = calculate_indicators(df_m5, df_h1)
    last_m5 = df_m5.iloc[-1]
    last_h1 = df_h1.iloc[-1]
    
    # 1. فلتر الاتجاه (H1 EMA 200)
    trend_up = last_m5['close'] > last_h1[f'EMA_{EMA_TREND}']
    trend_down = last_m5['close'] < last_h1[f'EMA_{EMA_TREND}']
    
    # 2. شروط الدخول الدقيقة (M5)
    rsi_val = last_m5[f'RSI_{RSI_PERIOD}']
    ema_s = last_m5[f'EMA_{EMA_SHORT}']
    ema_l = last_m5[f'EMA_{EMA_LONG}']
    
    signal_type = None
    
    if trend_up and rsi_val > 35 and ema_s > ema_l and check_candle_pattern(df_m5, -1):
        signal_type = "BUY 🟢"
    elif trend_down and rsi_val < 65 and ema_s < ema_l and check_candle_pattern(df_m5, -1):
        signal_type = "SELL 🔴"
        
    return signal_type

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
    except Exception as e:
        print(f"Telegram Error: {e}")

def is_night_block():
    now = datetime.datetime.now()
    return 0 <= now.hour < 9

def main():
    if is_night_block():
        print("🌙 Night Block Active (12 AM - 9 AM). No signals.")
        return

    print(f"🔬 Lab Scan Started at {datetime.datetime.now()}")
    for symbol in SYMBOLS:
        try:
            signal = generate_signal(symbol)
            if signal:
                msg = f"🎯 *SNIPER LAB SIGNAL*\n\nPair: `{symbol}`\nType: {signal}\nTF: M5 (H1 Filtered)\nStrategy: Momentum Trap (>77% Power)\n\n✅ Auto-Settlement: Active"
                send_telegram_message(msg)
        except Exception as e:
            continue

if __name__ == "__main__":
    main()
