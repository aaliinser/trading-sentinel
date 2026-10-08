#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Extreme Reversal Bot - Layer 1 & 2 Integrated (Final Fix)
الإصلاح الحاسم: تحديث دوال Pandas لتجنب خطأ fillna(method=...).
"""
import os, sys, time, json, logging
from datetime import datetime, timezone
from pathlib import Path
import requests
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

# ═══════════════════════════════════════════════
# 1. الإعدادات العامة (Global Config)
# ═══════════════════════════════════════════════
BOT_NAME = "H1_Stoch_Bot"
STATE_FILE = "state_h1.json"
LOG_LEVEL = logging.INFO

# Telegram Credentials
TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(BOT_NAME)

# ═══════════════════════════════════════════════
# 2. أدوات مساعدة (Utilities)
# ═══════════════════════════════════════════════
def send_telegram(msg: str):
    """إرسال رسالة نصية إلى تليجرام."""
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return None
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("Message sent to Telegram.")
            return True
        else:
            log.error(f"Telegram API Error: {resp.status_code}")
            return False
    except Exception as e:
        log.error(f"Exception sending Telegram message: {e}")
        return False

# ═══════════════════════════════════════════════
# 3. إدارة الحالة (State Management)
# ═══════════════════════════════════════════════
class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = {"pending_trades": [], "history": []}
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    self.data.update(loaded_data)
                log.info("State loaded successfully.")
            except Exception as e:
                log.error(f"Failed to load state: {e}. Starting fresh.")
                self.save()
        else:
            log.info("No existing state file found. Creating new one.")
            self.save()

    def save(self):
        try:
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
            
            if self.filepath.exists():
                self.filepath.unlink()
            temp_file.rename(self.filepath)
            log.debug("State saved successfully.")
        except Exception as e:
            log.critical(f"CRITICAL ERROR saving state: {e}")
            raise

    def add_pending_trade(self, trade_info: dict):
        self.data["pending_trades"].append(trade_info)
        self.save()
        log.info(f"Added pending trade for {trade_info.get('symbol')}")

    def get_pending_trades(self):
        return self.data.get("pending_trades", [])

    def remove_pending_trade(self, trade_id: str):
        self.data["pending_trades"] = [
            t for t in self.data["pending_trades"] 
            if t.get("id") != trade_id
        ]
        self.save()
        log.info(f"Removed resolved trade {trade_id}")

    def archive_trade(self, trade_result: dict):
        self.data["history"].append(trade_result)
        if len(self.data["history"]) > 1000:
            self.data["history"] = self.data["history"][-1000:]
        self.save()

# ═══════════════════════════════════════════════
# 4. محرك البيانات والمؤشرات (Data Engine & Indicators) - LAYER 2 FIXED
# ═══════════════════════════════════════════════

def fetch_h1_data(symbol: str, period_days: int = 7):
    """يجلب بيانات الشموع للساعة الواحدة (H1)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{period_days}d", interval="1h", auto_adjust=False, actions=False)
        
        if df is None or df.empty:
            log.warning(f"No data returned for {symbol}")
            return None
            
        # تنظيف الأعمدة
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
            
        required_cols = ["Open", "High", "Low", "Close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log.error(f"Missing columns for {symbol}: {missing}")
            return None
            
        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated()].sort_index().dropna()
        
        # إزالة الشمعة الحالية غير المكتملة
        now_utc = pd.Timestamp.now(tz="UTC")
        if not df.empty and df.index[-1] + pd.Timedelta(hours=1) > now_utc:
             df = df.iloc[:-1]
             
        return df
    except Exception as e:
        log.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame):
    """
    يحسب EMA200 و Stochastic (%K, %D).
    ★★ الإصلاح هنا: استبدال .fillna(method='ffill') بـ .ffill() ★★
    """
    if df is None or len(df) < 200:
        return df
        
    close_prices = df["Close"]
    
    # 1. حساب EMA 200
    df["EMA_200"] = close_prices.ewm(span=200, adjust=False).mean()
    
    # 2. حساب Stochastic Oscillator (K=9, D=5)
    k_period = 9
    d_period = 5
    
    lowest_low = df["Low"].rolling(window=k_period).min()
    highest_high = df["High"].rolling(window=k_period).max()
    
    denom = highest_high - lowest_low
    denom.replace(0, np.nan, inplace=True) 
    
    raw_k = 100 * ((close_prices - lowest_low) / denom)
    
    # ★★★ التعديل الجوهري لحل المشكلة ★★★
    # بدلاً من: raw_k.fillna(method='ffill', inplace=True)
    # نستخدم:
    raw_k = raw_k.ffill()
    raw_k = raw_k.fillna(50) # تعبئة الباقي بقيمة محايدة
    
    df["STOCH_K"] = raw_k
    
    stoch_d_raw = raw_k.rolling(window=d_period).mean()
    # بدلاً من: stoch_d_raw.fillna(method='ffill', inplace=True)
    df["STOCH_D"] = stoch_d_raw.ffill()
    
    return df

def test_layer_2():
    """دالة اختبار سريعة للتأكد من عمل المحرك."""
    test_symbol = "EURUSD=X"
    log.info(f"Testing Layer 2 with symbol: {test_symbol}")
    
    # جلب بيانات كافية (15 يوم = ~360 شمعة) لضمان حساب EMA200
    df_raw = fetch_h1_data(test_symbol, period_days=15) 
    
    if df_raw is None:
        log.error("Failed to fetch raw data.")
        return False
        
    log.info(f"Fetched {len(df_raw)} candles.")
    
    df_processed = calculate_indicators(df_raw)
    if df_processed is None or "EMA_200" not in df_processed.columns:
        log.error("Failed to calculate indicators.")
        return False
        
    last_row = df_processed.iloc[-1]
    log.info("--- Sample Calculation Results ---")
    log.info(f"Candle Time: {last_row.name}")
    log.info(f"Close Price: {last_row['Close']:.5f}")
    log.info(f"EMA 200:     {last_row['EMA_200']:.5f}")
    log.info(f"Stoch %K:    {last_row['STOCH_K']:.2f}")
    log.info(f"Stoch %D:    {last_row['STOCH_D']:.2f}")
    
    trend_up = last_row['Close'] > last_row['EMA_200']
    log.info(f"Trend Direction: {'UP' if trend_up else 'DOWN'}")
    
    return True

# ═══════════════════════════════════════════════
# 5. الحلقة الرئيسية (Main Loop)
# ═══════════════════════════════════════════════
def main():
    log.info("="*50)
    log.info(f"Starting {BOT_NAME} - Final Integration Test")
    log.info("="*50)

    sm = StateManager(STATE_FILE)
    
    msg_l1 = f"✅ **{BOT_NAME} Online!**\nLayer 1 & 2 Active.\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    send_telegram(msg_l1)

    log.info(">>> Running Layer 2 Self-Test <<<")
    success = test_layer_2()
    
    if success:
        log.info("✅ Layer 2 Passed.")
        send_telegram("🚀 Data Engine Verified OK!\nReady for Strategy Logic.")
    else:
        log.error("❌ Layer 2 Failed.")
        send_telegram("⚠️ Warning: Data Engine Test Failed. Check Logs.")

    log.info("Cycle Complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped manually by user.")
    except Exception as e:
        log.exception(f"Fatal error occurred: {e}")
        send_telegram(f"🚨 **CRASH ALERT**\nError: `{str(e)[:200]}`")
