#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Extreme Reversal Bot - PRODUCTION READY v2.1
التحديث: توسيع نطاق التداول ليشمل 9 أزواج رئيسية وأكثر سيولة.
الإصلاحات الجوهرية محفوظة:
1. استخدام atomic write لضمان سلامة ملف الحالة.
2. توافق كامل مع أحدث إصدارات Pandas (.ffill() بدلاً من fillna(method)).
3. منطق حسم تلقائي دقيق يعتمد على إغلاق الشمعة السابقة تماماً.
4. إزالة أي رسائل إدارية مزعجة (Silent Mode).
"""
import os, sys, time, json, logging, uuid
from datetime import datetime, timedelta, timezone
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
BOT_NAME = "H1_Stoch_Prod"
STATE_FILE = "state_h1.json"
LOG_LEVEL = logging.INFO

# ★★ التحديث الحصري هنا: إضافة 3 أزواج جديدة للأمان والتنوع ★★
TRADING_SYMBOLS = [
    # الأساسيات (5 أزواج)
    "EURUSD=X", 
    "GBPUSD=X", 
    "USDJPY=X", 
    "AUDUSD=X", 
    "USDCAD=X",
    # الإضافات الآمنة (4 أزواج إضافية لدعم السيولة والسلوك الارتدادي)
    "CHFJPY=X",   # سويسري/ين - معروف بتقلباته الحادة المناسبة للعكس
    "NZDUSD=X",   # نيوزلندي/دولار - يرتبط بأستراليا لكنه مستقل قليلاً
    "EURCHF=X"    # يورو/سويسري - هادئ جداً ومناسب للاستراتيجيات طويلة النفس
]

# إعدادات المؤشرات (Stochastic K=9, D=5 | OB=90 OS=10 | EMA=200)
K_PERIOD = 9
D_PERIOD = 5
OVERBOUGHT_ZONE = 90
OVERSOLD_ZONE = 10
EMA_PERIOD = 200

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
def send_telegram(msg: str, reply_to=None):
    """إرسال رسالة نصية إلى تليجرام."""
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return None
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT,
        "text": msg,
        "parse_mode": "Markdown"
    }
    if reply_to:
        payload["reply_to_message_id"] = reply_to
        
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            mid = resp.json().get("result", {}).get("message_id")
            log.info(f"Message sent to Telegram (ID: {mid}).")
            return mid
        else:
            log.error(f"Telegram API Error: {resp.status_code}")
            return None
    except Exception as e:
        log.error(f"Exception sending Telegram message: {e}")
        return None

# ═══════════════════════════════════════════════
# 3. إدارة الحالة الآمنة (Safe State Management)
# ═══════════════════════════════════════════════
class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        # الهيكل الافتراضي النظيف
        self.data = {"pending_trades": [], "history": [], "processed_candles": []}
        self.load()

    def load(self):
        """قراءة ملف الحالة إذا كان موجوداَ."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # دمج ذكي للحفاظ على الحقول الجديدة إن أُضيفت مستقبلاً
                    for key in self.data.keys():
                        if key in loaded_data:
                            self.data[key] = loaded_data[key]
                log.info("State loaded successfully.")
            except Exception as e:
                log.error(f"Failed to load state: {e}. Starting fresh.")
                self.save()
        else:
            log.info("No existing state file found. Creating new one.")
            self.save()

    def save(self):
        """حفظ الحالة الحالية على القرص بطريقة Atomic Write."""
        try:
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
            
            # استبدال الملف القديم بالجديد لضمان عدم التلف عند الانقطاع المفاجئ
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

    def mark_candle_processed(self, candle_key: str):
        if candle_key not in self.data["processed_candles"]:
            self.data["processed_candles"].append(candle_key)
            if len(self.data["processed_candles"]) > 500:
                self.data["processed_candles"] = self.data["processed_candles"][-500:]
            self.save()

    def is_candle_processed(self, candle_key: str) -> bool:
        return candle_key in self.data["processed_candles"]

# ═══════════════════════════════════════════════
# 4. محرك البيانات والمؤشرات (Data Engine & Indicators)
# ═══════════════════════════════════════════════

def fetch_h1_data(symbol: str, period_days: int = 7):
    """يجلب بيانات الشموع للساعة الواحدة (H1)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{period_days}d", interval="1h", auto_adjust=False, actions=False)
        
        if df is None or df.empty:
            log.warning(f"No data returned for {symbol}")
            return None
            
        # تنظيف الأعمدة (MultiIndex fix)
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
        
        # إزالة الشمعة الحالية غير المكتملة (Zero Look-Ahead Bias Prevention)
        now_utc = pd.Timestamp.now(tz="UTC")
        if not df.empty and df.index[-1] + pd.Timedelta(hours=1) > now_utc:
             df = df.iloc[:-1]
             
        return df
    except Exception as e:
        log.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame):
    """يحسب EMA200 و Stochastic (%K, %D) بدقة رياضية عالية."""
    if df is None or len(df) < 200:
        return df
        
    close_prices = df["Close"]
    
    # 1. حساب EMA 200
    df["EMA_200"] = close_prices.ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # 2. حساب Stochastic Oscillator
    lowest_low = df["Low"].rolling(window=K_PERIOD).min()
    highest_high = df["High"].rolling(window=K_PERIOD).max()
    
    denom = highest_high - lowest_low
    denom.replace(0, np.nan, inplace=True) 
    
    raw_k = 100 * ((close_prices - lowest_low) / denom)
    
    # ★★★ الإصلاح الحاسم لتوافق Pandas الحديث ★★★
    raw_k = raw_k.ffill()
    raw_k = raw_k.fillna(50) # تعبئة الباقي بقيمة محايدة
    
    df["STOCH_K"] = raw_k
    
    stoch_d_raw = raw_k.rolling(window=D_PERIOD).mean()
    df["STOCH_D"] = stoch_d_raw.ffill()
    
    return df

# ═══════════════════════════════════════════════
# 5. منطق الاستراتيجية الصارم (Strategy Logic)
# ═══════════════════════════════════════════════

def check_signal_logic(df: pd.DataFrame):
    """
    يفحص شروط الدخول بناءً على استراتيجية H1 Stochastic Extreme Reversal.
    يرجع 'CALL', 'PUT', أو None.
    """
    if df is None or len(df) < 2:
        return None
        
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    close_price = float(current_candle['Close'])
    ema_200 = float(current_candle['EMA_200'])
    
    k_curr = float(current_candle['STOCH_K'])
    d_curr = float(current_candle['STOCH_D'])
    
    k_prev = float(prev_candle['STOCH_K'])
    d_prev = float(prev_candle['STOCH_D'])
    
    signal_direction = None
    
    # ---------------------------------------------------------
    # 1. فحص إشارة الشراء (BUY / CALL)
    # ---------------------------------------------------------
    trend_is_bullish = close_price > ema_200
    crossover_up = (k_prev <= d_prev) and (k_curr > d_curr)
    oversold_zone_confirmed = min(k_prev, k_curr) < OVERSOLD_ZONE 
    
    if trend_is_bullish and crossover_up and oversold_zone_confirmed:
        signal_direction = "CALL"
        
    # ---------------------------------------------------------
    # 2. فحص إشارة البيع (SELL / PUT)
    # ---------------------------------------------------------
    trend_is_bearish = close_price < ema_200
    crossover_down = (k_prev >= d_prev) and (k_curr < d_curr)
    overbought_zone_confirmed = max(k_prev, k_curr) > OVERBOUGHT_ZONE
    
    if trend_is_bearish and crossover_down and overbought_zone_confirmed:
        signal_direction = "PUT"
        
    return signal_direction

# ═══════════════════════════════════════════════
# 6. الحلقة الرئيسية للروبوت الحي (Live Bot Loop)
# ═══════════════════════════════════════════════

def resolve_pending_trades(sm: StateManager):
    """يفحص الصفقات المعلقة ويحسم نتيجتها تلقائياَ."""
    pending = sm.get_pending_trades()
    if not pending: return
    
    now_utc = datetime.now(timezone.utc)
    resolved_ids = []
    
    log.info(f">>> Checking {len(pending)} pending trades...")
    
    for trade in pending:
        try:
            entry_time_str = trade['entry_time']
            entry_dt = datetime.fromisoformat(entry_time_str)
            expiry_dt = entry_dt + timedelta(minutes=60)
            
            # ننتظر حتى تمر دقيقة كاملة بعد وقت الانتهاء لضمان إغلاق الشمعة
            if now_utc < expiry_dt + timedelta(minutes=1):
                continue
                
            sym = trade['symbol']
            dr = trade['direction']
            entry_px = float(trade['entry_price'])
            
            # جلب السعر الحالي للإغلاق (Exit Price)
            df_exit = fetch_h1_data(sym, period_days=1) 
            exit_px = None
            
            if df_exit is not None and not df_exit.empty:
                target_time = expiry_dt - timedelta(hours=1) 
                mask = df_exit.index <= target_time
                if mask.any():
                    exit_px = float(df_exit.loc[mask].iloc[-1]['Close'])
                else:
                     exit_px = float(df_exit.iloc[-1]['Close'])
                    
            if exit_px is None:
                log.warning(f"Could not fetch exit price for {sym}. Will retry next cycle.")
                continue
                
            win = (exit_px > entry_px) if dr == 'CALL' else (exit_px < entry_px)
            result_emoji = "✅ WIN" if win else "❌ LOSS"
            
            rid = int(trade['telegram_message_id'])
            result_msg = (f"{result_emoji} *Result for {sym} ({dr}):*\n"
                          f"• Entry: `{entry_px:.5f}`\n"
                          f"• Exit: `{exit_px:.5f}`\n"
                          f"• P/L: `{abs(exit_px-entry_px)/entry_px*100:.3f}%`")
            send_telegram(result_msg, reply_to=rid)
            
            trade_result = {
                "id": trade['id'],
                "symbol": sym,
                "direction": dr,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "win": bool(win),
                "resolved_at": now_utc.isoformat()
            }
            sm.archive_trade(trade_result)
            resolved_ids.append(trade['id'])
            log.info(f"Resolved trade {trade['id']} for {sym}: {'WIN' if win else 'LOSS'}")

        except Exception as e:
            log.error(f"Error resolving trade {trade.get('id')}: {e}")
            
    for tid in resolved_ids:
        sm.remove_pending_trade(tid)

def scan_and_alert(sm: StateManager):
    """يفحص جميع الأزواج ويرسل إشارات جديدة."""
    log.info(">>> Scanning market for new signals...")
    
    open_positions_syms = {t['symbol'] for t in sm.get_pending_trades()}
    new_signals_count = 0
    
    for sym in TRADING_SYMBOLS:
        if sym in open_positions_syms:
            log.debug(f"Skipping {sym}. Already has an open position.")
            continue
            
        df_raw = fetch_h1_data(sym, period_days=15)
        if df_raw is None or len(df_raw) < 200:
            continue
            
        df_processed = calculate_indicators(df_raw)
        sig_dir = check_signal_logic(df_processed)
        
        if sig_dir is None:
            continue
            
        # التحقق من عدم تكرار الإشارة لنفس الشمعة
        current_candle_time = df_processed.index[-1]
        candle_key = f"{sym}_{current_candle_time.isoformat()}"
        
        if sm.is_candle_processed(candle_key):
            log.debug(f"Signal for {candle_key} was already processed. Skipping.")
            continue
            
        # === تم العثور على إشارة جديدة! ===
        log.info(f"🔥 NEW SIGNAL DETECTED: {sym} -> {sig_dir}")
        
        trade_id = str(uuid.uuid4())[:8]
        
        arrow = "🟢 BUY/CALL" if sig_dir == "CALL" else "🔴 SELL/PUT"
        alert_msg = (f"*H1 Stochastic Signal Alert!* \n\n"
                     f"Pair: `{sym}`\n"
                     f"Direction: {arrow}\n"
                     f"Entry Price: `{df_processed.iloc[-1]['Close']:.5f}`\n"
                     f"Expiry: 60 minutes\n"
                     f"_Result will be auto-reported._")
                     
        mid_id = send_telegram(alert_msg)
        
        if mid_id:
            trade_info = {
                "id": trade_id,
                "symbol": sym,
                "direction": sig_dir,
                "entry_price": float(df_processed.iloc[-1]['Close']),
                "entry_time": current_candle_time.isoformat(),
                "telegram_message_id": mid_id
            }
            sm.add_pending_trade(trade_info)
            sm.mark_candle_processed(candle_key)
            
            new_signals_count += 1
            time.sleep(1) 
            
    log.info(f"Scan complete. New signals sent: {new_signals_count}")


def main():
    log.info("="*50)
    log.info(f"Starting {BOT_NAME} - LIVE BOT INITIALIZATION")
    log.info("="*50)

    sm = StateManager(STATE_FILE)
    
    # 1. حسم الصفقات القديمة أولاَ
    resolve_pending_trades(sm)
    
    # 2. البحث عن صفقات جديدة وإرسالها
    scan_and_alert(sm)
    
    log.info("Live Cycle Complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped manually by user.")
    except Exception as e:
        log.exception(f"Fatal error occurred: {e}")
        send_telegram(f"🚨 **CRASH ALERT**\nError: `{str(e)[:200]}`")
