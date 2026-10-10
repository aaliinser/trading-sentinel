#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Ultimate Bot - v3.1 (No Loss Streak Protection)
الإصلاح: إزالة ميزة التوقف الآلي بعد 3 خسائر متتالية.
الميزات النشطة:
- 🌙 حظر ليلي: لا إشارات من 00:00 إلى 09:00 UTC
- ❄️ تبريد لكل زوج: إشارة واحدة حتى حسم النتيجة
- 📊 ملخص يومي عند منتصف الليل
- 📅 ملخص أسبوعي السبت (أيام + أوقات + أزواج + أنماط خسارة)
- 🗓️ ملخص شهري أول كل شهر
- ✨ بدون سقف تنبيهات — لن تضيع إشارة
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
BOT_NAME = "H1_Ultimate_Bot_V3_1"
STATE_FILE = "state_h1_ultimate.json"
LOG_LEVEL = logging.INFO

TRADING_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", 
    "USDCAD=X", "CHFJPY=X", "NZDUSD=X", "EURCHF=X",
    "GBPJPY=X", "EURGBP=X", "AUDJPY=X", "EURAUD=X", "USDCHF=X"
]

K_PERIOD = 9
D_PERIOD = 5
OVERBOUGHT_ZONE = 90
OVERSOLD_ZONE = 10
EMA_PERIOD = 200

# قواعد الإدارة
NIGHT_START_HOUR = 0    # 12 AM UTC
NIGHT_END_HOUR = 9      # 9 AM UTC

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(BOT_NAME)

# ═══════════════════════════════════════════════
# 2. أدوات مساعدة ذكية (Smart Utilities)
# ═══════════════════════════════════════════════

def get_pip_multiplier(symbol: str) -> float:
    if "JPY" in symbol.upper(): return 0.01
    return 0.0001

def send_telegram(msg: str, reply_to=None):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return None
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
        
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            mid = resp.json().get("result", {}).get("message_id")
            log.info(f"Message sent to Telegram (ID: {mid}).")
            return mid
        else:
            log.error(f"Telegram API Error: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        log.error(f"Exception sending Telegram message: {e}")
        return None

# ═══════════════════════════════════════════════
# 3. إدارة الحالة الآمنة والمتقدمة (Advanced State Management)
# ═══════════════════════════════════════════════
class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        # هيكل البيانات (تمت إزالة cooldown_until لأنها لم تعد مستخدمة)
        self.data = {
            "pending_trades": [], 
            "history": [], 
            "processed_candles": [],
            "last_daily_report_date": "", # تاريخ آخر تقرير يومي (YYYY-MM-DD)
            "last_weekly_report_stamp": "", # بصمة الأسبوع الأخير (Year-Week)
            "last_monthly_report_stamp": "" # بصمة الشهر الأخير (Year-Month)
        }
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
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
        if len(self.data["history"]) > 2000: # زيادة السعة للتقارير الشهرية
            self.data["history"] = self.data["history"][-2000:]
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
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{period_days}d", interval="1h", auto_adjust=False, actions=False)
        
        if df is None or df.empty:
            log.warning(f"No data returned for {symbol}")
            return None
            
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
        
        now_utc = pd.Timestamp.now(tz="UTC")
        if not df.empty and df.index[-1] + pd.Timedelta(minutes=60, seconds=30) > now_utc:
             df = df.iloc[:-1]
             
        return df
    except Exception as e:
        log.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame):
    if df is None or len(df) < 200:
        return df
        
    close_prices = df["Close"]
    df["EMA_200"] = close_prices.ewm(span=EMA_PERIOD, adjust=False).mean()
    
    lowest_low = df["Low"].rolling(window=K_PERIOD).min()
    highest_high = df["High"].rolling(window=K_PERIOD).max()
    
    denom = highest_high - lowest_low
    denom.replace(0, np.nan, inplace=True) 
    
    raw_k = 100 * ((close_prices - lowest_low) / denom)
    raw_k = raw_k.ffill()
    raw_k = raw_k.fillna(50) 
    
    df["STOCH_K"] = raw_k
    
    stoch_d_raw = raw_k.rolling(window=D_PERIOD).mean()
    df["STOCH_D"] = stoch_d_raw.ffill()
    
    return df

# ═══════════════════════════════════════════════
# 5. منطق الاستراتيجية الصارم (Strategy Logic)
# ═══════════════════════════════════════════════

def check_signal_logic(df: pd.DataFrame):
    if df is None or len(df) < 2:
        return None
        
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    if pd.isna(current_candle['Close']) or pd.isna(prev_candle['STOCH_K']):
        return None

    close_price = float(current_candle['Close'])
    ema_200 = float(current_candle['EMA_200'])
    
    k_curr = float(current_candle['STOCH_K'])
    d_curr = float(current_candle['STOCH_D'])
    
    k_prev = float(prev_candle['STOCH_K'])
    d_prev = float(prev_candle['STOCH_D'])
    
    signal_direction = None
    
    trend_is_bullish = close_price > ema_200
    crossover_up = (k_prev <= d_prev) and (k_curr > d_curr)
    oversold_zone_confirmed = min(k_prev, k_curr) < OVERSOLD_ZONE 
    
    if trend_is_bullish and crossover_up and oversold_zone_confirmed:
        signal_direction = "CALL"
        
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
            
            if now_utc < expiry_dt + timedelta(minutes=2):
                continue
                
            sym = trade['symbol']
            dr = trade['direction']
            entry_px = float(trade['entry_price'])
            
            df_exit_full = fetch_h1_data(sym, period_days=2) 
            
            exit_px = None
            
            if df_exit_full is not None and not df_exit_full.empty:
                target_mask = df_exit_full.index >= expiry_dt
                
                if target_mask.any():
                    candidate_candle = df_exit_full[target_mask].iloc[0]
                    exit_px = float(candidate_candle['Close'])
                    
                    if abs(exit_px - entry_px) < 0.00001: 
                         idx_pos = df_exit_full.index.get_loc(candidate_candle.name)
                         next_pos = idx_pos + 1
                         if next_pos < len(df_exit_full):
                             exit_px = float(df_exit_full.iloc[next_pos]['Close'])
                else:
                    exit_px = float(df_exit_full.iloc[-1]['Close'])
                    
            if exit_px is None:
                log.warning(f"Could not fetch valid exit price for {sym}. Will retry next cycle.")
                continue
                
            diff = exit_px - entry_px
            
            if abs(diff) < 0.00001:
                 result_emoji = "⚪ TIE (Ignored)"
                 win_status = None
                 send_telegram(f"⚪ *Result for {sym} ({dr}):*\n• Entry & Exit identical.\n_Treated as Void._")
                 trade_result = {
                    "id": trade['id'], "symbol": sym, "direction": dr,
                    "entry_price": entry_px, "exit_price": exit_px,
                    "win": None, 
                    "resolved_at": now_utc.isoformat(),
                    "status": "VOID_DATA_LAG"
                 }
                 sm.archive_trade(trade_result)
                 resolved_ids.append(trade['id'])
                 continue

            win = (diff > 0) if dr == 'CALL' else (diff < 0)
            result_emoji = "✅ WIN" if win else "❌ LOSS"
            
            rid = int(trade['telegram_message_id'])
            result_msg = (f"{result_emoji} *Result for {sym} ({dr}):*\n"
                          f"• Entry: `{entry_px:.5f}`\n"
                          f"• Exit: `{exit_px:.5f}`\n"
                          f"• P/L: `{abs(diff)/entry_px*100:.3f}%`")
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
    """يفحص جميع الأزواج ويرسل إشارات جديدة مع تطبيق قواعد الحماية."""
    log.info(">>> Scanning market for new signals...")
    
    now_utc = datetime.now(timezone.utc)
    
    # 1. فحص الحظر الليلي (Night Ban)
    if NIGHT_START_HOUR <= now_utc.hour < NIGHT_END_HOUR:
        log.info("Night ban active (00:00 - 09:00 UTC). Skipping scan.")
        return

    # ★★ تمت إزالة فحص التبريد (Cooldown) هنا ★★

    open_positions_syms = {t['symbol'] for t in sm.get_pending_trades()}
    new_signals_count = 0
    
    for sym in TRADING_SYMBOLS:
        # 2. تبريد لكل زوج (One signal per pair until resolved)
        if sym in open_positions_syms:
            log.debug(f"Skipping {sym}. Already has an open position.")
            continue
            
        df_raw = fetch_h1_data(sym, period_days=15)
        if df_raw is None or len(df_raw) < 200:
            continue
            
        df_processed = calculate_indicators(df_raw)
        
        if df_processed is None or "STOCH_K" not in df_processed.columns or "EMA_200" not in df_processed.columns:
            log.warning(f"Indicators calculation failed for {sym}. Skipping.")
            continue

        sig_dir = check_signal_logic(df_processed)
        
        if sig_dir is None:
            continue
            
        current_candle_time = df_processed.index[-1]
        candle_key = f"{sym}_{current_candle_time.isoformat()}"
        
        if sm.is_candle_processed(candle_key):
            log.debug(f"Signal for {candle_key} was already processed. Skipping.")
            continue
            
        # === تم العثور على إشارة جديدة! ===
        log.info(f"🔥 NEW SIGNAL DETECTED: {sym} -> {sig_dir}")
        
        trade_id = str(uuid.uuid4())[:8]
        
        last_row = df_processed.iloc[-1]
        live_price = float(last_row['Close'])
        ema_val = float(last_row['EMA_200'])
        k_val = float(last_row['STOCH_K'])
        d_val = float(last_row['STOCH_D'])
        
        pip_mult = get_pip_multiplier(sym)
        spread_buffer = 2 * pip_mult 
        
        optimal_entry_min = live_price - spread_buffer
        optimal_entry_max = live_price + spread_buffer
        
        arrow = "🟢 BUY/CALL" if sig_dir == "CALL" else "🔴 SELL/PUT"
        
        alert_msg = (f"*H1 Stochastic Signal Alert!* \n\n"
                     f"الزوج: `{sym}`\n"
                     f"الاتجاه: {arrow}\n"
                     f"السعر الحي الآن: `{live_price:.5f}`\n\n"
                     f"*مناطق الدخول الموصى بها:*\n"
                     f"• قوية (عند الباند): `{optimal_entry_min:.5f} - {optimal_entry_max:.5f}`\n"
                     f"• وسطى (مقبولة): `{live_price:.5f}`\n\n"
                     f"*التعليمات:*\n"
                     f"✅ ادخل فورا خلال 60 ثانية\n"
                     f"(السعر عند الباند — فرصة مثالية)\n\n"
                     f"*السبب:* التقاطع حدث في منطقة التشبع المتطرف (Stoch K={k_val:.1f}, D={d_val:.1f}) متوافقا مع اتجاه EMA200.\n"
                     f"_النتيجة ستسجل تلقائيا عند الانتهاء._")
                     
        mid_id = send_telegram(alert_msg)
        
        if mid_id:
            trade_info = {
                "id": trade_id,
                "symbol": sym,
                "direction": sig_dir,
                "entry_price": live_price,
                "entry_time": current_candle_time.isoformat(),
                "telegram_message_id": mid_id
            }
            sm.add_pending_trade(trade_info)
            sm.mark_candle_processed(candle_key)
            
            new_signals_count += 1
            time.sleep(1) 
            
    log.info(f"Scan complete. New signals sent: {new_signals_count}")

# ═══════════════════════════════════════════════
# 7. نظام التقارير الذكية (Smart Reporting System)
# ═══════════════════════════════════════════════

def generate_reports(sm: StateManager):
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    year, week_num, _ = now_utc.isocalendar()
    week_stamp = f"{year}-{week_num}"
    month_stamp = f"{year}-{now_utc.month}"
    
    history = sm.data.get("history", [])
    
    # --- 1. الملخص اليومي (Daily Report) ---
    # الشرط: اليوم هو أمس بالنسبة لآخر تقرير، والساعة تجاوزت منتصف الليل (00:00)
    # نرسله مرة واحدة فقط في اليوم
    if now_utc.hour >= 0 and sm.data.get("last_daily_report_date") != today_str:
        # نتحقق إذا كان هناك نشاط بالأمس
        yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        daily_trades = [t for t in history if t['resolved_at'].startswith(yesterday_str)]
        
        if daily_trades:
            wins = sum(1 for t in daily_trades if t.get('win') is True)
            losses = sum(1 for t in daily_trades if t.get('win') is False)
            voids = sum(1 for t in daily_trades if t.get('win') is None)
            total = len(daily_trades)
            wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            
            emoji_wr = "🟢" if wr >= 60 else ("🟡" if wr >= 40 else "🔴")
            
            report_msg = (
                f"*📊 H1 Daily Report ({yesterday_str})*\n\n"
                f"*Total Signals:* `{total}`\n"
                f"*Wins:* ✅ `{wins}`\n"
                f"*Losses:* ❌ `{losses}`\n"
                f"*Void/Tie:* ⚪ `{voids}`\n\n"
                f"*Win Rate:* {emoji_wr} `{wr:.1f}%`\n"
            )
            send_telegram(report_msg)
            sm.data["last_daily_report_date"] = today_str
            sm.save()
            log.info("Daily report sent.")

    # --- 2. الملخص الأسبوعي (Weekly Report) ---
    # الشرط: اليوم هو السبت، والساعة >= 9 صباحاَ، والبصمة مختلفة
    if now_utc.weekday() == 5 and now_utc.hour >= 9 and sm.data.get("last_weekly_report_stamp") != week_stamp:
        seven_days_ago = now_utc - timedelta(days=7)
        weekly_trades = []
        for t in history:
            try:
                resolved_dt = datetime.fromisoformat(t['resolved_at'])
                if resolved_dt >= seven_days_ago:
                    weekly_trades.append(t)
            except:
                continue
        
        if weekly_trades:
            total = len(weekly_trades)
            wins = sum(1 for t in weekly_trades if t.get('win') is True)
            losses = sum(1 for t in weekly_trades if t.get('win') is False)
            voids = sum(1 for t in weekly_trades if t.get('win') is None)
            wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            
            # تحليل أفضل زوج وأسوأ نمط خسارة
            pair_stats = {}
            loss_patterns = {} # لتتبع الأنماط مثل PUT losses in bearish trend etc.
            
            for t in weekly_trades:
                sym = t['symbol']
                if sym not in pair_stats:
                    pair_stats[sym] = {'wins': 0, 'total': 0}
                pair_stats[sym]['total'] += 1
                if t.get('win'):
                    pair_stats[sym]['wins'] += 1
            
            best_pair = ""
            best_wr = 0
            for sym, stats in pair_stats.items():
                s_wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                if s_wr > best_wr:
                    best_wr = s_wr
                    best_pair = sym

            emoji_wr = "🟢" if wr >= 60 else ("🟡" if wr >= 40 else "🔴")
            
            report_msg = (
                f"*📅 H1 Weekly Report (Week {week_num})*\n\n"
                f"*Period:* Last 7 Days\n"
                f"*Total Signals:* `{total}`\n"
                f"*Wins:* ✅ `{wins}`\n"
                f"*Losses:* ❌ `{losses}`\n"
                f"*Void/Tie:* ⚪ `{voids}`\n\n"
                f"*Overall Win Rate:* {emoji_wr} `{wr:.1f}%`\n\n"
                f"*Top Performer:* `{best_pair}` ({best_wr:.1f}% WR)\n"
                f"\n_Next report: Next Saturday 09:00 UTC_"
            )
            send_telegram(report_msg)
            sm.data["last_weekly_report_stamp"] = week_stamp
            sm.save()
            log.info("Weekly report sent.")

    # --- 3. الملخص الشهري (Monthly Report) ---
    # الشرط: أول يوم في الشهر، والساعة >= 9 صباحاَ، والبصمة مختلفة
    if now_utc.day == 1 and now_utc.hour >= 9 and sm.data.get("last_monthly_report_stamp") != month_stamp:
        thirty_days_ago = now_utc - timedelta(days=30)
        monthly_trades = []
        for t in history:
            try:
                resolved_dt = datetime.fromisoformat(t['resolved_at'])
                if resolved_dt >= thirty_days_ago:
                    monthly_trades.append(t)
            except:
                continue
        
        if monthly_trades:
            total = len(monthly_trades)
            wins = sum(1 for t in monthly_trades if t.get('win') is True)
            losses = sum(1 for t in monthly_trades if t.get('win') is False)
            wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            
            report_msg = (
                f"*🗓️ H1 Monthly Report ({month_stamp})*\n\n"
                f"*Period:* Last 30 Days\n"
                f"*Total Signals:* `{total}`\n"
                f"*Wins:* ✅ `{wins}`\n"
                f"*Losses:* ❌ `{losses}`\n\n"
                f"*Monthly Win Rate:* `{wr:.1f}%`\n"
                f"\n_Consistency is key. Keep going!_"
            )
            send_telegram(report_msg)
            sm.data["last_monthly_report_stamp"] = month_stamp
            sm.save()
            log.info("Monthly report sent.")

    # ★★ تمت إزالة قسم تفعيل الحماية من الخسائر المتتالية هنا ★★


def main():
    log.info("="*50)
    log.info(f"Starting {BOT_NAME} - ULTIMATE EDITION (NO LOSS PROTECTION)")
    log.info("="*50)

    sm = StateManager(STATE_FILE)
    
    # 1. حسم الصفقات القديمة أولاَ (لتحديث التاريخ قبل الفحص والحماية)
    resolve_pending_trades(sm)
    
    # 2. توليد التقارير (يتم فحص الشروط داخلياَ)
    generate_reports(sm)
    
    # 3. البحث عن صفقات جديدة وإرسالها (مع تطبيق قواعد الحماية)
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
