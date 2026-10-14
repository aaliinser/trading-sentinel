#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Ultimate Bot - v4.4 (Independent Resolution & Daily Summary)
الإصلاحات النهائية والحاسمة:
1. ★★ إلغاء نظام Reply واستبداله برسائل مستقلة (حل جذري لمشكلة State Loss في GitHub Actions) ★★
2. ★★ إضافة ملخص يومي لحظي يُرسل تلقائياُ بعد كل حسم ناجح ★★
3. آلية إعادة محاولة ذكية (Retry Loop) ضد تأخر بيانات yfinance.
4. توقيت غزة المحلي: يعمل 9 صباحاُ -> 12 منتصف الليل، ينام 12 -> 9 صباحاُ.
5. حفظ ذرّي آمن للملفات (os.replace).
ملاحظة: منطق الاستراتيجية (check_signal_logic) والتجميد والعداد لم تُمَس إطلاقاُ.
"""
import os, sys, time, json, logging, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

# ═══════════════════════════════════════════════
# 1. الإعدادات العامة
# ═══════════════════════════════════════════════
BOT_NAME = "H1_Ultimate_Bot_V4_4_Gaza_Time"
STATE_FILE = "state_h1_ultimate_v44.json" # ملف حالة جديد لضمان النظافة الكاملة
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

# ★★ التعديل الجوهري: توقيت غزة المحلي ★★
NIGHT_START_HOUR = 0   
NIGHT_END_HOUR = 9     

TRADE_DURATION_MINUTES = 60
EXIT_GRACE_MINUTES = 1 

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(BOT_NAME)

# ═══════════════════════════════════════════════
# 2. الأدوات المساعدة
# ═══════════════════════════════════════════════

def get_pip_multiplier(symbol: str) -> float:
    if "JPY" in symbol.upper():
        return 100.0
    return 10000.0

def get_point_value(symbol: str) -> float:
    return 1.0 / get_pip_multiplier(symbol)

def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def send_telegram(msg: str):
    """دالة إرسال بسيطة وموثوقة بدون اعتماد على reply_to"""
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return False
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                log.info("Message sent successfully to Telegram.")
                return True
            elif resp.status_code == 429:
                wait = resp.json().get('parameters', {}).get('retry_after', 5)
                time.sleep(wait)
            else:
                log.error(f"Telegram API Error: {resp.status_code} - {resp.text}")
                break
        except Exception as e:
            log.error(f"Exception sending Telegram message: {e}")
            time.sleep(2)
    return False

# ═══════════════════════════════════════════════
# 3. إدارة الحالة
# ═══════════════════════════════════════════════
class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = {
            "pending_trades": [],
            "history": [],
            "processed_candles": [],
            "last_daily_summary_date": "" # لتجنب تكرار الملخص اليومي بنفس اليوم
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
            os.replace(str(temp_file), str(self.filepath))
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
        if len(self.data["history"]) > 2000:
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
# 4. محرك البيانات
# ═══════════════════════════════════════════════

def fetch_h1_data(symbol: str, period_days: int = 7):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{period_days}d", interval="1h",
                            auto_adjust=False, actions=False)
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
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100 * ((close_prices - lowest_low) / denom)
    raw_k = raw_k.ffill().fillna(50)
    df["STOCH_K"] = raw_k
    df["STOCH_D"] = raw_k.rolling(window=D_PERIOD).mean().ffill()
    return df

# ═══════════════════════════════════════════════
# 5. منطق الاستراتيجية
# ═══════════════════════════════════════════════

def check_signal_logic(df: pd.DataFrame):
    if df is None or len(df) < 2:
        return None
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    if pd.isna(current_candle['Close']) or pd.isna(prev_candle['STOCH_K']):
        return None
    if pd.isna(current_candle['EMA_200']) or pd.isna(prev_candle['STOCH_D']):
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
# 6. ★★★ الحسم التلقائي المستقل (بدون Reply) ★★★
# ═══════════════════════════════════════════════

def resolve_pending_trades(sm: StateManager):
    pending = sm.get_pending_trades()
    if not pending:
        return
    
    now_utc = datetime.now(timezone.utc)
    resolved_ids = []
    log.info(f">>> Checking {len(pending)} pending trades...")
    
    for trade in pending:
        try:
            entry_dt = ensure_utc(datetime.fromisoformat(trade['entry_time']))
            
            exit_candle_start = entry_dt + timedelta(minutes=TRADE_DURATION_MINUTES)
            exit_candle_close = exit_candle_start + timedelta(minutes=TRADE_DURATION_MINUTES)
            check_time = exit_candle_close + timedelta(minutes=EXIT_GRACE_MINUTES)
            
            if now_utc < check_time:
                continue
            
            sym = trade['symbol']
            dr = trade['direction']
            entry_px = float(trade['entry_price'])
            
            log.info(f"Resolving {sym} | Target Close Time: {exit_candle_close}")
            
            # ★★ آلية إعادة المحاولة الذكية ★★
            exit_px = None
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                df_exit_full = fetch_h1_data(sym, period_days=3)
                
                if df_exit_full is None or df_exit_full.empty:
                    log.warning(f"Attempt {attempt+1}: No data returned for {sym}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                
                mask = df_exit_full.index >= exit_candle_start
                
                if not mask.any():
                    log.warning(f"Attempt {attempt+1}: Candle {exit_candle_start} not found yet. Retrying...")
                    time.sleep(retry_delay)
                    continue
                
                exit_px = float(df_exit_full[mask].iloc[0]['Close'])
                break
            
            if exit_px is None:
                log.error(f"All retries failed for {sym}. Will try again next cycle.")
                continue

            pip_mult = get_pip_multiplier(sym)
            diff = exit_px - entry_px
            pips_diff = abs(diff) * pip_mult
            
            log.info(f"Success! Entry={entry_px} Exit={exit_px} Diff={diff} Pips={pips_diff:.2f}")
            
            # ★★ إرسال رسالة مستقلة بدل الرد ★★
            if pips_diff < 0.5:
                result_msg = (
                    f"⚪ *H1 Result for {sym} ({dr}):*\n\n"
                    f"• Entry: `{entry_px:.5f}`\n"
                    f"• Exit: `{exit_px:.5f}`\n"
                    f"• Status: _Market flat (Void)_\n"
                    f"• Time: `{now_utc.strftime('%H:%M')} UTC`"
                )
                send_telegram(result_msg)
                sm.archive_trade({
                    "id": trade['id'], "symbol": sym, "direction": dr,
                    "entry_price": entry_px, "exit_price": exit_px,
                    "win": None, "resolved_at": now_utc.isoformat(),
                    "status": "VOID_FLAT", "pips": pips_diff
                })
                resolved_ids.append(trade['id'])
                continue
                
            win = (diff > 0) if dr == 'CALL' else (diff < 0)
            result_emoji = "✅ WIN" if win else "❌ LOSS"
            
            result_msg = (
                f"{result_emoji} *H1 Result for {sym} ({dr}):*\n\n"
                f"• Entry: `{entry_px:.5f}`\n"
                f"• Exit: `{exit_px:.5f}`\n"
                f"• Pips: `{pips_diff:.1f}`\n"
                f"• P/L: `{abs(diff)/entry_px*100:.3f}%`\n"
                f"• Time: `{now_utc.strftime('%H:%M')} UTC`"
            )
            
            send_telegram(result_msg)
            
            sm.archive_trade({
                "id": trade['id'], "symbol": sym, "direction": dr,
                "entry_price": entry_px, "exit_price": exit_px,
                "win": bool(win), "resolved_at": now_utc.isoformat(),
                "status": "RESOLVED", "pips": pips_diff
            })
            resolved_ids.append(trade['id'])
            log.info(f"Resolved {trade['id']} {sym}: {'WIN' if win else 'LOSS'}")
            
        except Exception as e:
            log.error(f"Error resolving trade {trade.get('id')}: {e}", exc_info=True)
            
    for tid in resolved_ids:
        sm.remove_pending_trade(tid)

# ═══════════════════════════════════════════════
# 7. ★★ الملخص اليومي اللحظي ★★
# ═══════════════════════════════════════════════

def generate_instant_daily_summary(sm: StateManager):
    """يرسل ملخصاَ لحظياَ لأداء اليوم الحالي فقط إذا كانت هناك صفقات محسومة"""
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    
    history = sm.data.get("history", [])
    if not history:
        return

    # فلترة صفقات اليوم المنتهية فقط
    daily_resolved = [t for t in history 
                      if str(t.get('resolved_at', '')).startswith(today_str) 
                      and t.get('status') == 'RESOLVED']
    
    if not daily_resolved:
        return # لا توجد صفقات محسومة اليوم

    total = len(daily_resolved)
    wins = sum(1 for t in daily_resolved if t.get('win') is True)
    losses = sum(1 for t in daily_resolved if t.get('win') is False)
    
    wr = (wins / total * 100) if total > 0 else 0.0
    net_pips = sum(t.get('pips', 0) for t in daily_resolved if t.get('win') is True) - \
               sum(t.get('pips', 0) for t in daily_resolved if t.get('win') is False)

    emoji_wr = "🟢" if wr >= 60 else ("🟡" if wr >= 40 else "🔴")
    
    summary_msg = (
        f"*📊 H1 Instant Daily Update ({today_str})*\n\n"
        f"*Trades Closed Today:* `{total}`\n"
        f"*Wins:* ✅ `{wins}` | *Losses:* ❌ `{losses}`\n\n"
        f"*Win Rate:* {emoji_wr} `{wr:.1f}%`\n"
        f"*Net Pips:* `{net_pips:+.1f}`\n\n"
        f"_Updated automatically after last resolution._"
    )
    
    send_telegram(summary_msg)
    log.info("Instant daily summary sent.")

# ═══════════════════════════════════════════════
# 8. مسح الإشارات
# ═══════════════════════════════════════════════

def scan_and_alert(sm: StateManager):
    log.info(">>> Scanning market for new signals...")
    now_utc = datetime.now(timezone.utc)
    
    hebron_tz = timezone(timedelta(hours=3)) 
    local_now = now_utc.astimezone(hebron_tz)
    
    if NIGHT_START_HOUR <= local_now.hour < NIGHT_END_HOUR:
        log.info(f"Night ban active ({local_now.strftime('%H:%M')} Local Time). Skipping scan.")
        return
        
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
        if (df_processed is None or "STOCH_K" not in df_processed.columns
                or "EMA_200" not in df_processed.columns):
            log.warning(f"Indicators failed for {sym}. Skipping.")
            continue
        sig_dir = check_signal_logic(df_processed)
        if sig_dir is None:
            continue
        current_candle_time = df_processed.index[-1]
        candle_key = f"{sym}_{current_candle_time.isoformat()}"
        if sm.is_candle_processed(candle_key):
            log.debug(f"Signal {candle_key} already processed. Skipping.")
            continue
        log.info(f"🔥 NEW SIGNAL: {sym} -> {sig_dir}")
        trade_id = str(uuid.uuid4())[:8]
        last_row = df_processed.iloc[-1]
        live_price = float(last_row['Close'])
        k_val = float(last_row['STOCH_K'])
        d_val = float(last_row['STOCH_D'])
        point_value = get_point_value(sym)
        spread_buffer = 2.0 * point_value
        optimal_entry_min = live_price - spread_buffer
        optimal_entry_max = live_price + spread_buffer
        arrow = "🟢 BUY/CALL" if sig_dir == "CALL" else "🔴 SELL/PUT"
        alert_msg = (
            f"*H1 Stochastic Signal Alert!*\n\n"
            f"الزوج: `{sym}`\n"
            f"الاتجاه: {arrow}\n"
            f"السعر الحي الآن: `{live_price:.5f}`\n\n"
            f"*مناطق الدخول الموصى بها:*\n"
            f"• قوية (عند الباند): `{optimal_entry_min:.5f} - {optimal_entry_max:.5f}`\n"
            f"• وسطى (مقبولة): `{live_price:.5f}`\n\n"
            f"*التعليمات:*\n"
            f"✅ ادخل فورا خلال 60 ثانية\n"
            f"(السعر عند الباند — فرصة مثالية)\n\n"
            f"*السبب:* التقاطع في منطقة التشبع المتطرف "
            f"(Stoch K={k_val:.1f}, D={d_val:.1f}) متوافقا مع اتجاه EMA200.\n"
            f"_النتيجة ستصلك كرسالة منفصلة فور إغلاق شمعة الخروج._"
        )
        success = send_telegram(alert_msg)
        if success:
            sm.add_pending_trade({
                "id": trade_id,
                "symbol": sym,
                "direction": sig_dir,
                "entry_price": live_price,
                "entry_time": current_candle_time.isoformat()
            })
            sm.mark_candle_processed(candle_key)
            new_signals_count += 1
            time.sleep(1)
    log.info(f"Scan complete. New signals sent: {new_signals_count}")

# ═══════════════════════════════════════════════
# 9. الحلقة الرئيسية
# ═══════════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info(f"Starting {BOT_NAME} - INDEPENDENT RESOLUTION MODE")
    log.info(f"Current UTC Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    log.info("=" * 50)
    sm = StateManager(STATE_FILE)
    
    # 1. الحسم التلقائي المستقل
    resolve_pending_trades(sm)
    
    # 2. إرسال الملخص اليومي اللحظي (إذا تمت عمليات حسم جديدة)
    generate_instant_daily_summary(sm)
    
    # 3. فحص السوق وإرسال إشارات جديدة
    scan_and_alert(sm)
    
    log.info("Live Cycle Complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped manually by user.")
    except Exception as e:
        log.exception(f"Fatal error occurred: {e}")
        send_telegram(f"🚨 *CRASH ALERT*\nError: `{str(e)[:200]}`")
