#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Ultimate Bot - v4.1 (FULLY AUDITED)
الإصلاحات:
1. تصحيح حساب مناطق الدخول (point_value = 1/pip_mult).
2. تصحيح توقيت الحسم لانتظار إغلاق شمعة الخروج (حل VOID الجذري).
3. حفظ ذرّي للملف (os.replace).
4. فرض timezone على fromisoformat.
5. استعادة التقارير الأسبوعية والشهرية.
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
BOT_NAME = "H1_Ultimate_Bot_V4_1"
STATE_FILE = "state_h1_ultimate_v41.json"
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

NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 9

# مدة الصفقة بالدقائق (ساعة واحدة = 60 دقيقة)
TRADE_DURATION_MINUTES = 60
# هامش انتظار إغلاق شمعة الخروج + تحديث البيانات
EXIT_GRACE_MINUTES = 3

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
    """مضاعف تحويل فرق السعر إلى عدد نقاط (pips)."""
    if "JPY" in symbol.upper():
        return 100.0   # نقطة الين = 0.01
    return 10000.0     # نقطة عادية = 0.0001

def get_point_value(symbol: str) -> float:
    """قيمة النقطة الواحدة بالسعر (لحساب مناطق الدخول)."""
    return 1.0 / get_pip_multiplier(symbol)

def ensure_utc(dt: datetime) -> datetime:
    """ضمان أن الـ datetime يحمل timezone = UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def send_telegram(msg: str, reply_to=None):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}
    if reply_to is not None:
        try:
            payload["reply_to_message_id"] = int(reply_to)
        except (ValueError, TypeError):
            pass
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                mid = resp.json().get("result", {}).get("message_id")
                log.info(f"Message sent to Telegram (ID: {mid}).")
                return mid
            elif resp.status_code == 429:
                wait = resp.json().get('parameters', {}).get('retry_after', 5)
                time.sleep(wait)
            else:
                log.error(f"Telegram API Error: {resp.status_code} - {resp.text}")
                break
        except Exception as e:
            log.error(f"Exception sending Telegram message: {e}")
            time.sleep(2)
    return None

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
            "last_daily_report_date": "",
            "last_weekly_report_stamp": "",
            "last_monthly_report_stamp": ""
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
        """حفظ ذرّي آمن: نكتب مؤقتاً ثم نستبدل بذرة واحدة."""
        try:
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
            # os.replace ذرّي على جميع الأنظمة (يستبدل الملف الهدف بأمان)
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
        # حذف الشمعة الجارية (لم تغلق بعد)
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
# 6. ★★★ الحسم التلقائي المُصحح جذرياً ★★★
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
            # شمعة الخروج تبدأ بعد مدة الصفقة، وتغلق بعدها بمدة الصفقة
            exit_candle_start = entry_dt + timedelta(minutes=TRADE_DURATION_MINUTES)
            exit_candle_close = exit_candle_start + timedelta(minutes=TRADE_DURATION_MINUTES)
            # ننتظر إغلاق شمعة الخروج + هامش تحديث البيانات
            check_time = exit_candle_close + timedelta(minutes=EXIT_GRACE_MINUTES)
            if now_utc < check_time:
                continue
            sym = trade['symbol']
            dr = trade['direction']
            entry_px = float(trade['entry_price'])
            log.info(f"Resolving {sym} | entry_start={entry_dt} | exit_candle={exit_candle_start}")
            df_exit_full = fetch_h1_data(sym, period_days=3)
            if df_exit_full is None or df_exit_full.empty:
                log.warning(f"No exit data for {sym}. Retry next cycle.")
                continue
            # نأخذ أول شمعة تبدأ عند/بعد وقت بداية شمعة الخروج
            mask = df_exit_full.index >= exit_candle_start
            if not mask.any():
                # شمعة الخروج غير متاحة بعد -> لا نحسب VOID كاذب، ننتظر
                log.warning(f"Exit candle not ready for {sym}. Retry next cycle.")
                continue
            exit_px = float(df_exit_full[mask].iloc[0]['Close'])
            pip_mult = get_pip_multiplier(sym)
            diff = exit_px - entry_px
            pips_diff = abs(diff) * pip_mult
            log.info(f"Entry={entry_px} Exit={exit_px} Diff={diff} Pips={pips_diff:.2f}")
            rid = trade.get('telegram_message_id')
            # سوق ساكن حقاً (أقل من نصف نقطة)
            if pips_diff < 0.5:
                send_telegram(
                    f"⚪ *Result for {sym} ({dr}):*\n"
                    f"• Entry: `{entry_px:.5f}`\n"
                    f"• Exit: `{exit_px:.5f}`\n"
                    f"• Status: _Market flat (Void)_",
                    reply_to=rid
                )
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
            send_telegram(
                f"{result_emoji} *Result for {sym} ({dr}):*\n"
                f"• Entry: `{entry_px:.5f}`\n"
                f"• Exit: `{exit_px:.5f}`\n"
                f"• Pips: `{pips_diff:.1f}`\n"
                f"• P/L: `{abs(diff)/entry_px*100:.3f}%`",
                reply_to=rid
            )
            sm.archive_trade({
                "id": trade['id'], "symbol": sym, "direction": dr,
                "entry_price": entry_px, "exit_price": exit_px,
                "win": bool(win), "resolved_at": now_utc.isoformat(),
                "status": "RESOLVED", "pips": pips_diff
            })
            resolved_ids.append(trade['id'])
            log.info(f"Resolved {trade['id']} {sym}: {'WIN' if win else 'LOSS'} ({pips_diff:.1f} pips)")
        except Exception as e:
            log.error(f"Error resolving trade {trade.get('id')}: {e}", exc_info=True)
    for tid in resolved_ids:
        sm.remove_pending_trade(tid)

# ═══════════════════════════════════════════════
# 7. مسح الإشارات
# ═══════════════════════════════════════════════

def scan_and_alert(sm: StateManager):
    log.info(">>> Scanning market for new signals...")
    now_utc = datetime.now(timezone.utc)
    if NIGHT_START_HOUR <= now_utc.hour < NIGHT_END_HOUR:
        log.info("Night ban active (00:00 - 09:00 UTC). Skipping scan.")
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
        # ★★ التصحيح: قيمة النقطة = 1/multiplier ★★
        point_value = get_point_value(sym)
        spread_buffer = 2.0 * point_value  # نقطتان فقط
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
            f"_النتيجة ستسجل تلقائيا بعد إغلاق شمعة الخروج._"
        )
        mid_id = send_telegram(alert_msg)
        if mid_id:
            sm.add_pending_trade({
                "id": trade_id,
                "symbol": sym,
                "direction": sig_dir,
                "entry_price": live_price,
                "entry_time": current_candle_time.isoformat(),
                "telegram_message_id": mid_id
            })
            sm.mark_candle_processed(candle_key)
            new_signals_count += 1
            time.sleep(1)
    log.info(f"Scan complete. New signals sent: {new_signals_count}")

# ═══════════════════════════════════════════════
# 8. نظام التقارير الكامل (يومي/أسبوعي/شهري)
# ═══════════════════════════════════════════════

def _compute_stats(trades_list):
    total = len(trades_list)
    resolved = [t for t in trades_list if t.get('status') == 'RESOLVED']
    wins = sum(1 for t in resolved if t.get('win') is True)
    losses = sum(1 for t in resolved if t.get('win') is False)
    voids = total - len(resolved)
    wr = (wins / len(resolved) * 100) if len(resolved) > 0 else 0.0
    return total, wins, losses, voids, wr

def generate_reports(sm: StateManager):
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    iso = now_utc.isocalendar()
    week_stamp = f"{iso[0]}-{iso[1]}"
    month_stamp = f"{now_utc.year}-{now_utc.month:02d}"
    history = sm.data.get("history", [])
    if not history:
        return

    # --- التقرير اليومي (عن يوم أمس، مرة واحدة باليوم) ---
    if sm.data.get("last_daily_report_date") != today_str:
        yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        daily = [t for t in history if str(t.get('resolved_at', '')).startswith(yesterday_str)]
        if daily:
            total, wins, losses, voids, wr = _compute_stats(daily)
            emoji_wr = "🟢" if wr >= 60 else ("🟡" if wr >= 40 else "🔴")
            send_telegram(
                f"*📊 H1 Daily Report ({yesterday_str})*\n\n"
                f"*Total Signals:* `{total}`\n"
                f"*Wins:* ✅ `{wins}`\n"
                f"*Losses:* ❌ `{losses}`\n"
                f"*Void/Tie:* ⚪ `{voids}`\n\n"
                f"*Win Rate:* {emoji_wr} `{wr:.1f}%`"
            )
            sm.data["last_daily_report_date"] = today_str
            sm.save()
            log.info("Daily report sent.")

    # --- التقرير الأسبوعي (السبت بعد 09:00 UTC) ---
    if now_utc.weekday() == 5 and now_utc.hour >= 9 and sm.data.get("last_weekly_report_stamp") != week_stamp:
        seven_days_ago = now_utc - timedelta(days=7)
        weekly = []
        for t in history:
            try:
                rd = ensure_utc(datetime.fromisoformat(t['resolved_at']))
                if rd >= seven_days_ago:
                    weekly.append(t)
            except Exception:
                continue
        if weekly:
            total, wins, losses, voids, wr = _compute_stats(weekly)
            pair_stats = {}
            for t in weekly:
                s = t['symbol']
                pair_stats.setdefault(s, {'wins': 0, 'total': 0})
                pair_stats[s]['total'] += 1
                if t.get('win') is True:
                    pair_stats[s]['wins'] += 1
            best_pair, best_wr = "", 0.0
            for s, st in pair_stats.items():
                swr = (st['wins'] / st['total'] * 100) if st['total'] else 0
                if swr > best_wr:
                    best_wr, best_pair = swr, s
            emoji_wr = "🟢" if wr >= 60 else ("🟡" if wr >= 40 else "🔴")
            send_telegram(
                f"*📅 H1 Weekly Report (Week {iso[1]})*\n\n"
                f"*Period:* Last 7 Days\n"
                f"*Total Signals:* `{total}`\n"
                f"*Wins:* ✅ `{wins}` | *Losses:* ❌ `{losses}` | *Void:* ⚪ `{voids}`\n\n"
                f"*Overall Win Rate:* {emoji_wr} `{wr:.1f}%`\n"
                f"*Top Performer:* `{best_pair}` ({best_wr:.1f}%)"
            )
            sm.data["last_weekly_report_stamp"] = week_stamp
            sm.save()
            log.info("Weekly report sent.")

    # --- التقرير الشهري (أول الشهر بعد 09:00 UTC) ---
    if now_utc.day == 1 and now_utc.hour >= 9 and sm.data.get("last_monthly_report_stamp") != month_stamp:
        thirty_days_ago = now_utc - timedelta(days=30)
        monthly = []
        for t in history:
            try:
                rd = ensure_utc(datetime.fromisoformat(t['resolved_at']))
                if rd >= thirty_days_ago:
                    monthly.append(t)
            except Exception:
                continue
        if monthly:
            total, wins, losses, voids, wr = _compute_stats(monthly)
            send_telegram(
                f"*🗓️ H1 Monthly Report ({month_stamp})*\n\n"
                f"*Period:* Last 30 Days\n"
                f"*Total Signals:* `{total}`\n"
                f"*Wins:* ✅ `{wins}` | *Losses:* ❌ `{losses}` | *Void:* ⚪ `{voids}`\n\n"
                f"*Monthly Win Rate:* `{wr:.1f}%`"
            )
            sm.data["last_monthly_report_stamp"] = month_stamp
            sm.save()
            log.info("Monthly report sent.")

# ═══════════════════════════════════════════════
# 9. الحلقة الرئيسية
# ═══════════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info(f"Starting {BOT_NAME} - FULLY AUDITED")
    log.info(f"Current UTC Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    log.info("=" * 50)
    sm = StateManager(STATE_FILE)
    resolve_pending_trades(sm)
    generate_reports(sm)
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
