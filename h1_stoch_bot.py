#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 RSI14 Sniper Bot - v1.0 (RSI14_25 | All | 10m | Report Catch-Up Edition)
يعمل داخل نفس الملف h1_stoch_bot.py باستراتيجية الـ 5 دقائق المعتمدة:
BB(20, 2.0) + Wilder RSI(14) <25/>75 + Wilder ADX(14) <25 | 28 زوجاً | 24 ساعة | انتهاء 10 دقائق.
"""
import os
import sys
import time
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

# ═══════════════════════════════════════════════
# 0. ثوابت التوقيت الآمنة (Asia/Hebron)
# ═══════════════════════════════════════════════
try:
    HEBRON_TZ = ZoneInfo("Asia/Hebron")
except Exception:
    try:
        HEBRON_TZ = ZoneInfo("Asia/Gaza")
    except Exception:
        HEBRON_TZ = None


def _is_palestine_summer(dt_utc: datetime) -> bool:
    def last_saturday(year, month):
        nxt = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
        last_day = nxt - timedelta(days=1)
        return last_day - timedelta(days=(last_day.weekday() - 5) % 7)
    return last_saturday(dt_utc.year, 3) <= dt_utc < last_saturday(dt_utc.year, 10)


def to_gaza(dt_obj: datetime) -> datetime:
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    if HEBRON_TZ is not None:
        return dt_obj.astimezone(HEBRON_TZ)
    off = 3 if _is_palestine_summer(dt_obj.astimezone(timezone.utc)) else 2
    return dt_obj.astimezone(timezone(timedelta(hours=off)))


def get_palestine_now_safe() -> datetime:
    return to_gaza(datetime.now(timezone.utc))


def _strip_tz(d: datetime) -> datetime:
    """تجريد المنطقة الزمنية للمقارنات التقويمية الآمنة."""
    return d.replace(tzinfo=None) if d.tzinfo is not None else d


def format_pair(symbol: str) -> str:
    try:
        s = str(symbol).replace("=X", "").replace("/", "").strip().upper()
        if len(s) == 6:
            return f"{s[:3]}/{s[3:]}"
        return s
    except Exception:
        return str(symbol)


def get_live_price(symbol: str):
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m",
                                       auto_adjust=False, actions=False)
        if df is None or df.empty:
            return None
        val = float(df['Close'].iloc[-1])
        if not np.isfinite(val) or val <= 0:
            return None
        return val
    except Exception:
        return None


def build_drift_text(live_px, signal_px: float, symbol: str, signal_type: str):
    if live_px is None:
        return "_غير متاح_", None
    try:
        pip_mult = 100.0 if "JPY" in symbol else 10000.0
        raw = (float(live_px) - float(signal_px)) * pip_mult
        drift = raw if signal_type == "CALL" else -raw
        txt = f"`{live_px:.5f}` (انجراف: `{drift:+.1f}` نقطة)"
        if drift < -DRIFT_WARN_PIPS:
            txt += " ⚠️ _السعر ابتعد ضد الاتجاه — يُفضّل التجاهل_"
        elif drift > DRIFT_WARN_PIPS:
            txt += " ℹ️ _السعر تحرك لصالح الاتجاه مسبقاً_"
        return txt, round(drift, 1)
    except Exception:
        return "_غير متاح_", None


# ═══════════════════════════════════════════════
# 1. الإعدادات العامة (استراتيجية الـ 5 دقائق)
# ═══════════════════════════════════════════════
BOT_NAME = "M5_RSI14_Sniper_V1"
STATE_FILE = "state_m5_rsi14_v1.json"
LOG_LEVEL = logging.INFO

# جميع أزواج الفوركس الـ 28 (شامل أزواج الين JPY)
TRADING_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X",
    "EURGBP=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPAUD=X", "GBPCHF=X", "GBPCAD=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
    "USDJPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
]

CANDLE_INTERVAL = "5m"
CANDLE_PERIOD = "5d"
CANDLE_DURATION_MINUTES = 5
TRADE_DURATION_MINUTES = 10      # مدة الصفقة: 10 دقائق (شمعتين)
FREEZE_DURATION_MINUTES = 0      # بدون تجميد إضافي بعد انتهاء الصفقة (تجميد فقط طوال الـ 10 دقائق)

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 25.0
RSI_OVERBOUGHT = 75.0
ADX_PERIOD = 14
ADX_MAX = 25.0

RISK_PER_TRADE_USD = 10.0
PAYOUT_RATE = 0.85
DRIFT_WARN_PIPS = 3.0            # تنبيه الانجراف المناسب لفريم 5 دقائق

TG_MAX_LEN = 3800
HISTORY_CAP = 20000

MAX_DAILY_CATCHUP = 31
MAX_WEEKLY_CATCHUP = 8
MAX_MONTHLY_CATCHUP = 6
MAX_REPORTS_PER_RUN = 3

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(BOT_NAME)


# ═══════════════════════════════════════════════
# 2. أدوات الاتصال والحالة
# ═══════════════════════════════════════════════

def _split_message(msg: str, limit: int = TG_MAX_LEN) -> list:
    if len(msg) <= limit:
        return [msg]
    parts = []
    cur = ""
    for line in msg.split("\n"):
        while len(line) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            if cur:
                parts.append(cur)
            cur = line
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        parts.append(cur)
    return parts


def _send_one(msg: str) -> bool:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                return True
            elif resp.status_code == 429:
                try:
                    wait = int(resp.json().get('parameters', {}).get('retry_after', 5))
                except Exception:
                    wait = 5
                time.sleep(wait)
            elif resp.status_code == 400:
                log.warning("TG Markdown parse error, retrying as plain text...")
                plain_text = msg.replace("*", "").replace("`", "").replace("_", "")
                plain = {"chat_id": TG_CHAT, "text": plain_text}
                resp_plain = requests.post(url, json=plain, timeout=15)
                if resp_plain.status_code == 200:
                    return True
                break
            else:
                log.error(f"TG API Error: {resp.status_code} | {resp.text}")
                break
        except Exception as e:
            log.error(f"TG Send Failed (Attempt {attempt + 1}): {e}")
            time.sleep(2)
    return False


def send_telegram(msg: str) -> bool:
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return False
    ok = True
    for chunk in _split_message(msg):
        if not _send_one(chunk):
            ok = False
    return ok


class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.defaults = {
            "frozen_pairs": {},
            "open_trades": [],
            "trade_history": [],
            "daily_counter_date": "",
            "daily_signal_count": 0,
            "last_daily_report": "",
            "last_weekly_report": "",
            "last_monthly_report": ""
        }
        self.data = json.loads(json.dumps(self.defaults))
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    for key, default_val in self.defaults.items():
                        if key not in loaded_data:
                            continue
                        val = loaded_data[key]
                        if isinstance(default_val, int) and not isinstance(default_val, bool):
                            if isinstance(val, bool):
                                continue
                            try:
                                self.data[key] = int(float(val))
                            except (TypeError, ValueError):
                                pass
                        elif isinstance(val, type(default_val)):
                            self.data[key] = val
                self.cleanup_expired_freezes()
                log.info(f"State loaded successfully from {self.filepath}")
            except Exception as e:
                log.error(f"Failed to load state: {e}. Starting with safe defaults.")
                self.save()
        else:
            log.info("State file does not exist yet. Creating initial state file.")
            self.save()

    def save(self):
        try:
            tmp = self.filepath.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self.filepath))
        except Exception as e:
            log.critical(f"STATE SAVE CRITICAL FAILURE: {e}")

    def cleanup_expired_freezes(self):
        try:
            expired = [s for s in list(self.data["frozen_pairs"].keys()) if not self.is_frozen(s)]
            if expired:
                for s in expired:
                    self.data["frozen_pairs"].pop(s, None)
                self.save()
        except Exception as e:
            log.error(f"Error cleaning freezes: {e}")

    def log_summary(self):
        try:
            frozen_active = [s for s in self.data["frozen_pairs"] if self.is_frozen(s)]
            log.info(
                f"STATE SUMMARY | open={len(self.data['open_trades'])} | "
                f"frozen={len(frozen_active)} | history={len(self.data['trade_history'])} | "
                f"daily={self.data.get('last_daily_report')} | "
                f"weekly={self.data.get('last_weekly_report')} | "
                f"monthly={self.data.get('last_monthly_report')}"
            )
        except Exception as e:
            log.error(f"State summary failed: {e}")

    def get_next_signal_number(self) -> int:
        today_str = get_palestine_now_safe().strftime("%Y-%m-%d")
        if self.data.get("daily_counter_date") != today_str:
            self.data["daily_counter_date"] = today_str
            self.data["daily_signal_count"] = 0
        try:
            current_count = int(float(self.data.get("daily_signal_count", 0)))
        except (TypeError, ValueError):
            current_count = 0
        self.data["daily_signal_count"] = current_count + 1
        self.save()
        return self.data["daily_signal_count"]

    def is_frozen(self, symbol: str) -> bool:
        if symbol not in self.data["frozen_pairs"]:
            return False
        freeze_until_str = self.data["frozen_pairs"][symbol]
        try:
            freeze_until = datetime.fromisoformat(str(freeze_until_str))
            if freeze_until.tzinfo is None:
                freeze_until = freeze_until.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < freeze_until
        except Exception as e:
            log.error(f"Error checking freeze for {symbol}: {e}")
            return False

    def freeze_pair(self, symbol: str, base_time: datetime, duration_minutes: int = FREEZE_DURATION_MINUTES):
        # تجميد الزوج حتى انتهاء الصفقة (إغلاق شمعة الإشارة + 10 دقائق)
        release_time = base_time + timedelta(
            minutes=CANDLE_DURATION_MINUTES + TRADE_DURATION_MINUTES + duration_minutes
        )
        now_utc = datetime.now(timezone.utc)
        if release_time <= now_utc:
            release_time = now_utc + timedelta(minutes=TRADE_DURATION_MINUTES)
        self.data["frozen_pairs"][symbol] = release_time.isoformat()
        self.save()

    def add_open_trade(self, trade: dict):
        self.data["open_trades"].append(trade)
        self.save()

    def remove_open_trade(self, tid: str):
        self.data["open_trades"] = [t for t in self.data["open_trades"] if t.get('id') != tid]
        self.save()

    def archive_trade(self, result: dict):
        self.data["trade_history"].append(result)
        if len(self.data["trade_history"]) > HISTORY_CAP:
            self.data["trade_history"] = self.data["trade_history"][-HISTORY_CAP:]
        self.save()


# ═══════════════════════════════════════════════
# 3. محرك البيانات والاستراتيجية (M5 RSI14_25 | 10m)
# ═══════════════════════════════════════════════

def get_data(symbol: str, interval: str = CANDLE_INTERVAL, period: str = CANDLE_PERIOD):
    df = None
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False)
            if df is not None and not df.empty:
                break
            time.sleep(2)
        except Exception:
            time.sleep(2)

    if df is None or df.empty:
        return None

    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated()].sort_index().dropna()

        # استبعاد الشمعة الحالية غير المغلقة على فريم 5 دقائق
        now_utc = pd.Timestamp.now(tz="UTC")
        while not df.empty and (df.index[-1] + pd.Timedelta(minutes=CANDLE_DURATION_MINUTES)) > now_utc:
            df = df.iloc[:-1]

        if len(df) < 35:
            return None

        c = df['Close']
        h = df['High']
        l = df['Low']

        # 1. Bollinger Bands (20, 2.0)
        sma20 = c.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean()
        std20 = c.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std()
        df['BB_UPPER'] = sma20 + (BB_STD * std20)
        df['BB_LOWER'] = sma20 - (BB_STD * std20)

        # 2. Wilder RSI (14)
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df['RSI14'] = 100.0 - (100.0 / (1.0 + rs))

        # 3. Wilder ADX (14)
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean()

        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr_safe = atr.replace(0.0, np.nan)
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean() / atr_safe
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean() / atr_safe
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        df['ADX14'] = dx.ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        log.error(f"Data Processing Error [{symbol}]: {e}")
        return None


def analyze_market_smart(symbol: str):
    df = get_data(symbol)
    if df is None or len(df) < 2:
        return None

    curr = df.iloc[-1]
    candle_time = df.index[-1]

    # تجنب إرسال إشارات قديمة جداً (إذا أغلقت الشمعة منذ أكثر من 8 دقائق)
    candle_close_time = candle_time + timedelta(minutes=CANDLE_DURATION_MINUTES)
    if (datetime.now(timezone.utc) - candle_close_time) > timedelta(minutes=8):
        return None

    open_p = float(curr['Open'])
    close_p = float(curr['Close'])
    bb_u = float(curr['BB_UPPER'])
    bb_l = float(curr['BB_LOWER'])
    rsi14 = float(curr['RSI14'])
    adx14 = float(curr['ADX14'])

    if adx14 >= ADX_MAX:
        return None

    signal_type = None
    if close_p < open_p and close_p < bb_l and rsi14 < RSI_OVERSOLD:
        signal_type = "CALL"
    elif close_p > open_p and close_p > bb_u and rsi14 > RSI_OVERBOUGHT:
        signal_type = "PUT"

    if signal_type is None:
        return None

    return {
        "symbol": symbol,
        "type": signal_type,
        "price": close_p,
        "time": candle_time,
        "rsi14": rsi14,
        "adx14": adx14,
        "bb_upper": bb_u,
        "bb_lower": bb_l,
    }


# ═══════════════════════════════════════════════
# 4. نظام الحسم التلقائي (انتهاء 10 دقائق = شمعتين 5m)
# ═══════════════════════════════════════════════

def resolve_pending_trades(sm: StateManager):
    if not sm.data["open_trades"]:
        log.info("No open trades pending resolution.")
        return

    now_utc = datetime.now(timezone.utc)
    remaining_trades = []

    log.info(f"Checking {len(sm.data['open_trades'])} open trade(s) for resolution...")

    for trade in sm.data["open_trades"]:
        try:
            trade_id = trade.get('id') or str(uuid.uuid4())[:8]
            trade['id'] = trade_id

            entry_dt_raw = datetime.fromisoformat(str(trade['entry_time']))
            if entry_dt_raw.tzinfo is None:
                entry_dt_raw = entry_dt_raw.replace(tzinfo=timezone.utc)

            # إغلاق شمعة الإشارة = وقت الدخول الفعلي
            actual_entry_close_time = entry_dt_raw + timedelta(minutes=CANDLE_DURATION_MINUTES)
            # وقت انتهاء الصفقة = وقت الدخول + 10 دقائق
            expected_exit_close_time = actual_entry_close_time + timedelta(minutes=TRADE_DURATION_MINUTES)
            check_ready_time = expected_exit_close_time + timedelta(seconds=30)

            if now_utc < check_ready_time:
                mins_left = max(1, int((check_ready_time - now_utc).total_seconds() // 60))
                log.info(f"Trade {trade.get('symbol')} ({trade_id}) still active. ~{mins_left}m remaining.")
                remaining_trades.append(trade)
                continue

            sym = trade['symbol']
            dr = trade.get('direction') or trade.get('type')
            if dr is None:
                log.error(f"Trade {trade_id} missing direction/type. Dropping corrupted record.")
                continue

            entry_px = float(trade['entry_price'])
            log.info(f"Resolving {sym} ({trade_id}) | Target Exit Close: {expected_exit_close_time.strftime('%Y-%m-%d %H:%M')} UTC")

            df_exit_full = get_data(sym)
            if df_exit_full is None or df_exit_full.empty:
                log.warning(f"No market data returned for {sym}. Keeping trade open for next cycle.")
                remaining_trades.append(trade)
                continue

            # الشمعة التي تغلق عند expected_exit_close_time تبدأ قبلها بـ 5 دقائق
            target_candle_start = expected_exit_close_time - timedelta(minutes=CANDLE_DURATION_MINUTES)

            mask = (
                (df_exit_full.index >= target_candle_start - timedelta(minutes=1)) &
                (df_exit_full.index <= target_candle_start + timedelta(minutes=1))
            )
            matching_candles = df_exit_full[mask]

            if not matching_candles.empty:
                exit_px = float(matching_candles.iloc[0]['Close'])
            else:
                future_candles = df_exit_full[df_exit_full.index >= target_candle_start]
                if not future_candles.empty:
                    exit_px = float(future_candles.iloc[0]['Close'])
                    log.info(f"Used first available candle >= {target_candle_start} for {sym}.")
                else:
                    if now_utc < expected_exit_close_time + timedelta(minutes=30):
                        log.warning(
                            f"Exit candle ({target_candle_start.strftime('%H:%M')} UTC) not published yet for {sym}. "
                            f"Waiting for next pulse..."
                        )
                        remaining_trades.append(trade)
                        continue
                    else:
                        exit_px = float(df_exit_full.iloc[-1]['Close'])
                        log.warning(f"Timeout waiting for exact candle on {sym}; using latest available close.")

            pip_mult = 100.0 if "JPY" in sym else 10000.0
            diff = exit_px - entry_px
            pips_diff = abs(diff) * pip_mult

            if pips_diff < 0.1:
                emoji, is_win, status_text = "⚪ تعادل/لاغي", None, "_(Market Flat)_"
            else:
                is_win = (diff > 0) if dr == 'CALL' else (diff < 0)
                emoji = "✅ صفقة رابحة" if is_win else "❌ صفقة خاسرة"
                status_text = ""

            entry_gaza_str = to_gaza(actual_entry_close_time).strftime('%H:%M')
            exit_gaza_str = to_gaza(expected_exit_close_time).strftime('%H:%M')

            result_msg = (
                f"{emoji} *| نتيجة استراتيجية الـ 5 دقائق:* `{format_pair(sym)}`\n\n"
                f"📈 *الاتجاه:* `{dr}`\n"
                f"⏰ *وقت الدخول:* `{actual_entry_close_time.strftime('%H:%M')} UTC` (`{entry_gaza_str}` غزة)\n"
                f"🏁 *وقت الإغلاق (10م):* `{expected_exit_close_time.strftime('%H:%M')} UTC` (`{exit_gaza_str}` غزة)\n"
                f"💰 *سعر الدخول:* `{entry_px:.5f}`\n"
                f"💸 *سعر الإغلاق:* `{exit_px:.5f}`\n"
                f"📊 *الفارق:* `{pips_diff:+.1f} نقطة (Pips)` {status_text}"
            )

            sent = send_telegram(result_msg)
            if not sent:
                log.error(f"Telegram failed to send resolution for {sym} ({trade_id}). Will retry next cycle.")
                remaining_trades.append(trade)
                continue

            sm.archive_trade({
                "id": trade_id,
                "symbol": sym,
                "type": dr,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "entry_time": trade['entry_time'],
                "exit_time": expected_exit_close_time.isoformat(),
                "win": is_win,
                "pips": round(float(pips_diff), 2)
            })
            log.info(f"Trade {sym} ({trade_id}) resolved & archived: win={is_win}, pips={pips_diff:.2f}")

        except Exception as e:
            log.error(f"Critical Error resolving trade {trade.get('id')}: {e}", exc_info=True)
            remaining_trades.append(trade)

    sm.data["open_trades"] = remaining_trades
    sm.save()


# ═══════════════════════════════════════════════
# 5. نظام التقارير والتحليل الشامل
# ═══════════════════════════════════════════════

AR_MONTHS = {
    "01": "يناير", "02": "فبراير", "03": "مارس", "04": "أبريل",
    "05": "مايو", "06": "يونيو", "07": "يوليو", "08": "أغسطس",
    "09": "سبتمبر", "10": "أكتوبر", "11": "نوفمبر", "12": "ديسمبر"
}
AR_DAYS = {
    0: 'الاثنين', 1: 'الثلاثاء', 2: 'الأربعاء', 3: 'الخميس',
    4: 'الجمعة', 5: 'السبت', 6: 'الأحد'
}


def _to_utc_dt(value):
    try:
        d = pd.to_datetime(str(value), errors='coerce', utc=True)
        if pd.isna(d):
            return None
        return d.to_pydatetime()
    except Exception:
        return None


def convert_utc_to_gaza_date(time_str) -> str:
    d = _to_utc_dt(time_str)
    return to_gaza(d).strftime("%Y-%m-%d") if d else ""


def _gaza_hour(time_str) -> int:
    d = _to_utc_dt(time_str)
    return to_gaza(d).hour if d else -1


def _signed_pips(t: dict) -> float:
    try:
        p = float(t.get('pips', 0.0) or 0.0)
    except (TypeError, ValueError):
        p = 0.0
    w = t.get('win')
    return p if w is True else (-p if w is False else 0.0)


def _stats(trades: list) -> dict:
    total = len(trades)
    wins = sum(1 for t in trades if t.get('win') is True)
    losses = sum(1 for t in trades if t.get('win') is False)
    voids = total - wins - losses
    valid = wins + losses
    wr = (wins / valid * 100.0) if valid else 0.0
    net = sum(_signed_pips(t) for t in trades)
    return {"total": total, "wins": wins, "losses": losses,
            "voids": voids, "valid": valid, "wr": wr, "net": net}


def _wr_emoji(wr: float) -> str:
    return "🟢 ممتاز" if wr >= 60 else ("🟡 جيد" if wr >= 54 else "🔴 يحتاج مراجعة")


def _pair_analysis(trades: list) -> dict:
    pm = {}
    for t in trades:
        sym = str(t.get('symbol', 'UNKNOWN'))
        d = pm.setdefault(sym, {"trades": 0, "wins": 0, "losses": 0, "net": 0.0})
        d["trades"] += 1
        w = t.get('win')
        if w is True:
            d["wins"] += 1
        elif w is False:
            d["losses"] += 1
        d["net"] += _signed_pips(t)
    for d in pm.values():
        v = d["wins"] + d["losses"]
        d["wr"] = (d["wins"] / v * 100.0) if v else 0.0
    return pm


def _hour_analysis(trades: list) -> dict:
    hm = {}
    for t in trades:
        h = _gaza_hour(t.get('entry_time') or t.get('exit_time'))
        if h < 0:
            continue
        d = hm.setdefault(h, {"trades": 0, "wins": 0, "losses": 0, "net": 0.0})
        d["trades"] += 1
        w = t.get('win')
        if w is True:
            d["wins"] += 1
        elif w is False:
            d["losses"] += 1
        d["net"] += _signed_pips(t)
    for d in hm.values():
        v = d["wins"] + d["losses"]
        d["wr"] = (d["wins"] / v * 100.0) if v else 0.0
    return hm


def _first_trade_per_day(trades: list) -> list:
    by_day = {}
    for t in trades:
        src = t.get('entry_time') or t.get('exit_time')
        d = _to_utc_dt(src)
        if d is None:
            continue
        key = to_gaza(d).strftime("%Y-%m-%d")
        if key not in by_day or d < by_day[key][0]:
            by_day[key] = (d, t)
    return [(k, by_day[k][1]) for k in sorted(by_day.keys())]


def _streaks(seq: list) -> tuple:
    mw = ml = cw = cl = 0
    for w in seq:
        if w is True:
            cw += 1
            cl = 0
        elif w is False:
            cl += 1
            cw = 0
        else:
            cw = cl = 0
        mw = max(mw, cw)
        ml = max(ml, cl)
    return mw, ml


def build_first_trade_section(trades: list, show_details: bool = False) -> str:
    picks = _first_trade_per_day(trades)
    if not picks:
        return ("*🎯 استراتيجية الصفقة الأولى يومياً:*\n"
                "ℹ️ _لا توجد صفقات كافية لحساب هذه الاستراتيجية في هذه الفترة._\n\n")

    fl = [t for _, t in picks]
    s = _stats(fl)
    mw, ml = _streaks([t.get('win') for t in fl])
    capital = s["wins"] * RISK_PER_TRADE_USD * PAYOUT_RATE - s["losses"] * RISK_PER_TRADE_USD

    lines = [
        "*🎯 استراتيجية الصفقة الأولى يومياً:*",
        "ℹ️ _محاكاة: صفقة واحدة كل يوم — أول إشارة ثم إغلاق المنصة._",
        f"• *عدد الأيام المتداولة:* `{s['total']}` يوم",
        f"• *النتيجة:* ✅ `{s['wins']}` رابحة | ❌ `{s['losses']}` خاسرة | ⚪ `{s['voids']}` تعادل",
        f"• *نسبة النجاح:* {_wr_emoji(s['wr'])} `{s['wr']:.1f}%`",
        f"• *صافي النقاط:* `{s['net']:+.1f} Pips`",
        f"• *أطول سلسلة ربح:* `{mw}` | *أطول سلسلة خسارة:* `{ml}`",
        f"• *محاكاة رأس المال:* مخاطرة `{RISK_PER_TRADE_USD:.0f}$` وعائد `{PAYOUT_RATE*100:.0f}%` → `{capital:+.2f}$`"
    ]

    if show_details:
        lines.append("")
        lines.append("*📅 سجل الأيام:*")
        for day, t in picks:
            w = t.get('win')
            mark = "✅" if w is True else ("❌" if w is False else "⚪")
            hh = _gaza_hour(t.get('entry_time') or t.get('exit_time'))
            hs = f"{hh:02d}:00" if hh >= 0 else "--:--"
            sp = _signed_pips(t)
            lines.append(
                f"• `{day}` {hs} | `{format_pair(t.get('symbol', 'UNKNOWN'))}` | "
                f"{t.get('type', '?')} | {mark} `{sp:+.1f}`"
            )

    return "\n".join(lines) + "\n\n"


def build_report_message(trades_list: list, report_type: str, period_name: str,
                         include_first: bool = True, show_details: bool = False) -> str:
    header = (f"*📊 التقرير {report_type} — بوت استراتيجية الـ 5 دقائق (RSI14_25)*\n"
              f"🗓 *الفترة:* `{period_name}`\n\n")

    if not trades_list:
        return (header
                + "📈 *إجمالي الصفقات:* `0`\n"
                + "ℹ️ _لم تُسجل أي صفقات مكتملة خلال هذه الفترة._\n\n"
                + (build_first_trade_section([], show_details) if include_first else "")
                + "_البوت يعمل بكفاءة ويراقب الفرص وفق الفلاتر الصارمة!_")

    s = _stats(trades_list)
    body = (
        f"📈 *إجمالي الصفقات:* `{s['total']}` (✅ رابحة: `{s['wins']}` | ❌ خاسرة: `{s['losses']}` | ⚪ تعادل: `{s['voids']}`)\n"
        f"🏆 *نسبة الفوز (Win Rate):* {_wr_emoji(s['wr'])} `{s['wr']:.1f}%`\n"
        f"💰 *صافي النقاط (Net Pips):* `{s['net']:+.1f} Pips`\n\n"
    )

    pm = _pair_analysis(trades_list)
    best_txt = worst_txt = "غير متاح"
    if pm:
        sp = sorted(pm.items(), key=lambda kv: (kv[1]["net"], kv[1]["wr"]), reverse=True)
        bs, bi = sp[0]
        best_txt = (f"🥇 `{format_pair(bs)}` ({bi['net']:+.1f} Pips | "
                    f"فوز: {bi['wr']:.0f}% | {bi['trades']} صفقات)")
        if len(sp) > 1:
            ws, wi = sp[-1]
            worst_txt = (f"⚠️ `{format_pair(ws)}` ({wi['net']:+.1f} Pips | "
                         f"فوز: {wi['wr']:.0f}% | {wi['trades']} صفقات)")
        else:
            worst_txt = "لا يوجد (تم التداول على زوج واحد فقط)"

    body += "*🔍 تحليل أداء الأزواج:*\n"
    body += f"• *أفضل زوج:* {best_txt}\n"
    body += f"• *أسوأ زوج:* {worst_txt}\n\n"

    hm = _hour_analysis(trades_list)
    bh_txt = wh_txt = "غير متاح"
    if hm:
        sh = sorted(hm.items(), key=lambda kv: (kv[1]["wr"], kv[1]["net"]), reverse=True)
        bh, bd = sh[0]
        bh_txt = (f"`{bh:02d}:00 - {(bh+1)%24:02d}:00` (فوز: {bd['wr']:.0f}% | "
                  f"{bd['net']:+.1f} Pips | {bd['trades']} صفقات)")
        if len(sh) > 1:
            wh, wd = sh[-1]
            wh_txt = (f"`{wh:02d}:00 - {(wh+1)%24:02d}:00` (فوز: {wd['wr']:.0f}% | "
                      f"{wd['net']:+.1f} Pips | {wd['trades']} صفقات)")
        else:
            wh_txt = "لا يوجد (ساعة تداول واحدة فقط)"

    body += "*🕒 تحليل أفضل وقت تداول (توقيت غزة):*\n"
    body += f"• *أفضل ساعة:* {bh_txt}\n"
    body += f"• *أسوأ ساعة:* {wh_txt}\n\n"

    if include_first:
        body += build_first_trade_section(trades_list, show_details)

    body += f"_انتهت جلسة التقرير {report_type}. استمر في الانضباط!_"
    return header + body


def _pending_days(last_daily: str, today_local: datetime, max_back: int = MAX_DAILY_CATCHUP) -> list:
    today_local = _strip_tz(today_local)
    yesterday = (today_local - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = None
    if last_daily:
        try:
            start = datetime.strptime(str(last_daily), "%Y-%m-%d") + timedelta(days=1)
        except Exception:
            start = None
    if start is None:
        start = yesterday
    if start > yesterday:
        return []
    if (yesterday - start).days + 1 > max_back:
        start = yesterday - timedelta(days=max_back - 1)
    out = []
    cur = start
    while cur <= yesterday:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _pending_weeks(last_weekly: str, today_local: datetime, max_back: int = MAX_WEEKLY_CATCHUP) -> list:
    today_local = _strip_tz(today_local)
    this_monday = (today_local - timedelta(days=today_local.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    cur = this_monday - timedelta(days=7)

    def pack(d):
        yy, wn, _ = d.isocalendar()
        return (f"{yy}-W{wn:02d}",
                d.strftime("%Y-%m-%d"),
                (d + timedelta(days=6)).strftime("%Y-%m-%d"))

    if not last_weekly:
        return [pack(cur)]

    out = []
    for _ in range(max_back):
        wid, a, b = pack(cur)
        if wid <= str(last_weekly):
            break
        out.append((wid, a, b))
        cur -= timedelta(days=7)
    return list(reversed(out))


def _pending_months(last_monthly: str, today_local: datetime, max_back: int = MAX_MONTHLY_CATCHUP) -> list:
    today_local = _strip_tz(today_local)
    first_this = today_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cur = first_this - timedelta(days=1)

    def pack(d):
        return (d.strftime("%Y-%m"),
                d.replace(day=1).strftime("%Y-%m-%d"),
                d.strftime("%Y-%m-%d"))

    if not last_monthly:
        return [pack(cur)]

    out = []
    for _ in range(max_back):
        mid, a, b = pack(cur)
        if mid <= str(last_monthly):
            break
        out.append((mid, a, b))
        cur = cur.replace(day=1) - timedelta(days=1)
    return list(reversed(out))


def generate_advanced_reports(sm: StateManager):
    local_now = get_palestine_now_safe()
    history_raw = sm.data.get("trade_history", [])

    records = []
    if isinstance(history_raw, list):
        for t in history_raw:
            if not isinstance(t, dict):
                continue
            src = t.get('exit_time') or t.get('entry_time')
            key = convert_utc_to_gaza_date(src)
            if not key:
                continue
            r = dict(t)
            r["_day"] = key
            records.append(r)

    sent_count = 0

    for day_str in _pending_days(sm.data.get("last_daily_report", ""), local_now):
        if sent_count >= MAX_REPORTS_PER_RUN:
            break
        try:
            d = datetime.strptime(day_str, "%Y-%m-%d")
            recs = [r for r in records if r["_day"] == day_str]
            name = f"{AR_DAYS[d.weekday()]} {d.day} {AR_MONTHS[d.strftime('%m')]} {d.year}"
            msg = build_report_message(recs, "اليومي", name,
                                       include_first=True, show_details=False)
            if send_telegram(msg):
                sm.data["last_daily_report"] = day_str
                sm.save()
                sent_count += 1
            else:
                break
        except Exception as e:
            log.error(f"Daily report error for {day_str}: {e}", exc_info=True)
            break

    for wid, a, b in _pending_weeks(sm.data.get("last_weekly_report", ""), local_now):
        if sent_count >= MAX_REPORTS_PER_RUN:
            break
        try:
            ds = datetime.strptime(a, "%Y-%m-%d")
            de = datetime.strptime(b, "%Y-%m-%d")
            wn = int(wid.split("-W")[1])
            recs = [r for r in records if a <= r["_day"] <= b]
            name = (f"الأسبوع {wn} ({ds.day} {AR_MONTHS[ds.strftime('%m')]} – "
                    f"{de.day} {AR_MONTHS[de.strftime('%m')]} {de.year})")
            msg = build_report_message(recs, "الأسبوعي", name,
                                       include_first=True, show_details=True)
            if send_telegram(msg):
                sm.data["last_weekly_report"] = wid
                sm.save()
                sent_count += 1
            else:
                break
        except Exception as e:
            log.error(f"Weekly report error for {wid}: {e}", exc_info=True)
            break

    for mid, a, b in _pending_months(sm.data.get("last_monthly_report", ""), local_now):
        if sent_count >= MAX_REPORTS_PER_RUN:
            break
        try:
            de = datetime.strptime(b, "%Y-%m-%d")
            recs = [r for r in records if a <= r["_day"] <= b]
            name = f"{AR_MONTHS[de.strftime('%m')]} {de.year}"
            msg = build_report_message(recs, "الشهري", name,
                                       include_first=True, show_details=True)
            if send_telegram(msg):
                sm.data["last_monthly_report"] = mid
                sm.save()
                sent_count += 1
            else:
                break
        except Exception as e:
            log.error(f"Monthly report error for {mid}: {e}", exc_info=True)
            break

    if sent_count == 0:
        log.info("Reports up to date. Nothing pending.")


# ═══════════════════════════════════════════════
# 6. الحلقة الرئيسية
# ═══════════════════════════════════════════════

def main():
    local_now = get_palestine_now_safe()
    log.info("=" * 60)
    log.info(f"STARTING {BOT_NAME} | M5 -> 10m (28 Pairs | 24/5)")
    log.info(
        f"UTC Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Gaza/Hebron Time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    log.info("=" * 60)

    if not TG_TOKEN or not TG_CHAT:
        log.critical("MISSING TELEGRAM CREDS. ABORTING.")
        return

    sm = StateManager(STATE_FILE)
    sm.log_summary()

    resolve_pending_trades(sm)
    generate_advanced_reports(sm)

    if local_now.weekday() >= 5:
        log.info("Weekend. Skipping new signals.")
        return

    signals_found = 0

    for sym in TRADING_SYMBOLS:
        if sm.is_frozen(sym):
            continue

        result = analyze_market_smart(sym)
        if result is None:
            continue

        signals_found += 1
        signal_num = sm.get_next_signal_number()
        arrival_time_gaza = get_palestine_now_safe().strftime("%H:%M")

        arrow = "🟢 CALL (صعود)" if result['type'] == "CALL" else "🔴 PUT (هبوط)"
        entry_close_utc = result['time'] + timedelta(minutes=CANDLE_DURATION_MINUTES)
        expiry_utc = entry_close_utc + timedelta(minutes=TRADE_DURATION_MINUTES)
        expiry_gaza = to_gaza(expiry_utc).strftime("%H:%M")

        live_px = get_live_price(sym)
        drift_txt, drift_val = build_drift_text(live_px, result['price'], sym, result['type'])
        if drift_val is not None:
            log.info(f"{sym} {result['type']} | signal={result['price']:.5f} live={live_px:.5f} drift={drift_val:+.1f} pips")
        else:
            log.info(f"{sym} {result['type']} | live price unavailable (display only)")

        msg = (
            f"*⚡ إشارة استراتيجية الـ 5 دقائق #{signal_num}*\n\n"
            f"📊 *الزوج:* `{format_pair(result['symbol'])}`\n"
            f"📈 *الاتجاه:* {arrow}\n"
            f"💰 *سعر الإشارة (إغلاق 5m):* `{result['price']:.5f}`\n"
            f"⚡ *السعر الآن:* {drift_txt}\n\n"
            f"*تحليل الشروط (RSI14_25):*\n"
            f"✅ تشبّع حاد: `RSI(14) = {result['rsi14']:.1f}`\n"
            f"✅ سوق عرضي مناسب: `ADX(14) = {result['adx14']:.1f} < 25`\n"
            f"✅ إغلاق خارج `BB(20, 2.0)` مع تأكيد لون الشمعة\n\n"
            f"⏱️ *مدة الصفقة:* `10 دقائق` (شمعتين)\n"
            f"⌛ *انتهاء الصفقة:* `{expiry_utc.strftime('%H:%M')} UTC` (`{expiry_gaza}` غزة)\n"
            f"💰 *شرط الـ payout:* 85% فأعلى فقط\n\n"
            f"🕒 *وقت وصول الإشارة:* {arrival_time_gaza} (غزة)\n"
            f"📝 *الاستراتيجية:* M5 RSI14_25 | All | 10m"
        )

        if send_telegram(msg):
            sm.freeze_pair(sym, base_time=result['time'])
            sm.add_open_trade({
                "id": str(uuid.uuid4())[:8],
                "symbol": result['symbol'],
                "type": result['type'],
                "entry_price": float(result['price']),
                "entry_time": result['time'].isoformat(),
                "signal_sent": True
            })
            log.info(f"Signal Sent & Saved to open_trades: {sym} {result['type']} (#{signal_num})")
        else:
            log.error(f"Failed to send signal for {sym}")
        time.sleep(1.0)

    if signals_found == 0:
        log.info("No valid setups. Market quiet or filters active.")
    else:
        log.info(f"Cycle complete. {signals_found} new signals sent.")

    sm.log_summary()


if __name__ == "__main__":
    main()
