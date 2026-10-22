#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Ultimate Bot - v4.6 (Guaranteed-Reports & Fresh-Data Release)
الإصلاحات الجذرية في v4.6:
1. ★★ إصلاح قاتل للتأخير الزمني: نافذة حذف الشمعة الجارية كانت 90 دقيقة، فمع أي تأخر
   بسيط من yfinance كانت تُحذف الشمعة المغلقة الطازجة ويُحلل على شمعة عمرها 65+ دقيقة
   → إشارات متأخرة وتقاطعات انعكست قبل الدخول (هذا سبب "قليلة وخاسرة" بشكل رئيسي).
   أصبحت 62 دقيقة (60 إغلاق + دقيقتان هامش بيانات) — شمعة مغلقة دائمًا طازجة.
2. ★★ تقارير مضمونة التسليم: إلغاء نافذة 00:00-00:10 بالكامل — يُحاول الإرسال في كل
   نبضة بعد منتصف الليل حتى ينجح مرة واحدة (مانع التكرار last_daily_report_date)،
   مع رسالة "لم تُسجَّل صفقات" عند الفراغ بدل الصمت، وتواريخ عربية واضحة.
3. ★★ ZoneInfo("Asia/Gaza") مع fallback آمن — ضبط التوقيت الصيفي/الشتوي تلقائيًا
   بدل الإزاحة الثابتة +3 التي تُخرج الحظر الليلي والتقارير عن الصواب نصف السنة.
4. إضافة فلتر عطلة نهاية الأسبوع (السبت والأحد بتوقيت غزة) — منع الإشارات على
   بيانات سوق مغلق/متجمدة.
5. جلب بيانات الحسم بنافذة 7 أيام بدل 3 (صفقات عطلة نهاية الأسبوع لا تضيع).
6. Zero-Crash: تقرير محصّن ضد المفاتيح الناقصة والتواريخ التالفة (coerce+dropna).
الإصلاحات النهائية والحاسمة (مطبقة من منطق الـ M15 v3.8.2):
1. ★★ إصلاح جذري لنافذة زمنية الحسم (استخدام وقت بداية الشمعة بدقة رياضية) ★★
   - شمعة الدخول: entry_dt -> entry_dt + 60 دقيقة.
   - شمعة الخروج: entry_dt + 60 -> entry_dt + 120 دقيقة.
   - سعر الخروج = إغلاق الشمعة الثانية.
2. ★★ إلغاء نظام Reply واستبداله برسائل مستقلة (حل لمشكلة State Loss في GitHub Actions) ★★
3. ★★ إضافة آلية Retry Loop ذكية لجلب بيانات yfinance ضد التأخير اللحظي ★★
4. ★★ تعديل التقارير لمنع السبام وإرسالها مرة واحدة فقط عند منتصف الليل بتوقيت غزة ★★
5. توقيت غزة المحلي: يعمل 9 صباحاُ -> 12 منتصف الليل، ينام 12 -> 9 صباحاُ.
ملاحظة: منطق الاستراتيجية (check_signal_logic) لم يُمَس إطلاقاُ.
"""
import os, sys, time, json, logging, uuid
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
    print("pip install yfinance"); sys.exit(1)

try:
    GAZA_TZ = ZoneInfo("Asia/Gaza")
except Exception:
    GAZA_TZ = timezone(timedelta(hours=3))

def get_gaza_now() -> datetime:
    """توقيت غزة الحقيقي مع دعم التوقيت الصيفي/الشتوي"""
    return datetime.now(timezone.utc).astimezone(GAZA_TZ)

# ═══════════════════════════════════════════════
# 1. الإعدادات العامة
# ═══════════════════════════════════════════════
BOT_NAME = "H1_Ultimate_Bot_V4_6_Guaranteed_Reports"
STATE_FILE = "state_h1_ultimate_v45.json" # ملف حالة جديد لضمان النظافة
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

TRADE_DURATION_MINUTES = 60 # مدة الصفقة ساعة واحدة
EXIT_GRACE_MINUTES = 2      # هامش انتظار تحديث البيانات بعد الإغلاق

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
            elif resp.status_code == 400:
                log.warning("TG Markdown parse error, retrying as plain text...")
                plain = {"chat_id": TG_CHAT, "text": msg.replace("*", "").replace("`", "").replace("_", "")}
                try:
                    resp_plain = requests.post(url, json=plain, timeout=15)
                    if resp_plain.status_code == 200:
                        log.info("Message sent successfully to Telegram (plain).")
                        return True
                except Exception:
                    pass
                break
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
            "last_daily_report_date": "" # ★★ متغير جديد لمنع تكرار التقرير ★★
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
        """حفظ ذرّي آمن"""
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
# 4. محرك البيانات (مع Retry Loop)
# ═══════════════════════════════════════════════

def fetch_h1_data(symbol: str, period_days: int = 7):
    """★★ v4.5: Retry Loop حقيقي ضد تأخر yfinance ★★"""
    df = None
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{period_days}d", interval="1h",
                                auto_adjust=False, actions=False)
            if df is not None and not df.empty:
                break
            log.warning(f"[{symbol}] Empty data from yfinance (attempt {attempt+1}/3). Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            log.error(f"Error fetching data for {symbol} (attempt {attempt+1}/3): {e}")
            time.sleep(3)

    if df is None or df.empty:
        return None

    try:
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
        # حذف الشمعة الجارية فقط: 62 د = ساعة إغلاق + دقيقتان هامش بيانات (لا تحذف الشمعة المغلقة الطازجة)
        if not df.empty and df.index[-1] + pd.Timedelta(minutes=62) > now_utc:
            df = df.iloc[:-1]
            
        return df
    except Exception as e:
        log.error(f"Data Processing Error [{symbol}]: {e}")
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
# 5. منطق الاستراتيجية (لم يُمَس)
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
# 6. ★★★ الحسم التلقائي المصحح جذريا (منطق Kimi) ★★★
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
            
            # ★★ التصحيح الجوهري هنا ★★
            # entry_dt هو وقت *بداية* شمعة الدخول (مثال 10:00).
            # شمعة الدخول تغلق عند 11:00.
            # شمعة الخروج تبدأ 11:00 وتغلق 12:00.
            exit_candle_start = entry_dt + timedelta(minutes=TRADE_DURATION_MINUTES) # 11:00
            exit_candle_close = exit_candle_start + timedelta(minutes=TRADE_DURATION_MINUTES) # 12:00
            
            # ننتظر إغلاق شمعة الخروج + هامش تحديث البيانات
            check_ready_time = exit_candle_close + timedelta(minutes=EXIT_GRACE_MINUTES) # 12:02
            
            if now_utc < check_ready_time:
                continue
            
            sym = trade['symbol']
            dr = trade['direction']
            entry_px = float(trade['entry_price'])
            
            log.info(f"Resolving {sym} | Target Exit Close: {exit_candle_close}")
            
            df_exit_full = fetch_h1_data(sym, period_days=7)  # v4.6: نافذة أوسع لتغطية عطلة الأسبوع
            if df_exit_full is None or df_exit_full.empty:
                log.warning(f"No exit data for {sym}. Retry next cycle.")
                continue
                
            # البحث عن الشمعة التي تبدأ عند/بعد وقت بداية شمعة الخروج
            target_ts = pd.Timestamp(exit_candle_start)
            future_mask = df_exit_full.index >= target_ts
            
            if not future_mask.any():
                # احتياطي لنقص البيانات
                past_mask = df_exit_full.index < target_ts
                if past_mask.any():
                    exit_px = float(df_exit_full[past_mask].iloc[-1]['Close'])
                    log.warning(f"Exit candle not ready for {sym}, using fallback last close.")
                else:
                    log.error(f"No candles at all for {sym}. Skipping.")
                    continue
            else:
                exit_px = float(df_exit_full[future_mask].iloc[0]['Close'])

            pip_mult = get_pip_multiplier(sym)
            diff = exit_px - entry_px
            pips_diff = abs(diff) * pip_mult
            
            log.info(f"Success! Entry={entry_px} Exit={exit_px} Diff={diff} Pips={pips_diff:.2f}")
            
            rid = trade.get('telegram_message_id') # retained just in case, but we won't use it for reply
            
            # سوق ساكن حقاُ
            if pips_diff < 0.5:
                result_msg = (f"⚪ *H1 Result for {sym} ({dr}):*\n\n"
                              f"• Entry: `{entry_px:.5f}`\n"
                              f"• Exit: `{exit_px:.5f}`\n"
                              f"• Status: _Market flat (Void)_\n"
                              f"• Time: `{now_utc.strftime('%H:%M')} UTC`")
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
            
            # ★★ إرسال رسالة مستقلة بدل الرد ★★
            result_msg = (f"{result_emoji} *H1 Result for {sym} ({dr}):*\n\n"
                          f"• Entry: `{entry_px:.5f}`\n"
                          f"• Exit: `{exit_px:.5f}`\n"
                          f"• Pips: `{pips_diff:.1f}`\n"
                          f"• P/L: `{abs(diff)/entry_px*100:.3f}%`\n"
                          f"• Time: `{now_utc.strftime('%H:%M')} UTC`")
            
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
            log.error(f"Critical Error resolving trade {trade.get('id')}: {e}", exc_info=True)
            
    for tid in resolved_ids:
        sm.remove_pending_trade(tid)

# ═══════════════════════════════════════════════
# 7. نظام التقارير (معدل لمنع السبام)
# ═══════════════════════════════════════════════

AR_MONTHS = {
    "01": "يناير", "02": "فبراير", "03": "مارس", "04": "أبريل",
    "05": "مايو", "06": "يونيو", "07": "يوليو", "08": "أغسطس",
    "09": "سبتمبر", "10": "أكتوبر", "11": "نوفمبر", "12": "ديسمبر"
}
AR_DAYS = {0: 'الاثنين', 1: 'الثلاثاء', 2: 'الأربعاء', 3: 'الخميس',
           4: 'الجمعة', 5: 'السبت', 6: 'الأحد'}


def generate_reports(sm: StateManager):
    """
    ★★ v4.6 — تقرير يومي مضمون التسليم ★★
    لا يوجد أي قيد hour/minute. يُحاول الإرسال في كل نبضة بعد منتصف الليل بتوقيت غزة
    حتى ينجح مرة واحدة فقط (last_daily_report_date يخزن تاريخ "أمس" ويمنع التكرار).
    """
    local_now = get_gaza_now()
    yesterday = local_now - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    period_name = f"{AR_DAYS[yesterday.weekday()]} {yesterday.day} {AR_MONTHS[yesterday.strftime('%m')]} {yesterday.year}"

    # التقرير مستحق طالما لم نُرسله لأمس بعد — بأي ساعة من الليل أو النهار
    if sm.data.get("last_daily_report_date") == yesterday_str:
        return

    log.info(f"Daily report due for {yesterday_str} — attempting send...")

    # جمع صفقات أمس بتواريخ محصّنة (coerce + skip تالف) — Zero-Crash
    daily_trades = []
    for t in sm.data.get("history", []):
        try:
            exit_time_str = t.get('resolved_at', '')
            if not exit_time_str:
                continue
            exit_dt_utc = pd.to_datetime(exit_time_str, utc=True, errors='coerce')
            if pd.isna(exit_dt_utc):
                continue
            exit_dt_gaza = exit_dt_utc.tz_convert(GAZA_TZ)
            if exit_dt_gaza.date() == yesterday.date():
                daily_trades.append(t)
        except Exception:
            continue

    if not daily_trades:
        # ★★ v4.6: رسالة اطمئنان بدل الصمت التام ★★
        msg = (
            f"*🌙 ملخص أداء H1 ({period_name})*\n\n"
            f"📊 *إجمالي الصفقات:* `0`\n"
            f"ℹ️ _لم تُسجَّل صفقات خلال هذه الفترة._\n\n"
            f"_البوت يعمل بكفاءة ويراقب السوق بانتظار الفرص القادمة!_"
        )
        if send_telegram(msg):
            sm.data["last_daily_report_date"] = yesterday_str
            sm.save()
            log.info(f"Empty daily report CONFIRMED for {yesterday_str}")
        else:
            log.warning("Daily report send failed — will retry on next pulse (guaranteed delivery).")
        return

    total = len(daily_trades)
    wins = sum(1 for t in daily_trades if t.get('win') is True)
    losses = sum(1 for t in daily_trades if t.get('win') is False)
    voids = total - wins - losses

    # صافي النقاط الموجّه (ربح موجب / خسارة سالبة / لاغي صفر)
    net_pips = 0.0
    for t in daily_trades:
        try:
            p = float(t.get('pips', 0) or 0)
        except (TypeError, ValueError):
            p = 0.0
        if t.get('win') is True:
            net_pips += p
        elif t.get('win') is False:
            net_pips -= p

    wr = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
    emoji_wr = "🟢 ممتاز" if wr >= 60 else ("🟡 جيد" if wr >= 40 else "🔴 يحتاج مراجعة")

    report_msg = (
        f"*🌙 ملخص أداء H1 ({period_name})*\n\n"
        f"📊 *إجمالي الصفقات:* `{total}`\n"
        f"✅ *الرابحة:* `{wins}` | ❌ *الخاسرة:* `{losses}` | ⚪ *لاغية:* `{voids}`\n\n"
        f"🏆 *نسبة الفوز:* {emoji_wr} `{wr:.1f}%`\n"
        f"💰 *صافي النقاط:* `{net_pips:+.1f} Pips`\n\n"
        f"_انتهت جلسة التقرير اليومي. استمر في الانضباط!_"
    )

    if send_telegram(report_msg):
        sm.data["last_daily_report_date"] = yesterday_str
        sm.save()
        log.info(f"Daily report CONFIRMED for {yesterday_str}")
    else:
        log.warning("Daily report send failed — will retry on next pulse (guaranteed delivery).")


# ═══════════════════════════════════════════════
# 8. مسح الإشارات
# ═══════════════════════════════════════════════

def scan_and_alert(sm: StateManager):
    log.info(">>> Scanning market for new signals...")
    local_now = get_gaza_now()

    # ★★ v4.6: فلتر عطلة نهاية الأسبوع — منع الإشارات على بيانات سوق متجمدة ★★
    if local_now.weekday() >= 5:
        log.info("Weekend (Sat/Sun Gaza). Skipping scan.")
        return

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
        
        arrow = "🟢 BUY/CALL" if sig_dir == "CALL" else "🔴 SELL/PUT"
        alert_msg = (
            f"*H1 Stochastic Signal Alert!*\n\n"
            f"الزوج: `{sym}`\n"
            f"الاتجاه: {arrow}\n"
            f"السعر الحي الآن: `{live_price:.5f}`\n\n"
            f"*التعليمات:*\n"
            f"✅ ادخل فوراُ خلال 60 ثانية\n\n"
            f"*السبب:* التقاطع في منطقة التشبع المتطرف "
            f"(Stoch K={k_val:.1f}, D={d_val:.1f}) متوافقاُ مع اتجاه EMA200.\n"
            f"_النتيجة ستصلك كرسالة منفصلة فور إغلاق شمعة الخروج._"
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
# 9. الحلقة الرئيسية
# ═══════════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info(f"Starting {BOT_NAME} - GAZA TIME OPTIMIZED & FIXED RESOLUTION")
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
